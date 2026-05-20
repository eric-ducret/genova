"""
pipeline/pipeline.py — Top-level orchestrator.

Wires together:
  - LLM client (vLLM in production, HF in dev)
  - Tool instances
  - Manifest loader
  - Executor
  - Stage functions (retrieval → triage → reasoning → cross_analysis → conclusion → final_conclusion)

Design rule: server.py calls pipeline.run() and knows nothing about variants.
All logic lives here and in the stages/tools/core modules.

Public API:
    Pipeline(model_name, llm_kwargs)   — create and configure
    Pipeline.run(variants, phenotype)  — execute the full pipeline, return report string

Per-variant flow:

  retrieval:         variants → list[str]  (one context slice per variant)
  triage:            per slice → KEEP/DISCARD  (skipped if n <= TRIAGE_ENABLED_THRESHOLD)
  reasoning:         kept slices → dict[int, str]  (one reasoning block per kept variant)
  cross_analysis:    per gene with ≥2 tier-1 variants → dict[gene, str]
  conclusion:        tier-1 slices + reasoning + cross_analysis → dict[int, str]
  final_conclusion:  tier-1 conclusions → str  (the # Clinical Conclusion paragraph)
"""

from __future__ import annotations

import re
import logging
from collections import defaultdict
from typing import Optional

from pipeline.config          import (
    DEFAULT_GENOME_BUILD,
    MAX_CONCLUSION_VARIANTS,
    MIN_CONCLUSION_VARIANTS,
    TRIAGE_ENABLED_THRESHOLD,
)
from pipeline.core.manifest  import ManifestLoader
from pipeline.core.executor  import Executor
from pipeline.core.normalizer import TARGET_COLUMNS
from pipeline.llm.registry   import get_client
from pipeline.tools          import AutoPVS1Tool, LitVar2SummaryTool, SpliceAITool, WebSearchAgentTool
from pipeline.stages         import retrieval, reasoning, conclusion, cross_analysis, final_conclusion, triage

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# VARIANT STRING → DICT CONVERSION
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_variant_str(variant_str: str) -> dict:
    """
    Parse a canonical key=value variant string into a dict.

    The normalizer produces strings of the form:
        "Variant=chr6:100896130 T>C, Chromosome=chr6, Position=100896130, ..."

    We extract each TARGET_COLUMN by matching:
        <ColumnName>=<value>
    where value runs up to the next ", <NextColumnName>=" or end of string.

    This is more robust than a naive split on ", " because field values may
    themselves contain commas (e.g. OMIM_phenotype).
    """
    result: dict[str, str] = {}
    for col in TARGET_COLUMNS:
        # Pattern: "Col=<capture>" stopped at ", NextCol=" or end of string
        m = re.search(
            rf"(?:^|,\s*){re.escape(col)}=(.*?)(?=,\s*(?:{'|'.join(re.escape(c) for c in TARGET_COLUMNS)})=|$)",
            variant_str,
        )
        result[col] = m.group(1).strip() if m else "NA"
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

# Matches the shared PATIENT DATA … VARIANT DATA:\n\n header within each slice
_SLICE_HEADER_RE = re.compile(r"^(PATIENT DATA:.*?VARIANT DATA:\n\n)", re.DOTALL)


def _group_by_gene(variants: list[dict]) -> dict[str, list[int]]:
    """
    Return a mapping of gene symbol → list of 0-based variant indices.

    Variants with Gene == "NA" or missing Gene are excluded (they cannot
    participate in gene-level cross-analysis).
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for i, variant in enumerate(variants):
        gene = variant.get("Gene", "NA")
        if gene and gene != "NA":
            groups[gene].append(i)
    return dict(groups)


def _reconstruct_full_context(context_slices: list[str]) -> str:
    """
    Reconstruct the full augmented context string from per-variant slices.

    Used only for display in the AUGMENTED CONTEXT section of the final output.
    Each slice contains a PATIENT DATA header + one VARIANT block; this helper
    emits the header once, then concatenates the VARIANT blocks from all slices.
    """
    if not context_slices:
        return ""

    m = _SLICE_HEADER_RE.match(context_slices[0])
    if not m:
        # Fallback: concatenate slices separated by a blank line
        return "\n\n".join(s.rstrip() for s in context_slices) + "\n"

    header     = m.group(1)
    header_len = len(header)

    variant_sections = [s[header_len:].rstrip() for s in context_slices]
    return header + "\n\n".join(variant_sections) + "\n"


def _build_process_details(
    process_log: list[dict],
    variants: list[dict],
) -> str:
    """
    Build the PROCESS DETAILS section from the executor's process log.
    One block per variant, one entry per tool.
    """
    SEP_VARIANT = "─" * 56
    lines = []

    by_variant: dict[int, list[dict]] = defaultdict(list)
    for entry in process_log:
        by_variant[entry["variant_index"]].append(entry)

    for i, variant in enumerate(variants):
        gene  = variant.get("Gene", "?")
        hgvs  = variant.get("HGVS", variant.get("Variant", "?"))
        lines.append(f"\n{SEP_VARIANT}")
        lines.append(f"VARIANT {i + 1} — {gene}  {hgvs}")
        lines.append(SEP_VARIANT)

        entries = by_variant.get(i, [])
        if not entries:
            lines.append("  (no tool entries recorded)")
            continue

        for entry in entries:
            tool_name = entry["tool_name"].upper()
            gate      = entry["gate"]

            if gate == "SKIP":
                reason = entry.get("gate_reason") or ""
                lines.append(f"\n[{tool_name}]  GATE: SKIPPED  ({reason})")
                continue

            lines.append(f"\n[{tool_name}]  GATE: PASS")

            metadata = entry.get("metadata") or {}

            # ── LitVar2 metadata ────────────────────────────────────────
            if "query" in metadata:
                query   = metadata.get("query") or "?"
                n_found = metadata.get("n_found") or 0
                kept    = metadata.get("kept") or []
                lines.append(f"  Query     : {query}")
                lines.append(f"  Articles  : {len(kept)} kept from {n_found} found")
                if kept:
                    lines.append("  Kept articles:")
                    for art in kept:
                        lines.append(f"    [PMID:{art['pmid']}] {art['title']}")

            # ── WebSearch trace ──────────────────────────────────────────
            if "trace" in metadata:
                trace = metadata.get("trace") or []
                if trace:
                    lines.append("  ReAct trace:")
                    for step in trace:
                        lines.append(f"    {step}")

            # ── Raw output ───────────────────────────────────────────────
            raw = entry.get("raw_output")
            if raw:
                lines.append("  Raw output:")
                for raw_line in raw.splitlines():
                    lines.append(f"    {raw_line}")
            else:
                lines.append("  Output: (none)")

    return "\n".join(lines)


def _select_tier1_indices(
    kept_indices: list[int],
    relevance_scores: dict[int, int],
    gene_groups: dict[str, list[int]],
    max_variants: int,
    min_variants: int,
) -> list[int]:
    """
    Select which kept variants go to Tier 1 (conclusion).

    Rules:
    - Same-gene groups are atomic: the group score = max score of its members.
      The whole group moves together.
    - Take all groups with score == 3 first.
    - If total variants selected < min_variants, backfill with score == 2 groups.
    - Hard cap at max_variants total variants selected.
    - Singletons (no gene partner among kept variants) are treated as
      single-member groups.

    Returns list of 0-based variant indices that are Tier 1.
    """
    kept_set = set(kept_indices)

    # Build index → gene group members (only among kept_indices)
    # A variant whose gene appears only once among kept_indices is its own group.
    index_to_group: dict[int, frozenset[int]] = {}
    for gene, idxs in gene_groups.items():
        members = frozenset(i for i in idxs if i in kept_set)
        if members:
            for i in members:
                index_to_group[i] = members

    # Singletons: kept variants not covered by any gene group entry
    for i in kept_indices:
        if i not in index_to_group:
            index_to_group[i] = frozenset([i])

    # Deduplicate groups
    seen_groups: set[frozenset[int]] = set()
    unique_groups: list[frozenset[int]] = []
    for i in kept_indices:
        grp = index_to_group[i]
        if grp not in seen_groups:
            seen_groups.add(grp)
            unique_groups.append(grp)

    # Assign each group a score = max relevance score among its members
    def group_score(grp: frozenset[int]) -> int:
        return max(relevance_scores.get(i, 2) for i in grp)

    # Sort: score=3 first, then score=2, then score=1
    unique_groups.sort(key=group_score, reverse=True)

    # Greedily select groups
    selected: list[int] = []
    for grp in unique_groups:
        score = group_score(grp)
        if len(selected) >= max_variants:
            break
        # Always take score-3; backfill score-2 if below min_variants
        if score == 3 or (score == 2 and len(selected) < min_variants):
            new_members = sorted(grp)
            if len(selected) + len(new_members) <= max_variants:
                selected.extend(new_members)
            else:
                # Partial group addition would break atomicity — skip this group
                remaining = max_variants - len(selected)
                if remaining >= len(new_members):
                    selected.extend(new_members)
                # else: skip the whole group to preserve atomicity

    return sorted(set(selected))


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

class Pipeline:
    """
    Full variant analysis pipeline.

    Instantiate once at server startup; call run() per request.

    Args:
        model_name: Key in pipeline.llm.registry.MODELS (default "qwen3.5-9b").
        llm_kwargs: Forwarded to get_client() — e.g. base_url for VLLMClient.
    """

    def __init__(
        self,
        model_name: str = "qwen3.5-9b",
        llm_kwargs: Optional[dict] = None,
        llm=None,                              # pre-built LLMClient (overrides model_name)
    ):
        self._model_name = model_name
        self._llm_kwargs = llm_kwargs or {}

        # ── LLM client (loaded once at startup) ──────────────────────────────
        if llm is not None:
            logger.info("[Pipeline] Using pre-built LLM client: %s", type(llm).__name__)
            self._llm = llm
        else:
            logger.info("[Pipeline] Loading LLM client: %s", model_name)
            self._llm = get_client(model_name, **self._llm_kwargs)

        # ── Manifests ─────────────────────────────────────────────────────────
        loader = ManifestLoader()
        self._manifests = loader.load_all()
        logger.info(
            "[Pipeline] Loaded %d manifests: %s",
            len(self._manifests),
            [m["name"] for m in self._manifests],
        )

        # ── Tool instances (one per pipeline tool) ────────────────────────────
        self._tools = [
            LitVar2SummaryTool(),
            SpliceAITool(),
            AutoPVS1Tool(),
            WebSearchAgentTool(),
        ]

        # ── Executor ──────────────────────────────────────────────────────────
        self._executor = Executor(
            llm=self._llm,
            tools=self._tools,
            manifests=self._manifests,
        )
        logger.info("[Pipeline] Executor ready with %d tools.", len(self._executor._paired))

    # ── public ────────────────────────────────────────────────────────────────

    async def run(
        self,
        variants: list[dict],
        patient_phenotype: str,
        genome_build: str = DEFAULT_GENOME_BUILD,
        raw_rows: list[dict] | None = None,
    ) -> str:
        """
        Execute the full per-variant pipeline and return the combined output.

        Pipeline flow:
            Stage 1   — retrieval:        variants → list[str] (one context slice per variant)
            Triage    — (if n > threshold) kept/discarded per variant
            Stage 2   — reasoning:        kept slices → dict[int, str]
            Tier select — parse RELEVANCE scores, pick Tier 1 via _select_tier1_indices
            Stage 3.5 — cross_analysis:   per gene with ≥2 tier-1 variants → dict[gene, str]
            Stage 4   — conclusion:        tier-1 variants → dict[int, str]
            Stage 5   — final_conclusion:  tier-1 conclusions → str

        When n <= TRIAGE_ENABLED_THRESHOLD the pipeline behaves identically to
        the no-triage path: all variants go through all stages, and the
        Tier 2/Tier 3 appendix sections are omitted from the output.

        Args:
            variants:          List of normalized variant dicts from normalize_upload().
            patient_phenotype: Free-text patient phenotype string from the request.
            genome_build:      Reference genome build for this run ("hg19" or "hg38").
            raw_rows:          Full original CSV rows, one dict per variant.

        Returns:
            Combined output string with PROCESS DETAILS, AUGMENTED CONTEXT,
            REASONING, and FINAL REPORT sections separated by === banners.

        Raises:
            ValueError if variants is empty.
        """
        if not variants:
            raise ValueError("variants list is empty — nothing to analyze.")

        n = len(variants)
        logger.info("[Pipeline] Processing %d variants.", n)

        SEP         = "=" * 60
        SEP_VARIANT = "─" * 56

        # ── Stage 1: Retrieval ────────────────────────────────────────────────
        context_slices: list[str] = retrieval.run(
            variants=variants,
            patient_phenotype=patient_phenotype,
            executor=self._executor,
            genome_build=genome_build,
            raw_rows=raw_rows,
        )

        process_details = _build_process_details(self._executor.process_log, variants)

        # ── Triage ────────────────────────────────────────────────────────────
        # triage_results maps original variant index → ("KEEP"|"DISCARD", justification)
        triage_results: dict[int, tuple[str, str]] = {}
        triage_ran = n > TRIAGE_ENABLED_THRESHOLD

        if triage_ran:
            # Identify multi-variant genes (compound het candidates) — exempt from discard
            gene_groups_all = _group_by_gene(variants)
            multi_gene_indices: set[int] = set()
            for gene, idxs in gene_groups_all.items():
                if len(idxs) >= 2:
                    multi_gene_indices.update(idxs)

            for i, (variant_ctx, variant) in enumerate(zip(context_slices, variants)):
                gene = variant.get("Gene", "?")
                hgvs = variant.get("HGVS", variant.get("Variant", "?"))
                logger.info(
                    "[Pipeline] Triage %d/%d — %s %s...", i + 1, n, gene, hgvs
                )
                decision, justification = triage.run_one(
                    variant_context=variant_ctx,
                    patient_phenotype=patient_phenotype,
                    llm=self._llm,
                )
                # Compound het exemption: override DISCARD → KEEP
                if decision == "DISCARD" and i in multi_gene_indices:
                    logger.info(
                        "[Pipeline] Triage override: variant %d is compound-het candidate "
                        "in gene %s — forced KEEP",
                        i + 1, gene,
                    )
                    justification = (
                        f"[compound-het exempt — original triage: DISCARD: {justification}]"
                    )
                    decision = "KEEP"
                triage_results[i] = (decision, justification)
        else:
            # No triage — all variants kept
            for i in range(n):
                triage_results[i] = ("KEEP", "")

        kept_indices     = [i for i, (dec, _) in triage_results.items() if dec == "KEEP"]
        discarded_indices = [i for i, (dec, _) in triage_results.items() if dec == "DISCARD"]
        logger.info(
            "[Pipeline] Triage complete: %d kept, %d discarded.",
            len(kept_indices), len(discarded_indices),
        )

        # ── Stage 2: Reasoning (only on kept variants) ───────────────────────
        reasonings: dict[int, str] = {}
        relevance_scores: dict[int, int] = {}

        for pos, i in enumerate(kept_indices):
            variant = variants[i]
            gene    = variant.get("Gene", "?")
            logger.info(
                "[Pipeline] Reasoning %d/%d (variant idx %d) — %s...",
                pos + 1, len(kept_indices), i, gene,
            )
            r = reasoning.run_one(variant_context=context_slices[i], llm=self._llm)
            reasonings[i]       = r
            relevance_scores[i] = reasoning.parse_relevance_score(r)

        # ── Tier 1 selection ──────────────────────────────────────────────────
        # Build gene groups restricted to kept variants, using original indices
        gene_groups_kept_orig: dict[str, list[int]] = defaultdict(list)
        for i in kept_indices:
            gene = variants[i].get("Gene", "NA")
            if gene and gene != "NA":
                gene_groups_kept_orig[gene].append(i)

        tier1_indices = _select_tier1_indices(
            kept_indices=kept_indices,
            relevance_scores=relevance_scores,
            gene_groups=dict(gene_groups_kept_orig),
            max_variants=MAX_CONCLUSION_VARIANTS,
            min_variants=MIN_CONCLUSION_VARIANTS,
        )
        tier1_set    = set(tier1_indices)
        tier2_indices = [i for i in kept_indices if i not in tier1_set]

        logger.info(
            "[Pipeline] Tier split: %d Tier 1, %d Tier 2.",
            len(tier1_indices), len(tier2_indices),
        )

        # ── Stage 3.5: Gene-level cross-analysis (Tier 1 only) ───────────────
        gene_groups_tier1: dict[str, list[int]] = defaultdict(list)
        for i in tier1_indices:
            gene = variants[i].get("Gene", "NA")
            if gene and gene != "NA":
                gene_groups_tier1[gene].append(i)

        cross_analyses: dict[str, str] = {}
        for gene, indices in gene_groups_tier1.items():
            if len(indices) >= 2:
                logger.info(
                    "[Pipeline] Cross-analysis for gene %s (%d variants)...",
                    gene, len(indices),
                )
                cross_analyses[gene] = cross_analysis.run(
                    gene=gene,
                    variant_contexts=[context_slices[j] for j in indices],
                    reasonings=[reasonings[j] for j in indices],
                    llm=self._llm,
                )

        # ── Stage 4: Per-variant conclusion (Tier 1 only) ────────────────────
        conclusions: dict[int, str] = {}
        for pos, i in enumerate(tier1_indices):
            variant = variants[i]
            gene    = variant.get("Gene", "NA")
            ca      = cross_analyses.get(gene) if gene != "NA" else None
            logger.info(
                "[Pipeline] Conclusion %d/%d (variant idx %d) — %s...",
                pos + 1, len(tier1_indices), i, gene,
            )
            conclusions[i] = conclusion.run_one(
                variant_context=context_slices[i],
                reasoning=reasonings[i],
                cross_analysis=ca,
                llm=self._llm,
            )

        # ── Stage 5: Final clinical conclusion (Tier 1 only) ─────────────────
        logger.info("[Pipeline] Generating final clinical conclusion...")
        final_summary = final_conclusion.run(
            conclusions=[conclusions[i] for i in tier1_indices],
            patient_phenotype=patient_phenotype,
            llm=self._llm,
        )

        # ── Assemble final output ─────────────────────────────────────────────

        # PROCESS DETAILS — all variants + triage summary block
        triage_summary = _build_triage_summary(
            n=n,
            kept_indices=kept_indices,
            discarded_indices=discarded_indices,
            tier1_indices=tier1_indices,
            tier2_indices=tier2_indices,
            triage_results=triage_results,
            variants=variants,
            triage_ran=triage_ran,
        )
        process_section = process_details + "\n\n" + triage_summary

        # AUGMENTED CONTEXT — all variants (unchanged)
        full_context_display = _reconstruct_full_context(context_slices)

        # REASONING — Tier 1 + Tier 2 only, in original variant order
        reasoning_parts = []
        for i in kept_indices:
            variant = variants[i]
            gene    = variant.get("Gene", "?")
            hgvs    = variant.get("HGVS", variant.get("Variant", "?"))
            tier_tag = "[TIER 1]" if i in tier1_set else "[TIER 2]"
            reasoning_parts.append(
                f"{SEP_VARIANT}\nVARIANT {i + 1} — {gene}  {hgvs}  {tier_tag}\n"
                f"{SEP_VARIANT}\n{reasonings[i]}"
            )
        reasoning_display = "\n\n".join(reasoning_parts)

        # FINAL REPORT — Tier 1 conclusions + final paragraph + appendix
        tier1_report_blocks = [conclusions[i] for i in tier1_indices]
        final_report = "\n\n---\n\n".join(tier1_report_blocks) + "\n\n---\n\n" + final_summary

        if triage_ran and (tier2_indices or discarded_indices):
            final_report += _build_report_appendix(
                tier2_indices=tier2_indices,
                discarded_indices=discarded_indices,
                triage_results=triage_results,
                relevance_scores=relevance_scores,
                variants=variants,
            )

        output = (
            f"{SEP}\nPROCESS DETAILS\n{SEP}\n"
            + process_section + "\n\n"
            + f"{SEP}\nAUGMENTED CONTEXT\n{SEP}\n"
            + full_context_display + "\n\n"
            + f"{SEP}\nREASONING\n{SEP}\n"
            + reasoning_display + "\n\n"
            + f"{SEP}\nFINAL REPORT\n{SEP}\n"
            + final_report + "\n"
        )
        return output


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT ASSEMBLY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _build_triage_summary(
    n: int,
    kept_indices: list[int],
    discarded_indices: list[int],
    tier1_indices: list[int],
    tier2_indices: list[int],
    triage_results: dict[int, tuple[str, str]],
    variants: list[dict],
    triage_ran: bool,
) -> str:
    if not triage_ran:
        return ""

    SEP = "─" * 56
    lines = [
        f"\n{SEP}",
        "TRIAGE SUMMARY",
        SEP,
        f"Total variants  : {n}",
        f"Kept (Tier 1+2) : {len(kept_indices)}",
        f"Discarded       : {len(discarded_indices)}",
    ]

    if discarded_indices:
        lines.append("\nDiscarded variants:")
        for i in discarded_indices:
            variant = variants[i]
            gene    = variant.get("Gene", "?")
            hgvs    = variant.get("HGVS", variant.get("Variant", "?"))
            _, just = triage_results[i]
            lines.append(f"  Variant {i + 1} — {gene} ({hgvs}): {just}")

    # Compound-het exemptions: kept variants whose justification records the override
    exemptions = [
        i for i in kept_indices
        if triage_results[i][1].startswith("[compound-het exempt")
    ]
    if exemptions:
        lines.append("\nCompound-het exemptions (forced KEEP):")
        for i in exemptions:
            variant = variants[i]
            gene    = variant.get("Gene", "?")
            hgvs    = variant.get("HGVS", variant.get("Variant", "?"))
            lines.append(f"  Variant {i + 1} — {gene} ({hgvs})")

    tier1_labels = ", ".join(f"Variant {i + 1}" for i in tier1_indices) or "(none)"
    tier2_labels = ", ".join(f"Variant {i + 1}" for i in tier2_indices) or "(none)"
    lines.append(f"\nTier 1 (→ conclusion): {tier1_labels}")
    lines.append(f"Tier 2 (→ reasoning only): {tier2_labels}")

    return "\n".join(lines)


def _build_report_appendix(
    tier2_indices: list[int],
    discarded_indices: list[int],
    triage_results: dict[int, tuple[str, str]],
    relevance_scores: dict[int, int],
    variants: list[dict],
) -> str:
    lines = ["\n\n---\n\n## Variants not included in conclusion"]

    if tier2_indices:
        lines.append("\n### Tier 2 — Reasoning available (see REASONING section)")
        for i in tier2_indices:
            variant = variants[i]
            gene    = variant.get("Gene", "?")
            hgvs    = variant.get("HGVS", variant.get("Variant", "?"))
            score   = relevance_scores.get(i, 2)
            lines.append(f"- Variant {i + 1} — {gene} ({hgvs})  [RELEVANCE: {score}]")

    if discarded_indices:
        lines.append("\n### Tier 3 — Discarded at triage")
        for i in discarded_indices:
            variant = variants[i]
            gene    = variant.get("Gene", "?")
            hgvs    = variant.get("HGVS", variant.get("Variant", "?"))
            _, just = triage_results[i]
            lines.append(f"- Variant {i + 1} — {gene} ({hgvs}): {just}")

    return "\n".join(lines)
