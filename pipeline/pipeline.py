"""
pipeline/pipeline.py — Top-level orchestrator.

Wires together:
  - LLM client (vLLM in production, HF in dev)
  - Tool instances
  - Manifest loader
  - Executor
  - Stage functions (retrieval → first_triage → reasoning → cross_analysis → conclusion → final_conclusion)

Design rule: server.py calls pipeline.run() and knows nothing about variants.
All logic lives here and in the stages/tools/core modules.

Public API:
    Pipeline(model_name, llm_kwargs)   — create and configure
    Pipeline.run(variants, phenotype)  — execute the full pipeline, return report string

Per-variant flow:

  retrieval:         variants → list[str]  (one context slice per variant)
  first_triage:      per slice → KEEP/DISCARD  (skipped if n <= TRIAGE_ENABLED_THRESHOLD)
  reasoning:         kept slices → dict[int, str]  (one reasoning block per kept variant)
  cross_analysis:    per gene with ≥2 tier-1 variants → dict[gene, str]
  conclusion:        tier-1 slices + reasoning + cross_analysis → dict[int, str]
  final_conclusion:  tier-1 conclusions → str  (the # Clinical Conclusion paragraph)
"""

from __future__ import annotations

import re
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

MAX_WORKERS = 8

from pipeline.config          import (
    DEFAULT_GENOME_BUILD,
    TRIAGE_ENABLED_THRESHOLD,
)
from pipeline.core.manifest  import ManifestLoader
from pipeline.core.executor  import Executor
from pipeline.core.normalizer import TARGET_COLUMNS
from pipeline.llm.registry   import get_client
from pipeline.tools          import AutoPVS1Tool, LitVar2SummaryTool, SpliceAITool, WebSearchAgentTool
from pipeline.stages         import retrieval, reasoning, conclusion, cross_analysis, final_conclusion, first_triage

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
            Tier select — split kept variants into included/excluded via inclusion_decisions
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

            def _triage_one(i: int) -> tuple[int, str, str]:
                decision, justification = first_triage.run_one(
                    variant_context=context_slices[i],
                    patient_phenotype=patient_phenotype,
                    llm=self._llm,
                )
                return i, decision, justification

            workers = min(MAX_WORKERS, n)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_triage_one, i): i for i in range(n)}
                for future in as_completed(futures):
                    i, decision, justification = future.result()
                    gene = variants[i].get("Gene", "?")
                    if decision == "DISCARD" and i in multi_gene_indices:
                        logger.info(
                            "[Pipeline] Triage override: variant %d is compound-het "
                            "candidate in gene %s — forced KEEP", i + 1, gene,
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
        inclusion_decisions: dict[int, str] = {}
        second_triage_justifications: dict[int, str] = {}

        def _reason_one(i: int) -> tuple[int, str, str, str]:
            r = reasoning.run_one(variant_context=context_slices[i], llm=self._llm)
            decision, justification = reasoning.parse_inclusion_decision(r)
            return i, r, decision, justification

        workers = min(MAX_WORKERS, len(kept_indices))
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(_reason_one, i): i for i in kept_indices}
            for future in as_completed(futures):
                i, r, decision, justification = future.result()
                gene = variants[i].get("Gene", "?")
                reasonings[i] = r
                inclusion_decisions[i] = decision
                second_triage_justifications[i] = justification
                logger.info(
                    "[Pipeline] Variant %d (%s) second triage → %s", i + 1, gene, decision
                )

        # ── Inclusion split (direct from second_triage decisions) ────────────
        include_indices = [i for i in kept_indices if inclusion_decisions.get(i) == "INCLUDE"]
        exclude_indices = [i for i in kept_indices if inclusion_decisions.get(i) == "EXCLUDE"]

        logger.info(
            "[Pipeline] Second triage: %d included, %d excluded.",
            len(include_indices), len(exclude_indices),
        )

        # ── Stage 3.5: Gene-level cross-analysis (included variants only) ──────
        gene_groups_included: dict[str, list[int]] = defaultdict(list)
        for i in include_indices:
            gene = variants[i].get("Gene", "NA")
            if gene and gene != "NA":
                gene_groups_included[gene].append(i)

        cross_analyses: dict[str, str] = {}
        for gene, indices in gene_groups_included.items():
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

        # ── Stage 4: Per-variant conclusion (included variants only) ─────────
        conclusions: dict[int, str] = {}
        def _conclude_one(i: int) -> tuple[int, str]:
            gene = variants[i].get("Gene", "NA")
            ca   = cross_analyses.get(gene) if gene != "NA" else None
            return i, conclusion.run_one(
                variant_context=context_slices[i],
                reasoning=reasonings[i],
                cross_analysis=ca,
                llm=self._llm,
            )

        workers = min(MAX_WORKERS, len(include_indices))
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(_conclude_one, i): i for i in include_indices}
            for future in as_completed(futures):
                i, conc = future.result()
                conclusions[i] = conc
                logger.info(
                    "[Pipeline] Conclusion done for variant %d (%s).",
                    i + 1, variants[i].get("Gene", "?"),
                )

        # ── Stage 5: Final clinical conclusion (Tier 1 only) ─────────────────
        logger.info("[Pipeline] Generating final clinical conclusion...")
        final_summary = final_conclusion.run(
            conclusions=[conclusions[i] for i in include_indices],
            patient_phenotype=patient_phenotype,
            llm=self._llm,
        )

        # ── Assemble final output ─────────────────────────────────────────────

        # PROCESS DETAILS — all variants + triage summary + second triage summary
        triage_summary = _build_triage_summary(
            n=n,
            kept_indices=kept_indices,
            discarded_indices=discarded_indices,
            include_indices=include_indices,
            exclude_indices=exclude_indices,
            triage_results=triage_results,
            variants=variants,
            triage_ran=triage_ran,
        )
        second_triage_summary = _build_second_triage_summary(
            kept_indices=kept_indices,
            inclusion_decisions=inclusion_decisions,
            second_triage_justifications=second_triage_justifications,
            variants=variants,
        )
        process_parts = [process_details]
        if triage_summary:
            process_parts.append(triage_summary)
        process_parts.append(second_triage_summary)
        process_section = "\n\n".join(process_parts)

        # AUGMENTED CONTEXT — all variants (unchanged)
        full_context_display = _reconstruct_full_context(context_slices)

        # REASONING — all kept variants, in original variant order
        reasoning_parts = []
        for i in kept_indices:
            variant  = variants[i]
            gene     = variant.get("Gene", "?")
            hgvs     = variant.get("HGVS", variant.get("Variant", "?"))
            decision = inclusion_decisions.get(i, "INCLUDE")
            tier_tag = f"[{decision}]"
            reasoning_parts.append(
                f"{SEP_VARIANT}\nVARIANT {i + 1} — {gene}  {hgvs}  {tier_tag}\n"
                f"{SEP_VARIANT}\n{reasonings[i]}"
            )
        reasoning_display = "\n\n".join(reasoning_parts)

        # FINAL REPORT — included conclusions + final paragraph + appendix
        included_report_blocks = [conclusions[i] for i in include_indices]
        if included_report_blocks:
            final_report = "\n\n---\n\n".join(included_report_blocks) + "\n\n---\n\n" + final_summary
        else:
            final_report = final_summary

        if triage_ran and (exclude_indices or discarded_indices):
            final_report += _build_report_appendix(
                exclude_indices=exclude_indices,
                discarded_indices=discarded_indices,
                triage_results=triage_results,
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
    include_indices: list[int],
    exclude_indices: list[int],
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

    include_labels = ", ".join(f"Variant {i + 1}" for i in include_indices) or "(none)"
    exclude_labels = ", ".join(f"Variant {i + 1}" for i in exclude_indices) or "(none)"
    lines.append(f"\nIncluded (→ conclusion): {include_labels}")
    lines.append(f"Excluded (→ reasoning only): {exclude_labels}")

    return "\n".join(lines)


def _build_second_triage_summary(
    kept_indices: list[int],
    inclusion_decisions: dict[int, str],
    second_triage_justifications: dict[int, str],
    variants: list[dict],
) -> str:
    SEP = "─" * 56
    n_included = sum(1 for i in kept_indices if inclusion_decisions.get(i) == "INCLUDE")
    n_excluded  = sum(1 for i in kept_indices if inclusion_decisions.get(i) == "EXCLUDE")
    lines = [
        f"\n{SEP}",
        "SECOND TRIAGE (INCLUSION DECISIONS)",
        SEP,
        f"Evaluated : {len(kept_indices)}",
        f"Included  : {n_included}",
        f"Excluded  : {n_excluded}",
        "",
    ]
    for i in kept_indices:
        variant  = variants[i]
        gene     = variant.get("Gene", "?")
        hgvs     = variant.get("HGVS", variant.get("Variant", "?"))
        decision = inclusion_decisions.get(i, "INCLUDE")
        just     = second_triage_justifications.get(i, "")
        lines.append(f"  Variant {i + 1} — {gene}  {hgvs}  [{decision}]")
        if just:
            lines.append(f"    {just}")
    return "\n".join(lines)


def _build_report_appendix(
    exclude_indices: list[int],
    discarded_indices: list[int],
    triage_results: dict[int, tuple[str, str]],
    variants: list[dict],
) -> str:
    lines = ["\n\n---\n\n## Variants not included in conclusion"]

    if exclude_indices:
        lines.append("\n### Excluded — Reasoning available (see REASONING section)")
        for i in exclude_indices:
            variant = variants[i]
            gene    = variant.get("Gene", "?")
            hgvs    = variant.get("HGVS", variant.get("Variant", "?"))
            lines.append(f"- Variant {i + 1} — {gene} ({hgvs})")

    if discarded_indices:
        lines.append("\n### Discarded at first triage")
        for i in discarded_indices:
            variant = variants[i]
            gene    = variant.get("Gene", "?")
            hgvs    = variant.get("HGVS", variant.get("Variant", "?"))
            _, just = triage_results[i]
            lines.append(f"- Variant {i + 1} — {gene} ({hgvs}): {just}")

    return "\n".join(lines)
