"""
pipeline/pipeline.py — Top-level orchestrator.

Wires together:
  - LLM client (vLLM in production, HF in dev)
  - Tool instances
  - Manifest loader
  - Executor
  - Stage functions (retrieval → first_triage → reasoning → cross_analysis →
    conclusion → MOI layers → final_conclusion → actionable)

Design rule: server.py calls pipeline.run() and knows nothing about variants.
All logic lives here and in the stages/tools/core modules.

Public API:
    Pipeline(model_name, llm_kwargs)   — create and configure
    Pipeline.run(variants, phenotype)  — execute the full pipeline, return report string

Per-variant flow:

  retrieval:          variants → list[str]  (one context slice per variant)
  ACMG-SF detection:   all variants → actionable_indices  (independent of triage)
  first_triage:        per slice → KEEP/DISCARD  (skipped if n <= TRIAGE_ENABLED_THRESHOLD;
                        DISCARD overridden to KEEP for actionable/compound-het candidates)
  MOI precompute:      kept variants → gene_mode_cache, recessive_gene_groups, pertinence_cache
  reasoning (Call 1):  kept slices (+ sibling/inheritance-mode context) → dict[int, str]
  second_triage (Call 2): reasoning + sibling context → INCLUDE/EXCLUDE
  cross_analysis:      per gene with ≥2 included variants → dict[gene, str]
  conclusion (Layer 2, base ACMG): kept-minus-failed slices → dict[int, str]
  MOI layers (3-6):    included variants, gated by inheritance mode/chromosome →
                        add mode-of-inheritance-specific ACMG criteria on top of base score
  final_conclusion:    cross-MOI synthesis (two-pass draft+revise) → str
  actionable:          ACMG-SF-flagged variants → str
"""

from __future__ import annotations

import itertools
import re
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

MAX_WORKERS          = 32   # retrieval/first_triage (network tools, I/O bound)
MAX_WORKERS_LLM       = 16   # reasoning/conclusion/MOI layers (vLLM calls, GPU-bound)

from pipeline.config          import (
    DEFAULT_GENOME_BUILD,
    TRIAGE_ENABLED_THRESHOLD,
    FORCE_GENERIC_PHENOTYPE,
    GENERIC_PHENOTYPE_TEXT,
)
from pipeline.core.manifest   import ManifestLoader
from pipeline.core.executor   import Executor
from pipeline.core.normalizer import TARGET_COLUMNS, normalize_gene
from pipeline.core import moi, segregation, acmg_sf
from pipeline.llm.registry    import get_client
from pipeline.tools           import (
    AutoPVS1Tool, LitVar2SummaryTool, SpliceAITool, WebSearchAgentTool,
    GnomadConstraintTool, ClinVarGeneStatsTool, ClinGenAlleleTool,
)
from pipeline.tools.gnomad_constraint    import classify_pli, classify_loeuf
from pipeline.tools.clinvar_gene_stats   import classify_consequence_counts
from pipeline.stages import (
    retrieval, reasoning, conclusion, cross_analysis, final_conclusion, first_triage,
    moi_denovo, moi_dominant, moi_recessive, moi_xlinked, actionable,
)

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
    participate in gene-level cross-analysis or MOI grouping). Gene symbols
    are normalized via normalize_gene() so a multi-gene CSV field (e.g.
    "BRCA1;BRCA2") groups under its first listed gene rather than forming
    its own disconnected singleton group.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for i, variant in enumerate(variants):
        gene = normalize_gene(variant.get("Gene", "NA"))
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


# ── ACMG points / classification parsing (SLM writes the arithmetic; Python
#    only reads it back for sorting/ranking — never recomputes it) ──────────

# Matches "ACMG points: N → Classification" or "Total ACMG points: N → Classification",
# tolerating markdown bold markers and a fractional point value (e.g. PM3 solo +0.5).
_ACMG_LINE_RE = re.compile(
    r"(?:Total\s+)?ACMG\s+points:?\*{0,2}\s*(-?\d+(?:\.\d+)?)\s*\*{0,2}\s*→\s*\*{0,2}\s*([A-Za-z /().-]+)",
    re.IGNORECASE,
)

_CLASSIFICATION_RANK_UNKNOWN = 5


def _extract_acmg_points_value(text: str) -> float | None:
    m = _ACMG_LINE_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _extract_classification_label(text: str) -> str | None:
    m = _ACMG_LINE_RE.search(text)
    if not m:
        return None
    return m.group(2).strip().rstrip("*").strip()


def _classification_rank(label: str | None) -> int:
    """Sort key so report sections read Pathogenic → Likely Pathogenic → VUS →
    Likely Benign → Benign → unknown. Checked longest/most-specific substring
    first since "likely pathogenic" itself contains "pathogenic"."""
    if not label:
        return _CLASSIFICATION_RANK_UNKNOWN
    low = label.strip().lower()
    if "likely pathogenic" in low:
        return 1
    if "pathogenic" in low:
        return 0
    if "uncertain" in low or low == "vus":
        return 2
    if "likely benign" in low:
        return 3
    if "benign" in low:
        return 4
    return _CLASSIFICATION_RANK_UNKNOWN


def _sort_blocks_by_classification(blocks: list[str]) -> list[str]:
    return sorted(blocks, key=lambda b: _classification_rank(_extract_classification_label(b)))


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
            GnomadConstraintTool(),
            ClinVarGeneStatsTool(),
            ClinGenAlleleTool(),
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
        parental_ab: list[dict] | None = None,
        header_mapping_summary: str | None = None,
    ) -> str:
        """
        Execute the full per-variant pipeline and return the combined output.

        Args:
            variants:          List of normalized variant dicts from normalize_upload().
            patient_phenotype: Free-text patient phenotype string from the request.
            genome_build:      Reference genome build for this run ("hg19" or "hg38").
            raw_rows:          Full original CSV rows, one dict per variant.
            parental_ab:       Per-variant allelic balance dicts from normalize_upload()
                               (proband/mother/father), or None if no trio data.
            header_mapping_summary: Human-readable SLM column-mapping summary from
                               normalize_upload(), surfaced at the top of the report.

        Returns:
            Combined output string with (optional) COLUMN HEADER INTERPRETATION,
            PROCESS DETAILS, (optional) SEGREGATION ANALYSIS, AUGMENTED CONTEXT,
            REASONING, and FINAL REPORT sections separated by === banners.

        Raises:
            ValueError if variants is empty.
        """
        if not variants:
            raise ValueError("variants list is empty — nothing to analyze.")

        if FORCE_GENERIC_PHENOTYPE:
            logger.info(
                "[Pipeline] FORCE_GENERIC_PHENOTYPE active — overriding phenotype "
                "with generic placeholder for this run."
            )
            patient_phenotype = GENERIC_PHENOTYPE_TEXT

        # Reset per-run state: _disease_query is derived from patient_phenotype and
        # must not bleed across run() calls when the same Pipeline instance is reused.
        for _tool in self._tools:
            if hasattr(_tool, "_disease_query"):
                _tool._disease_query = None

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
            parental_ab=parental_ab,
        )

        process_details = _build_process_details(self._executor.process_log, variants)

        # ── ACMG-SF actionable detection (all variants, independent of triage) ─
        litvar2_raw_by_variant: dict[int, str | None] = {}
        for entry in self._executor.process_log:
            if entry.get("tool_name") == "litvar2_summary" and entry.get("gate") == "PASS":
                litvar2_raw_by_variant[entry["variant_index"]] = entry.get("raw_output")

        actionable_indices, actionable_reasons = acmg_sf.build_actionable_set(
            variants, litvar2_raw_by_variant, self._llm
        )
        if actionable_indices:
            logger.info(
                "[Pipeline] ACMG-SF actionable findings: %d variant(s) — %s",
                len(actionable_indices),
                sorted(i + 1 for i in actionable_indices),
            )

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
                    if decision == "DISCARD" and i in actionable_indices:
                        logger.info(
                            "[Pipeline] Triage override: variant %d is an ACMG-SF "
                            "actionable finding — forced KEEP", i + 1,
                        )
                        justification = (
                            f"[ACMG-SF actionable exempt — original triage: DISCARD: {justification}]"
                        )
                        decision = "KEEP"
                    elif decision == "DISCARD" and i in multi_gene_indices:
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

        # ── MOI precompute (once globally, before reasoning) ──────────────────
        gene_mode_cache = moi.build_gene_mode_cache(
            variants, kept_indices, context_slices, _group_by_gene
        )
        recessive_gene_groups = moi.build_recessive_gene_groups(
            variants, gene_mode_cache, kept_indices, _group_by_gene
        )
        pertinence_cache = moi.build_pertinence_cache(variants, kept_indices, context_slices)

        # ── Context-block builders (sibling / inheritance-mode / zygosity) ────

        def _inheritance_mode_block(i: int) -> str:
            gene = normalize_gene(variants[i].get("Gene", "NA"))
            mode = gene_mode_cache.get(gene, "")
            pertinence = pertinence_cache.get(i)
            return (
                f"\nGENE INHERITANCE MODE (backend-determined): {moi.MODE_LABELS.get(mode, moi.MODE_LABELS[''])}"
                f"\nGENE-PHENOTYPE LITERATURE PERTINENCE (backend-determined): "
                f"{moi.PERTINENCE_LABELS.get(pertinence, moi.PERTINENCE_LABELS[None])}"
            )

        def _pertinence_block(i: int) -> str:
            pertinence = pertinence_cache.get(i)
            return (
                f"\nGENE-PHENOTYPE LITERATURE PERTINENCE (backend-determined): "
                f"{moi.PERTINENCE_LABELS.get(pertinence, moi.PERTINENCE_LABELS[None])}"
            )

        def _zygosity_note_block(i: int) -> str:
            if not parental_ab or i >= len(parental_ab) or not parental_ab[i]:
                return ""
            proband_ab = parental_ab[i].get("proband")
            if proband_ab is None:
                return ""
            ab_class = segregation.classify_ab_ratio(proband_ab)
            return (
                f"\nPROBAND ZYGOSITY FROM ALLELIC BALANCE (backend-determined): "
                f"AB={proband_ab} → {ab_class}"
            )

        def _sibling_evidence_block(i: int) -> str:
            """Sibling context built from raw evidence, for reasoning Call 1
            (no reasoning exists yet at that point). A confirmed-homozygous
            variant already explains a recessive disease on its own and
            doesn't need sibling context."""
            if moi.zygosity_is_confirmed_hom(variants[i].get("Zygosity", "NA")):
                return ""
            gene = normalize_gene(variants[i].get("Gene", "NA"))
            group = recessive_gene_groups.get(gene)
            if not group or i not in group:
                return ""
            siblings = [j for j in group if j != i]
            if not siblings:
                return ""
            parts = ["\nSIBLING VARIANT(S) IN SAME GENE (for compound-het/homozygous recessive evaluation):"]
            for j in siblings:
                hgvs = variants[j].get("HGVS", variants[j].get("Variant", "?"))
                zyg  = variants[j].get("Zygosity", "NA")
                parts.append(f"\n--- Sibling: Variant {j + 1} — {gene} ({hgvs}), Zygosity={zyg} ---\n{context_slices[j]}")
            return "\n".join(parts)

        def _sibling_reasoning_block(i: int, reasoning_only: dict[int, str]) -> str:
            """Sibling context built from sibling reasoning (Call 1 output),
            for second_triage (Call 2) — richer than raw evidence since it
            includes the sibling's own step-by-step analysis."""
            if moi.zygosity_is_confirmed_hom(variants[i].get("Zygosity", "NA")):
                return ""
            gene = normalize_gene(variants[i].get("Gene", "NA"))
            group = recessive_gene_groups.get(gene)
            if not group or i not in group:
                return ""
            siblings = [j for j in group if j != i and j in reasoning_only]
            if not siblings:
                return ""
            parts = ["\nSIBLING VARIANT(S) IN SAME GENE — reasoning already produced:"]
            for j in siblings:
                hgvs = variants[j].get("HGVS", variants[j].get("Variant", "?"))
                parts.append(f"\n--- Sibling: Variant {j + 1} — {gene} ({hgvs}) reasoning ---\n{reasoning_only[j]}")
            return "\n".join(parts)

        # ── Stage 2a: Reasoning Call 1 (all kept variants) ────────────────────
        reasoning_only: dict[int, str] = {}
        reasoning_failed: set[int] = set()

        def _reason_one(i: int) -> tuple[int, str]:
            r = reasoning.run_reasoning(
                variant_context=context_slices[i],
                llm=self._llm,
                sibling_context_block=_sibling_evidence_block(i),
                inheritance_mode_block=_inheritance_mode_block(i),
            )
            return i, r

        workers = min(MAX_WORKERS_LLM, len(kept_indices))
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(_reason_one, i): i for i in kept_indices}
            for future in as_completed(futures):
                i = futures[future]
                gene = variants[i].get("Gene", "?")
                try:
                    _, r = future.result()
                except Exception as exc:
                    logger.error(
                        "[Pipeline] Reasoning Call 1 failed for variant %d (%s): %s",
                        i + 1, gene, exc,
                    )
                    reasoning_only[i] = f"[REASONING FAILED: {exc}]"
                    reasoning_failed.add(i)
                    continue
                reasoning_only[i] = r

        # ── Stage 2b: Second triage Call 2 (kept, non-failed variants) ────────
        reasonings: dict[int, str] = {}
        inclusion_decisions: dict[int, str] = {}
        second_triage_justifications: dict[int, str] = {}

        second_triage_targets = [i for i in kept_indices if i not in reasoning_failed]

        def _second_triage_one(i: int) -> tuple[int, str, str, str]:
            combined = reasoning.run_second_triage(
                variant_context=context_slices[i],
                reasoning_text=reasoning_only[i],
                llm=self._llm,
                sibling_context_block=_sibling_reasoning_block(i, reasoning_only),
            )
            decision, justification = reasoning.parse_inclusion_decision(combined)
            return i, combined, decision, justification

        workers = min(MAX_WORKERS_LLM, len(second_triage_targets))
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(_second_triage_one, i): i for i in second_triage_targets}
            for future in as_completed(futures):
                i = futures[future]
                gene = variants[i].get("Gene", "?")
                try:
                    _, combined, decision, justification = future.result()
                except Exception as exc:
                    logger.error(
                        "[Pipeline] Second triage failed for variant %d (%s): %s — marking EXCLUDE",
                        i + 1, gene, exc,
                    )
                    reasonings[i] = reasoning_only[i] + f"\n\n[SECOND TRIAGE FAILED: {exc}]"
                    inclusion_decisions[i] = "EXCLUDE"
                    second_triage_justifications[i] = f"second triage error: {exc}"
                    continue
                reasonings[i] = combined
                inclusion_decisions[i] = decision
                second_triage_justifications[i] = justification
                logger.info(
                    "[Pipeline] Variant %d (%s) second triage → %s", i + 1, gene, decision
                )

        for i in reasoning_failed:
            reasonings[i] = reasoning_only[i]
            inclusion_decisions[i] = "EXCLUDE"
            second_triage_justifications[i] = "reasoning Call 1 failed — excluded"

        # ── ACMG-SF second-triage override ────────────────────────────────────
        for i in kept_indices:
            if i in actionable_indices and inclusion_decisions.get(i) != "INCLUDE":
                logger.info(
                    "[Pipeline] ACMG-SF override: variant %d forced INCLUDE (actionable finding)",
                    i + 1,
                )
                prior = second_triage_justifications.get(i, "")
                second_triage_justifications[i] = (
                    f"[ACMG-SF actionable — forced INCLUDE: {actionable_reasons.get(i, '')}] {prior}".strip()
                )
                inclusion_decisions[i] = "INCLUDE"

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
            gene = normalize_gene(variants[i].get("Gene", "NA"))
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

        # ── Stage 4: Conclusion / base ACMG score (Layer 2) ───────────────────
        # Runs for ALL kept variants minus reasoning failures — not just included
        # ones — so excluded variants still carry a documented base score for
        # the report appendix.
        conclusion_targets = [i for i in kept_indices if i not in reasoning_failed]
        conclusions: dict[int, str] = {}
        base_acmg_points: dict[int, float | None] = {}

        def _conclude_one(i: int) -> tuple[int, str]:
            gene = normalize_gene(variants[i].get("Gene", "NA"))
            ca   = cross_analyses.get(gene) if gene != "NA" else None
            reasoning_text = reasonings.get(i, reasoning_only.get(i, "")) + _pertinence_block(i) + _zygosity_note_block(i)
            return i, conclusion.run_one(
                variant_context=context_slices[i],
                reasoning=reasoning_text,
                cross_analysis=ca,
                llm=self._llm,
            )

        workers = min(MAX_WORKERS_LLM, len(conclusion_targets))
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(_conclude_one, i): i for i in conclusion_targets}
            for future in as_completed(futures):
                i = futures[future]
                try:
                    _, conc = future.result()
                except Exception as exc:
                    logger.error(
                        "[Pipeline] Conclusion failed for variant %d (%s): %s",
                        i + 1, variants[i].get("Gene", "?"), exc,
                    )
                    conc = f"[CONCLUSION FAILED: {exc}]"
                conclusions[i] = conc
                base_acmg_points[i] = _extract_acmg_points_value(conc)
                logger.info(
                    "[Pipeline] Conclusion done for variant %d (%s).",
                    i + 1, variants[i].get("Gene", "?"),
                )

        # ── MOI-layer precompute (segregation + chromosome caches) ────────────
        segregation_cache: dict[int, str] = {}
        for i in include_indices:
            ab_entry   = parental_ab[i] if (parental_ab and i < len(parental_ab)) else None
            proband_ab = ab_entry.get("proband") if ab_entry else None
            mother_ab  = ab_entry.get("mother") if ab_entry else None
            father_ab  = ab_entry.get("father") if ab_entry else None
            segregation_cache[i] = segregation.classify_segregation(proband_ab, mother_ab, father_ab)

        gene_chrom_cache = moi.build_gene_chrom_cache(variants, kept_indices, _group_by_gene)

        layer_outputs: dict[str, list[str]] = {
            "De Novo": [], "Dominant-Inherited": [], "Recessive": [], "X-Linked": [],
        }
        moi_covered_indices: set[int] = set()

        # ── Layer 3: De Novo ───────────────────────────────────────────────────
        denovo_targets = [
            i for i in include_indices
            if gene_mode_cache.get(normalize_gene(variants[i].get("Gene", "NA")))
            in ("AD", "AD_AR", "XLD", "XLR", "XLD_XLR", "XL")
        ]

        def _denovo_one(i: int) -> tuple[int, str]:
            ab_entry = parental_ab[i] if (parental_ab and i < len(parental_ab)) else None
            has_trio = segregation.has_parent_data(ab_entry)
            has_one_parent = (not has_trio) and segregation.has_any_parent_data(ab_entry)
            result = moi_denovo.run_one(
                variant_context=context_slices[i],
                base_conclusion=conclusions.get(i, ""),
                segregation=segregation_cache.get(i, "insufficient_data"),
                has_trio=has_trio,
                has_one_parent=has_one_parent,
                llm=self._llm,
            )
            return i, result

        workers = min(MAX_WORKERS_LLM, len(denovo_targets))
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(_denovo_one, i): i for i in denovo_targets}
            for future in as_completed(futures):
                i = futures[future]
                try:
                    _, result = future.result()
                except Exception as exc:
                    logger.error("[Pipeline] De novo layer failed for variant %d: %s", i + 1, exc)
                    continue
                layer_outputs["De Novo"].append(result)
                moi_covered_indices.add(i)

        # ── Layer 4: Dominant-Inherited ────────────────────────────────────────
        dominant_targets = [
            i for i in include_indices
            if gene_mode_cache.get(normalize_gene(variants[i].get("Gene", "NA"))) in ("AD", "AD_AR", "XLD")
        ]

        def _dominant_one(i: int) -> tuple[int, str]:
            ab_entry = parental_ab[i] if (parental_ab and i < len(parental_ab)) else None
            seg = segregation_cache.get(i, "insufficient_data")
            proband_ab_class = segregation.classify_ab_ratio(ab_entry.get("proband")) if ab_entry else "uncertain"
            if seg == "maternal":
                parent_ab_class = segregation.classify_ab_ratio(ab_entry.get("mother")) if ab_entry else "uncertain"
            elif seg == "paternal":
                parent_ab_class = segregation.classify_ab_ratio(ab_entry.get("father")) if ab_entry else "uncertain"
            else:
                parent_ab_class = "uncertain"
            result = moi_dominant.run_one(
                variant_context=context_slices[i],
                base_conclusion=conclusions.get(i, ""),
                segregation=seg,
                proband_ab_class=proband_ab_class,
                transmitting_parent_ab_class=parent_ab_class,
                llm=self._llm,
            )
            return i, result

        workers = min(MAX_WORKERS_LLM, len(dominant_targets))
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(_dominant_one, i): i for i in dominant_targets}
            for future in as_completed(futures):
                i = futures[future]
                try:
                    _, result = future.result()
                except Exception as exc:
                    logger.error("[Pipeline] Dominant layer failed for variant %d: %s", i + 1, exc)
                    continue
                layer_outputs["Dominant-Inherited"].append(result)
                moi_covered_indices.add(i)

        # ── Layer 5: Recessive — pair path (compound heterozygous) ────────────
        recessive_gene_groups_included: dict[str, list[int]] = {}
        for gene, idxs in recessive_gene_groups.items():
            in_incl = [i for i in idxs if i in include_indices]
            if len(in_incl) >= 2:
                recessive_gene_groups_included[gene] = in_incl

        recessive_pair_targets: list[tuple[str, int, int, str]] = []
        for gene, idxs in recessive_gene_groups_included.items():
            for i, j in itertools.combinations(sorted(idxs), 2):
                phase = segregation.classify_phase(
                    segregation_cache.get(i, "insufficient_data"),
                    segregation_cache.get(j, "insufficient_data"),
                )
                if phase == "cis":
                    # Hard block — a CIS pair is never compound heterozygous;
                    # never reaches the LLM.
                    continue
                recessive_pair_targets.append((gene, i, j, phase))

        def _recessive_pair_one(target: tuple[str, int, int, str]) -> tuple[int, int, str]:
            gene, i, j, phase = target
            label_i = f"Variant {i + 1} — {gene} ({variants[i].get('HGVS', variants[i].get('Variant', '?'))})"
            label_j = f"Variant {j + 1} — {gene} ({variants[j].get('HGVS', variants[j].get('Variant', '?'))})"
            result = moi_recessive.run_pair(
                gene=gene,
                variant_a_label=label_i, variant_a_context=context_slices[i],
                variant_a_base_conclusion=conclusions.get(i, ""),
                variant_b_label=label_j, variant_b_context=context_slices[j],
                variant_b_base_conclusion=conclusions.get(j, ""),
                phase=phase, cross_analysis=cross_analyses.get(gene), llm=self._llm,
            )
            return i, j, result

        workers = min(MAX_WORKERS_LLM, len(recessive_pair_targets))
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(_recessive_pair_one, t): t for t in recessive_pair_targets}
            for future in as_completed(futures):
                target = futures[future]
                try:
                    i, j, result = future.result()
                except Exception as exc:
                    logger.error("[Pipeline] Recessive pair layer failed for %s: %s", target, exc)
                    continue
                layer_outputs["Recessive"].append(result)
                moi_covered_indices.add(i)
                moi_covered_indices.add(j)

        # ── Layer 5: Recessive — solo path (confirmed homozygous) ─────────────
        paired_genes = set(recessive_gene_groups_included.keys())
        solo_targets = [
            i for i in include_indices
            if gene_mode_cache.get(normalize_gene(variants[i].get("Gene", "NA")))
            in ("AR", "XLR", "AD_AR", "XLD_XLR")
            and moi.zygosity_is_confirmed_hom(variants[i].get("Zygosity", "NA"))
            and normalize_gene(variants[i].get("Gene", "NA")) not in paired_genes
        ]

        def _recessive_solo_one(i: int) -> tuple[int, str]:
            gene = normalize_gene(variants[i].get("Gene", "NA"))
            label = f"Variant {i + 1} — {gene} ({variants[i].get('HGVS', variants[i].get('Variant', '?'))})"
            result = moi_recessive.run_solo(
                gene=gene, variant_label=label, variant_context=context_slices[i],
                variant_base_conclusion=conclusions.get(i, ""),
                segregation=segregation_cache.get(i, "insufficient_data"),
                cross_analysis=cross_analyses.get(gene), llm=self._llm,
            )
            return i, result

        workers = min(MAX_WORKERS_LLM, len(solo_targets))
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(_recessive_solo_one, i): i for i in solo_targets}
            for future in as_completed(futures):
                i = futures[future]
                try:
                    _, result = future.result()
                except Exception as exc:
                    logger.error("[Pipeline] Recessive solo layer failed for variant %d: %s", i + 1, exc)
                    continue
                layer_outputs["Recessive"].append(result)
                moi_covered_indices.add(i)

        # ── Layer 6: X-linked ───────────────────────────────────────────────────
        xlinked_targets = [
            i for i in include_indices
            if gene_chrom_cache.get(normalize_gene(variants[i].get("Gene", "NA"))) == "X"
        ]

        def _xlinked_one(i: int) -> tuple[int, str]:
            ab_entry   = parental_ab[i] if (parental_ab and i < len(parental_ab)) else None
            proband_ab = ab_entry.get("proband") if ab_entry else None
            mother_ab  = ab_entry.get("mother") if ab_entry else None
            pattern = segregation.classify_xlinked_ab(proband_ab, mother_ab)
            result = moi_xlinked.run_one(
                variant_context=context_slices[i], base_conclusion=conclusions.get(i, ""),
                xlinked_pattern=pattern, llm=self._llm,
            )
            return i, result

        workers = min(MAX_WORKERS_LLM, len(xlinked_targets))
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(_xlinked_one, i): i for i in xlinked_targets}
            for future in as_completed(futures):
                i = futures[future]
                try:
                    _, result = future.result()
                except Exception as exc:
                    logger.error("[Pipeline] X-linked layer failed for variant %d: %s", i + 1, exc)
                    continue
                layer_outputs["X-Linked"].append(result)
                moi_covered_indices.add(i)

        # ── Unclassified (included, but no MOI layer fired) ──────────────────
        unclassified_indices = [i for i in include_indices if i not in moi_covered_indices]
        unclassified_conclusions = [conclusions[i] for i in unclassified_indices if i in conclusions]

        # ── Stage 5: Final clinical conclusion (cross-MOI synthesis) ─────────
        if not include_indices:
            final_summary = "No variants were included after second triage — no clinical conclusion generated."
        else:
            logger.info("[Pipeline] Generating final clinical conclusion...")
            layer_outputs_for_summary = {k: v for k, v in layer_outputs.items() if v}
            final_summary = final_conclusion.run(
                layer_outputs=layer_outputs_for_summary,
                unclassified_conclusions=unclassified_conclusions,
                patient_phenotype=patient_phenotype,
                llm=self._llm,
            )

        # ── ACMG-SF actionable variants section ───────────────────────────────
        actionable_flagged = []
        for i in sorted(actionable_indices):
            if i not in conclusions:
                continue
            gene = variants[i].get("Gene", "NA")
            actionable_flagged.append({
                "index": i,
                "gene": gene,
                "hgvs": variants[i].get("HGVS", variants[i].get("Variant", "?")),
                "condition": acmg_sf.ACMG_SF_CONDITIONS.get(gene, "unknown condition"),
                "reason": actionable_reasons.get(i, ""),
                "zygosity": variants[i].get("Zygosity", "NA"),
                "classification": _extract_classification_label(conclusions[i]),
            })
        actionable_section = (
            actionable.run(actionable_flagged, patient_phenotype, self._llm)
            if actionable_flagged else ""
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

        # SEGREGATION ANALYSIS — included variants with a computed segregation call
        segregation_section = _build_segregation_analysis(include_indices, segregation_cache, variants)

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
                f"{SEP_VARIANT}\n{reasonings.get(i, reasoning_only.get(i, ''))}"
            )
        reasoning_display = "\n\n".join(reasoning_parts)

        # FINAL REPORT — gene-evidence table + MOI-layer sections + unclassified
        # appendix + clinical conclusion + actionable section + exclude/discard appendix
        gene_evidence_table = _build_gene_evidence_table(variants, kept_indices, genome_build)

        final_report_parts: list[str] = []
        if gene_evidence_table:
            final_report_parts.append(f"## Gene-Level Evidence Summary\n\n{gene_evidence_table}")

        for layer_title in ("De Novo", "Dominant-Inherited", "Recessive", "X-Linked"):
            blocks = layer_outputs.get(layer_title, [])
            if blocks:
                ordered = _sort_blocks_by_classification(blocks)
                final_report_parts.append(f"## {layer_title}\n\n" + "\n\n---\n\n".join(ordered))

        if unclassified_conclusions:
            ordered_unclassified = _sort_blocks_by_classification(unclassified_conclusions)
            final_report_parts.append(
                "## Unclassified (base ACMG score only)\n\n" + "\n\n---\n\n".join(ordered_unclassified)
            )

        final_report_parts.append(final_summary)

        if actionable_section:
            final_report_parts.append(actionable_section)

        final_report = "\n\n---\n\n".join(final_report_parts)

        if triage_ran and (exclude_indices or discarded_indices):
            final_report += _build_report_appendix(
                exclude_indices=exclude_indices,
                discarded_indices=discarded_indices,
                triage_results=triage_results,
                variants=variants,
            )

        header_section = (
            f"{SEP}\nCOLUMN HEADER INTERPRETATION\n{SEP}\n{header_mapping_summary}\n\n"
            if header_mapping_summary else ""
        )
        segregation_display = (
            f"{SEP}\nSEGREGATION ANALYSIS\n{SEP}\n{segregation_section}\n\n"
            if segregation_section else ""
        )

        output = (
            header_section
            + f"{SEP}\nPROCESS DETAILS\n{SEP}\n"
            + process_section + "\n\n"
            + segregation_display
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

    # Exemptions: kept variants whose justification records an override
    exemptions = [
        i for i in kept_indices
        if triage_results[i][1].startswith("[compound-het exempt")
        or triage_results[i][1].startswith("[ACMG-SF actionable exempt")
    ]
    if exemptions:
        lines.append("\nExemptions (forced KEEP):")
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


def _build_segregation_analysis(
    include_indices: list[int],
    segregation_cache: dict[int, str],
    variants: list[dict],
) -> str:
    if not segregation_cache:
        return ""
    lines = []
    for i in include_indices:
        if i not in segregation_cache:
            continue
        gene = variants[i].get("Gene", "?")
        hgvs = variants[i].get("HGVS", variants[i].get("Variant", "?"))
        lines.append(f"  Variant {i + 1} — {gene} ({hgvs}): {segregation_cache[i]}")
    return "\n".join(lines)


def _build_gene_evidence_table(
    variants: list[dict],
    kept_indices: list[int],
    genome_build: str,
) -> str:
    """
    Per-gene evidence summary (gnomAD constraint + ClinVar consequence verdict)
    for every gene with a kept variant. Reads from the tools' own class-level
    caches (populated during retrieval) rather than re-fetching — a pure
    presentation step over data already gathered.
    """
    reference_genome = "GRCh38" if genome_build == "hg38" else "GRCh37"
    genes = sorted({
        normalize_gene(variants[i].get("Gene", "NA"))
        for i in kept_indices
        if normalize_gene(variants[i].get("Gene", "NA")) != "NA"
    })
    if not genes:
        return ""

    gnomad_tool  = GnomadConstraintTool()
    clinvar_tool = ClinVarGeneStatsTool()

    lines = [
        "| Gene | pLI | LOEUF | ClinVar P/LP missense | ClinVar P/LP nonsense/frameshift | Verdict |",
        "|---|---|---|---|---|---|",
    ]
    for gene in genes:
        try:
            constraint = gnomad_tool._get_constraint(gene, reference_genome) or {}
        except Exception:
            constraint = {}
        pli, loeuf = constraint.get("pli"), constraint.get("loeuf")
        pli_s   = f"{pli:.3f} ({classify_pli(pli)})" if pli is not None else "n/a"
        loeuf_s = f"{loeuf:.3f} ({classify_loeuf(loeuf)})" if loeuf is not None else "n/a"

        try:
            stats = clinvar_tool._get_stats(gene)
            missense, nonsense = stats["missense"], stats["nonsense"]
            verdict = classify_consequence_counts(missense, nonsense)
        except Exception:
            missense, nonsense, verdict = "n/a", "n/a", "not available"

        lines.append(f"| {gene} | {pli_s} | {loeuf_s} | {missense} | {nonsense} | {verdict} |")

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
