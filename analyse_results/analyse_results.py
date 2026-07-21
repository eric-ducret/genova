#!/usr/bin/env python3
"""
analyse_results.py — Post-process pipeline results into a per-case analysis CSV.

For each patient in results/, reads the most recent report.txt and:
  1. Extracts the final clinical conclusion and calls the SLM to rank variant
     identifiers (GENE (HGVS)) from most to least probable causative (JSON array).
  2. Parses the report for stage-level funnel statistics (first triage, second
     triage, inclusion decisions, conclusion membership).
  3. Cross-references with pheno_prompts.csv for all metadata including the
     expert solution gene (solution column), is_solution_here, and
     resolution_status.

pheno_prompts.csv expected columns:
    patient_id, prompt, n_variants, resolution_status, phenotype_present,
    solution, is_solution_here

Output: results_table_{YYYYMMDD}.csv — one row per case with all §2 fields
        from analysis_strategy.md. Ready for use in analysis_notebook.ipynb.

Usage:
    python analyse_results.py
    python analyse_results.py --results-dir /path/to/results
    python analyse_results.py --pheno-csv /path/to/pheno_prompts.csv
    python analyse_results.py --vllm-url http://localhost:8001
    python analyse_results.py --workers 8
    python analyse_results.py --output-dir /path/to/out
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Per-model payload extras — mirrors pipeline/llm/registry.py MODELS entries.
_MODEL_CONFIGS = {
    "qwen3.5-9b": {
        "model_id":      "Qwen/Qwen3.5-9B",
        "extra_payload": {"chat_template_kwargs": {"enable_thinking": False}},
    },
    "gemma4-e4b": {
        "model_id":      "google/gemma-4-E4B-it",
        "extra_payload": {},
    },
}

# Gemma turn tokens that leak into responses when stop is not applied correctly.
_GEMMA_TURN_START_RE = re.compile(r"^(?:<start_of_turn>\w+\n)+")


def _strip_gemma_artifacts(text):
    # type: (str) -> str
    text = _GEMMA_TURN_START_RE.sub("", text, count=1)
    end = text.find("<end_of_turn>")
    if end != -1:
        text = text[:end]
    return text.strip() or text


class _VLLMClient(object):
    """Minimal stdlib-only vLLM HTTP client (no httpx/requests dependency)."""

    def __init__(self, base_url="http://localhost:8001", model_id="Qwen/Qwen3.5-9B",
                 timeout=300, extra_payload=None):
        self.base_url        = base_url.rstrip("/")
        self.model_id        = model_id
        self.timeout         = timeout
        self._extra_payload  = extra_payload or {}

    def generate(self, system, user, max_tokens=512, temperature=0.0, **kwargs):
        # type: (str, str, int, float) -> str
        body = {
            "model":       self.model_id,
            "messages":    [{"role": "system", "content": system},
                            {"role": "user",   "content": user}],
            "max_tokens":  max_tokens,
            "temperature": temperature,
        }
        body.update(self._extra_payload)
        payload = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            "{0}/v1/chat/completions".format(self.base_url),
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        raw = data["choices"][0]["message"]["content"].strip()
        return _strip_gemma_artifacts(raw)

    def is_available(self):
        # type: () -> bool
        try:
            urllib.request.urlopen(
                "{0}/health".format(self.base_url), timeout=5
            )
            return True
        except Exception:
            return False


_DEFAULT_RESULTS_DIR = Path(__file__).parent.parent / "results"
_DEFAULT_OUTPUT_DIR  = Path(__file__).parent
_DEFAULT_PHENO_CSV   = Path(__file__).parent.parent / "pheno_prompts.csv"

# Folder names follow the pattern: {patient_stem}_{YYYYMMDD_HHMMSS}
_TIMESTAMP_RE = re.compile(r"_(\d{8}_\d{6})$")

# Prompts matching these strings (case-insensitive, stripped) are classified
# as generic phenotype (default "visual impairment" assigned when no specific
# phenotype was available in the clinical metadata).
_GENERIC_PHENOTYPE_STRINGS = {"visual impairment", "blindness", "unknown"}

# ─── SLM ranking prompt ────────────────────────────────────────────────────────

_RANKING_SYSTEM = "You are an expert clinical geneticist."
_RANKING_PROMPT = """\
Below is the final clinical conclusion section of a variant analysis report.

{text}

Task: List all variants mentioned above, ordered from MOST to LEAST likely \
causative of the patient's condition, based on the reasoning provided.

Rules:
- Output ONLY a valid JSON array of strings on a single line.
- Each element should be in the format "GENE (HGVS)" if HGVS is available, \
or just "GENE" if not.
- Deduplicate: each variant appears at most once.
- Include ALL variants mentioned in the conclusion, even those considered less likely.
- Do not add explanations, markdown code fences, or any other text.

Example output: ["SIM1 (NM_005068.3:c.744-2A>G p.?)", "CCNP (NM_024877.4:c.305dup p.(Val103CysfsTer25))"]
"""


# ─── report parsing — final conclusion extraction ─────────────────────────────

def _extract_final_conclusion(report_text):
    # type: (str) -> Optional[str]
    """
    Extract the # Reasoning + # Clinical Conclusion synthesis block from
    the FINAL REPORT section. The synthesis is always the last block before
    the optional appendix (## Variants not included in conclusion).

    Falls back to the last ---‑separated block when the SLM omits the
    expected markdown headers (e.g. outputs ## instead of #, or plain text).
    """
    final_report_match = re.search(
        r"={60}\nFINAL REPORT\n={60}\n(.*?)(?=\n={60}|\Z)",
        report_text,
        re.DOTALL,
    )
    if not final_report_match:
        return None

    final_report = final_report_match.group(1)

    # Strip the appendix before any search so rfind targets the synthesis block.
    appendix_start = re.search(r"\n---\n\n##\s*Variants not included", final_report)
    if appendix_start:
        final_report = final_report[: appendix_start.start()]

    # Primary: look for exact markdown headers produced by the prompt template.
    # Also accept ## (two hashes) in case the SLM added an extra level.
    for header in ("# Reasoning\n", "# Clinical Conclusion\n",
                   "## Reasoning\n", "## Clinical Conclusion\n"):
        pos = final_report.rfind(header)
        if pos != -1:
            return final_report[pos:].strip() or None

    # Fallback: the final_summary is always the last ---‑separated chunk.
    chunks = re.split(r"\n\n---\n\n", final_report)
    if chunks:
        last = chunks[-1].strip()
        if last:
            logger.warning(
                "[analyse] Final conclusion headers missing — using last --- block as fallback"
            )
            return last

    return None


# ─── ranked variant parsing ───────────────────────────────────────────────────

def _dedupe_identifiers(parsed):
    # type: (List) -> List[str]
    """
    Dedupe a parsed ranking array by leading gene token (case-insensitive),
    preserving first-occurrence order. Guards against models that repeat the
    whole array (or individual entries) instead of emitting a stop token.
    """
    items = [str(item).strip() for item in parsed if str(item).strip()]
    seen = set()
    result = []
    for item in items:
        gene = re.split(r"[\s(]", item, 1)[0].strip().upper()
        if gene and gene not in seen:
            seen.add(gene)
            result.append(item)
    if len(result) < len(items):
        logger.warning(
            "[analyse] Ranking response had %d repeated/duplicate entries — deduped %d → %d",
            len(items) - len(result), len(items), len(result),
        )
    return result


def _parse_ranked_variants(response):
    # type: (str) -> List[str]
    """
    Parse the SLM ranking response into an ordered list of variant identifiers.

    Handles JSON array output (primary), and falls back to comma/newline
    splitting if JSON parsing fails. Returns identifiers in the form
    "GENE (HGVS)" or "GENE".
    """
    text = response.strip()

    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    text = text.strip()

    # Try full JSON parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return _dedupe_identifiers(parsed)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try extracting array content with greedy outer bracket match
    array_match = re.search(r"\[(.+)\]", text, re.DOTALL)
    if array_match:
        try:
            parsed = json.loads("[{0}]".format(array_match.group(1)))
            if isinstance(parsed, list):
                return _dedupe_identifiers(parsed)
        except (json.JSONDecodeError, ValueError):
            pass

    # Truncated array fallback: extract all double-quoted strings in order.
    # Handles the case where max_tokens cut the JSON array before the closing ].
    quoted = re.findall(r'"([^"]+)"', text)
    if quoted:
        logger.warning("[analyse] Truncated JSON array — extracting quoted strings directly")
        seen = set()
        result = []
        for item in quoted:
            item = item.strip()
            if item and item.upper() not in seen:
                seen.add(item.upper())
                result.append(item)
        if result:
            return result

    # Last resort: split and extract gene-like tokens
    logger.warning("[analyse] JSON parse failed for ranking response; falling back to token split")
    return _fallback_parse_gene_tokens(text)


def _fallback_parse_gene_tokens(text):
    # type: (str) -> List[str]
    """Fallback: extract gene-like tokens from unstructured or truncated ranking response."""
    text = re.sub(
        r"^(?:(?:the )?(?:ranked |ordered )?genes?(?:[^:]*):?\s*)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    parts = re.split(r"[,\n]+", text)
    seen = set()
    result = []
    for part in parts:
        token = part.strip()
        token = re.sub(r"^\d+[\.\)\-]\s*", "", token).strip()
        # Strip markdown chars AND JSON quote characters so truncated JSON arrays
        # like ["RPGRIP1 (NM_...)","SAMD11 don't fail the gene fullmatch below.
        token = re.sub(r'[*_`#"\']', "", token).strip()
        gene = re.split(r"[\s(]", token)[0].strip()
        if gene and re.fullmatch(r"[A-Za-z][A-Za-z0-9\-]{1,19}", gene):
            key = gene.upper()
            if key not in seen:
                seen.add(key)
                result.append(token)
    return result


# ─── report parsing — stage funnel ────────────────────────────────────────────

def _split_gene_field(raw_list):
    # type: (List[str]) -> List[str]
    """
    Flatten a list of raw gene captures into individual unique gene names.

    The pipeline uses the variant's Gene field verbatim, which can be "SAG"
    or "SAG;SAG" (compound het entry). Split on ";" and deduplicate while
    preserving first-occurrence order.
    """
    return list(dict.fromkeys(
        g.strip() for raw in raw_list for g in raw.split(";") if g.strip()
    ))


def _parse_report_funnel(report_text):
    # type: (str) -> Dict
    """
    Extract stage-level funnel statistics and per-variant decisions from report.

    Returns a dict with:
        n_input              : total variants in this run (int or None)
        triage_ran           : whether first triage ran (n > threshold)
        n_discarded_triage   : variants discarded at first triage
        n_kept_after_triage  : variants kept after first triage
        n_evaluated_second   : variants evaluated at second triage
        n_included_second    : variants included (→ conclusion)
        n_excluded_second    : variants excluded (reasoning only)
        n_in_conclusion      : conclusion blocks in FINAL REPORT
        discarded_genes      : gene names discarded at first triage
        excluded_genes       : gene names excluded at second triage
        included_genes       : gene names included at second triage
        conclusion_genes     : gene names with conclusion blocks in FINAL REPORT
    """
    result = {
        "n_input": None,
        "triage_ran": False,
        "n_discarded_triage": 0,
        "n_kept_after_triage": None,
        "n_evaluated_second": None,
        "n_included_second": None,
        "n_excluded_second": None,
        "n_in_conclusion": 0,
        "discarded_genes": [],
        "excluded_genes": [],
        "included_genes": [],
        "conclusion_genes": [],
    }

    # TRIAGE SUMMARY block (only present when first triage ran)
    # Format uses Unicode BOX DRAWINGS LIGHT HORIZONTAL (─) as separator
    triage_m = re.search(
        r"TRIAGE SUMMARY\n[─\-]+\n"
        r"Total variants\s*:\s*(\d+)\n"
        r"Kept \(Tier 1\+2\)\s*:\s*(\d+)\n"
        r"Discarded\s*:\s*(\d+)",
        report_text,
    )
    if triage_m:
        result["triage_ran"] = True
        result["n_input"] = int(triage_m.group(1))
        result["n_kept_after_triage"] = int(triage_m.group(2))
        result["n_discarded_triage"] = int(triage_m.group(3))

    # Discarded variants list in TRIAGE SUMMARY
    # Format: "  Variant N — GENE (HGVS): justification"
    # Gene can be "SAG" or "SAG;SAG" (compound het in same gene) — include ";" in
    # character class and split afterwards.
    discard_block_m = re.search(
        r"Discarded variants:\n((?:  Variant .+\n)+)",
        report_text,
    )
    if discard_block_m:
        raw = re.findall(
            r"Variant \d+ — ([A-Za-z][A-Za-z0-9;\-]*)\s*[\s(]",
            discard_block_m.group(1),
        )
        result["discarded_genes"] = _split_gene_field(raw)

    # Also check FINAL REPORT appendix (redundant but catches edge cases)
    appendix_discard_m = re.search(
        r"### Discarded at first triage\n((?:- .+\n)*)",
        report_text,
    )
    if appendix_discard_m:
        raw_extra = re.findall(
            r"- Variant \d+ — ([A-Za-z][A-Za-z0-9;\-]*)",
            appendix_discard_m.group(1),
        )
        existing_upper = {g.upper() for g in result["discarded_genes"]}
        for g in _split_gene_field(raw_extra):
            if g.upper() not in existing_upper:
                result["discarded_genes"].append(g)
                existing_upper.add(g.upper())

    # SECOND TRIAGE (INCLUSION DECISIONS) block
    second_m = re.search(
        r"SECOND TRIAGE \(INCLUSION DECISIONS\)\n[─\-]+\n"
        r"Evaluated\s*:\s*(\d+)\n"
        r"Included\s*:\s*(\d+)\n"
        r"Excluded\s*:\s*(\d+)",
        report_text,
    )
    if second_m:
        result["n_evaluated_second"] = int(second_m.group(1))
        result["n_included_second"] = int(second_m.group(2))
        result["n_excluded_second"] = int(second_m.group(3))

    # Per-variant inclusion decisions
    # Format: "  Variant N — GENE  HGVS  [INCLUDE|EXCLUDE]"
    # Gene can be "SAG;SAG" — ";" included in character class, split afterwards.
    decisions = re.findall(
        r"  Variant \d+ — ([A-Za-z][A-Za-z0-9;\-]*)  .+?  \[(INCLUDE|EXCLUDE)\]",
        report_text,
    )
    result["included_genes"] = _split_gene_field([g for g, d in decisions if d == "INCLUDE"])
    result["excluded_genes"] = _split_gene_field([g for g, d in decisions if d == "EXCLUDE"])

    # n_input fallback: when triage didn't run, all variants went to second triage
    if result["n_input"] is None and result["n_evaluated_second"] is not None:
        result["n_input"] = result["n_evaluated_second"]

    # Count conclusion blocks and collect gene names from FINAL REPORT
    final_report_m = re.search(
        r"={60}\nFINAL REPORT\n={60}\n(.*?)(?=\n={60}|\Z)",
        report_text,
        re.DOTALL,
    )
    if final_report_m:
        # Non-greedy [^\n]*? stops at the first dash (the em-dash separator),
        # avoiding false matches on hyphens inside HGVS notation.
        #
        # Since the ACMG-MOI-layer merge, a variant's base conclusion block
        # ("# Variant N — GENE (HGVS)") is only displayed as-is for variants
        # under "## Unclassified" — a variant covered by a MOI layer (De
        # Novo/Dominant-Inherited/Recessive/X-Linked) instead surfaces under
        # that layer's own "# <Layer> Analysis — GENE (...)" header. Matching
        # both header shapes keeps this count from silently collapsing to
        # just the Unclassified section. Note: a variant whose gene qualifies
        # for more than one layer (e.g. AD_AR) is deliberately not
        # deduplicated by the pipeline, so it can be counted more than once
        # here — this reflects layer coverage, not distinct variants; treat
        # n_in_conclusion as a QC signal, not an exact variant count, for
        # any report containing MOI-layer sections. A compound-het "Recessive
        # Analysis" pair block also contributes only one gene mention (its
        # own header), not two, since both variants share the same gene.
        conclusion_genes = re.findall(
            r"^#\s*(?:Variant[^\n]*?[—\-]|"
            r"(?:De Novo|Dominant-Inherited|Recessive|X-Linked) Analysis\s*[—\-])"
            r"\s*([A-Za-z][A-Za-z0-9\-]+)",
            final_report_m.group(1),
            re.MULTILINE,
        )
        result["n_in_conclusion"] = len(conclusion_genes)
        result["conclusion_genes"] = conclusion_genes

    return result


# ─── expert variant tracking ─────────────────────────────────────────────────

def _parse_expert_genes(expert_gene):
    # type: (Optional[str]) -> List[str]
    """
    Split a potentially compound solution string into individual gene names.

    The solution column can be a single gene ("SAG") or semicolon-separated
    genes ("HSD17B10;HSD17B10"). Returns a deduplicated list of uppercased
    gene names, preserving order. Empty list if input is blank.
    """
    if not expert_gene:
        return []
    return list(dict.fromkeys(
        g.strip().upper() for g in expert_gene.split(";") if g.strip()
    ))


def _expert_stage_info(funnel, expert_gene):
    # type: (Dict, Optional[str]) -> Dict
    """
    Determine whether any expert gene survived each pipeline stage.

    For compound solutions ("GENE1;GENE2"), returns True if ANY expert gene
    passed the stage. Returns None when stage information is unavailable.
    """
    na = {"expert_first_triage_survived": None,
          "expert_second_triage_survived": None,
          "expert_in_conclusion": None}

    expert_genes = _parse_expert_genes(expert_gene)
    if not expert_genes:
        return na

    discarded_up  = {g.upper() for g in funnel["discarded_genes"]}
    excluded_up   = {g.upper() for g in funnel["excluded_genes"]}
    included_up   = {g.upper() for g in funnel["included_genes"]}
    conclusion_up = {g.upper() for g in funnel["conclusion_genes"]}

    if funnel["triage_ran"]:
        survived_t1 = any(eg not in discarded_up for eg in expert_genes)
    else:
        survived_t1 = True  # triage skipped → all variants are kept

    if not survived_t1:
        return {
            "expert_first_triage_survived": False,
            "expert_second_triage_survived": False,
            "expert_in_conclusion": False,
        }

    if included_up or excluded_up:
        survived_t2 = any(eg in included_up for eg in expert_genes)
    else:
        survived_t2 = None  # parse failure or very small case without decision data

    in_conclusion = any(eg in conclusion_up for eg in expert_genes)

    return {
        "expert_first_triage_survived": survived_t1,
        "expert_second_triage_survived": survived_t2,
        "expert_in_conclusion": in_conclusion,
    }


def _gene_from_identifier(identifier):
    # type: (str) -> str
    """
    Extract the leading gene symbol from a ranked-variant identifier.

    Handles "GENE (HGVS)", "GENE", and compound-het "GENE1;GENE2 (HGVS)" forms.
    Returns the first token before any whitespace, parenthesis, or semicolon,
    uppercased.
    """
    return re.split(r"[\s(;]", identifier)[0].strip().upper()


def _filter_ranked_variants(ranked, conclusion_genes):
    # type: (List[str], List[str]) -> List[str]
    """
    Anchor the SLM-ranked list to the genes confirmed by regex in the report.

    Two passes:
      1. Keep only ranked identifiers whose leading gene token is in
         conclusion_genes (removes SLM hallucinations).
      2. Append any conclusion_genes that the SLM omitted, preserving their
         report order (ensures the list is complete).
    """
    conclusion_set = {g.upper() for g in conclusion_genes}
    filtered = [v for v in ranked if _gene_from_identifier(v) in conclusion_set]
    seen = {_gene_from_identifier(v) for v in filtered}
    for gene in conclusion_genes:
        if gene.upper() not in seen:
            filtered.append(gene)
            seen.add(gene.upper())
    return filtered


def _find_expert_rank(ranked_variants, expert_gene):
    # type: (List[str], Optional[str]) -> Optional[int]
    """
    Find the best (lowest) 1-indexed rank of any expert gene in the ranked list.

    For compound solutions ("GENE1;GENE2"), returns the minimum rank across all
    expert genes. Returns None if no expert gene appears in the ranked list.
    """
    expert_genes = _parse_expert_genes(expert_gene)
    if not expert_genes or not ranked_variants:
        return None

    expert_genes_set = set(expert_genes)
    best = None
    for i, identifier in enumerate(ranked_variants):
        # Extract everything before the first space or parenthesis, then split on ";"
        # to handle SLM outputs like "SAG;SAG (HGVS)" for compound het cases.
        raw_gene = re.split(r"[\s(]", identifier)[0].strip().upper()
        identifier_genes = {g.strip() for g in raw_gene.split(";") if g.strip()}
        if identifier_genes & expert_genes_set:
            rank = i + 1
            if best is None or rank < best:
                best = rank
    return best


# ─── auxiliary data loaders ───────────────────────────────────────────────────

def _load_pheno_data(pheno_csv):
    # type: (Path) -> Dict[str, Dict]
    """
    Load all metadata from pheno_prompts.csv.

    Expected columns:
        patient_id, prompt, n_variants, resolution_status,
        phenotype_present, solution, is_solution_here

    Returns {patient_id: {phenotype_type, n_variants, prompt,
                           expert_gene, is_solution_here, resolution_status}}.

    phenotype_type is inferred from the prompt:
        - "generic" if the phenotype description (after "The patient has ") is
          one of the known vague strings (visual impairment, blindness, unknown)
        - "specific" otherwise
    """
    data = {}
    if not pheno_csv.exists():
        logger.warning(
            "[analyse] pheno_prompts.csv not found at %s — metadata fields will be empty",
            pheno_csv,
        )
        return data

    with open(pheno_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get("patient_id", "").strip()
            if not pid:
                continue

            prompt = row.get("prompt", "").strip()

            # Infer phenotype_type from the prompt text
            # Strip the common preamble "The patient has " to get the bare phenotype
            bare = re.sub(r"^the patient has\s+", "", prompt, flags=re.IGNORECASE).strip()
            phenotype_type = "generic" if bare.lower() in _GENERIC_PHENOTYPE_STRINGS else "specific"

            # Expert gene: the solution column (empty string for unsolved cases)
            expert_gene = row.get("solution", "").strip()

            # is_solution_here: 1 = the expert variant was present in the pipeline input
            try:
                is_solution_here = int(row.get("is_solution_here", "0").strip())
            except ValueError:
                is_solution_here = 0

            resolution_status = row.get("resolution_status", "").strip()

            data[pid] = {
                "phenotype_type":    phenotype_type,
                "n_variants":        row.get("n_variants", "").strip(),
                "prompt":            prompt,
                "expert_gene":       expert_gene,
                "is_solution_here":  is_solution_here,
                "resolution_status": resolution_status,
            }

    logger.info("[analyse] Loaded pheno data for %d patients from %s", len(data), pheno_csv)
    return data


# ─── result discovery ────────────────────────────────────────────────────────

def _extract_stem(folder_name):
    # type: (str) -> str
    return _TIMESTAMP_RE.sub("", folder_name)


def _find_latest_reports(results_dir):
    # type: (Path) -> Dict[str, Path]
    """
    For each patient stem, return the report.txt from the most recent run.
    Skips subdirectories with no report.txt or unparseable timestamps.
    """
    best = {}

    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir():
            continue
        report = subdir / "report.txt"
        if not report.exists():
            logger.debug("[analyse] No report.txt in %s — skipping", subdir.name)
            continue

        ts_match = _TIMESTAMP_RE.search(subdir.name)
        if not ts_match:
            logger.warning("[analyse] Cannot parse timestamp from %s — skipping", subdir.name)
            continue

        stem = _extract_stem(subdir.name)
        ts   = ts_match.group(1)

        existing = best.get(stem)
        if existing is None or ts > existing[0]:
            best[stem] = (ts, report)

    return {stem: report for stem, (_, report) in best.items()}


# ─── per-patient processing ──────────────────────────────────────────────────

def _process_patient(stem, report_path, llm, pheno_data):
    # type: (str, Path, object, Dict[str, Dict]) -> Dict
    """
    Process one patient report. Returns a per-case dict with all §2 fields.
    """
    text = report_path.read_text(encoding="utf-8")

    # Metadata from pheno CSV (expert gene comes from here, not a separate GT file)
    pheno       = pheno_data.get(stem, {})
    expert_gene = pheno.get("expert_gene", "")

    # Stage funnel parsing (pure regex, no SLM) — done first so conclusion_genes
    # is available to anchor and validate the SLM ranking below.
    funnel = _parse_report_funnel(text)

    # SLM-based variant ranking from final conclusion
    conclusion = _extract_final_conclusion(text)
    ranked_variants = []

    if conclusion:
        logger.info("[analyse] Ranking variants for %s ...", stem)
        prompt   = _RANKING_PROMPT.format(text=conclusion)
        response = llm.generate(
            system=_RANKING_SYSTEM,
            user=prompt,
            max_tokens=400,
            temperature=0.0,
        )
        ranked_variants = _parse_ranked_variants(response)
        logger.info("[analyse] %s → %d ranked: %s", stem, len(ranked_variants), ranked_variants)

        # Anti-hallucination + completeness: keep only genes confirmed by regex,
        # and append any the SLM missed (in report order).
        if funnel["conclusion_genes"]:
            ranked_variants = _filter_ranked_variants(
                ranked_variants, funnel["conclusion_genes"]
            )
    elif funnel["conclusion_genes"]:
        # Synthesis block missing entirely — fall back to the regex-confirmed
        # gene list in report order. No hallucinations possible.
        ranked_variants = list(funnel["conclusion_genes"])
        logger.warning(
            "[analyse] No synthesis block found for %s — using conclusion gene order as fallback",
            stem,
        )
    else:
        logger.warning("[analyse] No final conclusion found in %s", report_path)

    stage_info  = _expert_stage_info(funnel, expert_gene or None)
    expert_rank = _find_expert_rank(ranked_variants, expert_gene or None)

    return {
        "patient_id":                    stem,
        "phenotype_type":                pheno.get("phenotype_type", ""),
        "resolution_status":             pheno.get("resolution_status", ""),
        "is_solution_here":              pheno.get("is_solution_here", ""),
        "n_variants_pheno":              pheno.get("n_variants", ""),
        "n_input_report":                funnel["n_input"],
        "triage_ran":                    funnel["triage_ran"],
        "n_discarded_triage":            funnel["n_discarded_triage"],
        "n_kept_after_triage":           funnel["n_kept_after_triage"],
        "n_included_second":             funnel["n_included_second"],
        "n_excluded_second":             funnel["n_excluded_second"],
        "n_in_conclusion":               funnel["n_in_conclusion"],
        "expert_gene":                   expert_gene,
        "expert_first_triage_survived":  stage_info["expert_first_triage_survived"],
        "expert_second_triage_survived": stage_info["expert_second_triage_survived"],
        "expert_in_conclusion":          stage_info["expert_in_conclusion"],
        "expert_rank":                   expert_rank,
        "total_ranked":                  len(ranked_variants),
        "ranked_variants":               json.dumps(ranked_variants, ensure_ascii=False),
        "runtime_s":                     "",  # requires per-patient timing log
    }


# ─── entry point ─────────────────────────────────────────────────────────────

_OUTPUT_COLUMNS = [
    "patient_id",
    "phenotype_type",
    "resolution_status",
    "is_solution_here",
    "n_variants_pheno",
    "n_input_report",
    "triage_ran",
    "n_discarded_triage",
    "n_kept_after_triage",
    "n_included_second",
    "n_excluded_second",
    "n_in_conclusion",
    "expert_gene",
    "expert_first_triage_survived",
    "expert_second_triage_survived",
    "expert_in_conclusion",
    "expert_rank",
    "total_ranked",
    "ranked_variants",
    "runtime_s",
]


def main():
    # type: () -> None
    parser = argparse.ArgumentParser(
        description="Extract per-case analysis data from pipeline reports."
    )
    parser.add_argument(
        "--results-dir", type=Path, default=_DEFAULT_RESULTS_DIR, metavar="DIR",
        help="Folder containing per-patient result subdirectories (default: {0})".format(_DEFAULT_RESULTS_DIR),
    )
    parser.add_argument(
        "--pheno-csv", type=Path, default=_DEFAULT_PHENO_CSV, metavar="CSV",
        help=(
            "pheno_prompts.csv with columns: patient_id, prompt, n_variants, "
            "resolution_status, phenotype_present, solution, is_solution_here "
            "(default: {0})".format(_DEFAULT_PHENO_CSV)
        ),
    )
    parser.add_argument(
        "--vllm-url", default="http://localhost:8001", metavar="URL",
        help="vLLM HTTP server base URL (default: http://localhost:8001)",
    )
    parser.add_argument(
        "--model", default=os.environ.get("VARIANT_MODEL", "qwen3.5-9b"),
        choices=list(_MODEL_CONFIGS),
        help="Model name — qwen3.5-9b or gemma4-e4b (default: VARIANT_MODEL env var or qwen3.5-9b)",
    )
    parser.add_argument(
        "--workers", type=int, default=8, metavar="N",
        help="Number of parallel SLM workers (default: 8)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR, metavar="DIR",
        help="Directory for the output CSV (default: {0})".format(_DEFAULT_OUTPUT_DIR),
    )
    args = parser.parse_args()

    if not args.results_dir.is_dir():
        print("ERROR: {0} is not a directory".format(args.results_dir), file=sys.stderr)
        sys.exit(1)

    reports = _find_latest_reports(args.results_dir)
    if not reports:
        print("ERROR: No valid reports found in {0}".format(args.results_dir), file=sys.stderr)
        sys.exit(1)

    logger.info("[analyse] %d patient(s) found.", len(reports))

    pheno_data = _load_pheno_data(args.pheno_csv)

    n_with_gt = sum(1 for v in pheno_data.values() if v.get("expert_gene"))
    logger.info(
        "[analyse] Pheno data loaded: %d patients, %d with an expert gene (solution).",
        len(pheno_data), n_with_gt,
    )

    model_cfg = _MODEL_CONFIGS.get(args.model, _MODEL_CONFIGS["qwen3.5-9b"])
    logger.info("[analyse] Using model: %s (%s)", args.model, model_cfg["model_id"])
    llm = _VLLMClient(
        base_url=args.vllm_url,
        model_id=model_cfg["model_id"],
        extra_payload=model_cfg["extra_payload"],
    )

    if not llm.is_available():
        print("ERROR: vLLM server not reachable at {0}".format(args.vllm_url), file=sys.stderr)
        sys.exit(1)

    # Parallel SLM calls — one per patient
    all_results = {}
    workers = min(args.workers, len(reports))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_process_patient, stem, path, llm, pheno_data): stem
            for stem, path in reports.items()
        }
        for future in as_completed(futures):
            stem = futures[future]
            try:
                all_results[stem] = future.result()
            except Exception as exc:
                logger.error("[analyse] Failed for patient %s: %s", stem, exc)
                all_results[stem] = {
                    "patient_id": stem,
                    "ranked_variants": "[]",
                    "total_ranked": 0,
                }

    today    = datetime.now().strftime("%Y%m%d")
    filename = "results_table_{0}.csv".format(today)

    def _write_csv(path):
        # type: (Path) -> None
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_OUTPUT_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for stem in sorted(all_results):
                writer.writerow(all_results[stem])

    try:
        out_path = args.output_dir / filename
        _write_csv(out_path)
    except OSError as exc:
        # All patients were already processed by the SLM — an unwritable
        # output-dir must not throw that work away. Fall back to cwd.
        fallback_path = Path.cwd() / filename
        logger.error(
            "[analyse] Could not write to %s (%s) — falling back to %s",
            out_path, exc, fallback_path,
        )
        _write_csv(fallback_path)
        out_path = fallback_path

    logger.info("[analyse] Saved → %s", out_path)
    print(out_path)


if __name__ == "__main__":
    main()
