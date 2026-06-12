#!/usr/bin/env python3
"""
analyse_results.py — Post-process pipeline results into a per-case analysis CSV.

For each patient in results/, reads the most recent report.txt and:
  1. Extracts the final clinical conclusion and calls the SLM to rank variant
     identifiers (GENE (HGVS)) from most to least probable causative (JSON array).
  2. Parses the report for stage-level funnel statistics (first triage, second
     triage, inclusion decisions, conclusion membership).
  3. Cross-references with pheno_prompts.csv for phenotype type and n_variants.
  4. Cross-references with an optional ground-truth CSV to compute the expert
     variant's rank and per-stage survival.

Output: results_table_{YYYYMMDD}.csv — one row per case with all §2 fields from
        analysis_strategy.md. Ready for use in analysis_notebook.ipynb.

Usage:
    python analyse_results.py
    python analyse_results.py --results-dir /path/to/results
    python analyse_results.py --pheno-csv /path/to/pheno_prompts.csv
    python analyse_results.py --ground-truth /path/to/ground_truth.csv
    python analyse_results.py --vllm-url http://localhost:8001
    python analyse_results.py --workers 8
    python analyse_results.py --output-dir /path/to/out

Ground truth CSV format (patient_id column required):
    patient_id, expert_gene [, expert_hgvs]
    or
    patient_id, expert_variant  (will be treated as gene name)
"""

import argparse
import csv
import json
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_DEFAULT_RESULTS_DIR = Path(__file__).parent.parent / "results"
_DEFAULT_OUTPUT_DIR  = Path(__file__).parent
_DEFAULT_PHENO_CSV   = Path(__file__).parent.parent / "pheno_prompts.csv"

# Folder names follow the pattern: {patient_stem}_{YYYYMMDD_HHMMSS}
_TIMESTAMP_RE = re.compile(r"_(\d{8}_\d{6})$")

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

def _extract_final_conclusion(report_text: str) -> str | None:
    """
    Extract the # Reasoning + # Clinical Conclusion synthesis block from
    the FINAL REPORT section. The synthesis is always the last block before
    the optional appendix (## Variants not included in conclusion).
    """
    final_report_match = re.search(
        r"={60}\nFINAL REPORT\n={60}\n(.*?)(?=\n={60}|\Z)",
        report_text,
        re.DOTALL,
    )
    if not final_report_match:
        return None

    final_report = final_report_match.group(1)

    pos = final_report.rfind("# Reasoning\n")
    if pos == -1:
        pos = final_report.rfind("# Clinical Conclusion\n")
    if pos == -1:
        return None

    synthesis = final_report[pos:]

    appendix_match = re.search(r"\n---\n\n##\s*Variants not included", synthesis)
    if appendix_match:
        synthesis = synthesis[: appendix_match.start()]

    return synthesis.strip() or None


# ─── ranked variant parsing ───────────────────────────────────────────────────

def _parse_ranked_variants(response: str) -> list[str]:
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
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (json.JSONDecodeError, ValueError):
        pass

    # Try extracting array content with greedy outer bracket match
    array_match = re.search(r"\[(.+)\]", text, re.DOTALL)
    if array_match:
        try:
            parsed = json.loads(f"[{array_match.group(1)}]")
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except (json.JSONDecodeError, ValueError):
            pass

    # Last resort: split and extract gene-like tokens
    logger.warning("[analyse] JSON parse failed for ranking response; falling back to token split")
    return _fallback_parse_gene_tokens(text)


def _fallback_parse_gene_tokens(text: str) -> list[str]:
    """Fallback: extract gene-like tokens from unstructured ranking response."""
    text = re.sub(
        r"^(?:(?:the )?(?:ranked |ordered )?genes?(?:[^:]*):?\s*)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    parts = re.split(r"[,\n]+", text)
    seen: set[str] = set()
    result: list[str] = []
    for part in parts:
        token = part.strip()
        token = re.sub(r"^\d+[\.\)\-]\s*", "", token).strip()
        token = re.sub(r"[*_`#]", "", token).strip()
        gene = re.split(r"[\s(]", token)[0].strip()
        if gene and re.fullmatch(r"[A-Za-z][A-Za-z0-9\-]{1,19}", gene):
            key = gene.upper()
            if key not in seen:
                seen.add(key)
                result.append(token)
    return result


# ─── report parsing — stage funnel ────────────────────────────────────────────

def _parse_report_funnel(report_text: str) -> dict:
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
    result: dict = {
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
    discard_block_m = re.search(
        r"Discarded variants:\n((?:  Variant .+\n)+)",
        report_text,
    )
    if discard_block_m:
        result["discarded_genes"] = re.findall(
            r"Variant \d+ — ([A-Za-z][A-Za-z0-9\-]*)\s*[\s(]",
            discard_block_m.group(1),
        )

    # Also check FINAL REPORT appendix (redundant but catches edge cases)
    appendix_discard_m = re.search(
        r"### Discarded at first triage\n((?:- .+\n)*)",
        report_text,
    )
    if appendix_discard_m:
        extra = re.findall(
            r"- Variant \d+ — ([A-Za-z][A-Za-z0-9\-]*)",
            appendix_discard_m.group(1),
        )
        existing_upper = {g.upper() for g in result["discarded_genes"]}
        for g in extra:
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
    # Two spaces are used as field separators; HGVS may contain single spaces.
    decisions = re.findall(
        r"  Variant \d+ — ([A-Za-z][A-Za-z0-9\-]*)  .+?  \[(INCLUDE|EXCLUDE)\]",
        report_text,
    )
    result["included_genes"] = [g for g, d in decisions if d == "INCLUDE"]
    result["excluded_genes"] = [g for g, d in decisions if d == "EXCLUDE"]

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
        # Matches: "# Variant [N] — GENE ..." (em-dash U+2014 or regular hyphen)
        conclusion_genes = re.findall(
            r"^# Variant[^\n]*[—\-]\s*([A-Za-z][A-Za-z0-9\-]+)",
            final_report_m.group(1),
            re.MULTILINE,
        )
        result["n_in_conclusion"] = len(conclusion_genes)
        result["conclusion_genes"] = conclusion_genes

    return result


# ─── expert variant tracking ─────────────────────────────────────────────────

def _expert_stage_info(funnel: dict, expert_gene: str | None) -> dict:
    """
    Determine whether the expert variant survived each pipeline stage.

    Returns booleans for first triage survival, second triage survival,
    and presence in the final conclusion. Returns None for each field when
    the expert gene is unknown or the stage information is unavailable.
    """
    na = {"expert_first_triage_survived": None,
          "expert_second_triage_survived": None,
          "expert_in_conclusion": None}

    if not expert_gene:
        return na

    eg = expert_gene.strip().upper()
    discarded_up = {g.upper() for g in funnel["discarded_genes"]}
    excluded_up  = {g.upper() for g in funnel["excluded_genes"]}
    included_up  = {g.upper() for g in funnel["included_genes"]}
    conclusion_up = {g.upper() for g in funnel["conclusion_genes"]}

    if funnel["triage_ran"]:
        survived_t1 = eg not in discarded_up
    else:
        survived_t1 = True  # triage skipped → all variants are kept

    if not survived_t1:
        return {
            "expert_first_triage_survived": False,
            "expert_second_triage_survived": False,
            "expert_in_conclusion": False,
        }

    # Second triage: None if we have no decision data for this gene
    if included_up or excluded_up:
        survived_t2 = eg in included_up
    else:
        survived_t2 = None  # parse failure or very small case without decision data

    in_conclusion = eg in conclusion_up

    return {
        "expert_first_triage_survived": survived_t1,
        "expert_second_triage_survived": survived_t2,
        "expert_in_conclusion": in_conclusion,
    }


def _find_expert_rank(ranked_variants: list[str], expert_gene: str | None) -> int | None:
    """
    Find the 1-indexed rank of the expert gene in the ranked variant list.

    Matches by the gene name portion of each identifier (the part before any
    space or parenthesis). Returns None if the expert gene is absent from the
    ranked list or if expert_gene is not provided.
    """
    if not expert_gene or not ranked_variants:
        return None
    eg = expert_gene.strip().upper()
    for i, identifier in enumerate(ranked_variants):
        gene_part = re.split(r"[\s(]", identifier)[0].strip().upper()
        if gene_part == eg:
            return i + 1
    return None


# ─── auxiliary data loaders ───────────────────────────────────────────────────

def _load_pheno_data(pheno_csv: Path) -> dict[str, dict]:
    """
    Load phenotype type and n_variants from pheno_prompts.csv.

    Phenotype type is determined by:
      - An explicit 'phenotype_type' column if present, or
      - Inferred: prompt == "visual impairment" → generic, otherwise specific.
    """
    data: dict[str, dict] = {}
    if not pheno_csv.exists():
        logger.warning("[analyse] pheno_prompts.csv not found at %s — phenotype fields will be empty", pheno_csv)
        return data
    with open(pheno_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get("patient_id", "").strip()
            if not pid:
                continue
            prompt = row.get("prompt", "").strip()
            if "phenotype_type" in row:
                pt = row["phenotype_type"].strip()
            else:
                pt = "generic" if prompt.lower().strip() == "visual impairment" else "specific"
            data[pid] = {
                "phenotype_type": pt,
                "n_variants": row.get("n_variants", ""),
                "prompt": prompt,
            }
    logger.info("[analyse] Loaded pheno data for %d patients from %s", len(data), pheno_csv)
    return data


def _load_ground_truth(gt_csv: Path | None) -> dict[str, str]:
    """
    Load expert gene names from a ground truth CSV.

    Expected columns: patient_id, expert_gene  (or expert_variant as alias).
    Returns {patient_id: expert_gene_string}.
    """
    if not gt_csv or not gt_csv.exists():
        if gt_csv:
            logger.warning("[analyse] Ground truth CSV not found at %s", gt_csv)
        return {}
    data: dict[str, str] = {}
    with open(gt_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get("patient_id", "").strip()
            gene = (row.get("expert_gene") or row.get("expert_variant") or "").strip()
            if pid:
                data[pid] = gene
    logger.info("[analyse] Loaded ground truth for %d patients from %s", len(data), gt_csv)
    return data


# ─── result discovery ────────────────────────────────────────────────────────

def _extract_stem(folder_name: str) -> str:
    return _TIMESTAMP_RE.sub("", folder_name)


def _find_latest_reports(results_dir: Path) -> dict[str, Path]:
    """
    For each patient stem, return the report.txt from the most recent run.
    Skips subdirectories with no report.txt or unparseable timestamps.
    """
    best: dict[str, tuple[str, Path]] = {}

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

def _process_patient(
    stem: str,
    report_path: Path,
    llm,
    pheno_data: dict[str, dict],
    gt_data: dict[str, str],
) -> dict:
    """
    Process one patient report. Returns a per-case dict with all §2 fields.
    """
    text = report_path.read_text(encoding="utf-8")

    # SLM-based variant ranking from final conclusion
    conclusion = _extract_final_conclusion(text)
    ranked_variants: list[str] = []

    if conclusion:
        logger.info("[analyse] Ranking variants for %s ...", stem)
        prompt   = _RANKING_PROMPT.format(text=conclusion)
        response = llm.generate(
            system=_RANKING_SYSTEM,
            user=prompt,
            max_tokens=200,
            temperature=0.0,
        )
        ranked_variants = _parse_ranked_variants(response)
        logger.info("[analyse] %s → %d ranked: %s", stem, len(ranked_variants), ranked_variants)
    else:
        logger.warning("[analyse] No final conclusion found in %s", report_path)

    # Stage funnel parsing
    funnel     = _parse_report_funnel(text)
    expert_gene = gt_data.get(stem, "")
    stage_info  = _expert_stage_info(funnel, expert_gene or None)
    expert_rank = _find_expert_rank(ranked_variants, expert_gene or None)

    # Phenotype metadata
    pheno = pheno_data.get(stem, {})

    return {
        "patient_id":                    stem,
        "phenotype_type":                pheno.get("phenotype_type", ""),
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract per-case analysis data from pipeline reports."
    )
    parser.add_argument(
        "--results-dir", type=Path, default=_DEFAULT_RESULTS_DIR, metavar="DIR",
        help=f"Folder containing per-patient result subdirectories (default: {_DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--pheno-csv", type=Path, default=_DEFAULT_PHENO_CSV, metavar="CSV",
        help=f"pheno_prompts.csv with patient_id, prompt, n_variants columns (default: {_DEFAULT_PHENO_CSV})",
    )
    parser.add_argument(
        "--ground-truth", type=Path, default=None, metavar="CSV",
        help="Ground truth CSV with patient_id and expert_gene columns (optional)",
    )
    parser.add_argument(
        "--vllm-url", default="http://localhost:8001", metavar="URL",
        help="vLLM HTTP server base URL (default: http://localhost:8001)",
    )
    parser.add_argument(
        "--workers", type=int, default=8, metavar="N",
        help="Number of parallel SLM workers (default: 8)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR, metavar="DIR",
        help=f"Directory for the output CSV (default: {_DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    if not args.results_dir.is_dir():
        print(f"ERROR: {args.results_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    reports = _find_latest_reports(args.results_dir)
    if not reports:
        print(f"ERROR: No valid reports found in {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    logger.info("[analyse] %d patient(s) found.", len(reports))

    pheno_data = _load_pheno_data(args.pheno_csv)
    gt_data    = _load_ground_truth(args.ground_truth)

    if gt_data:
        logger.info("[analyse] Ground truth loaded for %d patient(s).", len(gt_data))
    else:
        logger.info("[analyse] No ground truth provided — expert_rank and stage survival will be empty.")

    from pipeline.llm.vllm_client import VLLMClient
    llm = VLLMClient(base_url=args.vllm_url)

    if not llm.is_available():
        print(f"ERROR: vLLM server not reachable at {args.vllm_url}", file=sys.stderr)
        sys.exit(1)

    # Parallel SLM calls — one per patient
    all_results: dict[str, dict] = {}
    workers = min(args.workers, len(reports))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_process_patient, stem, path, llm, pheno_data, gt_data): stem
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

    today   = datetime.now().strftime("%Y%m%d")
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"results_table_{today}.csv"

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for stem in sorted(all_results):
            writer.writerow(all_results[stem])

    logger.info("[analyse] Saved → %s", out_path)
    print(out_path)


if __name__ == "__main__":
    main()
