#!/usr/bin/env python3
"""
augment_litvar2.py — Augment a results_table CSV with LITVAR2 failure metadata.

Reads existing report.txt files from a results directory (the same directory
used by analyse_results.py), parses the PROCESS DETAILS section of each report
to detect LITVAR2_SUMMARY failures (including 429 rate-limit bursts), and adds
new columns to the CSV produced by analyse_results.py.

No GPU, no vLLM, no pipeline imports required — pure standard-library regex.

New columns added to the output CSV:
  n_litvar2_failures    int   — count of variants where LITVAR2 failed (any error)
  n_litvar2_429         int   — subset where the failure was a 429 rate-limit error
  litvar2_failed_genes  str   — JSON list of genes with LITVAR2 failures
  expert_litvar2_failed bool  — True if the expert gene had any LITVAR2 failure
  expert_litvar2_429    bool  — True if the expert gene specifically hit a 429

Usage:
    python augment_litvar2.py
    python augment_litvar2.py --results-dir /path/to/results --csv results_table_20260610.csv
    python augment_litvar2.py --output-dir /path/to/out --suffix _aug
"""

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_DEFAULT_RESULTS_DIR = Path(__file__).parent.parent / "results"
_DEFAULT_OUTPUT_DIR  = Path(__file__).parent

# Unicode box-drawing character used as separator in reports (U+2500, repeated 56×)
_SEP56 = "─" * 56

# Folder names follow the pattern: {patient_stem}_{YYYYMMDD_HHMMSS}
_TIMESTAMP_RE = re.compile(r"_(\d{8}_\d{6})$")

# New columns appended by this script (in insertion order)
_NEW_COLUMNS = [
    "n_litvar2_failures",
    "n_litvar2_429",
    "litvar2_failed_genes",
    "expert_litvar2_failed",
    "expert_litvar2_429",
]


# ─── report discovery ─────────────────────────────────────────────────────────

def _extract_stem(folder_name: str) -> str:
    return _TIMESTAMP_RE.sub("", folder_name)


def _find_latest_reports(results_dir: Path) -> dict[str, Path]:
    """
    For each patient stem, return the report.txt from the most recent run folder.
    Mirrors the logic in analyse_results.py so both scripts operate on the same set.
    """
    best: dict[str, tuple[str, Path]] = {}
    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir():
            continue
        report = subdir / "report.txt"
        if not report.exists():
            continue
        ts_match = _TIMESTAMP_RE.search(subdir.name)
        if not ts_match:
            logger.warning("Cannot parse timestamp from %s — skipping", subdir.name)
            continue
        stem = _extract_stem(subdir.name)
        ts   = ts_match.group(1)
        existing = best.get(stem)
        if existing is None or ts > existing[0]:
            best[stem] = (ts, report)
    return {stem: report for stem, (_, report) in best.items()}


# ─── LITVAR2 failure parsing ───────────────────────────────────────────────────

def _parse_litvar2_failures(report_text: str) -> dict:
    """
    Parse the PROCESS DETAILS section of a report.txt to find LITVAR2_SUMMARY
    failures.

    The report format inside PROCESS DETAILS is:

        ────────────────────────────────────────────────────────
        VARIANT N — GENE  HGVS
        ────────────────────────────────────────────────────────

        [LITVAR2_SUMMARY]  GATE: PASS
          Raw output:
            [FAILED — ToolFetchError]: PubMed esearch failed for gene=GENE: 429 ...

        [SPLICEAI]  GATE: PASS
          ...

    Returns a dict:
        n_total      : int          — LITVAR2 failures for any error type
        n_429        : int          — subset where failure was a 429 rate-limit
        failed_genes : list[str]    — deduplicated gene names with failures
        all_failures : list[dict]   — per-failure details (gene, error_type, is_429, message)
    """
    # Extract the PROCESS DETAILS section (ends at the next ======= banner)
    process_m = re.search(
        r"={60}\nPROCESS DETAILS\n={60}\n(.*?)(?=\n={60}|\Z)",
        report_text,
        re.DOTALL,
    )
    if not process_m:
        logger.debug("No PROCESS DETAILS section found in report")
        return {"n_total": 0, "n_429": 0, "failed_genes": [], "all_failures": []}

    process_text = process_m.group(1)

    # Split process text into per-variant chunks.
    # Each variant block starts with:  \n─{56}\nVARIANT N — GENE  HGVS\n─{56}
    # The split pattern removes "\n─{56}\nVARIANT N — " so each chunk starts
    # with "GENE  HGVS\n─{56}\n..." and contains all the tool entries for that
    # variant.  The preamble before the first VARIANT (index 0) is discarded.
    variant_chunks = re.split(
        r"\n" + re.escape(_SEP56) + r"\nVARIANT \d+ — ",
        process_text,
    )

    all_failures: list[dict] = []

    for chunk in variant_chunks[1:]:  # skip preamble before first VARIANT
        # First non-whitespace token = gene symbol (may be "GENE;GENE" for compound-het)
        gene_m = re.match(r"([^\s]+)", chunk)
        gene   = gene_m.group(1) if gene_m else "?"

        # Each tool block starts with "\n[TOOL_NAME]  GATE: ..."
        # Find all LITVAR2_SUMMARY blocks that have GATE: PASS (i.e. the tool ran)
        # Capture everything up to the next tool block or end of chunk.
        for litvar2_m in re.finditer(
            r"\[LITVAR2_SUMMARY\]\s+GATE: PASS\n(.*?)(?=\n\[[A-Z]|\Z)",
            chunk,
            re.DOTALL,
        ):
            content = litvar2_m.group(1)

            # Detect failure: raw output line starting with "    [FAILED — TYPE]: message"
            # (4-space indent from _build_process_details)
            fail_m = re.search(r"\[FAILED — ([^\]]+)\]:\s*(.+)", content)
            if fail_m:
                error_type = fail_m.group(1)
                error_msg  = fail_m.group(2).strip()
                is_429 = (
                    "429" in error_msg
                    or "too many requests" in error_msg.lower()
                )
                all_failures.append({
                    "gene":       gene,
                    "error_type": error_type,
                    "is_429":     is_429,
                    "message":    error_msg[:300],
                })

    # Deduplicate gene list while preserving encounter order
    failed_genes = list(dict.fromkeys(f["gene"] for f in all_failures))

    return {
        "n_total":      len(all_failures),
        "n_429":        sum(1 for f in all_failures if f["is_429"]),
        "failed_genes": failed_genes,
        "all_failures": all_failures,
    }


# ─── expert gene comparison ────────────────────────────────────────────────────

def _parse_gene_set(raw: str | None) -> set[str]:
    """
    Split a gene string (possibly "GENE" or "GENE;GENE2") into a set of
    uppercased symbols. Returns empty set for blank input.
    """
    if not raw:
        return set()
    return {g.strip().upper() for g in raw.split(";") if g.strip()}


def _expert_litvar2_status(expert_gene: str, failed_genes: list[str]) -> tuple[bool, bool]:
    """
    Return (any_failure, any_429) for whether any expert gene appears in the
    failed-genes list.
    """
    if not expert_gene:
        return False, False

    expert_genes = _parse_gene_set(expert_gene)
    any_fail = False
    any_429  = False

    for fg in failed_genes:
        fg_genes = _parse_gene_set(fg)
        if expert_genes & fg_genes:
            any_fail = True

    return any_fail, any_fail  # second value requires all_failures for 429 check


def _expert_litvar2_status_full(
    expert_gene: str,
    all_failures: list[dict],
) -> tuple[bool, bool]:
    """
    Richer version: returns (any_litvar2_failure, any_429_failure) for the
    expert gene, using the full per-failure list.
    """
    if not expert_gene:
        return False, False

    expert_genes = _parse_gene_set(expert_gene)
    any_fail = False
    any_429  = False

    for failure in all_failures:
        fail_genes = _parse_gene_set(failure["gene"])
        if expert_genes & fail_genes:
            any_fail = True
            if failure["is_429"]:
                any_429 = True

    return any_fail, any_429


# ─── CSV loading ───────────────────────────────────────────────────────────────

def _find_results_csv(csv_arg: Path | None, output_dir: Path) -> Path | None:
    """
    Locate the results CSV.  Precedence:
      1. Explicit --csv argument
      2. Most recent results_table_*.csv in the output directory
    """
    if csv_arg is not None:
        if not csv_arg.exists():
            logger.error("Specified CSV not found: %s", csv_arg)
            return None
        return csv_arg

    candidates = sorted(output_dir.glob("results_table_*.csv"), reverse=True)
    if not candidates:
        logger.error(
            "No results_table_*.csv found in %s — pass --csv explicitly", output_dir
        )
        return None
    logger.info("Auto-selected CSV: %s", candidates[0])
    return candidates[0]


def _load_csv(path: Path) -> tuple[list[str], list[dict]]:
    """Load a CSV, returning (fieldnames, rows)."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


# ─── entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Augment a results_table CSV with LITVAR2_SUMMARY failure metadata "
            "extracted from report.txt files. No GPU or vLLM required."
        )
    )
    parser.add_argument(
        "--results-dir", type=Path, default=_DEFAULT_RESULTS_DIR, metavar="DIR",
        help=f"Folder containing per-patient result subdirectories (default: {_DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--csv", type=Path, default=None, metavar="CSV",
        help=(
            "Path to the results_table_*.csv produced by analyse_results.py. "
            "If omitted, the most recent results_table_*.csv in --output-dir is used."
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR, metavar="DIR",
        help=f"Directory for the augmented output CSV (default: {_DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--suffix", default="_litvar2", metavar="SUFFIX",
        help=(
            "Suffix appended to the input CSV stem for the output file "
            "(default: '_litvar2'). Example: results_table_20260610_litvar2.csv"
        ),
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-variant failure details to stderr.",
    )
    args = parser.parse_args()

    if not args.results_dir.is_dir():
        print(f"ERROR: {args.results_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Locate input CSV
    csv_path = _find_results_csv(args.csv, args.output_dir)
    if csv_path is None:
        sys.exit(1)

    # Load existing CSV
    fieldnames, rows = _load_csv(csv_path)
    logger.info("Loaded %d rows from %s", len(rows), csv_path)

    # Build patient_id → row index lookup
    pid_to_row: dict[str, dict] = {r["patient_id"]: r for r in rows if r.get("patient_id")}

    # Find all report.txt files
    reports = _find_latest_reports(args.results_dir)
    logger.info("Found %d patient result directories in %s", len(reports), args.results_dir)

    # Parse LITVAR2 failures from each report
    n_processed = 0
    n_with_failures = 0
    n_expert_affected = 0

    for stem, report_path in sorted(reports.items()):
        try:
            text = report_path.read_text(encoding="utf-8")
        except OSError as e:
            logger.warning("Cannot read %s: %s", report_path, e)
            continue

        failure_info = _parse_litvar2_failures(text)
        n_processed += 1

        if failure_info["n_total"] > 0:
            n_with_failures += 1
            logger.info(
                "%s: %d LITVAR2 failure(s) (%d×429), genes: %s",
                stem,
                failure_info["n_total"],
                failure_info["n_429"],
                failure_info["failed_genes"],
            )
            if args.verbose:
                for f in failure_info["all_failures"]:
                    print(
                        f"  [{stem}] gene={f['gene']} type={f['error_type']} "
                        f"is_429={f['is_429']} | {f['message'][:120]}",
                        file=sys.stderr,
                    )

        # Match to CSV row and inject new columns
        row = pid_to_row.get(stem)
        if row is None:
            logger.warning("%s: report found but no matching row in CSV — skipping", stem)
            continue

        expert_gene = row.get("expert_gene", "")
        expert_fail, expert_429 = _expert_litvar2_status_full(
            expert_gene, failure_info["all_failures"]
        )

        row["n_litvar2_failures"]    = failure_info["n_total"]
        row["n_litvar2_429"]         = failure_info["n_429"]
        row["litvar2_failed_genes"]  = json.dumps(
            failure_info["failed_genes"], ensure_ascii=False
        )
        row["expert_litvar2_failed"] = expert_fail
        row["expert_litvar2_429"]    = expert_429

        if expert_fail:
            n_expert_affected += 1

    # Fill missing new columns (reports not found, or no matching report)
    for row in rows:
        for col in _NEW_COLUMNS:
            if col not in row:
                row[col] = ""

    # Build output CSV columns: original columns + new columns (skip duplicates)
    out_columns = fieldnames.copy()
    for col in _NEW_COLUMNS:
        if col not in out_columns:
            out_columns.append(col)

    # Write output
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_stem = csv_path.stem + args.suffix
    out_path = args.output_dir / (out_stem + ".csv")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=out_columns, extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Summary
    logger.info(
        "Done — %d reports parsed, %d case(s) with LITVAR2 failures, "
        "%d case(s) where expert gene was affected.",
        n_processed, n_with_failures, n_expert_affected,
    )
    logger.info("Augmented CSV saved → %s", out_path)
    print(out_path)


if __name__ == "__main__":
    main()
