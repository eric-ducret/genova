#!/usr/bin/env python3
"""
augment_pheno_acmg.py — Add ground-truth ACMG (2015) info to pheno_prompts.csv.

Reads the expert-annotated 557_genes.xlsx (one row per reported gene per
patient, "MA ID" column) and adds two columns to pheno_prompts.csv, matched
on (patient_id, solution gene):

  acmg_classification  str  — "ACMG [Classification guidlines 2015 ...]" column
                               (e.g. P, LP, VUS, LB, B)
  acmg_criteria        str  — "ACMG Classification Criteria" column
                               (e.g. "PVS1_Very strong, PM2_mod, PP5_sup")

Rows with no solution gene (unsolved cases, is_solution_here=0) get empty
strings for both columns. A few patients report two variants in the same
solution gene (compound-het); both xlsx rows are kept and joined with " | "
so no information is lost.

Usage:
    python augment_pheno_acmg.py
    python augment_pheno_acmg.py --xlsx path/to/557_genes.xlsx --pheno-csv path/to/pheno_prompts.csv
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).parent
_DEFAULT_XLSX = (
    Path(__file__).parent.parent.parent
    / "pakistani_cases" / "557_cases" / "557_genes.xlsx"
)
_DEFAULT_PHENO_CSV = _SCRIPT_DIR.parent / "pheno_prompts.csv"

_XLSX_CLASS_COL    = "ACMG [Classification guidlines 2015 (with latest recommendation, where applicable)]"
_XLSX_CRITERIA_COL = "ACMG Classification Criteria"


def _build_acmg_lookup(xlsx_path: Path) -> dict[tuple[str, str], tuple[str, str]]:
    """
    Return {(patient_id, gene): (classification, criteria)} from the xlsx.

    When a patient has more than one xlsx row for the same gene (compound-het,
    two causative variants), both rows' values are joined with " | " so both
    variants' ACMG info is preserved in a single cell.
    """
    df = pd.read_excel(xlsx_path)

    lookup: dict[tuple[str, str], tuple[list[str], list[str]]] = {}
    for _, row in df.iterrows():
        patient_id = str(row["MA ID"]).strip()
        gene = row.get("Gene")
        if pd.isna(gene) or not str(gene).strip():
            continue
        gene = str(gene).strip()

        classification = row.get(_XLSX_CLASS_COL)
        criteria = row.get(_XLSX_CRITERIA_COL)
        classification = "" if pd.isna(classification) else str(classification).strip()
        criteria = "" if pd.isna(criteria) else str(criteria).strip()

        key = (patient_id, gene)
        classes, crits = lookup.setdefault(key, ([], []))
        if classification:
            classes.append(classification)
        if criteria:
            crits.append(criteria)

    return {
        key: (" | ".join(classes), " | ".join(crits))
        for key, (classes, crits) in lookup.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add ground-truth ACMG classification/criteria columns to pheno_prompts.csv."
    )
    parser.add_argument("--xlsx", type=Path, default=_DEFAULT_XLSX, metavar="XLSX",
                         help=f"Path to 557_genes.xlsx (default: {_DEFAULT_XLSX})")
    parser.add_argument("--pheno-csv", type=Path, default=_DEFAULT_PHENO_CSV, metavar="CSV",
                         help=f"Path to pheno_prompts.csv, edited in place (default: {_DEFAULT_PHENO_CSV})")
    args = parser.parse_args()

    if not args.xlsx.exists():
        print(f"ERROR: {args.xlsx} not found", file=sys.stderr)
        sys.exit(1)
    if not args.pheno_csv.exists():
        print(f"ERROR: {args.pheno_csv} not found", file=sys.stderr)
        sys.exit(1)

    logger.info("Reading ground truth from %s", args.xlsx)
    lookup = _build_acmg_lookup(args.xlsx)

    with open(args.pheno_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    new_columns = ["acmg_classification", "acmg_criteria"]
    for col in new_columns:
        if col not in fieldnames:
            fieldnames.append(col)

    n_matched = 0
    for row in rows:
        patient_id = row.get("patient_id", "").strip()
        solution = (row.get("solution") or "").strip()

        classification, criteria = "", ""
        if solution:
            key = (patient_id, solution)
            if key not in lookup:
                logger.warning(
                    "%s: solution gene %r not found in xlsx", patient_id, solution
                )
            else:
                classification, criteria = lookup[key]
                if not classification and not criteria:
                    logger.info(
                        "%s: xlsx row for %r has no ACMG classification/criteria recorded",
                        patient_id, solution,
                    )
                n_matched += 1

        row["acmg_classification"] = classification
        row["acmg_criteria"] = criteria

    with open(args.pheno_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(
        "Done — %d/%d row(s) matched to ACMG ground truth. Updated %s",
        n_matched, len(rows), args.pheno_csv,
    )


if __name__ == "__main__":
    main()
