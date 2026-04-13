"""
pipeline/core/normalizer.py — Accept any CSV/Excel variant file and normalize it to the
standard TARGET_COLUMNS schema expected by the pipeline.

Unknown columns are silently ignored.
Missing target columns are filled with "NA".

Public function:
    normalize_upload(raw: bytes, filename: str) -> list[str]
        Accepts raw file bytes + original filename (used to detect .xlsx vs .csv).
        Returns a list of key=value variant strings ready for the pipeline.
"""

import io
import re
import csv
from typing import Optional

import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMA
# ═══════════════════════════════════════════════════════════════════════════════

TARGET_COLUMNS = [
    "Variant", "Chromosome", "Position", "RS_ID", "Ref_seq", "Var_seq",
    "Type", "HGVS", "Zygosity", "Gene", "OMIM_phenotype",
    "OMIM_inheritance", "Inheritance", "ClinVar_class",
    "Allelic_balance", "Frequency", "CADD_score",
]

COLUMN_ALIASES = {
    # Variant
    "variant": "Variant", "variant_id": "Variant", "snp": "Variant",
    "mutation": "Variant", "variant_name": "Variant",
    # Chromosome
    "chr": "Chromosome", "chrom": "Chromosome", "chromosome": "Chromosome",
    "contig": "Chromosome",
    # Position
    "position": "Position", "pos": "Position", "start": "Position",
    "genomic_position": "Position", "coord": "Position",
    # RS_ID
    "rs_id": "RS_ID", "rsid": "RS_ID", "dbsnp": "RS_ID",
    "rs": "RS_ID", "snp_id": "RS_ID",
    # Ref_seq
    "ref_seq": "Ref_seq", "ref": "Ref_seq", "reference": "Ref_seq",
    "ref_allele": "Ref_seq", "reference_allele": "Ref_seq",
    # Var_seq
    "var_seq": "Var_seq", "alt": "Var_seq", "alt_seq": "Var_seq",
    "alternate": "Var_seq", "alt_allele": "Var_seq",
    "alternate_allele": "Var_seq", "obs": "Var_seq",
    # Type
    "type": "Type", "variant_type": "Type", "mutation_type": "Type",
    "class": "Type", "var_type": "Type",
    # HGVS
    "hgvs": "HGVS", "cdna": "HGVS", "hgvs_c": "HGVS",
    "hgvs_p": "HGVS", "c_dot": "HGVS", "p_dot": "HGVS",
    "nucleotide_change": "HGVS",
    # Zygosity
    "zygosity": "Zygosity", "genotype": "Zygosity", "gt": "Zygosity",
    "zyg": "Zygosity",
    # Gene
    "gene": "Gene", "gene_name": "Gene", "gene_symbol": "Gene",
    "hugo": "Gene", "symbol": "Gene",
    # OMIM_phenotype
    "omim_phenotype": "OMIM_phenotype", "omim": "OMIM_phenotype",
    "phenotype": "OMIM_phenotype", "disease": "OMIM_phenotype",
    "condition": "OMIM_phenotype", "disorder": "OMIM_phenotype",
    # OMIM_inheritance
    "omim_inheritance": "OMIM_inheritance",
    # Inheritance
    "inheritance": "Inheritance", "inheritance_pattern": "Inheritance",
    "mode_of_inheritance": "Inheritance", "moi": "Inheritance",
    # ClinVar_class
    "clinvar_class": "ClinVar_class", "clinvar": "ClinVar_class",
    "clinical_significance": "ClinVar_class", "classification": "ClinVar_class",
    "pathogenicity": "ClinVar_class", "clinsig": "ClinVar_class",
    "interp": "ClinVar_class",
    # Allelic_balance
    "allelic_balance": "Allelic_balance", "ab": "Allelic_balance",
    "vaf": "Allelic_balance", "allele_fraction": "Allelic_balance",
    "allele_balance": "Allelic_balance",
    # Frequency
    "frequency": "Frequency", "gnomad": "Frequency", "maf": "Frequency",
    "af": "Frequency", "allele_frequency": "Frequency",
    "population_frequency": "Frequency", "gnomad_af": "Frequency",
    "exac_af": "Frequency",
    # CADD_score
    "cadd_score": "CADD_score", "cadd": "CADD_score", "phred": "CADD_score",
    "cadd_phred": "CADD_score",
}

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _clean(v) -> str:
    """Normalize a cell value to a clean string or 'NA'."""
    s = str(v).strip()
    return "NA" if s in ("", "nan", "NaN", "None", "none", "NULL", "null", ".") else s


def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns to TARGET_COLUMNS names using COLUMN_ALIASES.
    Columns that don't match any alias are kept as-is (they'll be ignored later).
    """
    df.columns = [str(c) for c in df.columns]
    mapping = {}
    for col in df.columns:
        canonical = COLUMN_ALIASES.get(col.strip().lower())
        if canonical:
            mapping[col] = canonical
    if mapping:
        df = df.rename(columns=mapping)
    return df


def _build_normalized_df(df: pd.DataFrame) -> pd.DataFrame:
    """Project df onto TARGET_COLUMNS, filling missing ones with 'NA'."""
    out = pd.DataFrame()
    for col in TARGET_COLUMNS:
        if col in df.columns:
            out[col] = df[col].apply(_clean)
        else:
            out[col] = "NA"
    return out


def _df_to_variant_strings(df: pd.DataFrame) -> list[str]:
    """Convert each row of a normalized DataFrame to a key=value string."""
    variants = []
    for _, row in df.iterrows():
        parts = [f"{col}={row[col]}" for col in TARGET_COLUMNS]
        variants.append(", ".join(parts))
    return variants


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_upload(raw: bytes, filename: str = "upload.csv") -> list[str]:
    """
    Accept raw file bytes from any CSV or Excel upload and return a list of
    normalized key=value variant strings for the pipeline.

    Supports:
      - .csv  (any delimiter — auto-detected)
      - .xlsx / .xls / .xlsm

    Raises:
      ValueError  if the file cannot be parsed or contains no rows.
    """
    fname = filename.lower()

    # ── Load into DataFrame ───────────────────────────────────────────────────
    try:
        if fname.endswith((".xlsx", ".xls", ".xlsm")):
            df = pd.read_excel(io.BytesIO(raw), dtype=str)
        else:
            # Try comma first, then tab, then semicolon
            text = raw.decode("utf-8", errors="replace")
            for sep in (",", "\t", ";"):
                try:
                    df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str)
                    if len(df.columns) > 1:
                        break
                except Exception:
                    continue
            else:
                raise ValueError("Could not parse CSV with common delimiters (,  \\t  ;)")
    except Exception as e:
        raise ValueError(f"File parsing failed: {e}") from e

    if df.empty:
        raise ValueError("File contains no data rows.")

    # ── Normalize ─────────────────────────────────────────────────────────────
    df = _map_columns(df)
    df = _build_normalized_df(df)

    if df.empty:
        raise ValueError("No variant rows found after normalization.")

    variants = _df_to_variant_strings(df)
    print(f"[csv_normalizer] {len(variants)} variants loaded from '{filename}'")

    # Log which target columns were actually populated
    populated = [c for c in TARGET_COLUMNS if any(df[c] != "NA")]
    missing   = [c for c in TARGET_COLUMNS if c not in populated]
    print(f"[csv_normalizer] Populated columns : {populated}")
    if missing:
        print(f"[csv_normalizer] Missing (→ NA)   : {missing}")

    return variants


# ═══════════════════════════════════════════════════════════════════════════════
# CLI convenience  (python -m pipeline.core.normalizer input.csv)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.core.normalizer <input_file> [output_file]")
        sys.exit(1)

    input_path  = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "normalized_output.csv"

    with open(input_path, "rb") as f:
        raw = f.read()

    variants = normalize_upload(raw, filename=input_path)

    # Write as CSV for inspection
    import csv as _csv
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = _csv.writer(f)
        writer.writerow(TARGET_COLUMNS)
        for v in variants:
            row = {}
            for part in v.split(", "):
                k, _, val = part.partition("=")
                row[k] = val
            writer.writerow([row.get(c, "NA") for c in TARGET_COLUMNS])

    print(f"Saved normalized CSV to {output_path}")
