"""
pipeline/core/normalizer.py — Accept any CSV/Excel variant file and normalize it to the
standard TARGET_COLUMNS schema expected by the pipeline.

Unknown columns are silently ignored.
Missing target columns are filled with "NA".

Public function:
    normalize_upload(raw: bytes, filename: str) -> tuple[list[dict], list[dict]]
        Accepts raw file bytes + original filename (used to detect .xlsx vs .csv).
        Returns (normalized_variant_dicts, raw_field_dicts).
"""

import io
import re

import pandas as pd


def normalize_gene(gene_field: str) -> str:
    """
    Return the first non-empty gene symbol from a field that may contain
    duplicates or multiple genes separated by ';', ',' or '/'.
    e.g. "BRCA1;BRCA1" -> "BRCA1",  "BRCA1;BRCA2" -> "BRCA1",  "BRCA1" -> "BRCA1"
    """
    for token in re.split(r"[;,/]", gene_field):
        token = token.strip()
        if token and token.upper() != "NA":
            return token
    return gene_field.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMA
# ═══════════════════════════════════════════════════════════════════════════════

TARGET_COLUMNS = [
    "Variant", "Chromosome", "Position", "RS_ID", "Ref_seq", "Var_seq",
    "Type", "HGVS", "Zygosity", "Gene", "OMIM_phenotype",
    "OMIM_inheritance", "Inheritance", "ClinVar_class",
    "Allelic_balance", "Frequency", "CADD_score", "SpliceAI_score",
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
    "rs": "RS_ID", "snp_id": "RS_ID", "rs id": "RS_ID",
    "avsnp150": "RS_ID",
    # Ref_seq
    "ref_seq": "Ref_seq", "ref": "Ref_seq", "reference": "Ref_seq",
    "ref_allele": "Ref_seq", "reference_allele": "Ref_seq",
    # Var_seq
    "var_seq": "Var_seq", "alt": "Var_seq", "alt_seq": "Var_seq",
    "alternate": "Var_seq", "alt_allele": "Var_seq",
    "alternate_allele": "Var_seq", "obs": "Var_seq",
    # Type — ExonicFunc takes priority; Func is a lower-priority fallback
    "type": "Type", "variant_type": "Type", "mutation_type": "Type",
    "class": "Type", "var_type": "Type",
    "exonicfunc": "Type",
    "func": "Type",
    # HGVS
    "hgvs": "HGVS", "cdna": "HGVS", "hgvs_c": "HGVS",
    "hgvs_p": "HGVS", "c_dot": "HGVS", "p_dot": "HGVS",
    "nucleotide_change": "HGVS",
    # Zygosity
    "zygosity": "Zygosity", "genotype": "Zygosity", "gt": "Zygosity",
    "zyg": "Zygosity",
    # Gene
    "gene": "Gene", "gene_name": "Gene", "gene_symbol": "Gene",
    "hugo": "Gene", "symbol": "Gene", "genes": "Gene",
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
    "clnsig": "ClinVar_class",
    # Allelic_balance
    "allelic_balance": "Allelic_balance", "ab": "Allelic_balance",
    "vaf": "Allelic_balance", "allele_fraction": "Allelic_balance",
    "allele_balance": "Allelic_balance",
    # Frequency
    "frequency": "Frequency", "gnomad": "Frequency", "maf": "Frequency",
    "af": "Frequency", "allele_frequency": "Frequency",
    "population_frequency": "Frequency", "gnomad_af": "Frequency",
    "exac_af": "Frequency",
    "gnomad30_af_popmax": "Frequency",
    "gnomad211_exome_af": "Frequency",
    # CADD_score
    "cadd_score": "CADD_score", "cadd": "CADD_score", "phred": "CADD_score",
    "cadd_phred": "CADD_score",
    "cadd_v17_phred": "CADD_score",
    # SpliceAI_score — pre-computed max delta score from any source
    "spliceai_score": "SpliceAI_score", "spliceai": "SpliceAI_score",
    "spliceai_v13": "SpliceAI_score", "spliceai_max_score": "SpliceAI_score",
    "spliceai_maxscore": "SpliceAI_score", "spliceai_ds_max": "SpliceAI_score",
    "spliceai_max": "SpliceAI_score",
}

# ═══════════════════════════════════════════════════════════════════════════════
# TYPE NORMALISATION MAP  (ANNOVAR ExonicFunc/Func → pipeline vocabulary)
# ═══════════════════════════════════════════════════════════════════════════════

_TYPE_MAP = {
    "nonsynonymous snv":          "SNV",
    "synonymous snv":             "synonymous",
    "stopgain":                   "nonsense",
    "stoploss":                   "stoploss",
    "frameshift substitution":    "frameshift",
    "frameshift insertion":       "frameshift",
    "frameshift deletion":        "frameshift",
    "nonframeshift substitution": "indel",
    "nonframeshift insertion":    "indel",
    "nonframeshift deletion":     "indel",
    "splicing":                   "splicing",
    "exonic":                     "SNV",
    "intronic":                   "intronic",
    "intergenic":                 "intergenic",
    "upstream":                   "upstream",
    "downstream":                 "downstream",
    "utr3":                       "utr3",
    "utr5":                       "utr5",
    "ncrna_exonic":               "synonymous",
    "ncrna_intronic":             "intronic",
}


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _clean(v) -> str:
    """Normalize a cell value to a clean string or 'NA'."""
    s = str(v).strip()
    return "NA" if s in ("", "nan", "NaN", "None", "none", "NULL", "null", ".") else s


def _normalize_type(raw: str) -> str:
    """Map ANNOVAR ExonicFunc/Func values to pipeline Type vocabulary."""
    return _TYPE_MAP.get(raw.strip().lower(), raw)


def _parse_spliceai_score(raw: str) -> str:
    """Normalize SpliceAI_score: ANNOVAR writes four pipe-separated delta scores → extract max."""
    if raw == "NA":
        return "NA"
    try:
        vals = [float(x) for x in raw.split("|") if x.strip() not in ("", ".")]
        return str(round(max(vals), 4)) if vals else "NA"
    except ValueError:
        return raw  # already a plain float string


def _parse_aachange(raw: str) -> str:
    """Extract a display HGVS string from ANNOVAR AAChange annotation."""
    if not raw or raw in ("NA", ".", "", "nan"):
        return "NA"
    first = raw.split("|")[0]
    parts = first.split(":")
    # expected: GENE, NM_xxx, exonN, c.xxx, p.xxx
    if len(parts) >= 5:
        return f"{parts[1]}:{parts[3]} {parts[4]}"
    elif len(parts) >= 4:
        return f"{parts[1]}:{parts[3]}"
    return "NA"


def _extract_raw_fields(df: pd.DataFrame) -> list[dict]:
    """Return full original rows as dicts, one per variant."""
    return [
        {col: _clean(row[col]) for col in df.columns}
        for _, row in df.iterrows()
    ]


def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns to TARGET_COLUMNS names using COLUMN_ALIASES.
    Columns that don't match any alias are kept as-is (they'll be ignored later).
    When two columns alias to the same target, ExonicFunc takes priority over Func
    for the Type field; otherwise first-encountered wins.
    """
    df.columns = [str(c) for c in df.columns]
    claimed: set[str] = set()
    mapping: dict[str, str] = {}

    # Sort so ExonicFunc is processed before Func, enforcing Type field priority
    cols = sorted(
        df.columns,
        key=lambda c: (0 if c.strip().lower() == "exonicfunc" else 1),
    )

    for col in cols:
        canonical = COLUMN_ALIASES.get(col.strip().lower())
        if canonical and canonical not in claimed:
            mapping[col] = canonical
            claimed.add(canonical)

    if mapping:
        df = df.rename(columns=mapping)
    return df


def _build_normalized_df(df: pd.DataFrame, df_original: pd.DataFrame) -> pd.DataFrame:
    """Project df onto TARGET_COLUMNS, filling missing ones with 'NA'.

    Post-processing:
      - If HGVS is all NA but AAChange exists in the original input, fill from AAChange.
      - Normalise Type values through the ANNOVAR vocabulary map.
    """
    out = pd.DataFrame()
    for col in TARGET_COLUMNS:
        if col in df.columns:
            out[col] = df[col].apply(_clean)
        else:
            out[col] = "NA"

    # Fill HGVS from AAChange if needed
    if (out["HGVS"] == "NA").all():
        aa_col = next(
            (c for c in df_original.columns if c.strip().lower() == "aachange"),
            None,
        )
        if aa_col is not None:
            out["HGVS"] = df_original[aa_col].apply(
                lambda v: _parse_aachange(_clean(v))
            ).values

    # Normalise Type vocabulary
    out["Type"] = out["Type"].apply(
        lambda v: _normalize_type(v) if v != "NA" else v
    )

    # Normalise SpliceAI_score: ANNOVAR pipe-separated "DS_AG|DS_AL|DS_DG|DS_DL" → max
    out["SpliceAI_score"] = out["SpliceAI_score"].apply(_parse_spliceai_score)

    return out


def _df_to_variant_dicts(df: pd.DataFrame) -> list[dict]:
    """Convert each row of a normalized DataFrame to a plain dict."""
    return [
        {col: row[col] for col in TARGET_COLUMNS}
        for _, row in df.iterrows()
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_upload(
    raw: bytes,
    filename: str = "upload.csv",
) -> tuple[list[dict], list[dict]]:
    """
    Accept raw file bytes from any CSV or Excel upload and return a tuple of
    (normalized_variant_dicts, raw_field_dicts).

    - normalized_variant_dicts: one dict per row, keyed by TARGET_COLUMNS
    - raw_field_dicts:          one dict per row, full original columns before normalization

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
    df_original = df.copy()          # preserve original before column mapping
    df = _map_columns(df)
    df = _build_normalized_df(df, df_original)

    if df.empty:
        raise ValueError("No variant rows found after normalization.")

    variants  = _df_to_variant_dicts(df)
    raw_rows  = _extract_raw_fields(df_original)

    print(f"[csv_normalizer] {len(variants)} variants loaded from '{filename}'")

    # Log which target columns were actually populated
    populated = [c for c in TARGET_COLUMNS if any(df[c] != "NA")]
    missing   = [c for c in TARGET_COLUMNS if c not in populated]
    print(f"[csv_normalizer] Populated columns : {populated}")
    if missing:
        print(f"[csv_normalizer] Missing (→ NA)   : {missing}")

    return variants, raw_rows


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

    variants, _ = normalize_upload(raw, filename=input_path)

    # Write as CSV for inspection
    import csv as _csv
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = _csv.writer(f)
        writer.writerow(TARGET_COLUMNS)
        for v in variants:
            writer.writerow([v.get(c, "NA") for c in TARGET_COLUMNS])

    print(f"Saved normalized CSV to {output_path}")
