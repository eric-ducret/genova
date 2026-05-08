"""
pipeline/tools/autopvs1.py — VCF coords → AutoPVS1 lookup (hg19).

Coords are extracted directly from the canonical variant dict fields
(Chromosome, Position, Ref_seq, Var_seq, Variant) — no Ensembl roundtrip needed.

Public functions (module-level, used internally by AutoPVS1Tool):
  vcf_to_autopvs1_id(chrom, pos, ref, alt)  → "6-100896130-T-C"
  parse_variant_coords(variant_str, ...)     → (chrom, pos, ref, alt) | None
  fetch_variant_data(variant_id)             → dict | None
  fetch_and_format_autopvs1(chrom, pos, ref, alt, hgvs="")
                                             → formatted string block | None

Pipeline class:
  AutoPVS1Tool — NetworkTool; gate uses the hybrid rule+LLM _pvs1_decision logic
                 ported from server_main._pvs1_decision().
"""

import re
import requests
from bs4 import BeautifulSoup

from pipeline.tools.base import NetworkTool
from pipeline.core.context import ToolContext
from pipeline.core.errors import ToolFetchError, ToolParseError


# ═══════════════════════════════════════════════════════════════════════════════
# COORD EXTRACTION  (no network call — pure string parsing)
# ═══════════════════════════════════════════════════════════════════════════════

# SNV: "chr2:21266223 T⇒C"  or  "chr2:21266223 T>C"  or  "chr2:21266223 T-C"
_SNV_RE = re.compile(
    r"chr(\w+):(\d+)\s+([ACGTacgt]+)\s*[=>\u21d2\-]+\s*([ACGTacgt]+)"
)
# Deletion: "chr18:44580785 delA"
_DEL_RE = re.compile(
    r"chr(\w+):(\d+)\s+del([ACGTacgt]+)",
    re.IGNORECASE,
)
# Insertion: "chrX:12345 insACG"
_INS_RE = re.compile(
    r"chr(\w+):(\d+)\s+ins([ACGTacgt]+)",
    re.IGNORECASE,
)
# DelIns: "chr1:12345 ACGdelinsT"
_DELINS_RE = re.compile(
    r"chr(\w+):(\d+)\s+([ACGTacgt]+)\s*delins\s*([ACGTacgt]+)",
    re.IGNORECASE,
)


def vcf_to_autopvs1_id(chrom: str, pos: str, ref: str, alt: str) -> str:
    """
    Build the AutoPVS1 variant ID string from raw VCF-style fields.
    Strips 'chr' prefix, normalises chrom name.
    e.g. ("chr6", "100896130", "T", "C") → "6-100896130-T-C"
    """
    chrom = chrom.strip().lower().removeprefix("chr")
    if chrom.isdigit():
        chrom = str(int(chrom))
    return f"{chrom}-{pos.strip()}-{ref.strip().upper()}-{alt.strip().upper()}"


def parse_variant_coords(
    variant_str: str,
    chrom_field: str = "",
    pos_field:   str = "",
    ref_field:   str = "",
    alt_field:   str = "",
) -> tuple[str, str, str, str] | None:
    """
    Extract (chrom, pos, ref, alt) from available CSV fields.

    Strategy (in order):
      1. If Ref_seq and Var_seq are non-NA → use them directly
      2. Parse the Variant column free-text string
      3. Fall back to chrom+pos from columns + loose base extraction from Variant string

    Returns None if extraction fails entirely.
    """

    def ok(v: str) -> bool:
        return bool(v) and v.strip().upper() not in ("NA", "", ".")

    # ── Strategy 1: explicit Ref_seq / Var_seq columns ────────────────────────
    if ok(ref_field) and ok(alt_field) and ok(chrom_field) and ok(pos_field):
        return chrom_field, pos_field, ref_field, alt_field

    # ── Strategy 2: parse the Variant column string ───────────────────────────
    v = variant_str.strip()

    m = _SNV_RE.match(v)
    if m:
        return f"chr{m.group(1)}", m.group(2), m.group(3).upper(), m.group(4).upper()

    m = _DEL_RE.match(v)
    if m:
        # AutoPVS1 accepts the deleted bases as REF and "-" as ALT for simple deletions
        return f"chr{m.group(1)}", m.group(2), m.group(3).upper(), "-"

    m = _INS_RE.match(v)
    if m:
        return f"chr{m.group(1)}", m.group(2), "-", m.group(3).upper()

    m = _DELINS_RE.match(v)
    if m:
        return f"chr{m.group(1)}", m.group(2), m.group(3).upper(), m.group(4).upper()

    # ── Strategy 3: chrom+pos known, try loose base extraction ────────────────
    if ok(chrom_field) and ok(pos_field):
        bases = re.findall(r"[ACGTacgt]+", v)
        if len(bases) >= 2:
            return chrom_field, pos_field, bases[0].upper(), bases[1].upper()

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# AutoPVS1 fetch + HTML parse
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_variant_data(variant_id: str) -> dict | None:
    url = f"https://autopvs1.bgi.com/variant/hg19/{variant_id}"
    headers = {
        "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
        "User-Agent": "Mozilla/5.0 (compatible; PythonScript/1.0)",
    }
    print(f"[AutoPVS1] Fetching: {url}")
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"[AutoPVS1] Request failed: {e}")
        return None
    return _parse_autopvs1_html(response.text)


def _parse_autopvs1_html(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    data = {}

    h3 = soup.find("h3")
    if h3:
        h3_text = h3.get_text(strip=True)
        m = re.match(r"^([\w\-]+):\s*([\d\w\-]+)$", h3_text)
        if m:
            data["variant_type"] = m.group(1)
            data["variant"]      = m.group(2)
        else:
            data["variant_type"] = None
            data["variant"]      = h3_text

    def get_field(label: str) -> str | None:
        tag = soup.find("b", string=re.compile(rf"^\s*{re.escape(label)}\s*$"))
        if not tag:
            return None
        val = ""
        for sib in tag.next_siblings:
            text = sib.get_text(strip=True) if hasattr(sib, "get_text") else str(sib).strip()
            if text:
                val = text
                break
        return None if val in ("", "-", "na", "N/A") else val

    data["gene"]               = get_field("Gene:")
    data["chgvs"]              = get_field("cHGVS:")
    data["phgvs"]              = get_field("pHGVS:")
    data["exon"]               = get_field("Exon:")
    data["intron"]             = get_field("Intron:")
    data["pli"]                = get_field("pLI:")
    data["haploinsufficiency"] = get_field("Haploinsufficiency:")

    for btn in soup.select("a.btn-primary"):
        label = btn.get_text(strip=True)
        href  = btn.get("href", "")
        if "clinvar" in label.lower():
            m = re.search(r"/variation/(\w+)", href)
            data["clinvar_id"]  = m.group(1) if m and m.group(1) != "na" else None
            data["clinvar_url"] = href if "na" not in href else None
        elif "gnomad" in label.lower():
            data["gnomad_url"] = href
        elif "omim" in label.lower():
            data["omim_url"] = href

    flowchart = soup.find("figure")
    if flowchart:
        caption = flowchart.find("figcaption")
        data["pvs1_path"] = caption.get_text(strip=True) if caption else None
        steps = [
            li.find("code").get_text(" ", strip=True)
            for li in flowchart.find_all("li")
            if li.find("code")
        ]
        data["pvs1_decision_steps"] = steps
        data["pvs1_strength"]       = steps[-1] if steps else None
    else:
        data["pvs1_path"]           = None
        data["pvs1_decision_steps"] = []
        data["pvs1_strength"]       = None

    table = soup.find("table", class_="table-bordered")
    if table:
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        rows = []
        for tr in table.find("tbody").find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            if cells and len(cells) == len(headers):
                rows.append(dict(zip(headers, cells)))
        data["disease_mechanism"] = rows if rows else None
    else:
        data["disease_mechanism"] = None

    full_text = soup.get_text()
    data["pvs1_applicable"] = not bool(
        re.search(r"incompatible with recommendations", full_text, re.I)
    )
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# High-level function  (new signature — no HGVS required, no Ensembl)
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_and_format_autopvs1(
    chrom: str,
    pos:   str,
    ref:   str,
    alt:   str,
    hgvs:  str = "",   # optional — used only as a label in the output block
) -> str | None:
    """
    Full pipeline: VCF coords → AutoPVS1 ID → fetch → formatted evidence block.

    Args:
        chrom : chromosome, e.g. "chr18" or "18"
        pos   : 1-based position string, e.g. "44580785"
        ref   : reference allele, e.g. "A"  (use "-" for pure insertions)
        alt   : alternate allele, e.g. "T"  (use "-" for pure deletions)
        hgvs  : optional label shown in the output header

    Returns a multi-line string ready to embed in the augmented context,
    or None if the fetch failed.
    """
    variant_id = vcf_to_autopvs1_id(chrom, pos, ref, alt)
    print(f"[AutoPVS1] ID: {variant_id}")

    d = fetch_variant_data(variant_id)
    if d is None:
        return None

    def _val(v) -> str:
        return str(v) if v not in (None, "", [], {}) else "—"

    steps_str   = " → ".join(d.get("pvs1_decision_steps") or []) or "—"
    disease_str = "—"
    if d.get("disease_mechanism"):
        disease_str = "; ".join(
            ", ".join(f"{k}: {v}" for k, v in row.items())
            for row in d["disease_mechanism"]
        )

    label = hgvs.split()[0] if hgvs else variant_id
    lines = [
        f"AUTOPVS1 EVIDENCE ({label}):",
        f"  Variant        : {variant_id}  ({_val(d.get('variant_type'))})",
        f"  Gene           : {_val(d.get('gene'))}",
        f"  cHGVS          : {_val(d.get('chgvs'))}",
        f"  pHGVS          : {_val(d.get('phgvs'))}",
        f"  Exon           : {_val(d.get('exon'))}   Intron: {_val(d.get('intron'))}",
        f"  pLI            : {_val(d.get('pli'))}",
        f"  Haploinsuff.   : {_val(d.get('haploinsufficiency'))}",
        f"  PVS1 path      : {_val(d.get('pvs1_path'))}",
        f"  PVS1 steps     : {steps_str}",
        f"  PVS1 strength  : {_val(d.get('pvs1_strength'))}",
        f"  PVS1 applicable: {d.get('pvs1_applicable', '—')}",
        f"  Disease mech.  : {disease_str}",
        f"  ClinVar        : {_val(d.get('clinvar_url'))}",
        f"  gnomAD         : {_val(d.get('gnomad_url'))}",
        f"  OMIM           : {_val(d.get('omim_url'))}",
    ]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# PVS1 GATE CONSTANTS  (ported from server_main._pvs1_decision)
# ═══════════════════════════════════════════════════════════════════════════════

# Variant types that unambiguously warrant PVS1.
# "stoploss" is intentionally excluded: stop-loss variants extend the protein
# rather than truncate it, so PVS1 does not apply and AutoPVS1 would give
# misleading output for that class.
_CLEAR_LOF_TYPES = {
    "frameshift", "nonsense", "stopgain",
    "splice-3", "splice-5", "splice_site", "splicing",
    "start_lost", "initiator_codon", "deletion", "indel",
}

# Variant types that unambiguously do NOT warrant PVS1
_CLEAR_NON_LOF_TYPES = {
    "missense", "synonymous", "silent", "utr", "utr3", "utr5",
    "intergenic", "upstream", "downstream",
}


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE TOOL CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class AutoPVS1Tool(NetworkTool):
    """Fetches ACMG PVS1 criterion evaluation from AutoPVS1 for LoF variants (hg19)."""

    name        = "autopvs1"
    description = "Fetches ACMG PVS1 criterion evaluation from AutoPVS1 for LoF variants (hg19)."

    def gate(self, variant: dict, context: ToolContext) -> bool:
        """
        Three-way hybrid gate — ported from server_main._pvs1_decision():
          - Clearly LoF        → True  (no LLM call)
          - Clearly not LoF    → False (no LLM call)
          - Ambiguous          → ask LLM (max_tokens=5, very cheap)

        Uses Type field first, then HGVS clues, then LLM.
        """
        type_val = variant.get("Type", "").lower()
        hgvs_val = variant.get("HGVS", "").lower()

        # ── Fast YES ─────────────────────────────────────────────────────────
        if any(t in type_val for t in _CLEAR_LOF_TYPES):
            print(f"[PVS1 gate] YES (rule) — type: {type_val}")
            return True

        # HGVS frameshift / stop clues
        if any(c in hgvs_val for c in ["fs", "ter", "del", "dup", "ins", "ext*"]):
            # but exclude if it's clearly synonymous p.(X=)
            if "=" not in hgvs_val:
                print(f"[PVS1 gate] YES (hgvs clue) — hgvs: {hgvs_val}")
                return True

        # ── Fast NO ──────────────────────────────────────────────────────────
        if any(t in type_val for t in _CLEAR_NON_LOF_TYPES):
            print(f"[PVS1 gate] NO (rule) — type: {type_val}")
            return False

        # Synonymous: protein annotation ends with = e.g. p.(Ser47=)
        if re.search(r"p\.\([a-z]+\d+=\)", hgvs_val):
            print(f"[PVS1 gate] NO (synonymous hgvs) — hgvs: {hgvs_val}")
            return False

        # Deep intronic: offset > 10 bp from exon boundary
        deep = re.search(r"[+\-](\d+)[acgt]", hgvs_val)
        if deep and int(deep.group(1)) > 10:
            print(f"[PVS1 gate] NO (deep intronic) — hgvs: {hgvs_val}")
            return False

        # UTR variants
        if re.match(r"c\.\*", hgvs_val) or re.match(r"c\.-", hgvs_val):
            print(f"[PVS1 gate] NO (UTR) — hgvs: {hgvs_val}")
            return False

        # ── Ambiguous → LLM judge ─────────────────────────────────────────────
        print(f"[PVS1 gate] Ambiguous — asking LLM. type={type_val} hgvs={hgvs_val}")
        answer = context.llm.generate(
            system=(
                "You are a clinical genetics assistant. "
                "Answer with a single word: YES or NO. No explanation."
            ),
            user=(
                "Should the PVS1 ACMG criterion be evaluated for this variant?\n"
                f"Type: {type_val}\n"
                f"HGVS: {hgvs_val}\n\n"
                "Answer YES only if this is a loss-of-function variant: frameshift, "
                "nonsense/stop-gain, canonical splice site (±1 or ±2 bp from exon), "
                "start-loss, or exon-spanning deletion/duplication.\n"
                "Answer NO for missense, synonymous, deep intronic, or UTR variants."
            ),
            max_tokens=5,
        )
        decision = "yes" in answer.strip().lower()
        print(f"[PVS1 gate] LLM answered: {answer.strip()!r} → {decision}")
        return decision

    def run(self, variant: dict, context: ToolContext) -> str | None:
        """
        Extract VCF coordinates from the canonical variant dict,
        then call fetch_and_format_autopvs1().
        Returns the formatted evidence block or None if coords cannot be resolved.
        """
        coords = parse_variant_coords(
            variant_str=variant.get("Variant", ""),
            chrom_field=variant.get("Chromosome", ""),
            pos_field=variant.get("Position", ""),
            ref_field=variant.get("Ref_seq", ""),
            alt_field=variant.get("Var_seq", ""),
        )
        if not coords:
            return None

        chrom, pos, ref, alt = coords
        try:
            result = fetch_and_format_autopvs1(
                chrom, pos, ref, alt,
                hgvs=variant.get("HGVS", ""),
            )
        except requests.exceptions.RequestException as e:
            raise ToolFetchError(
                f"AutoPVS1 request failed for {variant.get('Variant', '?')}: {e}"
            ) from e

        return result


# ═══════════════════════════════════════════════════════════════════════════════
# Quick test  (python -m pipeline.tools.autopvs1)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    test_cases = [
        ("chr18:44580785 delA",  "chr18", "44580785", "A", "-"),
        ("chr19:11242133 G⇒A",  "chr19", "11242133", "G", "A"),
        ("chr2:21266223 T⇒C",   "chr2",  "21266223", "T", "C"),
        ("chr1:55505651 C⇒T",   "chr1",  "55505651", "C", "T"),
        ("chr19:11230867 C⇒T",  "chr19", "11230867", "C", "T"),
    ]

    print("=== parse_variant_coords (from Variant string only) ===")
    for variant_str, *_ in test_cases:
        coords = parse_variant_coords(variant_str)
        print(f"  {variant_str!r:40s} → {coords}")

    print("\n=== vcf_to_autopvs1_id ===")
    for _, chrom, pos, ref, alt in test_cases:
        print(f"  {vcf_to_autopvs1_id(chrom, pos, ref, alt)}")

    print("\n=== Live fetch (chr19:11230867 C>T) ===")
    result = fetch_and_format_autopvs1("chr19", "11230867", "C", "T",
                                       hgvs="NM_000527.5:c.1945C>T")
    print(result or "No result")
