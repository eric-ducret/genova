"""
pipeline/tools/litvar2.py — LitVar2-powered literature summarisation tool.

Pipeline for a given rsID:
  1. Fetch all PMIDs from LitVar2 API           (max max_pmids)
  2. Fetch PubMed titles for every PMID
  3. Ask the SLM to keep only the N most relevant titles given the patient phenotype
  4. Fetch full abstracts for the selected PMIDs (via NCBI E-utilities)
  5. Ask the SLM to produce a concise evidence summary from those abstracts

Gene-level fallback (when rsID is absent but Gene is present):
  1. Fetch all variants for the gene from LitVar2
  2. Collect PMIDs across the first 10 variants
  3–5. Same filtering and summarisation pipeline, using resolved disease query

Changes from litvar2_tool.py:
  - LitVar2SummaryTool inherits from SLMTool instead of Tool (retrieval_agent)
  - __init__ initialises _disease_query as an instance attribute (thread-safe across runs)
  - max_pmids, top_n, max_chars, timeout are class-level attributes with defaults
  - _select_relevant_pmids and _summarise now accept context and use context.llm.generate()
  - run() signature changed to run(variant, context) per pipeline contract
  - gate() returns True when RS_ID is valid OR Gene is non-empty
  - _disease_query cached on first run via _resolve_disease_query()
  - _gene_fallback() handles variants with no rsID
"""

import ast
import re
import json
import time
import logging
import requests
import xml.etree.ElementTree as ET

from pipeline.tools.base import SLMTool
from pipeline.core.context import ToolContext
from pipeline.core.errors import ToolFetchError, ToolParseError

logger = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────

LITVAR2_BASE = "https://www.ncbi.nlm.nih.gov/research/litvar2-api"
LITVAR2_URL  = f"{LITVAR2_BASE}/variant/get/litvar@{{rsid}}%23%23/publications"
NCBI_BASE    = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_DELAY   = 0.34           # stay under 3 req/s

MAX_PMIDS     = 40             # hard cap on PMIDs fetched from LitVar2
TOP_N_PAPERS  = 8              # how many titles the SLM keeps
MAX_CHARS     = 600            # abstract truncation per paper
DEFAULT_TIMEOUT = 15


# ── helpers ───────────────────────────────────────────────────────────────────

def _ncbi_get(endpoint: str, params: dict, timeout: int = DEFAULT_TIMEOUT) -> requests.Response:
    time.sleep(NCBI_DELAY)
    resp = requests.get(f"{NCBI_BASE}/{endpoint}", params=params, timeout=timeout)
    resp.raise_for_status()
    return resp


def _clean_xml_text(raw: str) -> str:
    text = re.sub(r"<[^>]+>", " ", raw)
    return re.sub(r"\s+", " ", text).strip()


# ── main tool ─────────────────────────────────────────────────────────────────

class LitVar2SummaryTool(SLMTool):
    """
    Retrieves variant-specific literature via LitVar2, filters it with an SLM,
    then returns a concise evidence summary grounded in PubMed abstracts.

    gate():  runs when RS_ID is valid OR Gene is non-empty
    run():   resolves disease query once, then either:
             - rsID path: fetches PMIDs → titles → SLM filter → abstracts → SLM summary
             - gene fallback: gene → variants → PMIDs → titles → SLM filter → abstracts → SLM summary
    """

    name        = "litvar2_summary"
    description = (
        "Fetches variant-specific PubMed articles from LitVar2, selects the most "
        "relevant ones for the patient phenotype using an SLM, and returns a concise "
        "evidence summary derived from their abstracts."
    )

    # Class-level defaults — override at instantiation if needed
    max_pmids: int = MAX_PMIDS
    top_n:     int = TOP_N_PAPERS
    max_chars: int = MAX_CHARS
    timeout:   int = DEFAULT_TIMEOUT

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._disease_query: str | None = None
        self._last_query: str | None = None
        self._last_n_found: int = 0
        self._last_kept: list[dict] | None = None

    # ── gate ──────────────────────────────────────────────────────────────────

    def gate(self, variant: dict, context: ToolContext) -> bool:
        """Run when a valid rsID is present OR when a gene symbol is available."""
        rsid = context.field("RS_ID")
        gene = context.field("Gene")
        rsid_ok = rsid != "NA" and re.match(r"^rs\d+$", rsid, re.IGNORECASE) is not None
        gene_ok = gene != "NA" and gene.strip() != ""
        return rsid_ok or gene_ok

    # ── disease query resolution ──────────────────────────────────────────────

    def _resolve_disease_query(self, context: ToolContext) -> str:
        """
        Ask the SLM once to produce a compact search string (2-4 words) from the
        patient phenotype. Used as a PubMed title keyword filter.
        """
        system = (
            "Return only the shortest possible disease search string (2-4 words max) "
            "suitable for a PubMed title keyword filter. "
            "Lowercase only, no punctuation, no explanation."
        )
        return " ".join(
            context.llm.generate(system=system, user=context.patient_phenotype, max_tokens=20)
            .strip().lower().split()[:4]
        )

    # ── step 1: LitVar2 → PMIDs (rsID path) ──────────────────────────────────

    def _fetch_pmids(self, rsid: str) -> list[str]:
        url = LITVAR2_URL.format(rsid=rsid)
        logger.debug("LitVar2 fetch: %s", url)
        try:
            resp = requests.get(url, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            raise ToolFetchError(f"LitVar2 API request failed for {rsid}: {e}") from e

        # API returns {"pmids": [...], "pmcids": [...], "pmids_count": N}
        raw_pmids = data.get("pmids", [])
        pmids = [str(p) for p in raw_pmids if str(p).isdigit()][:self.max_pmids]

        logger.debug("LitVar2 returned %d PMIDs for %s", len(pmids), rsid)
        return pmids

    # ── step 2: PMIDs → titles ────────────────────────────────────────────────

    def _fetch_titles(self, pmids: list[str]) -> dict[str, str]:
        """Returns {pmid: title}."""
        if not pmids:
            return {}
        try:
            data = _ncbi_get(
                "esummary.fcgi",
                {"db": "pubmed", "id": ",".join(pmids), "retmode": "json"},
                self.timeout,
            ).json()
        except Exception as e:
            raise ToolFetchError(f"PubMed esummary request failed: {e}") from e

        result = data.get("result", {})
        try:
            return {
                pmid: result[pmid].get("title", "No title")
                for pmid in pmids
                if pmid in result
            }
        except Exception as e:
            raise ToolParseError(f"Failed to parse PubMed titles: {e}") from e

    # ── step 3: SLM title filter ──────────────────────────────────────────────

    def _select_relevant_pmids(
        self,
        titles: dict[str, str],
        question: str,
        context: ToolContext,
    ) -> list[str]:
        """Ask the SLM to pick the top_n most relevant PMIDs given the question."""

        numbered = "\n".join(
            f"{i+1}. [PMID:{pmid}] {title}"
            for i, (pmid, title) in enumerate(titles.items())
        )

        system = (
            "You are a biomedical literature relevance filter. "
            f"Given a list of paper titles and a research question, select the {self.top_n} "
            "most relevant papers. "
            "Output ONLY a JSON array of PMID strings, e.g. [\"12345678\", \"87654321\"]. "
            "No explanation, no markdown, no extra text."
        )
        user = (
            f"Research question: {question}\n\n"
            f"Paper list:\n{numbered}\n\n"
            f"Return the {self.top_n} most relevant PMIDs as a JSON array."
        )

        raw = context.llm.generate(system=system, user=user, max_tokens=256).strip()

        # Parse JSON array robustly
        try:
            match = re.search(r"\[.*?\]", raw, re.DOTALL)
            if match:
                selected = json.loads(match.group())
                # keep only valid PMIDs that exist in our titles dict
                valid = [str(p) for p in selected if str(p) in titles]
                if valid:
                    logger.debug("SLM selected %d PMIDs", len(valid))
                    return valid
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("Could not parse SLM selection JSON (%s) — falling back to first %d", e, self.top_n)

        # Fallback: just take the first top_n
        return list(titles.keys())[: self.top_n]

    # ── step 4: PMIDs → abstracts ─────────────────────────────────────────────

    def _fetch_abstracts(self, pmids: list[str]) -> dict[str, dict]:
        """Returns {pmid: {"title": ..., "abstract": ..., "url": ...}}."""
        if not pmids:
            return {}

        try:
            xml_text = _ncbi_get(
                "efetch.fcgi",
                {"db": "pubmed", "id": ",".join(pmids), "rettype": "xml", "retmode": "xml"},
                self.timeout,
            ).text
            root = ET.fromstring(xml_text)
        except Exception as e:
            raise ToolFetchError(f"PubMed efetch request failed: {e}") from e

        try:
            results = {}
            for article in root.iter("PubmedArticle"):
                pmid_el = article.find(".//PMID")
                if pmid_el is None:
                    continue
                pmid = pmid_el.text.strip()

                title_el = article.find(".//ArticleTitle")
                title    = _clean_xml_text(ET.tostring(title_el, encoding="unicode")) if title_el is not None else "No title"

                abstract_parts = [
                    _clean_xml_text(ET.tostring(a, encoding="unicode"))
                    for a in article.iter("AbstractText")
                ]
                abstract = " ".join(abstract_parts) if abstract_parts else "No abstract available."

                results[pmid] = {
                    "title":    title,
                    "abstract": abstract[: self.max_chars],
                    "url":      f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                }

            return results
        except Exception as e:
            raise ToolParseError(f"Failed to parse PubMed abstracts: {e}") from e

    # ── step 5: SLM summary ───────────────────────────────────────────────────

    def _summarise(
        self,
        papers: dict[str, dict],
        question: str,
        identifier: str,
        context: ToolContext,
    ) -> str:
        """Ask the SLM to produce a prose summary with inline PMID citations."""

        corpus = "\n\n".join(
            f"[PMID:{pmid}] {p['title']}\n{p['abstract']}"
            for pmid, p in papers.items()
        )

        system = (
            "You are a biomedical evidence synthesiser. "
            "Write a short paragraph (3-5 sentences) summarising findings from the abstracts "
            "that are relevant to the research question. "
            "Cite sources inline using (PMID:XXXXXXXX) immediately after the relevant claim. "
            "If no abstract contains relevant evidence, write one sentence saying so. "
            "Do NOT use bullet points, headers, or raw URLs. "
            "Do NOT mention abstracts or papers explicitly, just report the findings naturally."
        )
        user = (
            f"Research question: {question}\n"
            f"Variant/gene: {identifier}\n\n"
            f"Abstracts:\n{corpus}"
        )

        return context.llm.generate(system=system, user=user, max_tokens=512).strip()

    # ── gene-level fallback ───────────────────────────────────────────────────

    def _gene_fallback(self, gene: str, context: ToolContext) -> str | None:
        """
        Fetch gene-level literature when no rsID is available.

        Steps:
          1. Fetch all LitVar2 variants for the gene
          2. Collect and deduplicate PMIDs across the first 10 variants
          3–5. Reuse _fetch_titles, _select_relevant_pmids, _fetch_abstracts, _summarise
        """
        if self._disease_query is None:
            self._disease_query = self._resolve_disease_query(context)

        # Step 1: gene → variants
        url = f"{LITVAR2_BASE}/variant/search/gene/{gene}"
        try:
            resp = requests.get(url, timeout=self.timeout)
            resp.raise_for_status()
        except Exception as e:
            raise ToolFetchError(f"LitVar2 gene search failed for {gene}: {e}") from e

        try:
            all_variants = []
            for line in resp.text.strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    try:
                        parsed = ast.literal_eval(line)
                    except Exception:
                        continue
                all_variants.append(parsed)
        except Exception as e:
            raise ToolParseError(f"Failed to parse gene variant list for {gene}: {e}") from e

        variants = all_variants[:10]
        if not variants:
            return None

        # Step 2: variants → deduplicated PMIDs
        pmid_set: set[str] = set()
        for v in variants:
            variant_id = v.get("_id")
            if not variant_id:
                continue
            try:
                time.sleep(0.2)
                encoded = variant_id.replace("#", "%23")
                pub_url = f"{LITVAR2_BASE}/variant/get/{encoded}/publications"
                pub_resp = requests.get(pub_url, timeout=self.timeout)
                pub_resp.raise_for_status()
                data = pub_resp.json()
                for p in data.get("pmids", []):
                    if str(p).isdigit():
                        pmid_set.add(str(p))
            except Exception as e:
                logger.warning("LitVar2 gene fallback: could not fetch publications for %s: %s", variant_id, e)
                continue

        all_pmids = list(pmid_set)[: self.max_pmids]
        if not all_pmids:
            return None

        # Steps 3–5: reuse existing methods
        titles = self._fetch_titles(all_pmids)
        if not titles:
            return None

        selected_pmids = self._select_relevant_pmids(titles, self._disease_query, context)
        logger.info("Gene fallback %s: selected %d/%d papers", gene, len(selected_pmids), len(titles))

        self._last_query   = gene
        self._last_n_found = len(titles)
        self._last_kept    = [
            {"pmid": pmid, "title": titles[pmid]}
            for pmid in selected_pmids
            if pmid in titles
        ]

        papers = self._fetch_abstracts(selected_pmids)
        if not papers:
            return None

        summary = self._summarise(papers, context.patient_phenotype, gene, context)

        source_lines = "\n".join(
            f"  - [PMID:{pmid}] {p['title']}  {p['url']}"
            for pmid, p in papers.items()
        )
        return (
            f"LitVar2 gene-level evidence for {gene} (no rsID):\n\n"
            f"{summary}\n\n"
            f"Sources used:\n{source_lines}"
        )

    # ── public entry point ────────────────────────────────────────────────────

    def run(self, variant: dict, context: ToolContext) -> str | None:
        """
        Fetch LitVar2 evidence for the variant.

        - Resolves the disease query once (reused across all variants).
        - rsID present and valid  → rsID path, SLM filter uses resolved disease query
        - rsID absent, Gene present → gene-level fallback
        - Both absent              → None
        """
        self._last_query   = None
        self._last_n_found = 0
        self._last_kept    = None

        rsid = context.field("RS_ID")
        gene = context.field("Gene")

        # Resolve disease query once; reused for all subsequent variants
        if self._disease_query is None:
            self._disease_query = self._resolve_disease_query(context)

        rsid_ok = rsid != "NA" and re.match(r"^rs\d+$", rsid, re.IGNORECASE) is not None

        if rsid_ok:
            logger.info("LitVar2SummaryTool: rsid=%s", rsid)

            # ── step 1: PMIDs ─────────────────────────────────────────────────
            pmids = self._fetch_pmids(rsid)
            if not pmids:
                return None

            # ── step 2: titles ────────────────────────────────────────────────
            titles = self._fetch_titles(pmids)
            if not titles:
                return None

            # ── step 3: SLM filter (disease query, not raw phenotype) ─────────
            selected_pmids = self._select_relevant_pmids(titles, self._disease_query, context)
            logger.info("Selected %d/%d papers for abstract fetch", len(selected_pmids), len(titles))

            self._last_query   = rsid
            self._last_n_found = len(titles)
            self._last_kept    = [
                {"pmid": pmid, "title": titles[pmid]}
                for pmid in selected_pmids
                if pmid in titles
            ]

            # ── step 4: abstracts ─────────────────────────────────────────────
            papers = self._fetch_abstracts(selected_pmids)
            if not papers:
                return None

            # ── step 5: summary (full phenotype for richer output) ────────────
            summary = self._summarise(papers, context.patient_phenotype, rsid, context)

            source_lines = "\n".join(
                f"  - [PMID:{pmid}] {p['title']}  {p['url']}"
                for pmid, p in papers.items()
            )
            return (
                f"LitVar2 evidence summary for {rsid}:\n\n"
                f"{summary}\n\n"
                f"Sources used:\n{source_lines}"
            )

        elif gene != "NA" and gene.strip() != "":
            logger.info("LitVar2SummaryTool: gene fallback for gene=%s", gene)
            return self._gene_fallback(gene, context)

        else:
            return None


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)

    rsid = "rs121913527"  # swap for any rsid you want to test

    # Step 1: raw API response
    import requests
    url = f"https://www.ncbi.nlm.nih.gov/research/litvar2-api/variant/get/litvar@{rsid}%23%23/publications"
    resp = requests.get(url, timeout=15)
    data = resp.json()
    print("=== RAW RESPONSE (first 2 items) ===")
    items = data if isinstance(data, list) else data.get("publications", [])
    for item in items[:2]:
        print(item)

    # Step 2: fetch steps (no SLM — skips steps 3 and 5)
    tool = LitVar2SummaryTool()
    pmids = tool._fetch_pmids(rsid)
    print(f"\n=== PMIDs ({len(pmids)}) ===")
    print(pmids)

    titles = tool._fetch_titles(pmids)
    print(f"\n=== TITLES ===")
    for pmid, title in titles.items():
        print(f"  {pmid}: {title}")

    papers = tool._fetch_abstracts(list(titles.keys())[:3])
    print(f"\n=== ABSTRACTS (first 3) ===")
    for pmid, p in papers.items():
        print(f"\n  [{pmid}] {p['title']}\n  {p['abstract'][:200]}")
