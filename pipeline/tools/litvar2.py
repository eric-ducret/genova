"""
pipeline/tools/litvar2.py — LitVar2-powered literature summarisation tool.

Pipeline for a given rsID:
  1. Fetch all PMIDs from LitVar2 API           (max max_pmids)
  2. Fetch PubMed titles for every PMID
  3. Ask the SLM to keep only the N most relevant titles given the patient phenotype
  4. Fetch full abstracts for the selected PMIDs (via NCBI E-utilities)
  5. Ask the SLM to produce a concise evidence summary from those abstracts

Changes from litvar2_tool.py:
  - LitVar2SummaryTool inherits from SLMTool instead of Tool (retrieval_agent)
  - Removed __init__(self, slm, ...) — LLM comes from context, not constructor
  - max_pmids, top_n, max_chars, timeout are class-level attributes with defaults
  - _select_relevant_pmids and _summarise now accept context and use context.llm.generate()
  - run() signature changed to run(variant, context) per pipeline contract
  - gate() returns True only when RS_ID is available
"""

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

LITVAR2_URL = "https://www.ncbi.nlm.nih.gov/research/litvar2-api/variant/get/litvar@{rsid}%23%23/publications"
NCBI_BASE   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_DELAY    = 0.34           # stay under 3 req/s

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

    gate():  runs only when RS_ID is available (not "NA")
    run():   fetches PMIDs → titles → SLM filter → abstracts → SLM summary
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

    # ── gate ──────────────────────────────────────────────────────────────────

    def gate(self, variant: dict, context: ToolContext) -> bool:
        """Run only when a valid rsID is present."""
        rsid = context.field("RS_ID")
        return rsid != "NA" and re.match(r"^rs\d+$", rsid, re.IGNORECASE) is not None

    # ── step 1: LitVar2 → PMIDs ───────────────────────────────────────────────

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
        """Returns {pmid: {"title": ..., "abstract": ...}}."""
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
        rsid: str,
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
            f"Variant: {rsid}\n\n"
            f"Abstracts:\n{corpus}"
        )

        return context.llm.generate(system=system, user=user, max_tokens=512).strip()

    # ── public entry point ────────────────────────────────────────────────────

    def run(self, variant: dict, context: ToolContext) -> str | None:
        """
        Fetch LitVar2 evidence for the variant's rsID, filtered to the patient phenotype.
        Returns an evidence summary string, or None if no evidence is found.
        """
        rsid     = context.field("RS_ID")
        question = context.patient_phenotype

        logger.info("LitVar2SummaryTool: rsid=%s question=%r", rsid, question)

        # ── step 1: PMIDs ─────────────────────────────────────────────────────
        pmids = self._fetch_pmids(rsid)
        if not pmids:
            return None

        # ── step 2: titles ────────────────────────────────────────────────────
        titles = self._fetch_titles(pmids)
        if not titles:
            return None

        # ── step 3: SLM filter ────────────────────────────────────────────────
        selected_pmids = self._select_relevant_pmids(titles, question, context)
        logger.info("Selected %d/%d papers for abstract fetch", len(selected_pmids), len(titles))

        # ── step 4: abstracts ─────────────────────────────────────────────────
        papers = self._fetch_abstracts(selected_pmids)
        if not papers:
            return None

        # ── step 5: summary ───────────────────────────────────────────────────
        summary = self._summarise(papers, question, rsid, context)

        # Append a provenance block so downstream stages can cite sources
        source_lines = "\n".join(
            f"  - [PMID:{pmid}] {p['title']}  {p['url']}"
            for pmid, p in papers.items()
        )
        return (
            f"LitVar2 evidence summary for {rsid}:\n\n"
            f"{summary}\n\n"
            f"Sources used:\n{source_lines}"
        )


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
