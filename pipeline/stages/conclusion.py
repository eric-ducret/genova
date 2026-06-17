"""
pipeline/stages/conclusion.py â€” SLM structured report stage.

Ported from run_conclusion() in server_main.py.

Single SLM call per variant. Takes one variant's context string, its reasoning
output, and an optional gene-level cross-analysis text, then produces the
structured clinical block for that variant.

The Clinical Conclusion paragraph (overall synthesis across all variants) is
generated separately by stages/final_conclusion.py.

Prompt loaded from prompts/conclusion.txt.

Public API:
    run_one(variant_context, reasoning, cross_analysis, llm) -> str
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.llm.base import LLMClient

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "conclusion.txt"

MAX_NEW_TOKENS_REPORT = 700


def _validate_citations(result: str, context: str) -> str:
    """
    Strip inline citations from the model output that have no grounding in context.

    Two citation forms are checked:
      (PMID:12345678)  â€” removed if that PMID number does not appear anywhere in context.
      (https://...)    â€” removed if that URL does not appear as a substring in context.

    After removals, stray spaces before punctuation and double spaces are collapsed.
    """
    valid_pmids = set(re.findall(r"PMID:(\d+)", context))

    def _keep_pmid(m: re.Match) -> str:
        if m.group(1) in valid_pmids:
            return m.group(0)
        logger.warning("[Conclusion] Removed hallucinated citation: %s", m.group(0))
        return ""

    def _keep_url(m: re.Match) -> str:
        if m.group(1) in context:
            return m.group(0)
        logger.warning("[Conclusion] Removed hallucinated citation: %s", m.group(0))
        return ""

    result = re.sub(r"\(PMID:(\d+)\)", _keep_pmid, result)
    result = re.sub(r"\((https?://[^)\s]+)\)", _keep_url, result)
    result = re.sub(r" +([.,;:])", r"\1", result)
    result = re.sub(r"  +", " ", result)
    return result


def _load_prompt() -> str:
    if _PROMPT_PATH.exists():
        return _PROMPT_PATH.read_text(encoding="utf-8")
    raise FileNotFoundError(
        f"Conclusion prompt not found at {_PROMPT_PATH}. "
        "Run step 19 to extract prompts from server_main.py."
    )


def run_one(
    variant_context: str,
    reasoning: str,
    cross_analysis: str | None,
    llm: "LLMClient",
) -> str:
    """
    Stage 4 â€” Structured clinical report for one variant.

    Args:
        variant_context: Per-variant context string from Stage 1 (retrieval).
                         Contains PATIENT DATA header + one VARIANT block with tool outputs.
        reasoning:       Output from reasoning.run_one() for this variant.
        cross_analysis:  Gene-level cross-analysis text from cross_analysis.run(), or None
                         if this variant's gene has fewer than two variants in this run.
        llm:             Shared LLMClient instance.

    Returns:
        Structured report block for this variant:
            # Variant [N] â€” [GENE] ([HGVS])
            **Molecular mechanism:** ...
            **Phenotype fit:** ...
            **Inheritance check:** ...
            **Evidence strength:** ...
            **ACMG criteria**: ...
            **Comment:** ...
    """
    logger.info("[Conclusion] Generating structured report for variant...")

    template = _load_prompt()

    cross_analysis_block = (
        f"GENE-LEVEL CROSS-ANALYSIS:\n{cross_analysis}\n"
        if cross_analysis is not None else ""
    )

    user_prompt = (template
        .replace("{augmented_context}", variant_context)
        .replace("{reasoning_output}", reasoning)
        .replace("{cross_analysis_block}", cross_analysis_block))

    result = llm.generate(
        system="You are an expert clinical geneticist. Base every claim strictly on the evidence below.",
        user=user_prompt,
        max_tokens=MAX_NEW_TOKENS_REPORT,
    )
    return _validate_citations(result, variant_context)


