"""
pipeline/stages/conclusion.py — SLM structured report stage.

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

MAX_NEW_TOKENS_REPORT = 800


def _extract_pmids(text: str) -> list[str]:
    """Return deduplicated PMID:XXXXXXXX strings found in text, in order of first appearance."""
    hits = re.findall(r"PMID:(\d+)", text)
    return list(dict.fromkeys(f"PMID:{h}" for h in hits))


def _extract_urls(text: str) -> list[str]:
    """Return deduplicated http/https URLs found in text, in order of first appearance.

    Trailing punctuation common in prose (., ; : ) ] > " ') is stripped.
    """
    raw = re.findall(r"https?://\S+", text)
    cleaned = [u.rstrip(".,;:)>]\"'\\") for u in raw]
    return list(dict.fromkeys(u for u in cleaned if u))


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
    Stage 4 — Structured clinical report for one variant.

    Args:
        variant_context: Per-variant context string from Stage 1/2 (retrieval/compression).
                         Contains PATIENT DATA header + one VARIANT block.
        reasoning:       Output from reasoning.run_one() for this variant.
        cross_analysis:  Gene-level cross-analysis text from cross_analysis.run(), or None
                         if this variant's gene has fewer than two variants in this run.
        llm:             Shared LLMClient instance.

    Returns:
        Structured report block for this variant:
            # Variant [N] — [GENE] ([HGVS])
            **Molecular mechanism:** ...
            **Phenotype fit:** ...
            **Inheritance check:** ...
            **Evidence strength:** ...
            **ACMG criteria**: ...
            **Comment:** ...
    """
    logger.info("[Conclusion] Generating structured report for variant...")

    # Extract references deterministically from the evidence context before the SLM call
    # so they are appended as a Python-built block rather than relying on prompt instruction.
    pmids = _extract_pmids(variant_context)
    urls  = _extract_urls(variant_context)

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
        system="You are an expert clinical geneticist.",
        user=user_prompt,
        max_tokens=MAX_NEW_TOKENS_REPORT,
    )

    refs = pmids + urls
    if refs:
        result = result.rstrip() + "\n**References:** " + ", ".join(refs)

    return result
