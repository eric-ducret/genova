"""
pipeline/stages/conclusion.py — SLM structured report stage.

Ported from run_conclusion() in server_main.py.

Single SLM call. Takes augmented context + reasoning output and produces the
structured clinical report with ACMG criteria and a clinical conclusion.

Prompt loaded from prompts/conclusion.txt.

Public API:
    run(augmented_context, reasoning, llm) -> str
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.llm.base import LLMClient

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "conclusion.txt"

MAX_NEW_TOKENS_REPORT = 3000


def _load_prompt() -> str:
    if _PROMPT_PATH.exists():
        return _PROMPT_PATH.read_text(encoding="utf-8")
    raise FileNotFoundError(
        f"Conclusion prompt not found at {_PROMPT_PATH}. "
        "Run step 19 to extract prompts from server_main.py."
    )


def run(augmented_context: str, reasoning: str, llm: "LLMClient") -> str:
    """
    Stage 4 — Structured clinical report.

    Ported verbatim logic from server_main.run_conclusion():
        _chat(system, CONCLUSION_TEMPLATE.format(augmented_context=..., reasoning_output=...), 3000)

    Args:
        augmented_context: Output from Stage 1/2 (same context given to reasoning stage).
        reasoning:         Output from Stage 3 (reasoning.run()).
        llm:               Shared LLMClient instance.

    Returns:
        Structured report with per-variant findings and Clinical Conclusion section.
        Output format:
            # Variant [N] — [GENE] ([HGVS])
            **Molecular mechanism:** ...
            **Phenotype fit:** ...
            **Inheritance check:** ...
            **Evidence strength:** ...
            **ACMG criteria**: ...
            **Comment:** ...

            # Clinical Conclusion
            ...
    """
    logger.info("[Conclusion] Generating structured report...")
    template   = _load_prompt()
    user_prompt = template.format(
        augmented_context=augmented_context,
        reasoning_output=reasoning,
    )

    return llm.generate(
        system="You are an expert clinical geneticist.",
        user=user_prompt,
        max_tokens=MAX_NEW_TOKENS_REPORT,
    )
