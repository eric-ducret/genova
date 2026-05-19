"""
pipeline/stages/reasoning.py — SLM step-by-step clinical reasoning stage.

Ported from run_reasoning() in server_main.py.

Single SLM call per variant. Takes one variant's context string (from Stage 1/2)
and produces step-by-step clinical reasoning for that variant. Does NOT produce
a classification.

Prompt loaded from prompts/reasoning.txt.

Public API:
    run_one(variant_context, llm) -> str
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.llm.base import LLMClient

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "reasoning.txt"

MAX_NEW_TOKENS_REASONING = 1500


def _load_prompt() -> str:
    if _PROMPT_PATH.exists():
        return _PROMPT_PATH.read_text(encoding="utf-8")
    raise FileNotFoundError(
        f"Reasoning prompt not found at {_PROMPT_PATH}. "
        "Run step 19 to extract prompts from server_main.py."
    )


def run_one(variant_context: str, llm: "LLMClient") -> str:
    """
    Stage 3 — Step-by-step clinical reasoning for one variant.

    Args:
        variant_context: Per-variant context string from Stage 1/2 (retrieval/compression).
                         Contains PATIENT DATA header + one VARIANT block.
        llm:             Shared LLMClient instance.

    Returns:
        Step-by-step reasoning text for this variant. No final classification.
    """
    logger.info("[Reasoning] Running step-by-step reasoning...")
    template    = _load_prompt()
    user_prompt = template.replace("{augmented_context}", variant_context)

    return llm.generate(
        system="You are an expert clinical geneticist.",
        user=user_prompt,
        max_tokens=MAX_NEW_TOKENS_REASONING,
    )
