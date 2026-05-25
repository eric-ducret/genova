"""
pipeline/stages/reasoning.py — Two-call SLM clinical reasoning stage.

Splitting the original single call into two focused calls reduces hallucination
by separating the evidence-grounded narrative from the inclusion decision:

  Call 1 — reasoning:  step-by-step clinical reasoning using ONLY cited evidence.
                       Prompt: prompts/reasoning.txt
  Call 2 — second_triage: binary INCLUDE/EXCLUDE decision based on the reasoning.
                           Prompt: prompts/second_triage.txt

The two outputs are concatenated before returning so that the REASONING display
section in pipeline.py is self-contained.

Public API:
    run_one(variant_context, llm) -> str
    parse_inclusion_decision(reasoning_text) -> str   # "INCLUDE" | "EXCLUDE"
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.llm.base import LLMClient

logger = logging.getLogger(__name__)

_REASONING_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "reasoning.txt"
_SCORING_PROMPT_PATH   = Path(__file__).parent.parent.parent / "prompts" / "second_triage.txt"

# Call 1: step-by-step narrative — no score, citation-only rules apply
MAX_NEW_TOKENS_REASONING = 1200
# Call 2: single INCLUDE/EXCLUDE word — tiny budget
MAX_NEW_TOKENS_SCORING   = 10

_INCLUSION_RE = re.compile(r"DECISION:\s*(INCLUDE|EXCLUDE)", re.IGNORECASE)


# ── Prompt loaders ────────────────────────────────────────────────────────────

def _load_reasoning_prompt() -> str:
    if _REASONING_PROMPT_PATH.exists():
        return _REASONING_PROMPT_PATH.read_text(encoding="utf-8")
    raise FileNotFoundError(
        f"Reasoning prompt not found at {_REASONING_PROMPT_PATH}."
    )


def _load_scoring_prompt() -> str:
    if _SCORING_PROMPT_PATH.exists():
        return _SCORING_PROMPT_PATH.read_text(encoding="utf-8")
    raise FileNotFoundError(
        f"Second-triage prompt not found at {_SCORING_PROMPT_PATH}."
    )


# ── Score parser ──────────────────────────────────────────────────────────────

def parse_inclusion_decision(reasoning_text: str) -> str:
    """
    Extract the INCLUDE/EXCLUDE decision appended at the end of run_one output.

    Primary:  matches "DECISION: INCLUDE/EXCLUDE" — the structured prefix injected
              by run_one(), preventing false matches inside the reasoning body.
    Fallback: if the model added preamble to its answer, search the last 100
              characters for any bare INCLUDE/EXCLUDE word.

    Returns "INCLUDE" or "EXCLUDE". Defaults to "INCLUDE" on parse failure
    (safe default — never silently drops a potentially causative variant).
    """
    m = _INCLUSION_RE.search(reasoning_text)
    if m:
        return m.group(1).upper()

    # Fallback: model may have ignored the one-word constraint; search the tail only
    tail = reasoning_text[-100:]
    m2 = re.search(r"\b(INCLUDE|EXCLUDE)\b", tail, re.IGNORECASE)
    if m2:
        logger.warning(
            "[Reasoning] DECISION: prefix not found — matched bare word in tail: %r",
            m2.group(0),
        )
        return m2.group(1).upper()

    logger.error(
        "[Reasoning] Could not parse INCLUDE/EXCLUDE decision — defaulting to INCLUDE. "
        "Raw tail: %r",
        reasoning_text[-200:],
    )
    return "INCLUDE"


# ── Main entry point ──────────────────────────────────────────────────────────

def run_one(variant_context: str, llm: "LLMClient") -> str:
    """
    Stage 2 — Two-call clinical reasoning for one variant.

    Call 1 (reasoning):
        Generates step-by-step clinical reasoning grounded in cited evidence only.
        The model is instructed not to introduce external knowledge without a
        reference present in the provided context.

    Call 2 (second_triage):
        Takes the variant context and the reasoning produced in call 1 and emits
        a single INCLUDE or EXCLUDE word. Decoupling this decision from the
        narrative prevents the model from forcing a favourable decision to justify
        speculative reasoning.

    Args:
        variant_context: Per-variant context string from Stage 1 (retrieval).
                         Contains PATIENT DATA header + one VARIANT block with
                         tool outputs.
        llm:             Shared LLMClient instance.

    Returns:
        Combined text: step-by-step reasoning followed by the inclusion decision
        on its own line. Compatible with parse_inclusion_decision() and with the
        REASONING display section in pipeline.py.
    """
    # ── Call 1: evidence-grounded narrative ───────────────────────────────────
    logger.info("[Reasoning] Call 1/2 — step-by-step reasoning...")
    reasoning_template = _load_reasoning_prompt()
    reasoning_user     = reasoning_template.replace("{augmented_context}", variant_context)

    reasoning_text = llm.generate(
        system="You are an expert clinical geneticist.",
        user=reasoning_user,
        max_tokens=MAX_NEW_TOKENS_REASONING,
    )

    # ── Call 2: binary inclusion decision ────────────────────────────────────
    logger.info("[Reasoning] Call 2/2 — second_triage inclusion decision (INCLUDE/EXCLUDE)...")
    scoring_template = _load_scoring_prompt()
    scoring_user     = (
        scoring_template
        .replace("{augmented_context}", variant_context)
        .replace("{reasoning}", reasoning_text)
    )

    decision_text = llm.generate(
        system="You are an expert clinical geneticist.",
        user=scoring_user,
        max_tokens=MAX_NEW_TOKENS_SCORING,
    )

    # ── Combine for downstream display ───────────────────────────────────────
    # parse_inclusion_decision() and the REASONING display section both expect
    # the INCLUDE/EXCLUDE word to appear somewhere in the returned string.
    combined = reasoning_text.rstrip() + "\n\nDECISION: " + decision_text.strip()
    return combined
