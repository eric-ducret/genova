"""
pipeline/stages/reasoning.py â€” Two-call SLM clinical reasoning stage.

Splitting the original single call into two focused calls reduces hallucination
by separating the evidence-grounded narrative from the inclusion decision:

  Call 1 â€” reasoning:  step-by-step clinical reasoning using ONLY cited evidence.
                       Prompt: prompts/reasoning.txt
  Call 2 â€” second_triage: structured INCLUDE/EXCLUDE decision with justification.
                           Prompt: prompts/second_triage.txt
                           Format: Include-case / Exclude-case / Decision (3 lines)

The two outputs are concatenated before returning so that the REASONING display
section in pipeline.py is self-contained.

Public API:
    run_one(variant_context, llm) -> str
    parse_inclusion_decision(reasoning_text) -> tuple[str, str]
        Returns ("INCLUDE"|"EXCLUDE", "Include: <...> | Exclude: <...>").
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

# Call 1: step-by-step narrative â€” no score, citation-only rules apply
MAX_NEW_TOKENS_REASONING = 1500
# Call 2: three-line Include-case / Exclude-case / Decision block
# 200 gives headroom for any residual preamble before the 3 structured lines.
MAX_NEW_TOKENS_SCORING   = 200

_INCLUDE_CASE_RE = re.compile(r"Include-case:\s*(.+)", re.IGNORECASE)
_EXCLUDE_CASE_RE = re.compile(r"Exclude-case:\s*(.+)", re.IGNORECASE)
_DECISION_RE     = re.compile(r"Decision:\s*(INCLUDE|EXCLUDE)", re.IGNORECASE)


# â”€â”€ Prompt loaders â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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


# â”€â”€ Score parser â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def parse_inclusion_decision(reasoning_text: str) -> tuple[str, str]:
    """
    Parse the INCLUDE/EXCLUDE decision and justification from run_one output.

    Expected embedded format (after the "SECOND TRIAGE:" marker):
        Include-case: <strongest reason to include, â‰¤15 words>
        Exclude-case: <strongest reason to exclude, â‰¤15 words>
        Decision: INCLUDE or EXCLUDE

    Blank lines are stripped before parsing (Qwen without thinking sometimes
    inserts them between lines).

    Returns:
        ("INCLUDE" | "EXCLUDE", justification)
        justification = "Include: <include_case> | Exclude: <exclude_case>"
        Defaults to ("INCLUDE", "parse error â€” defaulting to INCLUDE") on failure.
    """
    cleaned = "\n".join(line for line in reasoning_text.splitlines() if line.strip())

    include_m  = _INCLUDE_CASE_RE.search(cleaned)
    exclude_m  = _EXCLUDE_CASE_RE.search(cleaned)
    decision_m = _DECISION_RE.search(cleaned)

    include_str = include_m.group(1).strip() if include_m else ""
    exclude_str = exclude_m.group(1).strip() if exclude_m else ""

    logger.debug(
        "[SecondTriage] Include-case: %r | Exclude-case: %r | raw Decision line: %r",
        include_str, exclude_str,
        decision_m.group(0) if decision_m else "(not found)",
    )

    if decision_m:
        decision = decision_m.group(1).upper()
        justification = (
            f"Include: {include_str} | Exclude: {exclude_str}"
            if (include_str or exclude_str)
            else decision
        )
        return (decision, justification)

    # Fallback: model may have ignored the three-line format; search the tail
    tail = reasoning_text[-150:]
    m2 = re.search(r"\b(INCLUDE|EXCLUDE)\b", tail, re.IGNORECASE)
    if m2:
        logger.warning(
            "[SecondTriage] Decision: prefix not found â€” matched bare word in tail: %r",
            m2.group(0),
        )
        return (m2.group(1).upper(), f"Include: {include_str} | Exclude: {exclude_str}")

    logger.warning(
        "[SecondTriage] Could not parse INCLUDE/EXCLUDE decision â€” defaulting to INCLUDE. "
        "Raw tail: %r",
        reasoning_text[-200:],
    )
    return ("INCLUDE", "parse error â€” defaulting to INCLUDE")


# â”€â”€ Main entry point â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run_one(variant_context: str, llm: "LLMClient") -> str:
    """
    Stage 2 â€” Two-call clinical reasoning for one variant.

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
    # â”€â”€ Call 1: evidence-grounded narrative â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    logger.info("[Reasoning] Call 1/2 â€” step-by-step reasoning...")
    reasoning_template = _load_reasoning_prompt()
    reasoning_user     = reasoning_template.replace("{augmented_context}", variant_context)

    reasoning_text = llm.generate(
        system="You are an expert clinical geneticist. Base every claim strictly on the evidence below.",
        user=reasoning_user,
        max_tokens=MAX_NEW_TOKENS_REASONING,
    )

    # â”€â”€ Call 2: binary inclusion decision â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    logger.info("[Reasoning] Call 2/2 â€” second_triage inclusion decision (INCLUDE/EXCLUDE)...")
    scoring_template = _load_scoring_prompt()
    scoring_user     = (
        scoring_template
        .replace("{augmented_context}", variant_context)
        .replace("{reasoning}", reasoning_text)
    )

    decision_text = llm.generate(
        system="You are an expert clinical geneticist. Base every claim strictly on the evidence below.",
        user=scoring_user,
        max_tokens=MAX_NEW_TOKENS_SCORING,
    )

    logger.debug("[SecondTriage] Full model output:\n%s", decision_text.strip())

    # â”€â”€ Combine for downstream display â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # parse_inclusion_decision() searches for the Decision: line within the
    # "SECOND TRIAGE:" block.  The full three-line output is preserved so the
    # REASONING display section shows Include-case / Exclude-case / Decision.
    combined = reasoning_text.rstrip() + "\n\nSECOND TRIAGE:\n" + decision_text.strip()
    return combined


