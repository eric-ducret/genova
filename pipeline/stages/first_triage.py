"""
pipeline/stages/first_triage.py — Post-retrieval SLM triage stage.

One small SLM call per variant immediately after retrieval.
Decides whether a variant can be discarded with high confidence before
the more expensive reasoning stage runs.

Expected SLM output format (from prompts/first_triage.txt):
    Thought: <decisive evidence phrase, ≤8 words>
    Decision: KEEP or DISCARD

Public API:
    run_one(variant_context, patient_phenotype, llm) -> tuple[str, str]
        Returns ("KEEP", thought) or ("DISCARD", thought).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.llm.base import LLMClient

logger = logging.getLogger(__name__)

_PROMPT_PATH   = Path(__file__).parent.parent.parent / "prompts" / "first_triage.txt"
_SYSTEM_PROMPT = "You are an expert clinical geneticist."


def _load_prompt() -> str:
    if _PROMPT_PATH.exists():
        return _PROMPT_PATH.read_text(encoding="utf-8")
    raise FileNotFoundError(f"First-triage prompt not found at {_PROMPT_PATH}.")


_THOUGHT_RE  = re.compile(r"Thought:\s*(.+)", re.IGNORECASE)
_DECISION_RE = re.compile(r"Decision:\s*(KEEP|DISCARD)", re.IGNORECASE)


def parse_triage(raw: str) -> tuple[str, str]:
    """
    Parse the raw SLM reply for a first-triage call.

    Expected format:
        Thought: <decisive evidence phrase, ≤8 words>
        Decision: KEEP or DISCARD

    Returns:
        ("KEEP",    thought)  if Decision is KEEP.
        ("DISCARD", thought)  if Decision is DISCARD.
        ("KEEP",    "parse error — defaulting to KEEP") on any parse failure.

    The thought is used as the justification in the report appendix.
    """
    thought_m  = _THOUGHT_RE.search(raw)
    decision_m = _DECISION_RE.search(raw)

    thought = thought_m.group(1).strip() if thought_m else ""

    if decision_m:
        decision = decision_m.group(1).upper()
        return (decision, thought)

    logger.warning("[FirstTriage] Could not parse response: %r — defaulting to KEEP", raw[:200])
    return ("KEEP", "parse error — defaulting to KEEP")


def run_one(
    variant_context: str,
    patient_phenotype: str,
    llm: "LLMClient",
) -> tuple[str, str]:
    """
    First triage — fast discard gate immediately after retrieval.

    Args:
        variant_context:   Per-variant context string from Stage 1 (retrieval).
        patient_phenotype: Free-text patient phenotype string.
        llm:               Shared LLMClient instance.

    Returns:
        ("KEEP", thought) or ("DISCARD", thought) where thought is the ≤8-word
        rationale from the Thought: line. On SLM or parse failure always returns
        ("KEEP", ...) — never silently drops a variant.
    """
    user_prompt = (
        _load_prompt()
        .replace("{patient_phenotype}", patient_phenotype)
        .replace("{variant_context}", variant_context)
    )

    try:
        raw = llm.generate(
            system=_SYSTEM_PROMPT,
            user=user_prompt,
            max_tokens=60,
        )
    except Exception as exc:
        logger.warning("[FirstTriage] SLM call failed: %s — defaulting to KEEP", exc)
        return ("KEEP", f"SLM error — defaulting to KEEP: {exc}")

    return parse_triage(raw)
