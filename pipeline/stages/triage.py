"""
pipeline/stages/triage.py — Fast SLM triage stage.

One small SLM call per variant immediately after retrieval.
Decides whether a variant can be discarded with high confidence.

Public API:
    run_one(variant_context, patient_phenotype, llm) -> tuple[str, str]
        Returns ("KEEP", "") or ("DISCARD", "<one sentence justification>").
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.llm.base import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = "You are an expert clinical geneticist performing rapid variant triage."

_USER_TEMPLATE = """\
Based solely on the retrieved evidence below, can we say with HIGH CONFIDENCE \
that this variant is NOT likely linked to the patient's phenotype?

Only answer DISCARD if the evidence strongly supports irrelevance \
(e.g. clearly benign, unrelated gene function, no phenotype overlap, \
very high population frequency with no pathogenic signal).
When in doubt, answer KEEP.

Patient phenotype: {patient_phenotype}

Retrieved evidence:
{variant_context}

Reply with exactly one of:
KEEP
DISCARD: <one sentence justification>\
"""

_DISCARD_RE = re.compile(r"DISCARD:\s*(.+)", re.IGNORECASE)
_KEEP_RE    = re.compile(r"^KEEP", re.IGNORECASE | re.MULTILINE)


def parse_triage(raw: str) -> tuple[str, str]:
    """
    Parse the raw SLM reply for a triage call.

    Returns:
        ("DISCARD", justification) if DISCARD detected.
        ("KEEP", "")              if KEEP detected.
        ("KEEP", "parse error — defaulting to KEEP") on any parse failure.
    """
    m = _DISCARD_RE.search(raw)
    if m:
        return ("DISCARD", m.group(1).strip())

    if _KEEP_RE.search(raw):
        return ("KEEP", "")

    logger.warning("[Triage] Could not parse triage response: %r — defaulting to KEEP", raw[:200])
    return ("KEEP", "parse error — defaulting to KEEP")


def run_one(
    variant_context: str,
    patient_phenotype: str,
    llm: "LLMClient",
) -> tuple[str, str]:
    """
    Triage one variant immediately after retrieval.

    Args:
        variant_context:   Per-variant context string from Stage 1 (retrieval).
        patient_phenotype: Free-text patient phenotype string.
        llm:               Shared LLMClient instance.

    Returns:
        ("KEEP", "") or ("DISCARD", "<one sentence justification>").
        On SLM or parse failure always returns ("KEEP", ...) — never silently drops.
    """
    user_prompt = _USER_TEMPLATE.replace(
        "{patient_phenotype}", patient_phenotype
    ).replace(
        "{variant_context}", variant_context
    )

    try:
        raw = llm.generate(
            system=_SYSTEM_PROMPT,
            user=user_prompt,
            max_tokens=40,
        )
    except Exception as exc:
        logger.warning("[Triage] SLM call failed: %s — defaulting to KEEP", exc)
        return ("KEEP", f"SLM error — defaulting to KEEP: {exc}")

    return parse_triage(raw)
