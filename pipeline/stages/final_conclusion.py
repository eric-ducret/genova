"""
pipeline/stages/final_conclusion.py â€” Overall clinical conclusion stage.

The last synthesis stage. Reads all per-variant conclusion strings produced by
stages/conclusion.py and generates the short "# Clinical Conclusion" paragraph
that closes the final report.

This is separated from stages/conclusion.py so each variant's structured block
can be generated individually (reducing SLM memory pressure) while the overall
summary still has visibility across all variants.

Prompt loaded from prompts/final_conclusion.txt.

Public API:
    run(conclusions, patient_phenotype, llm) -> str
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.llm.base import LLMClient

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "final_conclusion.txt"

MAX_NEW_TOKENS_FINAL_CONCLUSION = 600

# With --max-model-len 16384, leave 2000 tokens for prompt template + output.
# ~4 chars per token â†’ 14384 Ã— 4 = 57536 chars safe budget for the conclusions block.
_MAX_CONCLUSIONS_CHARS = 50_000


def _load_prompt() -> str:
    if _PROMPT_PATH.exists():
        return _PROMPT_PATH.read_text(encoding="utf-8")
    raise FileNotFoundError(
        f"Final conclusion prompt not found at {_PROMPT_PATH}."
    )


def run(
    conclusions: list[str],
    patient_phenotype: str,
    llm: "LLMClient",
) -> str:
    """
    Overall clinical conclusion synthesis.

    Args:
        conclusions:       List of per-variant conclusion strings from conclusion.run_one().
                           Each string is a structured block:
                               # Variant [N] â€” [GENE] ([HGVS])
                               **Molecular mechanism:** ...
                               ...
                               **Comment:** ...
        patient_phenotype: Free-text patient phenotype string from the request.
        llm:               Shared LLMClient instance.

    Returns:
        The "# Clinical Conclusion" paragraph (2â€“3 sentences) identifying the
        most significant findings and their relevance to the patient phenotype.
    """
    logger.info("[FinalConclusion] Synthesising overall clinical conclusion...")

    conclusions_text = "\n\n---\n\n".join(conclusions)

    if len(conclusions_text) > _MAX_CONCLUSIONS_CHARS:
        logger.warning(
            "[FinalConclusion] Combined conclusions (%d chars) exceeds budget â€” truncating.",
            len(conclusions_text),
        )
        conclusions_text = conclusions_text[:_MAX_CONCLUSIONS_CHARS] + "\n\n[... truncated ...]"

    template    = _load_prompt()
    user_prompt = (template
        .replace("{patient_phenotype}", patient_phenotype)
        .replace("{conclusions}", conclusions_text))

    return llm.generate(
        system="You are an expert clinical geneticist. Base every claim strictly on the evidence below.",
        user=user_prompt,
        max_tokens=MAX_NEW_TOKENS_FINAL_CONCLUSION,
    )


