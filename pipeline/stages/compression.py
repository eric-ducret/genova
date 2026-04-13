"""
pipeline/stages/compression.py — Global context compression stage.

This is Stage 2 of the pipeline — separate from per-tool compression
(which compresses individual tool outputs in the executor).

Global compression runs after all per-variant mini-docs are assembled.
It triggers when the total assembled context exceeds the token threshold
and compresses each VARIANT section independently to fit in the SLM context window.

Public API:
    run(augmented_context, llm, token_threshold, max_tokens_per_variant) -> str

Note: If the context is within the threshold, it is returned unchanged.
"""

from __future__ import annotations

import re
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.llm.base import LLMClient

logger = logging.getLogger(__name__)

# Rough token estimate: 1 token ≈ 4 characters
_CHARS_PER_TOKEN = 4

# Default thresholds
_DEFAULT_TOKEN_THRESHOLD    = 28_000   # trigger compression above this
_DEFAULT_MAX_TOKENS_PER_VAR = 1_200    # target per variant after compression


def _estimate_tokens(text: str) -> int:
    return len(text) // _CHARS_PER_TOKEN


# Matches "VARIANT N:\n..." sections produced by retrieval.py
_VARIANT_SECTION_RE = re.compile(
    r"(VARIANT \d+:.*?)(?=\nVARIANT \d+:|\Z)",
    re.DOTALL,
)


def _compress_section(
    section: str,
    llm: "LLMClient",
    max_tokens: int,
) -> str:
    """Compress a single VARIANT section using the SLM."""
    system = (
        "You are a clinical evidence compressor. "
        "Preserve ALL of: variant identifier, gene name, HGVS notation, zygosity, "
        "ACMG criteria, PVS1 strength, key functional evidence, phenotype fit, "
        "population frequency, and any URLs cited. "
        "Remove verbose restatement, raw API fields already interpreted, and repetition. "
        "Return only the compressed text — no commentary, no headers."
    )
    user = (
        f"Compress the following variant evidence section to {max_tokens} tokens maximum:\n\n"
        f"{section}"
    )
    try:
        compressed = llm.generate(
            system=system,
            user=user,
            max_tokens=max_tokens,
        )
        # Re-attach the VARIANT header if the SLM dropped it
        header_match = re.match(r"(VARIANT \d+:.*?\n)", section)
        header = header_match.group(1) if header_match else ""
        if header and not compressed.startswith("VARIANT"):
            return header + compressed.strip()
        return compressed.strip()
    except Exception as e:
        logger.warning(
            "[Compression] SLM compression failed for a variant section (%s) — keeping original.", e
        )
        return section


def run(
    augmented_context: str,
    llm: "LLMClient",
    token_threshold: int = _DEFAULT_TOKEN_THRESHOLD,
    max_tokens_per_variant: int = _DEFAULT_MAX_TOKENS_PER_VAR,
) -> str:
    """
    Stage 2 — Global context compression (optional).

    If the augmented context is within the token threshold, returns it unchanged.
    Otherwise, splits the context at VARIANT section boundaries and compresses
    each variant section independently using the SLM.

    The PATIENT DATA header and VARIANT DATA header are preserved unchanged.

    Args:
        augmented_context:      Output from stages/retrieval.run().
        llm:                    Shared LLMClient instance.
        token_threshold:        Context size (tokens) above which compression triggers.
        max_tokens_per_variant: Target token budget per variant after compression.

    Returns:
        Compressed (or unchanged) augmented context string.
    """
    estimated = _estimate_tokens(augmented_context)
    if estimated <= token_threshold:
        logger.info(
            "[Compression] Context within threshold (%d ≤ %d tokens) — skipping.",
            estimated, token_threshold,
        )
        return augmented_context

    logger.info(
        "[Compression] Context exceeds threshold (%d > %d tokens) — compressing.",
        estimated, token_threshold,
    )

    # Split into header (PATIENT DATA + VARIANT DATA preamble) and variant sections
    header_match = re.match(r"(.*?VARIANT DATA:\n\n)", augmented_context, re.DOTALL)
    header = header_match.group(1) if header_match else ""
    body   = augmented_context[len(header):]

    sections = _VARIANT_SECTION_RE.findall(body)
    if not sections:
        logger.warning("[Compression] Could not parse variant sections — returning as-is.")
        return augmented_context

    compressed_sections = []
    for i, section in enumerate(sections):
        logger.info("[Compression] Compressing variant section %d/%d", i + 1, len(sections))
        compressed_sections.append(
            _compress_section(section.strip(), llm, max_tokens_per_variant)
        )

    return header + "\n\n".join(compressed_sections) + "\n"
