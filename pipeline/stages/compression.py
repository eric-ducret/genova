"""
pipeline/stages/compression.py — Global context compression stage.

This is Stage 2 of the pipeline — separate from per-tool compression
(which compresses individual tool outputs in the executor).

Global compression runs after all per-variant context slices are assembled by
Stage 1 (retrieval). It triggers when the total assembled context exceeds the
token threshold and compresses each variant slice independently.

Each slice has the structure:
    PATIENT DATA:
    {patient_phenotype}

    VARIANT DATA:

    VARIANT N:
    {tool outputs}

The PATIENT DATA header is preserved within each slice; only the VARIANT body
is passed to the SLM for compression.

Public API:
    run(context_slices, llm, token_threshold, max_tokens_per_variant) -> list[str]

Note: If the context is within the threshold, the input list is returned unchanged.
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
_DEFAULT_TOKEN_THRESHOLD    = 4000   # trigger compression above this
_DEFAULT_MAX_TOKENS_PER_VAR = 600    # target per variant after compression

# Matches the shared header within each slice (PATIENT DATA … VARIANT DATA:\n\n)
_SLICE_HEADER_RE = re.compile(r"^(PATIENT DATA:.*?VARIANT DATA:\n\n)", re.DOTALL)


def _estimate_tokens(text: str) -> int:
    return len(text) // _CHARS_PER_TOKEN


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
    context_slices: list[str],
    llm: "LLMClient",
    token_threshold: int = _DEFAULT_TOKEN_THRESHOLD,
    max_tokens_per_variant: int = _DEFAULT_MAX_TOKENS_PER_VAR,
) -> list[str]:
    """
    Stage 2 — Global context compression (optional).

    If the total context is within the token threshold, returns the input list
    unchanged. Otherwise, compresses the VARIANT body of each slice independently
    using the SLM, preserving the PATIENT DATA header in each slice.

    Args:
        context_slices:         Output from stages/retrieval.run() — one str per variant.
        llm:                    Shared LLMClient instance.
        token_threshold:        Total context size (tokens) above which compression triggers.
        max_tokens_per_variant: Target token budget per variant body after compression.

    Returns:
        List of context slices, compressed or unchanged.
    """
    total_text = "".join(context_slices)
    estimated  = _estimate_tokens(total_text)

    if estimated <= token_threshold:
        logger.info(
            "[Compression] Context within threshold (%d ≤ %d tokens) — skipping.",
            estimated, token_threshold,
        )
        return context_slices

    logger.info(
        "[Compression] Context exceeds threshold (%d > %d tokens) — compressing %d slices.",
        estimated, token_threshold, len(context_slices),
    )

    compressed_slices: list[str] = []
    for i, slice_ in enumerate(context_slices):
        logger.info("[Compression] Compressing variant slice %d/%d", i + 1, len(context_slices))

        # Split each slice into shared header and variant body
        header_match = _SLICE_HEADER_RE.match(slice_)
        header       = header_match.group(1) if header_match else ""
        variant_body = slice_[len(header):]

        if not variant_body.strip():
            compressed_slices.append(slice_)
            continue

        compressed_body = _compress_section(variant_body.strip(), llm, max_tokens_per_variant)
        compressed_slices.append(header + compressed_body.strip() + "\n")

    return compressed_slices
