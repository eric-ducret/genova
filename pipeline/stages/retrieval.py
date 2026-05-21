"""
pipeline/stages/retrieval.py — Per-variant tool fan-out stage.

Ported from run_browsing() in server_main.py.

For each variant, calls Executor.run_variant() to run all tools in manifest order,
then returns a list of per-variant context strings. Each string is self-contained:
it includes the PATIENT DATA header and one VARIANT block with tool outputs.

This per-variant structure allows downstream stages (reasoning, conclusion) to
operate on one variant at a time, reducing SLM memory pressure.

Public API:
    run(variants, patient_phenotype, executor) -> list[str]

Each element of the returned list follows this format:
    PATIENT DATA:
    {patient_phenotype}

    VARIANT DATA:

    VARIANT N:
    {variant_key_value_str}

    {tool outputs}
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pipeline.config import DEFAULT_GENOME_BUILD
from pipeline.core.normalizer import TARGET_COLUMNS

if TYPE_CHECKING:
    from pipeline.core.executor import Executor

logger = logging.getLogger(__name__)


def _variant_dict_to_str(variant: dict) -> str:
    """Convert a variant dict back to the canonical key=value string for display."""
    return ", ".join(f"{k}={variant.get(k, 'NA')}" for k in TARGET_COLUMNS)


def run(
    variants: list[dict],
    patient_phenotype: str,
    executor: "Executor",
    genome_build: str = DEFAULT_GENOME_BUILD,
    raw_rows: list[dict] | None = None,
) -> list[str]:
    """
    Stage 1 — Per-variant tool fan-out.

    Args:
        variants:          List of normalized variant dicts.
        patient_phenotype: Free-text patient phenotype.
        executor:          Configured Executor instance (tools + manifests + LLM).
        genome_build:      Reference genome build for this run.
        raw_rows:          Full original CSV rows, one dict per variant.

    Returns:
        List of per-variant context strings. Each string contains the PATIENT DATA
        header followed by one VARIANT block with tool outputs. The list has one
        entry per input variant, in the same order.
    """
    total = len(variants)
    context_slices: list[str] = []

    for i, variant in enumerate(variants):
        logger.info("[Retrieval] Variant %d/%d: %s", i + 1, total,
                    variant.get("Variant", "?"))

        raw_fields = raw_rows[i] if (raw_rows and i < len(raw_rows)) else {}

        variant_str = _variant_dict_to_str(variant)
        mini_doc    = executor.run_variant(
            variant=variant,
            patient_phenotype=patient_phenotype,
            variant_index=i,
            total_variants=total,
            genome_build=genome_build,
            raw_fields=raw_fields,
        )

        slice_ = (
            f"PATIENT DATA:\n{patient_phenotype}\n\n"
            f"VARIANT DATA:\n\n"
            f"VARIANT {i + 1}:\n{variant_str}\n\n{mini_doc}"
        )
        context_slices.append(slice_)

        logger.info("[Retrieval] Variant %d/%d done.", i + 1, total)

    return context_slices
