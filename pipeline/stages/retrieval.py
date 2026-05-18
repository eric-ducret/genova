"""
pipeline/stages/retrieval.py — Per-variant tool fan-out stage.

Ported from run_browsing() in server_main.py.

For each variant, calls Executor.run_variant() to run all tools in manifest order,
then assembles the per-variant sections into the full augmented context string
that is passed to Stage 2 (compression) and Stage 3 (reasoning).

Public API:
    run(variants, patient_phenotype, executor) -> str

Output format matches server_main.py run_browsing() output:
    PATIENT DATA:
    {patient_phenotype}

    VARIANT DATA:

    VARIANT 1:
    {variant_key_value_str}

    {tool outputs}

    VARIANT 2:
    ...
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

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
    genome_build: str = "hg19",
    raw_rows: list[dict] | None = None,
) -> str:
    """
    Stage 1 — Per-variant tool fan-out.

    Args:
        variants:          List of normalized variant dicts.
        patient_phenotype: Free-text patient phenotype.
        executor:          Configured Executor instance (tools + manifests + LLM).

    Returns:
        Augmented context string: PATIENT DATA header + VARIANT sections.
        Each VARIANT section contains tool outputs assembled by the executor.
    """
    total = len(variants)
    sections: list[str] = []

    for i, variant in enumerate(variants):
        logger.info("[Retrieval] Variant %d/%d: %s", i + 1, total,
                    variant.get("Variant", "?"))

        raw_fields = raw_rows[i] if (raw_rows and i < len(raw_rows)) else {}

        variant_str  = _variant_dict_to_str(variant)
        mini_doc     = executor.run_variant(
            variant=variant,
            patient_phenotype=patient_phenotype,
            variant_index=i,
            total_variants=total,
            genome_build=genome_build,
            raw_fields=raw_fields,
        )

        section = f"VARIANT {i + 1}:\n{variant_str}\n\n{mini_doc}"
        sections.append(section)

        logger.info("[Retrieval] Variant %d/%d done.", i + 1, total)

    augmented_context = (
        f"PATIENT DATA:\n{patient_phenotype}\n\n"
        f"VARIANT DATA:\n\n"
        + "\n\n".join(sections)
        + "\n"
    )
    return augmented_context
