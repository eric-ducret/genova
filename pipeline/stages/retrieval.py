"""
pipeline/stages/retrieval.py — Per-variant tool fan-out stage.

For each variant, calls Executor.run_variant() to run all tools in manifest order,
then returns a list of per-variant context strings. Each string is self-contained:
it includes the PATIENT DATA header and one VARIANT block with tool outputs.

Variants are processed concurrently up to MAX_WORKERS using a ThreadPoolExecutor.
Tools within each variant still run in manifest order (sequential within the thread).
vLLM handles concurrent HTTP requests natively via continuous batching, so parallel
variants translate directly into higher GPU throughput.

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

from pipeline.config import DEFAULT_GENOME_BUILD
from pipeline.core.normalizer import TARGET_COLUMNS

if TYPE_CHECKING:
    from pipeline.core.executor import Executor

logger = logging.getLogger(__name__)

MAX_WORKERS = 8


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
    Stage 1 — Per-variant tool fan-out, parallelised across variants.

    Args:
        variants:          List of normalized variant dicts.
        patient_phenotype: Free-text patient phenotype.
        executor:          Configured Executor instance (tools + manifests + LLM).
        genome_build:      Reference genome build for this run.
        raw_rows:          Full original CSV rows, one dict per variant.

    Returns:
        List of per-variant context strings in the same order as the input.
        Sets executor.process_log to the accumulated, index-sorted log entries.
    """
    total   = len(variants)
    workers = min(MAX_WORKERS, total)

    context_slices: list[str | None] = [None] * total
    all_log_entries: list[dict]      = []

    def _process_one(i: int) -> tuple[int, str, list[dict]]:
        variant    = variants[i]
        raw_fields = raw_rows[i] if (raw_rows and i < len(raw_rows)) else {}
        variant_str = _variant_dict_to_str(variant)

        logger.info("[Retrieval] Variant %d/%d: %s", i + 1, total,
                    variant.get("Variant", "?"))

        mini_doc, log_entries = executor.run_variant(
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
        logger.info("[Retrieval] Variant %d/%d done.", i + 1, total)
        return i, slice_, log_entries

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_one, i): i for i in range(total)}
        for future in as_completed(futures):
            i = futures[future]
            try:
                idx, slice_, log_entries = future.result()
                context_slices[idx] = slice_
                all_log_entries.extend(log_entries)
            except Exception as exc:
                logger.error("[Retrieval] Variant %d failed: %s", i + 1, exc)
                variant_str = _variant_dict_to_str(variants[i])
                context_slices[i] = (
                    f"PATIENT DATA:\n{patient_phenotype}\n\n"
                    f"VARIANT DATA:\n\n"
                    f"VARIANT {i + 1}:\n{variant_str}\n\n"
                    f"[RETRIEVAL FAILED: {exc}]"
                )

    # Restore original variant order in the process log for the pipeline display
    executor.process_log = sorted(all_log_entries, key=lambda e: e["variant_index"])

    return context_slices  # type: ignore[return-value]  # all slots filled above
