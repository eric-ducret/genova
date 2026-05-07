"""
pipeline/pipeline.py — Top-level orchestrator.

Wires together:
  - LLM client (vLLM in production, HF in dev)
  - Tool instances
  - Manifest loader
  - Executor
  - Stage functions (retrieval → compression → reasoning → conclusion)

Design rule: server.py calls pipeline.run() and knows nothing about variants.
All logic lives here and in the stages/tools/core modules.

Public API:
    Pipeline(model_name, llm_kwargs)   — create and configure
    Pipeline.run(variants, phenotype)  — execute the full pipeline, return report string
"""

from __future__ import annotations

import re
import logging
from typing import Optional

from pipeline.core.manifest  import ManifestLoader
from pipeline.core.executor  import Executor
from pipeline.core.normalizer import TARGET_COLUMNS
from pipeline.llm.registry   import get_client
from pipeline.tools          import AutoPVS1Tool, LitVar2SummaryTool, SpliceAITool, WebSearchAgentTool
from pipeline.stages         import retrieval, compression, reasoning, conclusion

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# VARIANT STRING → DICT CONVERSION
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_variant_str(variant_str: str) -> dict:
    """
    Parse a canonical key=value variant string into a dict.

    The normalizer produces strings of the form:
        "Variant=chr6:100896130 T>C, Chromosome=chr6, Position=100896130, ..."

    We extract each TARGET_COLUMN by matching:
        <ColumnName>=<value>
    where value runs up to the next ", <NextColumnName>=" or end of string.

    This is more robust than a naive split on ", " because field values may
    themselves contain commas (e.g. OMIM_phenotype).
    """
    result: dict[str, str] = {}
    for col in TARGET_COLUMNS:
        # Pattern: "Col=<capture>" stopped at ", NextCol=" or end of string
        m = re.search(
            rf"(?:^|,\s*){re.escape(col)}=(.*?)(?=,\s*(?:{'|'.join(re.escape(c) for c in TARGET_COLUMNS)})=|$)",
            variant_str,
        )
        result[col] = m.group(1).strip() if m else "NA"
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

class Pipeline:
    """
    Full variant analysis pipeline.

    Instantiate once at server startup; call run() per request.

    Args:
        model_name: Key in pipeline.llm.registry.MODELS (default "qwen3.5-9b").
        llm_kwargs: Forwarded to get_client() — e.g. base_url for VLLMClient.
        enable_global_compression:
            If True, Stage 2 (global context compression) runs when the
            assembled context exceeds the token threshold. Default: True.
        global_compression_threshold:
            Token count above which Stage 2 triggers. Default: 28 000.
    """

    def __init__(
        self,
        model_name: str = "qwen3.5-9b",
        llm_kwargs: Optional[dict] = None,
        llm=None,                              # pre-built LLMClient (overrides model_name)
        enable_global_compression: bool = True,
        global_compression_threshold: int = 28_000,
    ):
        self._model_name = model_name
        self._llm_kwargs = llm_kwargs or {}
        self._enable_global_compression  = enable_global_compression
        self._global_compression_threshold = global_compression_threshold

        # ── LLM client (loaded once at startup) ──────────────────────────────
        if llm is not None:
            logger.info("[Pipeline] Using pre-built LLM client: %s", type(llm).__name__)
            self._llm = llm
        else:
            logger.info("[Pipeline] Loading LLM client: %s", model_name)
            self._llm = get_client(model_name, **self._llm_kwargs)

        # ── Manifests ─────────────────────────────────────────────────────────
        loader = ManifestLoader()
        self._manifests = loader.load_all()
        logger.info(
            "[Pipeline] Loaded %d manifests: %s",
            len(self._manifests),
            [m["name"] for m in self._manifests],
        )

        # ── Tool instances (one per pipeline tool) ────────────────────────────
        self._tools = [
            LitVar2SummaryTool(),
            SpliceAITool(),
            AutoPVS1Tool(),
            WebSearchAgentTool(),
        ]

        # ── Executor ──────────────────────────────────────────────────────────
        self._executor = Executor(
            llm=self._llm,
            tools=self._tools,
            manifests=self._manifests,
        )
        logger.info("[Pipeline] Executor ready with %d tools.", len(self._executor._paired))

    # ── public ────────────────────────────────────────────────────────────────

    async def run(self, variants: list[str], patient_phenotype: str) -> str:
        """
        Execute the full four-stage pipeline and return the combined output.

        Args:
            variants:          List of key=value variant strings from normalize_upload().
            patient_phenotype: Free-text patient phenotype string from the request.

        Returns:
            Combined output string:
                AUGMENTED CONTEXT section
                REASONING section
                FINAL REPORT section

        Raises:
            ValueError if variants is empty.
        """
        if not variants:
            raise ValueError("variants list is empty — nothing to analyze.")

        # ── String → dict conversion ──────────────────────────────────────────
        variant_dicts = [_parse_variant_str(v) for v in variants]
        logger.info("[Pipeline] Processing %d variants.", len(variant_dicts))

        # ── Stage 1: Retrieval ────────────────────────────────────────────────
        augmented_context = retrieval.run(
            variants=variant_dicts,
            patient_phenotype=patient_phenotype,
            executor=self._executor,
        )

        # ── Stage 2: Global compression (optional) ────────────────────────────
        if self._enable_global_compression:
            augmented_context = compression.run(
                augmented_context=augmented_context,
                llm=self._llm,
                token_threshold=self._global_compression_threshold,
            )

        # ── Stage 3: Reasoning ────────────────────────────────────────────────
        reasoning_output = reasoning.run(
            augmented_context=augmented_context,
            llm=self._llm,
        )

        # ── Stage 4: Conclusion ───────────────────────────────────────────────
        report = conclusion.run(
            augmented_context=augmented_context,
            reasoning=reasoning_output,
            llm=self._llm,
        )

        # ── Assemble final output (matches server_main.py /analyze format) ────
        SEP = "=" * 60
        output = (
            f"{SEP}\nAUGMENTED CONTEXT\n{SEP}\n"
            + augmented_context + "\n\n"
            + f"{SEP}\nREASONING\n{SEP}\n"
            + reasoning_output + "\n\n"
            + f"{SEP}\nFINAL REPORT\n{SEP}\n"
            + report + "\n"
        )
        return output
