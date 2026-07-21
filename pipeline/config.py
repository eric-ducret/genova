"""
pipeline/config.py — Global pipeline configuration constants.
"""

import os

EXECUTION_MODE = "async"   # "async" | "sequential"

DEFAULT_GENOME_BUILD = "hg19"  # fallback; override per-run when input format is known

TRIAGE_ENABLED_THRESHOLD  = 12   # skip first_triage if n_variants <= this

# Model selection — override with: export VARIANT_MODEL=gemma4-e4b
# Valid values are keys in pipeline/llm/registry.MODELS.
DEFAULT_MODEL = os.environ.get("VARIANT_MODEL", "qwen3.5-9b")

# Ablation flag: replace every incoming patient phenotype with a fixed generic
# placeholder before the pipeline runs. Used to test whether conclusions are
# actually driven by phenotype-specific evidence rather than variant data alone.
# Applied once in Pipeline.run() so it holds regardless of entry point
# (server, batch.py, batch_pheno.py, client.py).
# Override with: export VARIANT_FORCE_GENERIC_PHENOTYPE=0
FORCE_GENERIC_PHENOTYPE = os.environ.get("VARIANT_FORCE_GENERIC_PHENOTYPE", "1") == "1"
GENERIC_PHENOTYPE_TEXT  = "The patient presents with visual impairment."
