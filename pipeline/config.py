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
