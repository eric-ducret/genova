"""
pipeline/config.py — Global pipeline configuration constants.
"""

EXECUTION_MODE = "async"   # "async" | "sequential"

DEFAULT_GENOME_BUILD = "hg19"  # fallback; override per-run when input format is known

MAX_CONCLUSION_VARIANTS   = 12   # hard cap on variants entering conclusion
MIN_CONCLUSION_VARIANTS   = 5    # minimum to aim for when backfilling tier 2
TRIAGE_ENABLED_THRESHOLD  = MAX_CONCLUSION_VARIANTS  # skip triage if n_variants <= this
