#!/usr/bin/env python3
"""
submit_batches.py — Compute time-aware batch count and submit all sbatch jobs.

Reads pheno_prompts.csv, estimates runtime per patient from variant count, and
splits eligible patients into the smallest number of batches that each fit within
the walltime budget.  Then submits all batches via:

    sbatch batch_pheno_vllm.sh <batch_number> <jobs_folder> <n_batches>

Calibration:  171 variants ≈ 20 min (no API key) / 15 min (with NCBI_API_KEY).
Overhead:     vLLM startup adds ~10 min to every job.

Usage:
    python submit_batches.py <jobs_folder>
    python submit_batches.py <jobs_folder> --max-hours 6
    python submit_batches.py <jobs_folder> --patients-per-batch 10
    python submit_batches.py <jobs_folder> --dry-run
"""

import argparse
import csv
import math
import os
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR       = Path(__file__).parent
_PHENO_CSV        = _SCRIPT_DIR / "pheno_prompts.csv"
_SBATCH_SCRIPT    = _SCRIPT_DIR / "batch_pheno_vllm.sh"
_MAX_VARIANTS     = 600
_VLLM_STARTUP_MIN = 10      # minutes added per job for vLLM startup
# Calibration notes (171-variant patient):
#   Without NCBI API key (3 req/s, serialised):  ~20 min → 0.117 min/variant
#   With    NCBI API key (10 req/s, serialised):  ~15 min → 0.088 min/variant
# Free key: https://www.ncbi.nlm.nih.gov/account/  → set NCBI_API_KEY env var.
_MINS_PER_VARIANT = 0.088 if os.environ.get("NCBI_API_KEY") else 0.117


def _load_eligible() -> list[int]:
    """Return list of n_variants for each eligible patient (n_variants <= _MAX_VARIANTS)."""
    all_variants = []
    eligible_variants = []
    with open(_PHENO_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            n = int(row["n_variants"])
            all_variants.append(n)
            if n <= _MAX_VARIANTS:
                eligible_variants.append(n)
    n_excluded = len(all_variants) - len(eligible_variants)
    if n_excluded:
        print(f"  Excluded {n_excluded} patient(s) with >{_MAX_VARIANTS} variants")
    return eligible_variants


def _compute_n_batches(eligible: list[int], max_hours: float) -> tuple[int, int, float]:
    """Return (n_batches, patients_per_batch, estimated_minutes_per_batch)."""
    available_min = max_hours * 60 - _VLLM_STARTUP_MIN
    avg_variants  = sum(eligible) / len(eligible)
    avg_min       = avg_variants * _MINS_PER_VARIANT

    patients_per_batch = max(1, int(available_min / avg_min))
    n_batches          = math.ceil(len(eligible) / patients_per_batch)
    est_min_per_batch  = patients_per_batch * avg_min + _VLLM_STARTUP_MIN

    return n_batches, patients_per_batch, est_min_per_batch


def main() -> None:
    p = argparse.ArgumentParser(
        description="Submit all time-limited variant pipeline batches to SLURM."
    )
    p.add_argument("jobs_folder", type=Path,
                   help="Folder containing cut_viland_<patient_id>.csv files")
    p.add_argument("--max-hours", type=float, default=6.0, metavar="H",
                   help="Target max walltime per job in hours (default: 6.0)")
    p.add_argument("--patients-per-batch", type=int, default=None, metavar="N",
                   help="Override: fix how many patients per batch (skips auto-compute)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print sbatch commands without submitting")
    args = p.parse_args()

    if not args.jobs_folder.is_dir():
        print(f"ERROR: {args.jobs_folder} is not a directory", file=sys.stderr)
        sys.exit(1)

    if not _PHENO_CSV.exists():
        print(f"ERROR: {_PHENO_CSV} not found", file=sys.stderr)
        sys.exit(1)

    if not _SBATCH_SCRIPT.exists():
        print(f"ERROR: {_SBATCH_SCRIPT} not found", file=sys.stderr)
        sys.exit(1)

    eligible = _load_eligible()
    total    = len(eligible)

    avg_variants = sum(eligible) / len(eligible)

    if args.patients_per_batch:
        patients_per_batch = args.patients_per_batch
        n_batches          = math.ceil(total / patients_per_batch)
        est_min            = patients_per_batch * avg_variants * _MINS_PER_VARIANT + _VLLM_STARTUP_MIN
    else:
        n_batches, patients_per_batch, est_min = _compute_n_batches(eligible, args.max_hours)

    print(f"\nPatients eligible : {total}")
    print(f"Patients per batch: ~{patients_per_batch}  "
          f"(avg {avg_variants:.0f} variants/patient × "
          f"{_MINS_PER_VARIANT:.4f} min/variant)")
    print(f"Estimated runtime : ~{est_min:.0f} min / {est_min/60:.1f}h per job  "
          f"(incl. {_VLLM_STARTUP_MIN} min vLLM startup)")
    print(f"Total batches     : {n_batches}")
    print(f"Walltime limit    : {args.max_hours:.1f}h  "
          f"({'OK' if est_min/60 <= args.max_hours else 'WARNING: may exceed limit'})")
    print()

    if args.dry_run:
        print("=== DRY RUN — commands that would be submitted ===")

    submitted = 0
    for i in range(1, n_batches + 1):
        cmd = [
            "sbatch",
            str(_SBATCH_SCRIPT),
            str(i),
            str(args.jobs_folder),
            str(n_batches),
        ]
        if args.dry_run:
            print(f"  {' '.join(cmd)}")
        else:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  Batch {i}/{n_batches}: FAILED — {result.stderr.strip()}", file=sys.stderr)
            else:
                job_id = result.stdout.strip()
                print(f"  Batch {i}/{n_batches}: {job_id}")
            submitted += 1

    if not args.dry_run:
        print(f"\nSubmitted {submitted}/{n_batches} jobs.")
        print("Monitor with:  squeue -u $USER")


if __name__ == "__main__":
    main()
