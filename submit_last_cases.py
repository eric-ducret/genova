#!/usr/bin/env python3
"""
submit_last_cases.py — Submit one SLURM job per patient that has no completed report.

Usage:
    python submit_last_cases.py <jobs_folder>
    python submit_last_cases.py <jobs_folder> --dry-run
"""

import argparse
import csv
import math
import re
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR       = Path(__file__).parent
_PHENO_CSV        = _SCRIPT_DIR / "pheno_prompts.csv"
_RESULTS_DIR      = _SCRIPT_DIR / "results"
_SBATCH_SCRIPT    = _SCRIPT_DIR / "single_patient_vllm.sh"
_MAX_VARIANTS     = 600
_VLLM_STARTUP_MIN = 10
_MINS_PER_VARIANT = 0.117   # conservative (no NCBI API key)
_BUFFER_MIN       = 30      # safety margin


def _find_missing() -> list[tuple[str, int]]:
    seen: set[str] = set()
    eligible: list[tuple[str, int]] = []
    with open(_PHENO_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row["patient_id"]
            n   = int(row["n_variants"])
            if n <= _MAX_VARIANTS and pid not in seen:
                seen.add(pid)
                eligible.append((pid, n))

    done = {
        m.group(1)
        for p in _RESULTS_DIR.glob("*/report.txt")
        if (m := re.match(r"^(.+?)_\d{8}_\d{6}$", p.parent.name))
    }

    return [(pid, n) for pid, n in eligible if pid not in done]


def _walltime(n_variants: int) -> str:
    total_min = math.ceil(n_variants * _MINS_PER_VARIANT) + _VLLM_STARTUP_MIN + _BUFFER_MIN
    h, m = divmod(total_min, 60)
    return f"{h:02d}:{m:02d}:00"


def main() -> None:
    ap = argparse.ArgumentParser(description="Submit one SLURM job per missing patient report.")
    ap.add_argument("jobs_folder", type=Path,
                    help="Folder containing cut_viland_<patient_id>.csv files")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print sbatch commands without submitting")
    args = ap.parse_args()

    if not args.jobs_folder.is_dir():
        print(f"ERROR: {args.jobs_folder} is not a directory", file=sys.stderr)
        sys.exit(1)

    if not _SBATCH_SCRIPT.exists():
        print(f"ERROR: {_SBATCH_SCRIPT} not found — create it alongside this script",
              file=sys.stderr)
        sys.exit(1)

    missing = _find_missing()
    if not missing:
        print("No missing patients — all done.")
        return

    print(f"Missing patients: {len(missing)}")
    for pid, n in missing:
        wt  = _walltime(n)
        cmd = ["sbatch", f"--time={wt}", str(_SBATCH_SCRIPT), pid, str(args.jobs_folder), str(_SCRIPT_DIR)]
        if args.dry_run:
            print(f"  [DRY RUN] {' '.join(cmd)}  ({n} variants)")
        else:
            res = subprocess.run(cmd, capture_output=True, text=True)
            status = res.stdout.strip() if res.returncode == 0 else f"FAILED — {res.stderr.strip()}"
            print(f"  {pid} ({n} variants, walltime {wt}): {status}")


if __name__ == "__main__":
    main()
