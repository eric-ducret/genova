#!/usr/bin/env python3
"""
batch_pheno.py — Run the variant pipeline on one batch of patients.

Reads pheno_prompts.csv (patient_id, prompt, n_variants, …) from the same
directory as this script.  Patients with n_variants > 600 are excluded before
splitting.  The eligible patients are divided into --n-batches sequential
batches; pass 1..N as the first argument to select which batch to run.

Variant files are expected at:
    <jobs_folder>/cut_viland_<patient_id>.csv

Phenotype is taken from the `prompt` column in pheno_prompts.csv.

Usage:
    python batch_pheno.py <batch_number> <jobs_folder>
    python batch_pheno.py <batch_number> <jobs_folder> --n-batches 56
    python batch_pheno.py <batch_number> <jobs_folder> --backend vllm
    python batch_pheno.py <batch_number> <jobs_folder> --backend vllm --vllm-url http://localhost:8001
"""

import argparse
import asyncio
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_SCRIPT_DIR   = Path(__file__).parent
_PHENO_CSV    = _SCRIPT_DIR / "pheno_prompts.csv"
_RESULTS_DIR  = _SCRIPT_DIR / "results"
_MAX_VARIANTS = 600
_DEFAULT_N_BATCHES = 10


def _load_batch(batch_number: int, n_batches: int) -> list[tuple[str, str]]:
    """Return [(patient_id, prompt), …] for the requested batch."""
    df = pd.read_csv(_PHENO_CSV)

    eligible = df[df["n_variants"] <= _MAX_VARIANTS].reset_index(drop=True)
    excluded = len(df) - len(eligible)
    if excluded:
        logger.info(
            "[batch_pheno] Excluded %d patient(s) with >%d variants",
            excluded, _MAX_VARIANTS,
        )

    total = len(eligible)
    base, remainder = divmod(total, n_batches)
    # batches 1…remainder get (base+1), the rest get base
    sizes = [base + 1 if i < remainder else base for i in range(n_batches)]
    starts = [sum(sizes[:i]) for i in range(n_batches)]

    idx = batch_number - 1
    start = starts[idx]
    size  = sizes[idx]
    chunk = eligible.iloc[start : start + size]

    logger.info(
        "[batch_pheno] Batch %d/%d — patients %d–%d of %d eligible (%d patients)",
        batch_number, n_batches,
        start + 1, start + size, total, size,
    )
    return list(zip(chunk["patient_id"].astype(str), chunk["prompt"].astype(str)))


async def _run_all(
    patients: list[tuple[str, str]],
    jobs_folder: Path,
    pipeline,
) -> list[str | None]:
    from pipeline.core.normalizer import normalize_upload

    saved: list[str | None] = []
    for i, (patient_id, phenotype) in enumerate(patients, 1):
        csv_path = jobs_folder / f"cut_viland_{patient_id}.csv"
        logger.info("[batch_pheno] Job %d/%d — %s", i, len(patients), csv_path.name)

        # Resume: skip patients that already have a completed report
        existing = sorted(_RESULTS_DIR.glob(f"{patient_id}_*/report.txt"))
        if existing:
            logger.info(
                "[batch_pheno] %s already done (%s) — skipping",
                patient_id, existing[-1].parent.name,
            )
            saved.append(str(existing[-1].parent))
            continue

        if not csv_path.exists():
            logger.warning("[batch_pheno] %s not found — skipping", csv_path)
            saved.append(None)
            continue

        raw = csv_path.read_bytes()
        try:
            variants, raw_rows = normalize_upload(raw, csv_path.name)
        except ValueError as e:
            logger.error("[batch_pheno] Cannot normalize %s: %s — skipping", csv_path.name, e)
            saved.append(None)
            continue

        if not variants:
            logger.warning("[batch_pheno] %s has no variant rows — skipping", csv_path.name)
            saved.append(None)
            continue

        try:
            report = await pipeline.run(variants, phenotype, raw_rows=raw_rows)
        except Exception as e:
            logger.error("[batch_pheno] Patient %s pipeline failed: %s — skipping", patient_id, e)
            saved.append(None)
            continue

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = _RESULTS_DIR / f"{patient_id}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "report.txt").write_text(report, encoding="utf-8")
        logger.info("[batch_pheno] Saved → %s", run_dir)
        saved.append(str(run_dir))

    return saved


def _notify(email: str, batch_number: int, n_batches: int, n_done: int, n_failed: int) -> None:
    body = (
        f"Batch {batch_number}/{n_batches} complete.\n\n"
        f"  Jobs done   : {n_done}\n"
        f"  Jobs failed : {n_failed}\n"
    )
    try:
        subprocess.run(
            ["mail", "-s", f"Variant pipeline batch {batch_number}/{n_batches} complete", email],
            input=body.encode(),
            check=True,
            timeout=10,
        )
        logger.info("[batch_pheno] Notification sent to %s", email)
    except Exception as e:
        logger.warning("[batch_pheno] Could not send email: %s", e)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run the variant pipeline on one patient batch."
    )
    p.add_argument("batch_number", type=int, metavar="BATCH",
                   help="Batch number to run (1-indexed, must be ≤ --n-batches)")
    p.add_argument("jobs_folder", type=Path,
                   help="Folder containing cut_viland_<patient_id>.csv files")
    p.add_argument("--n-batches", type=int, default=_DEFAULT_N_BATCHES, metavar="N",
                   help=f"Total number of batches the dataset is split into (default: {_DEFAULT_N_BATCHES})")
    p.add_argument("--backend", choices=["hf", "vllm"], default="vllm",
                   help="LLM backend (default: vllm)")
    p.add_argument("--vllm-url", default="http://localhost:8001", metavar="URL",
                   help="vLLM base URL (default: http://localhost:8001)")
    p.add_argument("--notify-email", metavar="ADDRESS", default="eric.ducret@unil.ch",
                   help="Send email summary on completion")
    args = p.parse_args()

    if args.batch_number < 1 or args.batch_number > args.n_batches:
        print(f"ERROR: batch_number must be between 1 and {args.n_batches}", file=sys.stderr)
        sys.exit(1)

    if not args.jobs_folder.is_dir():
        print(f"ERROR: {args.jobs_folder} is not a directory", file=sys.stderr)
        sys.exit(1)

    patients = _load_batch(args.batch_number, args.n_batches)
    if not patients:
        print("ERROR: batch is empty", file=sys.stderr)
        sys.exit(1)

    from pipeline.pipeline import Pipeline
    if args.backend == "hf":
        from pipeline.llm.hf_client import HFClient
        llm = HFClient(model_id="Qwen/Qwen3.5-9B")
        pipeline = Pipeline(llm=llm)
    else:
        pipeline = Pipeline(
            model_name="qwen3.5-9b",
            llm_kwargs={"base_url": args.vllm_url},
        )

    results = asyncio.run(_run_all(patients, args.jobs_folder, pipeline))

    done   = [r for r in results if r is not None]
    failed = len(results) - len(done)
    logger.info("[batch_pheno] %d succeeded, %d failed", len(done), failed)
    for r in done:
        print(r)

    if args.notify_email:
        _notify(args.notify_email, args.batch_number, args.n_batches, len(done), failed)


if __name__ == "__main__":
    main()
