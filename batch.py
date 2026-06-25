#!/usr/bin/env python3
"""
batch.py — Run the variant analysis pipeline on a folder of job pairs.

Each job is a pair of files with the same stem inside the jobs folder:
    patient1.csv  — variants file (CSV or xlsx)
    patient1.txt  — patient phenotype description (plain text)

Results are saved to results/{stem}_{YYYYMMDD_HHMMSS}/report.txt,
following the same convention as the server.

Usage:
    python batch.py jobs/                                      # HF backend (local/dev)
    python batch.py jobs/ --backend vllm                       # vLLM backend
    python batch.py jobs/ --notify-email you@example.com       # email on completion
"""

import argparse
import asyncio
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_RESULTS_DIR = Path(__file__).parent / "results"

from pipeline.config import DEFAULT_MODEL
from pipeline.llm.registry import MODELS


def _find_jobs(folder: Path) -> list[tuple[Path, Path]]:
    """Return (csv, txt) pairs found in folder, matched by stem."""
    csv_files = sorted(folder.glob("*.csv")) + sorted(folder.glob("*.xlsx"))
    jobs = []
    for csv in csv_files:
        txt = csv.with_suffix(".txt")
        if txt.exists():
            jobs.append((csv, txt))
        else:
            logger.warning("[batch] No matching .txt for %s — skipping", csv.name)
    return jobs


async def _run_all(jobs: list[tuple[Path, Path]], pipeline) -> list[str | None]:
    from pipeline.core.normalizer import normalize_upload

    saved: list[str | None] = []
    for i, (csv_path, txt_path) in enumerate(jobs, 1):
        logger.info("[batch] Job %d/%d — %s", i, len(jobs), csv_path.name)

        phenotype = txt_path.read_text(encoding="utf-8").strip()
        raw = csv_path.read_bytes()

        try:
            variants, raw_rows = normalize_upload(raw, csv_path.name)
        except ValueError as e:
            logger.error("[batch] Cannot normalize %s: %s — skipping", csv_path.name, e)
            saved.append(None)
            continue

        if not variants:
            logger.warning("[batch] %s has no variant rows — skipping", csv_path.name)
            saved.append(None)
            continue

        report = await pipeline.run(variants, phenotype, raw_rows=raw_rows)

        stem = csv_path.stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = _RESULTS_DIR / f"{stem}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "report.txt").write_text(report, encoding="utf-8")
        logger.info("[batch] Saved → %s", run_dir)
        saved.append(str(run_dir))

    return saved


def _notify(email: str, n_done: int, n_failed: int, paths: list[str | None]) -> None:
    body = (
        f"Batch pipeline complete.\n\n"
        f"  Jobs done   : {n_done}\n"
        f"  Jobs failed : {n_failed}\n\n"
        f"Results:\n" + "\n".join(f"  {p}" for p in paths if p)
    )
    try:
        subprocess.run(
            ["mail", "-s", "Variant pipeline batch complete", email],
            input=body.encode(),
            check=True,
            timeout=10,
        )
        logger.info("[batch] Notification sent to %s", email)
    except Exception as e:
        logger.warning("[batch] Could not send email: %s", e)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run the variant analysis pipeline on a folder of CSV+txt job pairs."
    )
    p.add_argument("folder", type=Path,
                   help="Folder containing .csv + .txt job pairs (matched by stem)")
    p.add_argument("--backend", choices=["hf", "vllm"], default="hf",
                   help="LLM backend (default: hf)")
    p.add_argument("--vllm-url", default="http://localhost:8001", metavar="URL",
                   help="vLLM base URL when --backend=vllm (default: http://localhost:8001)")
    p.add_argument("--notify-email", metavar="ADDRESS", default="eric.ducret@unil.ch",
                   help="Send an email summary when all jobs finish (default: eric.ducret@unil.ch)")
    args = p.parse_args()

    if not args.folder.is_dir():
        print(f"ERROR: {args.folder} is not a directory", file=sys.stderr)
        sys.exit(1)

    jobs = _find_jobs(args.folder)
    if not jobs:
        print(f"ERROR: no CSV+txt pairs found in {args.folder}", file=sys.stderr)
        sys.exit(1)

    logger.info("[batch] %d job(s) found in %s", len(jobs), args.folder)

    from pipeline.pipeline import Pipeline
    if args.backend == "hf":
        from pipeline.llm.hf_client import HFClient
        llm = HFClient(model_id=MODELS[DEFAULT_MODEL]["model_id"])
        pipeline = Pipeline(llm=llm)
    else:
        pipeline = Pipeline(
            model_name=DEFAULT_MODEL,
            llm_kwargs={"base_url": args.vllm_url},
        )

    results = asyncio.run(_run_all(jobs, pipeline))

    done   = [r for r in results if r is not None]
    failed = len(results) - len(done)
    logger.info("[batch] %d succeeded, %d failed", len(done), failed)
    for r in done:
        print(r)

    if args.notify_email:
        _notify(args.notify_email, len(done), failed, results)


if __name__ == "__main__":
    main()