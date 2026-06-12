#!/bin/bash --login
#SBATCH --job-name=variant_single
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=eric.ducret@unil.ch

# Usage: sbatch --time=HH:MM:SS single_patient_vllm.sh <patient_id> <jobs_folder>
PATIENT_ID="${1:?Usage: sbatch single_patient_vllm.sh <patient_id> <jobs_folder> <pipeline_dir>}"
JOBS_FOLDER="${2:?Usage: sbatch single_patient_vllm.sh <patient_id> <jobs_folder> <pipeline_dir>}"
export PIPELINE_DIR="${3:?Usage: sbatch single_patient_vllm.sh <patient_id> <jobs_folder> <pipeline_dir>}"

CONDA_ENV="/work/PRTNR/CHUV/MED/fsantoni1/pitnet/AI/JING/GenMasterAI/eric_folder/env_vllm_0606"
VLLM_LOG="/work/PRTNR/CHUV/MED/fsantoni1/pitnet/AI/JING/GenMasterAI/eric_folder/vllm_startup.log"

dcsrsoft use 20241118
module load miniforge3
module load cuda
conda_init
set -e
conda activate "$CONDA_ENV"

echo "=== GPU INFO ===" && nvidia-smi && echo "================"
echo "Patient: $PATIENT_ID | Jobs folder: $JOBS_FOLDER"

# --- Start vLLM server ---
vllm serve Qwen/Qwen3.5-9B \
    --dtype bfloat16 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.90 \
    --port 8001 \
    --enable-prefix-caching \
    > "$VLLM_LOG" 2>&1 &

VLLM_PID=$!
trap "kill $VLLM_PID 2>/dev/null" EXIT

for i in $(seq 1 60); do
    curl -sf http://localhost:8001/v1/models > /dev/null 2>&1 && { echo "vLLM ready after $((i * 5))s"; break; }
    [ $i -eq 60 ] && { echo "ERROR: vLLM did not start after 10 minutes"; cat "$VLLM_LOG"; exit 1; }
    echo "  attempt $i/60..." && sleep 5
done

# --- Run single patient ---
python - "$PATIENT_ID" "$JOBS_FOLDER" <<'PYEOF'
import asyncio, csv, logging, os, sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

patient_id   = sys.argv[1]
jobs_folder  = Path(sys.argv[2])
pipeline_dir = Path(os.environ["PIPELINE_DIR"])
results_dir  = pipeline_dir / "results"
pheno_csv    = pipeline_dir / "pheno_prompts.csv"

existing = sorted(results_dir.glob(f"{patient_id}_*/report.txt"))
if existing:
    logger.info("%s already done (%s) — skipping", patient_id, existing[-1].parent.name)
    sys.exit(0)

with open(pheno_csv) as f:
    phenotype = next((r["prompt"] for r in csv.DictReader(f) if r["patient_id"] == patient_id), None)

if phenotype is None:
    logger.error("%s not found in %s", patient_id, pheno_csv)
    sys.exit(1)

csv_path = jobs_folder / f"cut_viland_{patient_id}.csv"
if not csv_path.exists():
    logger.error("%s not found", csv_path)
    sys.exit(1)

sys.path.insert(0, str(pipeline_dir))
from pipeline.pipeline import Pipeline
from pipeline.core.normalizer import normalize_upload

variants, raw_rows = normalize_upload(csv_path.read_bytes(), csv_path.name)
if not variants:
    logger.error("%s has no variant rows", csv_path.name)
    sys.exit(1)

pipeline_obj = Pipeline(model_name="qwen3.5-9b", llm_kwargs={"base_url": "http://localhost:8001"})

async def run():
    report = await pipeline_obj.run(variants, phenotype, raw_rows=raw_rows)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_dir / f"{patient_id}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "report.txt").write_text(report, encoding="utf-8")
    logger.info("Saved → %s", run_dir)

asyncio.run(run())
PYEOF
