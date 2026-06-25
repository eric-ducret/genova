#!/bin/bash --login
#SBATCH --job-name=analyse_results
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=eric.ducret@unil.ch

# Usage: sbatch analyse_results/analyse_results.sh [results_dir] [workers] [pheno_csv]
#   results_dir  — path to results folder       (default below)
#   workers      — parallel SLM workers          (default: 8)
#   pheno_csv    — path to pheno_prompts.csv     (default below)
RESULTS_DIR="${1:-/work/PRTNR/CHUV/MED/fsantoni1/pitnet/AI/JING/GenMasterAI/eric_folder/genova_vllm_556_0617/results}"
WORKERS="${2:-8}"
PHENO_CSV="${3:-/work/PRTNR/CHUV/MED/fsantoni1/pitnet/AI/JING/GenMasterAI/eric_folder/genova_vllm_556_0617/pheno_prompts.csv}"

PROJECT_DIR="/work/PRTNR/CHUV/MED/fsantoni1/pitnet/AI/JING/GenMasterAI/eric_folder/genova_vllm_556_0617"
CONDA_ENV="/work/PRTNR/CHUV/MED/fsantoni1/pitnet/AI/JING/GenMasterAI/eric_folder/env_vllm_0606"
VLLM_LOG="$PROJECT_DIR/vllm_startup.log"

# --- Software environment ---
dcsrsoft use 20241118
module load miniforge3
module load cuda
conda_init

set -e

conda activate "$CONDA_ENV"

# --- GPU info ---
echo "=== GPU INFO ==="
nvidia-smi
echo "================"

# --- Start vLLM server ---
echo "Starting vLLM server..."
vllm serve Qwen/Qwen3.5-9B \
    --dtype bfloat16 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.90 \
    --port 8001 \
    --enable-prefix-caching \
    > "$VLLM_LOG" 2>&1 &

VLLM_PID=$!
trap "kill $VLLM_PID 2>/dev/null" EXIT

# --- Wait for vLLM to be ready (up to 20 min — Qwen3.5 needs time on first load) ---
for i in $(seq 1 240); do
    if curl -sf http://localhost:8001/v1/models > /dev/null 2>&1; then
        echo "vLLM ready after $((i * 5))s"
        break
    fi
    if [ $i -eq 240 ]; then
        echo "ERROR: vLLM did not start after 20 minutes. Last lines of vllm_startup.log:"
        tail -30 "$VLLM_LOG"
        exit 1
    fi
    echo "  attempt $i/240..."
    sleep 5
done

# --- Build argument list ---
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

EXTRA_ARGS=""
[ -n "$RESULTS_DIR" ] && EXTRA_ARGS="$EXTRA_ARGS --results-dir $RESULTS_DIR"
[ -n "$PHENO_CSV" ]   && EXTRA_ARGS="$EXTRA_ARGS --pheno-csv $PHENO_CSV"

python analyse_results/analyse_results.py $EXTRA_ARGS --workers "$WORKERS"
