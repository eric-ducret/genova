#!/bin/bash --login
#SBATCH --job-name=analyse_results
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=eric.ducret@unil.ch

# Usage: sbatch analyse_results/analyse_results.sh [results_dir] [workers] [ground_truth_csv]
#   results_dir       — path to results folder (default: <project_root>/results)
#   workers           — parallel SLM workers   (default: 8)
#   ground_truth_csv  — CSV with patient_id,expert_gene columns (optional)
RESULTS_DIR="${1:-}"
WORKERS="${2:-8}"
GROUND_TRUTH="${3:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
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

# --- Wait for vLLM to be ready ---
for i in $(seq 1 60); do
    if curl -sf http://localhost:8001/v1/models > /dev/null 2>&1; then
        echo "vLLM ready after $((i * 5))s"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "ERROR: vLLM did not start after 10 minutes. vllm_startup.log:"
        cat "$VLLM_LOG"
        exit 1
    fi
    echo "  attempt $i/60..."
    sleep 5
done

# --- Build argument list ---
cd "$PROJECT_DIR"

EXTRA_ARGS=""
[ -n "$RESULTS_DIR" ]   && EXTRA_ARGS="$EXTRA_ARGS --results-dir $RESULTS_DIR"
[ -n "$GROUND_TRUTH" ]  && EXTRA_ARGS="$EXTRA_ARGS --ground-truth $GROUND_TRUTH"

python analyse_results/analyse_results.py $EXTRA_ARGS --workers "$WORKERS"
