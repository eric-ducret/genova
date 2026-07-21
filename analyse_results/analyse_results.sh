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
RESULTS_DIR="${1:-/work/PRTNR/CHUV/MED/fsantoni1/pitnet/AI/JING/GenMasterAI/eric_folder/genova_vllm_556_gemma4_0626/results}"
WORKERS="${2:-8}"
PHENO_CSV="${3:-/work/PRTNR/CHUV/MED/fsantoni1/pitnet/AI/JING/GenMasterAI/eric_folder/genova_vllm_556_gemma4_0626/pheno_prompts.csv}"

# Model selection — override at submission time:
#   VARIANT_MODEL=gemma4-e4b VARIANT_MODEL_HF_ID=google/gemma-4-E4B sbatch analyse_results/analyse_results.sh ...
VARIANT_MODEL="${VARIANT_MODEL:-qwen3.5-9b}"
VARIANT_MODEL_HF_ID="${VARIANT_MODEL_HF_ID:-Qwen/Qwen3.5-9B}"
VARIANT_MAX_MODEL_LEN="${VARIANT_MAX_MODEL_LEN:-16384}"

# Chat template override (rarely needed — the registered -it models ship their
# own chat template; only set this if a model truly lacks one).
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VARIANT_CHAT_TEMPLATE="${VARIANT_CHAT_TEMPLATE:-}"

CONDA_ENV="/work/PRTNR/CHUV/MED/fsantoni1/pitnet/AI/JING/GenMasterAI/eric_folder/env_vllm_0606"
VLLM_LOG="/work/PRTNR/CHUV/MED/fsantoni1/pitnet/AI/JING/GenMasterAI/eric_folder/vllm_startup.log"

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
echo "Starting vLLM server (model: $VARIANT_MODEL_HF_ID, max-model-len: $VARIANT_MAX_MODEL_LEN)..."
VLLM_CMD=(vllm serve "$VARIANT_MODEL_HF_ID"
    --dtype bfloat16
    --max-model-len "$VARIANT_MAX_MODEL_LEN"
    --gpu-memory-utilization 0.90
    --port 8001
    --enable-prefix-caching)
[ -n "$VARIANT_CHAT_TEMPLATE" ] && VLLM_CMD+=(--chat-template "$VARIANT_CHAT_TEMPLATE") && \
    echo "Using chat template: $VARIANT_CHAT_TEMPLATE"
"${VLLM_CMD[@]}" > "$VLLM_LOG" 2>&1 &

VLLM_PID=$!
trap "kill $VLLM_PID 2>/dev/null" EXIT

# --- Wait for vLLM to be ready (up to 20 min) ---
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

# --- Run analysis ---
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

VARIANT_MODEL="$VARIANT_MODEL" "$CONDA_ENV/bin/python" analyse_results/analyse_results.py \
    --results-dir "$RESULTS_DIR" \
    --pheno-csv   "$PHENO_CSV" \
    --workers     "$WORKERS" \
    --model       "$VARIANT_MODEL"
