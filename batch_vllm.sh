#!/bin/bash --login
#SBATCH --job-name=variant_batch
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=eric.ducret@unil.ch

# --- Software environment ---
dcsrsoft use 20241118
module load miniforge3
module load cuda
conda_init

set -e

JOBS_FOLDER="${1:?Usage: sbatch batch.sh <jobs_folder>}"
CONDA_ENV="/work/PRTNR/CHUV/MED/fsantoni1/pitnet/AI/JING/GenMasterAI/eric_folder/env_vllm_0606"
VLLM_LOG="/work/PRTNR/CHUV/MED/fsantoni1/pitnet/AI/JING/GenMasterAI/eric_folder/vllm_startup.log"

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
    --port 8001 \
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

# --- Run batch pipeline ---
python batch.py "$JOBS_FOLDER" --backend vllm