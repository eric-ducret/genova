#!/bin/bash --login
#SBATCH --job-name=variant_batch
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=eric.ducret@unil.ch
#SBATCH --output=/work/PRTNR/CHUV/MED/fsantoni1/pitnet/AI/JING/GenMasterAI/eric_folder/genova_vllm_556_qwen_same_pheno_0811/slurm-%j.out

# Usage: sbatch batch_pheno_vllm.sh <batch_number> <jobs_folder> [n_batches]
BATCH_NUMBER="${1:?Usage: sbatch batch_pheno_vllm.sh <batch_number> <jobs_folder> [n_batches]}"
JOBS_FOLDER="${2:?Usage: sbatch batch_pheno_vllm.sh <batch_number> <jobs_folder> [n_batches]}"
N_BATCHES="${3:-10}"

# Model selection — override at submission time:
#   VARIANT_MODEL=gemma4-e4b sbatch batch_pheno_vllm.sh ...
VARIANT_MODEL="${VARIANT_MODEL:-qwen3.5-9b}"
VARIANT_MAX_MODEL_LEN="${VARIANT_MAX_MODEL_LEN:-16384}"
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

echo "Starting batch $BATCH_NUMBER/$N_BATCHES from folder: $JOBS_FOLDER"

# Resolve HF model ID from registry (single source of truth)
VARIANT_MODEL_HF_ID=$("$CONDA_ENV/bin/python" -c "from pipeline.llm.registry import MODELS; print(MODELS['$VARIANT_MODEL']['model_id'])")

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

# --- Wait for vLLM to be ready ---
for i in $(seq 1 120); do
    if curl -sf http://localhost:8001/v1/models > /dev/null 2>&1; then
        echo "vLLM ready after $((i * 5))s"
        break
    fi
    if [ $i -eq 120 ]; then
        echo "ERROR: vLLM did not start after 10 minutes. vllm_startup.log:"
        cat "$VLLM_LOG"
        exit 1
    fi
    echo "  attempt $i/120..."
    sleep 5
done

# --- Run batch pipeline ---
VARIANT_MODEL="$VARIANT_MODEL" "$CONDA_ENV/bin/python" batch_pheno.py "$BATCH_NUMBER" "$JOBS_FOLDER" --backend vllm --n-batches "$N_BATCHES"
