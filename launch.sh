#!/bin/bash --login
#SBATCH --job-name=variant_pipeline

#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Model selection — override at submission time:
#   VARIANT_MODEL=gemma4-e4b VARIANT_MODEL_HF_ID=google/gemma-4-E4B sbatch launch.sh
VARIANT_MODEL="${VARIANT_MODEL:-qwen3.5-9b}"
VARIANT_MODEL_HF_ID="${VARIANT_MODEL_HF_ID:-Qwen/Qwen3.5-9B}"
VARIANT_MAX_MODEL_LEN="${VARIANT_MAX_MODEL_LEN:-32768}"

# Start vLLM server
vllm serve "$VARIANT_MODEL_HF_ID" \
    --dtype bfloat16 \
    --max-model-len "$VARIANT_MAX_MODEL_LEN" \
    --port 8001 &

# Wait for vLLM to be ready
sleep 60

# Start FastAPI server
VARIANT_MODEL="$VARIANT_MODEL" uvicorn server:app --host 0.0.0.0 --port 8000
