#!/bin/bash
#SBATCH --job-name=variant-pipeline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00

# Start vLLM server
vllm serve Qwen/Qwen3.5-9B \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --port 8001 &

# Wait for vLLM to be ready
sleep 60

# Start FastAPI server
uvicorn server:app --host 0.0.0.0 --port 8000
