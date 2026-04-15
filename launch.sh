#!/bin/bash --login
#SBATCH --job-name=variant_pipeline

#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Start vLLM server
vllm serve Qwen/Qwen3.5-9B \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --port 8001 &

# Wait for vLLM to be ready
sleep 60

# Start FastAPI server
uvicorn server:app --host 0.0.0.0 --port 8000
