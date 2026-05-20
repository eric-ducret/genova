#!/bin/bash --login
#SBATCH --job-name=variant_batch
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=eric.ducret@unil.ch

# --- Software environment ---
dcsrsoft use 20241118
module load miniforge3

# --- Conda initialization ---
# This sources the shell hook that makes `conda activate` work in scripts
conda_init

# Now safe to fail fast (after conda is set up)
set -e

# --- Activate your environment ---
JOBS_FOLDER="${1:?Usage: sbatch batch.sh <jobs_folder>}"
CONDA_ENV="/work/PRTNR/CHUV/MED/fsantoni1/pitnet/AI/JING/GenMasterAI/eric_folder/Benchmarking_200426/qwen_env"
conda activate "$CONDA_ENV"

python batch.py "$JOBS_FOLDER" --backend hf