#!/bin/bash
#SBATCH --job-name=retrain_vision
#SBATCH --output=logs/vision_%j.out
#SBATCH --time=02:00:00
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --exclude=ruche-gpu11,ruche-gpu16,ruche-gpu17,ruche-gpu19

module purge
PROJECT_ROOT="/gpfs/workdir/fernandeda/mini-genie"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
PYTHON_EXEC="/gpfs/workdir/fernandeda/conda_envs/mini-genie/bin/python"

echo "RETRAINING VQ-VAE (The Eyes)"
$PYTHON_EXEC -u src/train_vqvae.py