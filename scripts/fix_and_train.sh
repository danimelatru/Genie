#!/bin/bash
#SBATCH --job-name=active_train
#SBATCH --output=logs/active_%j.out
#SBATCH --time=04:00:00
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --exclude=ruche-gpu11,ruche-gpu16,ruche-gpu17,ruche-gpu19

# --- ENVIRONMENT SETUP ---
module purge
PROJECT_ROOT="/gpfs/workdir/fernandeda/mini-genie"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
PYTHON_EXEC="/gpfs/workdir/fernandeda/conda_envs/mini-genie/bin/python"

echo "----------------------------------------------------------------"
echo "🔧 STARTING NEW ACTIVE TRAINING PIPELINE"
echo "----------------------------------------------------------------"

# STEP 1: CLEAN UP OLD CHECKPOINTS
echo "🧹 Step 1: Removing old Transformer checkpoints..."
rm -f data/artifacts/action_net_transformer.pth
rm -f data/artifacts/world_model_transformer.pth
# Note: We KEEP the VQ-VAE.
echo "   Done. Old brains removed."

# 0. VQ-VAE Diagnostic (Verify reconstruction before training)
echo "--- Step 0: Running VQ-VAE Diagnostic ---"
$PYTHON_EXEC -u src/diagnostic_vqvae.py

# 1. Tokenize Data (Uses the VQ-VAE to convert images to codebook indices)
echo "--- Step 1: Tokenizing Episodes ---"
$PYTHON_EXEC -u src/tokenize_data.py

# 2. Train CNN Dynamics (The "Brain" - now using CNN architecture)
echo "--- Step 2: Training CNN WorldModel (Window size = 4) ---"
$PYTHON_EXEC -u src/train_transformer_dynamics.py

# 3. Visualization (t-SNE and Dreams)
echo "--- Step 3: Generating Visualizations ---"
$PYTHON_EXEC -u src/visualize_tsne.py
$PYTHON_EXEC -u src/generate_dream_gif.py

echo "----------------------------------------------------------------"
echo "✅ PIPELINE FINISHED"
echo "----------------------------------------------------------------"