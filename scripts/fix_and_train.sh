#!/bin/bash
#SBATCH --job-name=active_train
#SBATCH --output=logs/active_%j.out
#SBATCH --time=04:00:00
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1

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
# Note: We KEEP the VQ-VAE (eyes) because it works fine.
echo "   Done. Old brains removed."

# STEP 2: GENERATE NEW DATA (TOKENS)
 echo "🔄 Step 2: Tokenizing NEW active data..."
# Added -u to see progress in real-time
$PYTHON_EXEC -u src/tokenize_data.py

# STEP 3: TRAIN WITH ENTROPY REGULARIZATION
echo "🚀 Step 3: Training Transformer Dynamics (Entropy Regularization)..."
$PYTHON_EXEC -u src/train_transformer_dynamics.py

# STEP 4: VISUALIZE RESULTS
echo "🎨 Step 4: Generating final visualizations (t-SNE & GIF)..."
$PYTHON_EXEC -u src/visualize_tsne.py

echo "----------------------------------------------------------------"
echo "✅ PIPELINE FINISHED"
echo "----------------------------------------------------------------"