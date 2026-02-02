import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import sys
import os

# --- IMPORTS ---
try:
    from train_vqvae import VQVAE
    from train_transformer_dynamics import WorldModelCNN, ActionRecognitionNet, TokenTransitionsDataset
except ImportError:
    sys.path.append(str(Path(__file__).parent))
    from train_vqvae import VQVAE
    from train_transformer_dynamics import WorldModelCNN, ActionRecognitionNet, TokenTransitionsDataset

# --- CONFIG ---
BATCH_SIZE = 64
WINDOW_SIZE = 4
VOCAB_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "tokens"
ARTIFACTS_PATH = PROJECT_ROOT / "data" / "artifacts"

def main():
    print("🎨 Generating t-SNE visualization of latent action space...")
    
    # 1. Load Data
    print("Loading Dataset...")
    dataset = TokenTransitionsDataset(DATA_PATH, window_size=WINDOW_SIZE, limit=5000)
    if len(dataset) == 0:
        print("No data found.")
        return
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Load Action Recognition Model
    print("Loading Action Recognition Model...")
    action_net = ActionRecognitionNet().to(DEVICE)
    try:
        action_net.load_state_dict(torch.load(ARTIFACTS_PATH / "action_net_transformer.pth", map_location=DEVICE))
    except FileNotFoundError:
        print("❌ Model not found. Please train the model first.")
        return
    
    action_net.eval()
    
    print(f"Extracting latents for {len(dataset)} sequences...")
    
    all_latents = []
    all_actions = []
    
    with torch.no_grad():
        for z_seq, z_next, real_act in dataloader:
            z_seq, z_next = z_seq.to(DEVICE), z_next.to(DEVICE)
            
            # Use the last frame of the window + the target frame
            z_last = z_seq[:, -1, :, :]
            logits = action_net(z_last, z_next)
            
            all_latents.append(logits.cpu().numpy())
            all_actions.append(real_act[:, -1].cpu().numpy())
            
            # Limit to 2000 samples for faster t-SNE
            if len(all_latents) * BATCH_SIZE > 2000:
                break
    
    latents = np.concatenate(all_latents, axis=0)
    real_labels = np.concatenate(all_actions, axis=0)
    pred_labels = np.argmax(latents, axis=1)
    
    # 4. Run t-SNE
    print("Running t-SNE (Calculating 2D map)...")
    perp = min(30, len(latents) - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
    X_embedded = tsne.fit_transform(latents)
    
    # 5. Plot
    print(f"Plotting to {ARTIFACTS_PATH}...")
    
    # Create DataFrame for Seaborn
    df = pd.DataFrame({
        "x": X_embedded[:, 0],
        "y": X_embedded[:, 1],
        "Cluster": pred_labels,
        "Real Action": real_labels 
    })
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df, x="x", y="y", 
        hue="Cluster", 
        palette="tab10",
        s=60, alpha=0.7
    )
    plt.title("Mini-Genie Brain Map: Discovered Action Clusters")
    save_path = ARTIFACTS_PATH / "action_latent_space_tsne.png"
    plt.savefig(save_path)
    print(f"✅ t-SNE saved to {save_path}")

if __name__ == "__main__":
    main()