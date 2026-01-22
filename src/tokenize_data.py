import torch
import numpy as np
import os
from pathlib import Path
from train_vqvae import VQVAE  # Import your model definition

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "episodes"
TOKEN_OUTPUT_PATH = PROJECT_ROOT / "data" / "tokens"
ACTION_OUTPUT_PATH = PROJECT_ROOT / "data" / "actions"
ARTIFACTS_PATH = PROJECT_ROOT / "data" / "artifacts"

TOKEN_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
ACTION_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

def main():
    print("--- STARTING DATA TOKENIZATION ---")
    
    # 1. Load VQ-VAE (The "Eyes")
    print("Loading VQ-VAE model...")
    vqvae = VQVAE().to(DEVICE)
    model_path = ARTIFACTS_PATH / "vqvae.pth"
    if not model_path.exists():
        print(f"Error: VQ-VAE model not found at {model_path}")
        return
    
    vqvae.load_state_dict(torch.load(model_path, map_location=DEVICE))
    vqvae.eval()

    # 2. Get Files
    files = sorted(list(DATA_PATH.glob("*.npz")))
    if not files:
        print("No data found in data/episodes/")
        return
    
    print(f"Found {len(files)} episodes to tokenize...")

    # 3. Process
    with torch.no_grad():
        for i, f in enumerate(files):
            try:
                # Load Episode
                with np.load(f) as data:
                    frames = data['frames'] # (T, 64, 64, 3)
                    actions = data['action'] # (T,)

                # Preprocess Frames
                # VQVAE expects (B, 3, 64, 64)
                frames_torch = torch.from_numpy(frames).permute(0, 3, 1, 2).float().to(DEVICE) / 255.0
                
                # Encode -> Get Indices
                # We use the encoder and pre_vq_conv
                z = vqvae.encoder(frames_torch)
                z = vqvae._pre_vq_conv(z)
                
                # Get indices from the quantization layer
                _, _, indices = vqvae.vq_layer(z)
                
                # Reshape indices to (T, 16, 16)
                # indices comes out as (B*H*W, 1) or similar depending on implementation
                # Based on your VQVAE code: encoding_indices is (B*H*W, 1)
                # We need to reshape it back to the grid
                indices = indices.view(len(frames), 16, 16).cpu().numpy().astype(np.uint16)

                # Save
                token_name = f.stem + "_tokens.npy"
                action_name = f.stem + "_actions.npy"
                
                np.save(TOKEN_OUTPUT_PATH / token_name, indices)
                np.save(ACTION_OUTPUT_PATH / action_name, actions)
                
                if i % 10 == 0:
                    print(f"Tokenized episode {i}/{len(files)} | Shape: {indices.shape}")

            except Exception as e:
                print(f"Failed to process {f.name}: {e}")

    print("--- TOKENIZATION COMPLETE ---")

if __name__ == "__main__":
    main()