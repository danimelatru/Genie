import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import imageio
from pathlib import Path
import sys

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "tokens"
ARTIFACTS_PATH = PROJECT_ROOT / "data" / "artifacts"
DREAM_LENGTH = 50       
DREAM_ACTION_IDX = 0    
TEMPERATURE = 1.0       
TOP_K = 50              

def sample_top_k(logits, k, temperature):
    v, _ = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[..., [-1]]] = -float('Inf')
    probs = F.softmax(out / temperature, dim=-1)
    flat_probs = probs.view(-1, 512)
    sample = torch.multinomial(flat_probs, 1)
    return sample.view(logits.shape[:-1])

def generate_gif(vqvae, world_model):
    print(f"🎬 Starting Dream Generation (Forcing Action {DREAM_ACTION_IDX})...")
    
    vqvae.eval()
    world_model.eval()
    
    # 1. Load Seed
    token_files = sorted(list(DATA_PATH.glob("*.npy")))
    if not token_files:
        print("Error: No tokens found.")
        return

    try:
        start_tokens = np.load(token_files[0]) 
        # Ensure we have (1, 16, 16)
        z_t = torch.LongTensor(start_tokens[0]).unsqueeze(0).to(DEVICE) 
    except Exception as e:
        print(f"Error loading seed: {e}")
        return

    dream_tokens = [z_t]
    
    # 2. Dream Loop
    print(f"Dreaming {DREAM_LENGTH} frames...")
    with torch.no_grad():
        for i in range(DREAM_LENGTH):
            action = torch.zeros(1, 8).to(DEVICE)
            valid_action = min(DREAM_ACTION_IDX, 7) 
            action[0, valid_action] = 1.0 
            
            logits = world_model(z_t, action)
            z_next = sample_top_k(logits, TOP_K, TEMPERATURE)
            dream_tokens.append(z_next)
            z_t = z_next

    # 3. Decode Dream
    print("Decoding dream tokens to pixels using VQ-VAE...")
    dream_seq = torch.cat(dream_tokens, dim=0) 
    
    decoded_frames = []
    batch_size = 10
    
    with torch.no_grad():
        for i in range(0, len(dream_seq), batch_size):
            batch_indices = dream_seq[i : i + batch_size] 
            
            # --- FINAL FIX: USE _embedding (WITH UNDERSCORE) ---
            # Based on your train_vqvae.py line 74: self._embedding = ...
            z_q = vqvae.vq_layer._embedding(batch_indices) # (B, 16, 16, 64)
            
            z_q = z_q.permute(0, 3, 1, 2)
            recon = vqvae.decoder(z_q)
            
            recon = recon.permute(0, 2, 3, 1).cpu().numpy()
            recon = (np.clip(recon, 0, 1) * 255).astype(np.uint8)
            decoded_frames.extend([frame for frame in recon])

    # 4. Save GIF
    save_path = ARTIFACTS_PATH / "dream_real_data.gif"
    imageio.mimsave(save_path, decoded_frames, fps=10)
    print(f"✅ GIF saved to {save_path}")

if __name__ == "__main__":
    try:
        from train_vqvae import VQVAE
        from train_transformer_dynamics import WorldModelTransformer
        
        vqvae = VQVAE().to(DEVICE)
        vqvae.load_state_dict(torch.load(ARTIFACTS_PATH / "vqvae.pth", map_location=DEVICE))
        
        wm = WorldModelTransformer().to(DEVICE)
        wm.load_state_dict(torch.load(ARTIFACTS_PATH / "world_model_transformer.pth", map_location=DEVICE))
        
        generate_gif(vqvae, wm)
    except Exception as e:
        print(f"Test failed: {e}")