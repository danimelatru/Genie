"""
VQ-VAE Diagnostic Script
=========================
Generates a GIF from REAL episode frames -> tokenization -> reconstruction
to verify the VQ-VAE is working correctly (bypassing the WorldModel).
"""

import torch
import numpy as np
import imageio
from pathlib import Path

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = Path(__file__).parent.parent
EPISODES_PATH = PROJECT_ROOT / "data" / "episodes"
ARTIFACTS_PATH = PROJECT_ROOT / "data" / "artifacts"
NUM_FRAMES = 50  # How many frames to reconstruct

def main():
    print("🔍 VQ-VAE Diagnostic: Testing reconstruction quality...")
    
    # 1. Load VQ-VAE
    try:
        from train_vqvae import VQVAE
    except ImportError:
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from train_vqvae import VQVAE
    
    vqvae = VQVAE().to(DEVICE)
    vqvae_path = ARTIFACTS_PATH / "vqvae.pth"
    
    if not vqvae_path.exists():
        print(f"❌ VQ-VAE model not found at {vqvae_path}")
        return
    
    vqvae.load_state_dict(torch.load(vqvae_path, map_location=DEVICE))
    vqvae.eval()
    print(f"✅ Loaded VQ-VAE from {vqvae_path}")
    
    # 2. Load a real episode
    episode_files = sorted(list(EPISODES_PATH.glob("*.npz")))
    if not episode_files:
        print("❌ No episodes found!")
        return
    
    # Find an episode with enough frames
    for ep_file in episode_files:
        with np.load(ep_file) as data:
            frames = data['frames']
            if len(frames) >= NUM_FRAMES:
                break
    else:
        print(f"❌ No episode has {NUM_FRAMES} frames!")
        return
    
    frames = frames[:NUM_FRAMES]  # (N, 64, 64, 3)
    print(f"✅ Loaded {len(frames)} frames from {ep_file.name}")
    
    # 3. Process: Original -> Encode -> Quantize -> Decode -> Reconstruct
    original_frames = []
    reconstructed_frames = []
    
    with torch.no_grad():
        for i, frame in enumerate(frames):
            # Original frame
            original_frames.append(frame.astype(np.uint8))
            
            # Convert to tensor: (1, 3, 64, 64)
            frame_torch = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0
            
            # Encode
            z = vqvae.encoder(frame_torch)
            z = vqvae._pre_vq_conv(z)
            
            # Quantize (get indices and then look up embeddings)
            _, quantized, perplexity, indices = vqvae.vq_layer(z)
            
            # Decode
            recon = vqvae.decoder(quantized)
            
            # Convert back to image
            recon_np = (recon.squeeze().permute(1, 2, 0).cpu().numpy() * 255)
            recon_np = np.clip(recon_np, 0, 255).astype(np.uint8)
            reconstructed_frames.append(recon_np)
            
            if i == 0:
                print(f"   Frame 0: Perplexity = {perplexity.item():.1f}")
    
    # 4. Create side-by-side comparison GIF
    comparison_frames = []
    for orig, recon in zip(original_frames, reconstructed_frames):
        # Stack horizontally: Original | Reconstructed
        comparison = np.concatenate([orig, recon], axis=1)
        comparison_frames.append(comparison)
    
    # 5. Save comparison GIF only
    comparison_path = ARTIFACTS_PATH / "vqvae_diagnostic_comparison.gif"
    imageio.mimsave(comparison_path, comparison_frames, fps=10)
    print(f"✅ Comparison GIF saved to {comparison_path}")
    print("   (Left = Original, Right = Reconstructed)")
    
    # 6. Calculate reconstruction quality (MSE)
    orig_array = np.array(original_frames, dtype=np.float32)
    recon_array = np.array(reconstructed_frames, dtype=np.float32)
    mse = np.mean((orig_array - recon_array) ** 2)
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
    
    print(f"\n📊 Reconstruction Quality:")
    print(f"   MSE: {mse:.2f}")
    print(f"   PSNR: {psnr:.2f} dB")
    
    if psnr > 25:
        print("   ✅ VQ-VAE reconstruction is GOOD (PSNR > 25 dB)")
    elif psnr > 20:
        print("   ⚠️ VQ-VAE reconstruction is ACCEPTABLE (PSNR 20-25 dB)")
    else:
        print("   ❌ VQ-VAE reconstruction is POOR (PSNR < 20 dB)")
        print("   → Consider retraining VQ-VAE with more epochs or adjusting hyperparameters")

if __name__ == "__main__":
    main()
