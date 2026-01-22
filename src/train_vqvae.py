import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import matplotlib.pyplot as plt
import lpips  
import sys

# --- CONFIG ---
BATCH_SIZE = 64
LEARNING_RATE = 1e-3  
EPOCHS = 20
EMBED_DIM = 64
NUM_EMBEDDINGS = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- PATHS ---
current_path = Path(__file__).parent.resolve()
root_path = current_path.parent
DATA_PATH = root_path / "data" / "episodes"
ARTIFACTS_PATH = root_path / "data" / "artifacts"
ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)

# --- DATASET ---
class LowRAMEpisodesDataset(Dataset):
    def __init__(self, data_dir):
        self.files = sorted(list(Path(data_dir).glob("*.npz")))
        print(f"Found {len(self.files)} episodes.")
        
        self.episodes = [] 
        self.cumulative_sizes = []
        total_frames = 0
        
        limit = 2000 
        for i, f in enumerate(self.files[:limit]):
            try:
                with np.load(f) as d:
                    frames = d["frames"] # (T, 64, 64, 3)
                    # Transpose -> (T, 3, 64, 64) & Normalize
                    frames = np.transpose(frames, (0, 3, 1, 2)).astype(np.float32) / 255.0
                    self.episodes.append(frames)
                    total_frames += len(frames)
                    self.cumulative_sizes.append(total_frames)
            except Exception as e:
                print(f"Error loading {f}: {e}")
        
        self.cumulative_sizes = np.array(self.cumulative_sizes)
        print(f"Loaded {total_frames} frames. RAM Usage: ~{total_frames * 3 * 64 * 64 * 4 / 1e9:.2f} GB")

    def __len__(self):
        return self.cumulative_sizes[-1] if len(self.cumulative_sizes) > 0 else 0

    def __getitem__(self, idx):
        ep_idx = np.searchsorted(self.cumulative_sizes, idx, side='right')
        if ep_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_sizes[ep_idx - 1]
        return torch.from_numpy(self.episodes[ep_idx][local_idx])

# --- MODEL COMPONENTS ---
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim)
        
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
        
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), encoding_indices

class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()
        self._conv_1 = nn.Conv2d(in_channels, num_hiddens//2, 4, 2, 1)
        self._conv_2 = nn.Conv2d(num_hiddens//2, num_hiddens, 4, 2, 1)
        self._conv_3 = nn.Conv2d(num_hiddens, num_hiddens, 3, 1, 1)
        self._residual_stack = nn.Sequential(*[
            nn.Sequential(
                nn.ReLU(True),
                nn.Conv2d(num_hiddens, num_residual_hiddens, 3, 1, 1, bias=False),
                nn.ReLU(True),
                nn.Conv2d(num_residual_hiddens, num_hiddens, 1, 1, bias=False)
            ) for _ in range(num_residual_layers)
        ])
        
    def forward(self, x):
        x = self._conv_1(x)
        x = F.relu(x)
        x = self._conv_2(x)
        x = F.relu(x)
        x = self._conv_3(x)
        return self._residual_stack(x) + x # Skip connection simplified

class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()
        self._conv_1 = nn.Conv2d(in_channels, num_hiddens, 3, 1, 1)
        self._residual_stack = nn.Sequential(*[
            nn.Sequential(
                nn.ReLU(True),
                nn.Conv2d(num_hiddens, num_residual_hiddens, 3, 1, 1, bias=False),
                nn.ReLU(True),
                nn.Conv2d(num_residual_hiddens, num_hiddens, 1, 1, bias=False)
            ) for _ in range(num_residual_layers)
        ])
        self._conv_trans_1 = nn.ConvTranspose2d(num_hiddens, num_hiddens//2, 4, 2, 1)
        self._conv_trans_2 = nn.ConvTranspose2d(num_hiddens//2, 3, 4, 2, 1)

    def forward(self, x):
        x = self._conv_1(x)
        x = self._residual_stack(x) + x
        x = F.relu(self._conv_trans_1(x))
        return torch.sigmoid(self._conv_trans_2(x))

class VQVAE(nn.Module):
    def __init__(self):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(3, 128, 2, 32)
        self._pre_vq_conv = nn.Conv2d(128, EMBED_DIM, 1, 1)
        self.vq_layer = VectorQuantizer(NUM_EMBEDDINGS, EMBED_DIM)
        self.decoder = Decoder(EMBED_DIM, 128, 2, 32)

    def forward(self, x):
        z = self.encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity = self.vq_layer(z) 
        return loss, quantized, perplexity 

# --- TRAINING ---
def main():
    print(f"Training VQ-VAE (LPIPS + Spatial Loss) on {DEVICE}")
    
    dataset = LowRAMEpisodesDataset(DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False)
    
    model = VQVAE().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Loading LPIPS...")
    lpips_fn = lpips.LPIPS(net='vgg').to(DEVICE) 
    
    print("Starting Training...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for i, images in enumerate(dataloader):
            images = images.to(DEVICE)
            optimizer.zero_grad()
            
            vq_loss, quantized, _ = model(images)[0:3]
            recon_images = model.decoder(quantized)
            
            weights = torch.ones_like(images)
            weights[images > 0.05] = 10.0
            
            recon_loss = (F.l1_loss(recon_images, images, reduction='none') * weights).mean()
            p_loss = lpips_fn(recon_images * 2 - 1, images * 2 - 1).mean()
            
            loss = recon_loss + vq_loss + 0.5 * p_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(dataloader):.4f}")
        
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), ARTIFACTS_PATH / "vqvae.pth")

    torch.save(model.state_dict(), ARTIFACTS_PATH / "vqvae.pth")
    print("Training Complete.")

if __name__ == "__main__":
    main()