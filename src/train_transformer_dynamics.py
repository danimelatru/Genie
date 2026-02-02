import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from collections import Counter

# --- IMPORTS ---
try:
    from train_vqvae import VQVAE
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent))
    from train_vqvae import VQVAE

# --- CONFIG ---
BATCH_SIZE = 128  # Increased for A100 efficiency
LEARNING_RATE = 1e-4  # Aligned with Genie paper recommendation
EPOCHS = 5  # Reduced from 20 - model converges by epoch 2-3
WINDOW_SIZE = 4
VOCAB_SIZE = 512
NUM_ACTIONS = 8
ENTROPY_WEIGHT = 0.1
EMBED_DIM = 64
HIDDEN_DIM = 256  # Reduced from 512 for faster training
NUM_HEADS = 8
NUM_LAYERS = 2  # Reduced from 6 for speed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "tokens"
ARTIFACTS_PATH = PROJECT_ROOT / "data" / "artifacts"
ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)

# --- DATASET ---
class TokenTransitionsDataset(Dataset):
    def __init__(self, token_dir, window_size=WINDOW_SIZE, limit=None):
        self.files = sorted(list(Path(token_dir).glob("*.npy")))
        if limit: self.files = self.files[:limit]
        self.window_size = window_size
        self.data = []
        
        print(f"Loading {len(self.files)} episodes for windowed training (window={window_size})...")
        
        for idx, f in enumerate(self.files):
            if idx % 100 == 0 and idx > 0:
                print(f"  Processed {idx}/{len(self.files)} episodes, {len(self.data)} sequences so far...")
            try:
                tokens = np.load(f) # (T, 16, 16)
                if len(tokens.shape) != 3: continue

                action_file = str(f).replace("tokens", "actions")
                if not os.path.exists(action_file):
                     action_file = str(f).replace("_tokens.npy", "_actions.npy")
                
                if not os.path.exists(action_file): continue
                actions = np.load(action_file)

                limit_len = min(len(tokens), len(actions))
                
                # Create windowed sequences
                for i in range(limit_len - window_size):
                    seq_tokens = tokens[i:i+window_size] # (W, 16, 16)
                    seq_actions = actions[i:i+window_size] # (W,)
                    target_token = tokens[i+window_size] # (16, 16)
                    
                    self.data.append((seq_tokens, target_token, seq_actions))
            except Exception as e: 
                print(f"Error loading {f}: {e}")
                continue
        
        print(f"✅ Total Sequences: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq_tokens, target_token, seq_actions = self.data[idx]
        return torch.LongTensor(seq_tokens), torch.LongTensor(target_token), torch.LongTensor(seq_actions)

# --- MODELS ---
class ActionRecognitionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        # Process the last frame and the next frame to infer action
        self.conv_net = nn.Sequential(
            nn.Conv2d(EMBED_DIM * 2, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, NUM_ACTIONS)
        )

    def forward(self, z_prev, z_next):
        # z_prev is the last frame of the context window
        emb_prev = self.embedding(z_prev).permute(0, 3, 1, 2)
        emb_next = self.embedding(z_next).permute(0, 3, 1, 2) 
        x = torch.cat([emb_prev, emb_next], dim=1)
        return self.head(self.conv_net(x))

class WorldModelCNN(nn.Module):
    """CNN-based World Model that naturally preserves spatial structure.
    
    Uses 2D convolutions which inherently maintain local spatial coherence,
    instead of treating each position independently like the Transformer did.
    """
    def __init__(self):
        super().__init__()
        
        # Token embedding: convert discrete tokens to dense vectors
        self.token_emb = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        
        # Action embedding: project action to spatial feature map
        self.action_fc = nn.Linear(NUM_ACTIONS, 16 * 16)
        
        # Input channels: W frames * EMBED_DIM + 1 action channel
        input_channels = WINDOW_SIZE * EMBED_DIM + 1
        
        # CNN encoder with residual connections
        self.conv1 = nn.Conv2d(input_channels, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        # Residual projection (for skip connection)
        self.residual_proj = nn.Conv2d(input_channels, 256, kernel_size=1)
        
        # Output head: predict token logits for each position
        self.head = nn.Conv2d(256, VOCAB_SIZE, kernel_size=1)
        
        self.activation = nn.GELU()
        self.norm1 = nn.BatchNorm2d(128)
        self.norm2 = nn.BatchNorm2d(256)
        self.norm3 = nn.BatchNorm2d(256)

    def forward(self, z_seq, action_probs):
        """
        Args:
            z_seq: (B, W, 16, 16) - window of token indices
            action_probs: (B, 8) - action probabilities (one-hot or soft)
        Returns:
            logits: (B, 16, 16, VOCAB_SIZE) - next frame token predictions
        """
        B, W, H, W_grid = z_seq.shape
        
        # 1. Embed tokens: (B, W, 16, 16) -> (B, W, 16, 16, EMBED_DIM)
        z_emb = self.token_emb(z_seq)
        
        # 2. Reshape for convolution: (B, W*EMBED_DIM, 16, 16)
        z_emb = z_emb.permute(0, 1, 4, 2, 3)  # (B, W, EMBED_DIM, 16, 16)
        z_emb = z_emb.reshape(B, W * EMBED_DIM, H, W_grid)
        
        # 3. Create action feature map: (B, 1, 16, 16)
        act_map = self.action_fc(action_probs).view(B, 1, 16, 16)
        
        # 4. Concatenate: (B, W*EMBED_DIM + 1, 16, 16)
        x = torch.cat([z_emb, act_map], dim=1)
        
        # 5. Save input for residual connection
        residual = self.residual_proj(x)
        
        # 6. Process through CNN layers with residual connections
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.activation(self.norm2(self.conv2(x)))
        x = self.activation(self.norm3(self.conv3(x)))
        x = self.conv4(x)
        
        # 7. Add residual connection
        x = x + residual
        x = self.activation(x)
        
        # 8. Output logits: (B, VOCAB_SIZE, 16, 16)
        logits = self.head(x)
        
        # 9. Reshape to match expected output: (B, 16, 16, VOCAB_SIZE)
        return logits.permute(0, 2, 3, 1)

# --- MAIN ---
def main():
    print(f"Training Transformer Dynamics (Window={WINDOW_SIZE}) on {DEVICE}")
    print(f"Entropy Lambda: {ENTROPY_WEIGHT} | Learning Rate: {LEARNING_RATE}")
    
    dataset = TokenTransitionsDataset(DATA_PATH, window_size=WINDOW_SIZE, limit=1000)
    if len(dataset) == 0:
        print("❌ No data found!")
        return
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Loaded {len(dataset)} sequences, {len(dataloader)} batches per epoch.")
    
    action_net = ActionRecognitionNet().to(DEVICE)
    world_model = WorldModelCNN().to(DEVICE)
    optimizer = optim.AdamW(list(action_net.parameters()) + list(world_model.parameters()), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    print("Starting Training...")
    for epoch in range(EPOCHS):
        total_loss = 0
        total_acc = 0
        total_ent = 0
        
        for z_seq, z_next, real_act in dataloader:
            z_seq, z_next, real_act = z_seq.to(DEVICE), z_next.to(DEVICE), real_act.to(DEVICE)
            optimizer.zero_grad()
            
            # Predict action from the transition between last frame in window and the next frame
            z_last = z_seq[:, -1, :, :]
            action_logits = action_net(z_last, z_next)
            action_probs = torch.softmax(action_logits, dim=1)
            
            # Supervised action loss: predict the real action that caused the transition
            target_act = real_act[:, -1]  # The action that led to z_next
            loss_action = criterion(action_logits, target_act)
            
            # Predict next frame from sequence + action
            pred_logits = world_model(z_seq, action_probs)
            
            loss_recon = criterion(pred_logits.reshape(-1, VOCAB_SIZE), z_next.reshape(-1))
            
            # Entropy calculation for latent actions
            log_probs = torch.log_softmax(action_logits, dim=1)
            entropy = -(action_probs * log_probs).sum(dim=1).mean()
            
            # Total Loss: reconstruction + action prediction - entropy regularization
            loss = loss_recon + loss_action - (ENTROPY_WEIGHT * entropy)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_ent += entropy.item()
            
            # Accuracy: how well we predict the real action
            total_acc += (torch.argmax(action_probs, dim=1) == target_act).float().mean().item()
            
        avg_ent = total_ent/len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(dataloader):.4f} (Ent: {avg_ent:.4f}) | Acc: {total_acc/len(dataloader)*100:.1f}%")

    torch.save(action_net.state_dict(), ARTIFACTS_PATH / "action_net_transformer.pth")
    torch.save(world_model.state_dict(), ARTIFACTS_PATH / "world_model_transformer.pth")
    
    print("Generating Viz...")
    action_net.eval()
    all_preds, all_real = [], []
    with torch.no_grad():
        for z_seq, z_next, real_act in dataloader:
            z_seq, z_next = z_seq.to(DEVICE), z_next.to(DEVICE)
            z_last = z_seq[:, -1, :, :]
            all_preds.extend(torch.argmax(action_net(z_last, z_next), dim=1).cpu().numpy())
            all_real.extend(real_act[:, -1].cpu().numpy())
            
    cm = confusion_matrix(all_real, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('Ground Truth Action')
    plt.xlabel('Discovered Cluster')
    plt.title(f'Latent Action Discovery (Entropy w={ENTROPY_WEIGHT})')
    plt.savefig(ARTIFACTS_PATH / "transformer_confusion_matrix.png")
    
    try:
        import generate_dream_gif
        generate_dream_gif.generate_gif(VQVAE().to(DEVICE), world_model, "dream_during_training.gif")
    except: pass

if __name__ == "__main__":
    main()