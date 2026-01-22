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
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 10
HIDDEN_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 4
VOCAB_SIZE = 512
NUM_ACTIONS = 8
ENTROPY_WEIGHT = 0.001
EMBED_DIM = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "tokens"
ARTIFACTS_PATH = PROJECT_ROOT / "data" / "artifacts"
ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)

# --- DATASET ---
class TokenTransitionsDataset(Dataset):
    def __init__(self, token_dir, limit=None):
        self.files = sorted(list(Path(token_dir).glob("*.npy")))
        if limit: self.files = self.files[:limit]
        
        self.data = []
        all_actions_debug = []
        
        print(f"Loading {len(self.files)} episodes...")
        
        for f in self.files:
            try:
                tokens = np.load(f) 
                if len(tokens.shape) != 3: continue

                action_file = str(f).replace("tokens", "actions")
                if not os.path.exists(action_file):
                     action_file = str(f).replace("_tokens.npy", "_actions.npy")
                
                if not os.path.exists(action_file): continue
                actions = np.load(action_file)

                # Debug
                all_actions_debug.extend(actions.tolist())

                limit_len = min(len(tokens) - 1, len(actions))
                
                for i in range(limit_len):
                    curr = tokens[i]
                    nxt = tokens[i+1]
                    if np.array_equal(curr, nxt): continue 
                    self.data.append((curr, nxt, actions[i]))
            except: continue
        
        # Actions debug
        print(f"--- ACTION DISTRIBUTION REPORT ---")
        counts = Counter(all_actions_debug)
        print(f"Unique Actions found in files: {sorted(counts.keys())}")
        print(f"Counts: {dict(counts)}")
        print(f"Total Transitions: {len(self.data)}") 
        print(f"----------------------------------")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        curr, nxt, act = self.data[idx]
        return torch.LongTensor(curr), torch.LongTensor(nxt), torch.LongTensor([act])

# --- MODELS ---
class ActionRecognitionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.conv_net = nn.Sequential(
            nn.Conv2d(EMBED_DIM * 2, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, NUM_ACTIONS)
        )

    def forward(self, z_t, z_next):
        emb_t = self.embedding(z_t).permute(0, 3, 1, 2)
        emb_next = self.embedding(z_next).permute(0, 3, 1, 2) 
        x = torch.cat([emb_t, emb_next], dim=1)
        return self.head(self.conv_net(x))

class WorldModelTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.action_emb = nn.Linear(NUM_ACTIONS, HIDDEN_DIM)
        self.pos_emb = nn.Parameter(torch.randn(1, 16*16, HIDDEN_DIM))
        encoder_layer = nn.TransformerEncoderLayer(d_model=HIDDEN_DIM, nhead=NUM_HEADS, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        self.head = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, z_t, action_probs):
        B = z_t.shape[0]
        x = self.embedding(z_t.view(B, -1)) + self.pos_emb
        act_v = self.action_emb(action_probs).unsqueeze(1)
        x = x + act_v 
        out = self.transformer(x)
        return self.head(out).view(B, 16, 16, VOCAB_SIZE)

# --- MAIN ---
def main():
    print(f"Step 2: Training (Entropy Lambda={ENTROPY_WEIGHT})...")
    dataset = TokenTransitionsDataset(DATA_PATH, limit=5000)
    if len(dataset) == 0: return
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    action_net = ActionRecognitionNet().to(DEVICE)
    world_model = WorldModelTransformer().to(DEVICE)
    optimizer = optim.Adam(list(action_net.parameters()) + list(world_model.parameters()), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        total_acc = 0
        total_ent = 0
        
        for z_t, z_next, real_act in dataloader:
            z_t, z_next, real_act = z_t.to(DEVICE), z_next.to(DEVICE), real_act.to(DEVICE)
            optimizer.zero_grad()
            
            action_logits = action_net(z_t, z_next)
            action_probs = torch.softmax(action_logits, dim=1)
            pred_logits = world_model(z_t, action_probs)
            
            loss_recon = criterion(pred_logits.view(-1, VOCAB_SIZE), z_next.view(-1))
            
            # Entropy calculation
            log_probs = torch.log_softmax(action_logits, dim=1)
            entropy = -(action_probs * log_probs).sum(dim=1).mean()
            
            # Total Loss (Reduced weight)
            loss = loss_recon + (ENTROPY_WEIGHT * entropy)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_ent += entropy.item()
            total_acc += (torch.argmax(action_probs, dim=1) == real_act.squeeze()).float().mean().item()
            
        avg_ent = total_ent/len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(dataloader):.4f} (Ent: {avg_ent:.4f}) | Acc: {total_acc/len(dataloader)*100:.1f}%")

    torch.save(action_net.state_dict(), ARTIFACTS_PATH / "action_net_transformer.pth")
    torch.save(world_model.state_dict(), ARTIFACTS_PATH / "world_model_transformer.pth")
    
    print("Generating Viz...")
    action_net.eval()
    all_preds, all_real = [], []
    with torch.no_grad():
        for z_t, z_next, real_act in dataloader:
            z_t, z_next = z_t.to(DEVICE), z_next.to(DEVICE)
            all_preds.extend(torch.argmax(action_net(z_t, z_next), dim=1).cpu().numpy())
            all_real.extend(real_act.squeeze().numpy())
            
    cm = confusion_matrix(all_real, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('Ground Truth Action')
    plt.xlabel('Discovered Cluster')
    plt.title(f'Latent Action Discovery (Entropy w={ENTROPY_WEIGHT})')
    plt.savefig(ARTIFACTS_PATH / "transformer_confusion_matrix.png")
    
    try:
        import generate_dream_gif
        generate_dream_gif.generate_gif(VQVAE().to(DEVICE), world_model)
    except: pass

if __name__ == "__main__":
    main()