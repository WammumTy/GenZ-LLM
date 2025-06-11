# train.py
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader import load_dataset
from dataset import CharDataset
from model import TinyTransformer

# Hyperparameters
block_size = 64
batch_size = 32
max_iters = 2000
eval_interval = 200
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load data and tokenizer
data, tokenizer = load_dataset("data/conversations.txt", block_size=block_size)
split = int(0.9 * len(data))
train_data = data[:split]
val_data = data[split:]

train_dataset = CharDataset(train_data, block_size)
val_dataset = CharDataset(val_data, block_size)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Model
model = TinyTransformer(
    vocab_size=tokenizer.vocab_size,
    block_size=block_size,
    n_embed=128,
    n_heads=4,
    n_layers=4
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for iter in range(max_iters):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        break  # Only one batch per iteration for simplicity

    # Evaluation
    if iter % eval_interval == 0 or iter == max_iters - 1:
        model.eval()
        with torch.no_grad():
            losses = []
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                _, loss = model(xb, yb)
                losses.append(loss.item())
            avg_loss = sum(losses) / len(losses)
        print(f"Step {iter} | val loss {avg_loss:.4f}")

# Save model
torch.save(model.state_dict(), "genz_model.pt")
