# generate.py
import torch
from data_loader import load_dataset
from model import TinyTransformer

block_size = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load tokenizer and model
<<<<<<< HEAD
=======
# load_dataset defaults to data/conversations.txt
>>>>>>> 61763c6bbacfa18d94ef1faddd79795f6ac41ae1
data, tokenizer = load_dataset(block_size=block_size)

model = TinyTransformer(
    vocab_size=tokenizer.vocab_size,
    block_size=block_size,
    n_embed=128,
    n_heads=4,
    n_layers=4
).to(device)

model.load_state_dict(torch.load("genz_model.pt", map_location=device))
model.eval()

# Prompt the model with an initial string
start_prompt = "rizz"
start_ids = tokenizer.encode(start_prompt)
context = torch.tensor([start_ids], dtype=torch.long).to(device)

# Generate text
with torch.no_grad():
    generated_ids = model.generate(context, max_new_tokens=250)[0].tolist()

# Decode and print the result
generated_text = tokenizer.decode(generated_ids)
print("\n🧠 Gen Z Bot says:\n")
print(generated_text)
