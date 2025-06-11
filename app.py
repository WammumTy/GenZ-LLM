# app.py
import torch
import gradio as gr
from src.data_loader import load_dataset
from src.model import TinyTransformer

# Configuration
block_size = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load tokenizer and model
_, tokenizer = load_dataset("data/genz.txt", block_size=block_size)

model = TinyTransformer(
    vocab_size=tokenizer.vocab_size,
    block_size=block_size,
    n_embed=128,
    n_heads=4,
    n_layers=4
).to(device)

model.load_state_dict(torch.load("genz_model.pt", map_location=device))
model.eval()

# Generation function
def genz_reply(message, history):
    message = message.lower()  # normalize case
    encoded = tokenizer.encode(message)[-block_size:]
    idx = torch.tensor([encoded], dtype=torch.long).to(device)

    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=150)[0]

    response = tokenizer.decode(out)
    return response


# Launch Gradio chatbot interface
chat = gr.ChatInterface(fn=genz_reply, title="🧠 Gen Z LLM", description="Talk to your local rizzbot 🧃")
chat.launch()
