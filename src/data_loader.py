# data_loader.py
import torch

class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
    
    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]
    
    def decode(self, ids):
        return ''.join(self.itos[i] for i in ids if i in self.itos)


def load_dataset(file_path, block_size=64):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    return data, tokenizer
