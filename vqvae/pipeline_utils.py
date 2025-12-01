'''
Specifies dataset and model architecture for masked MRI token prediction pipeline
'''

import torch
import torch.nn as nn

MASK_ID = 64
IGNORE_VAL = -1 

class MRITokenDataset(torch.utils.data.Dataset):
    def __init__(self, tokens, context_slices, mask_prob):
        self.tokens = torch.tensor(tokens)
        self.context_window = 4096 * context_slices
        self.mask_prob = mask_prob
        self.num_samples = len(self.tokens) // self.context_window
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        tokens = self.tokens[idx * self.context_window : (idx + 1) * self.context_window].clone()
        mask = torch.rand(len(tokens)) < self.mask_prob
        labels = torch.where(mask, tokens, torch.tensor(IGNORE_VAL))
        tokens[mask] = MASK_ID
        return tokens, labels


class MRITransformer(nn.Module):
    def __init__(self, sequence_len, vocab_size=65, embedding_dim=256, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(sequence_len, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, x):
        sequence_len = x.shape[1]
        positions = torch.arange(sequence_len, device=x.device).unsqueeze(0)
        full_embedding = self.embedding(x) + self.position_embedding(positions)
        encoder_output = self.encoder(full_embedding)
        logits = self.linear(encoder_output)
        return logits