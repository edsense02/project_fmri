import torch
import torch.nn as nn
import numpy as np
import wandb

class MRITransformerEncoder(nn.Module):
    def __init__(self, vocab_size=512, seq_len=4096, embed_dim=128, heads=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, heads, dim_feedforward=512)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        return self.output(x)


def single_batch_train():
    wandb.init(project="transformerMRI", name='basic_next_token_prediction')

    batch_size = 32
    tokens = np.load('/home/mingjie/mri230/tokens/tokens.npy')  # shape (64 x 64 x batch_size, 1)
    tokens = tokens.reshape(batch_size, 4096)   # 64 x 64 = 4096
    tokens = torch.from_numpy(tokens).long()

    split = int(0.8 * batch_size)
    train_tokens = tokens[:split]
    val_tokens = tokens[split:]

    model = MRITransformerEncoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for i in range(len(train_tokens) - 1):
            input_seq = train_tokens[i].unsqueeze(0)    # shape (1, 4096)
            target_seq = train_tokens[i + 1].unsqueeze(0)   # shape (1, 4096)

            logits = model(input_seq)   # shape (1, 4096, vocab_size)
            loss = criterion(logits.reshape(-1, 512), target_seq.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / (len(train_tokens) - 1)
        wandb.log({"train_loss": avg_train_loss})

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for i in range(len(val_tokens) - 1):
                input_seq = val_tokens[i].unsqueeze(0)
                target_seq = val_tokens[i + 1].unsqueeze(0)
                logits = model(input_seq)
                loss = criterion(logits.reshape(-1, 512), target_seq.reshape(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / (len(val_tokens) - 1)
        wandb.log({"epoch": epoch, "val_loss": avg_val_loss})
        print(f"Epoch {epoch}: Train {avg_train_loss:.4f}, Val {avg_val_loss:.4f}")

    wandb.finish()


if __name__ == "__main__":
    single_batch_train()