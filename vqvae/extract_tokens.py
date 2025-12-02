import torch
import numpy as np
from models.vqvae import VQVAE

model_path = "./results/vqvae_data_tue_nov_11_21_09_30_2025.pth"
checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

model = VQVAE(
    h_dim=128, res_h_dim=32, n_res_layers=2,
    n_embeddings=512, embedding_dim=64, beta=0.25
)
model.load_state_dict(checkpoint["model"])
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

data = np.load("/home/mingjie/mri230/train_data/train_data.npy")    # shape (32, 1, 256, 256)
data = torch.from_numpy(data[:32]).float().to(device)

with torch.no_grad():
    _, _, _, tokens = model(data)
    
tokens = tokens.cpu().numpy()
    
np.save("/home/mingjie/mri230/tokens/tokens.npy", tokens)