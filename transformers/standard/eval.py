import os
import sys
import math
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import importlib.util

# imports from transformers/standard/utils.py 
try:
    # when used as a package: python -m transformers.standard.eval
    from .utils import (
        set_seed,
        get_device,
        mask_tokens,
        load_transformer,
    )
except ImportError:
    # when run as a script from transformers/standard/: python eval.py
    from utils import (
        set_seed,
        get_device,
        mask_tokens,
        load_transformer,
    )

## Set up paths to project root and vqvae folder 
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))  # project_fmri/
VQVAE_DIR = os.path.join(BASE_DIR, "vqvae")

# Load vqvae/utils.py as a separate module 
if VQVAE_DIR not in sys.path:
    sys.path.insert(0, VQVAE_DIR)

vqvae_utils_path = os.path.join(VQVAE_DIR, "utils.py")
vqvae_spec = importlib.util.spec_from_file_location("vqvae_utils", vqvae_utils_path)
vqvae_utils = importlib.util.module_from_spec(vqvae_spec)
vqvae_spec.loader.exec_module(vqvae_utils)

# Checkpoint Paths
vqvae_ckpt_path = os.path.join(
    BASE_DIR,
    "results",
    "vqvae_data_mon_nov_24_12_57_54_2025.pth",
)
transformer_ckpt_path = os.path.join(
    BASE_DIR,
    "results",
    "transformer_ckpt_Thu_Nov_27_18_46_00_2025.pth",
)


def _load_models():
    """
    Load transformer and VQVAE from their checkpoints.
    Also extract mask_token_id and mask_prob from the transformer hyperparameters.
    """
    device = get_device()

    ckpt_t = torch.load(transformer_ckpt_path, map_location="cpu", weights_only=False)
    hparams_t = ckpt_t.get("hyperparameters", {})
    mask_token_id = hparams_t.get("mask_token_id", 512)
    mask_prob = hparams_t.get("mask_prob", 0.1)

    transformer = load_transformer(transformer_ckpt_path, state="eval")
    transformer.to(device)
    transformer.eval()

    vqvae = vqvae_utils.load_model(vqvae_ckpt_path, state="eval")
    vqvae.to(device)
    vqvae.eval()

    return transformer, vqvae, mask_token_id, mask_prob, device


def _tokens_to_recon(vqvae_model: torch.nn.Module, tokens_flat: torch.LongTensor) -> torch.Tensor:
    """
    Decode VQ-VAE token indices back into an image reconstruction.

    tokens_flat: (B, L) with values in [0, K-1]
    returns: (B, 1, H, W)
    """
    device = next(vqvae_model.parameters()).device
    tokens_flat = tokens_flat.to(device)  # (B, L)

    B, L = tokens_flat.shape
    H_lat = int(math.sqrt(L))
    W_lat = H_lat
    if H_lat * W_lat != L:
        raise ValueError(f"tokens_flat length {L} is not a perfect square.")

    embedding = vqvae_model.vector_quantization.embedding.weight  # (K, C)
    K, C = embedding.shape

    flat_idx = tokens_flat.view(-1)  # (B * L,)
    if flat_idx.max().item() >= K:
        raise ValueError(
            f"Token index {flat_idx.max().item()} >= codebook size {K}. "
            "Did you accidentally keep the MASK token (id 512)?"
        )

    z_q_flat = embedding[flat_idx]  # (B * L, C)
    z_q = z_q_flat.view(B, H_lat, W_lat, C).permute(0, 3, 1, 2).contiguous()  # (B, C, H_lat, W_lat)

    x_recon = vqvae_model.decoder(z_q)  # (B, 1, H, W)
    return x_recon


def visualize_eval_grid(array_list, save_path: str):
    """
    Visualize a list of lists of arrays as a grid.

    array_list: list of rows, each row is a list of 2D arrays.
                e.g. [[x_tok, x_tok_masked, x_tok_pred],
                      [x_orig, x_tok_recon, x_tok_pred_recon]]

    Heuristic:
      - integer arrays -> 'viridis' (tokens)
      - float arrays   -> 'gray'    (images)

    Saves the figure to save_path.
    """
    n_rows = len(array_list)
    n_cols = len(array_list[0]) if n_rows > 0 else 0

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    # Handle the case n_rows == 1 or n_cols == 1 gracefully
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    for i in range(n_rows):
        for j in range(n_cols):
            arr = np.array(array_list[i][j])
            ax = axes[i][j]

            # Decide colormap by dtype
            if np.issubdtype(arr.dtype, np.integer):
                cmap = "viridis"
            else:
                cmap = "gray"

            ax.imshow(arr, cmap=cmap)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def eval(data: str, batch_size: int = 4) -> float:
    """
    Evaluation function.

    Inputs
    ------
    data : str
        Path to MRI data .npy file (N, 1, H, W).
    batch_size : int
        Batch size used for evaluation.

    Returns
    -------
    avg_mse : float
        Average MSE between x_orig and x_tok_pred_recon over all slices.
    """
    transformer, vqvae, mask_token_id, mask_prob, device = _load_models()

    if not os.path.exists(data):
        raise FileNotFoundError(f"Data file not found at {data}")
    arr = np.load(data)  # (N, 1, H, W)
    if arr.ndim != 4 or arr.shape[1] != 1:
        raise ValueError(f"Expected MRI data of shape (N, 1, H, W), got {arr.shape}")

    N = arr.shape[0]
    print(f"Loaded MRI data from {data} with {N} slices")

    tensor_data = torch.from_numpy(arr).float()
    dataset = torch.utils.data.TensorDataset(tensor_data)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Output dir for visualizations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_lower = data.lower()
    if "train" in data_lower:
        split = "train"
    elif "val" in data_lower:
        split = "val"
    elif "test" in data_lower:
        split = "test"
    else:
        split = "data"

    eval_dir = os.path.join(BASE_DIR, "results", "eval", timestamp)
    os.makedirs(eval_dir, exist_ok=True)
    print(f"Saving eval visualizations to: {eval_dir}")

    mse_values = []
    global_idx = 0
    example_counter = 1

    transformer.eval()
    vqvae.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x_orig_batch = batch[0].to(device)  # (B, 1, H, W)
            B = x_orig_batch.shape[0]

            # VQ-VAE: encode+quantize to tokens
            _, _, _, tokens = vqvae(x_orig_batch)
            tokens = tokens.view(-1)
            L = tokens.numel() // B
            tokens = tokens.view(B, L)  # (B, L), 0..511

            tokens_flat = tokens.clone().long()  # (B, L)

            # latent grid size for visualization
            H_lat = int(math.sqrt(L))
            W_lat = H_lat
            if H_lat * W_lat != L:
                raise ValueError(f"Token length {L} is not a perfect square.")

            # Mask tokens
            tokens_masked_flat, labels = mask_tokens(
                tokens_flat,
                mask_token_id=mask_token_id,
                p=mask_prob,
                ignore_index=-100,
            )

            # Transformer: predict masked tokens
            logits = transformer(tokens_masked_flat)  # (B, L, 513)
            logits_code = logits[..., :512]           # (B, L, 512)
            pred_ids = logits_code.argmax(dim=-1)     # (B, L)

            # Mix original and predicted tokens at masked positions
            tokens_pred_flat = tokens_flat.clone()
            mask = (tokens_masked_flat == mask_token_id)
            tokens_pred_flat[mask] = pred_ids[mask]

            # Reconstructions
            x_tok_recon_batch = _tokens_to_recon(vqvae, tokens_flat)        # (B, 1, H, W)
            x_tok_pred_recon_batch = _tokens_to_recon(vqvae, tokens_pred_flat)  # (B, 1, H, W)

            # MSE per slice
            mse_batch = F.mse_loss(
                x_tok_pred_recon_batch,
                x_orig_batch,
                reduction="none",
            )  # (B, 1, H, W)
            mse_batch = mse_batch.mean(dim=(1, 2, 3))  # (B,)

            mse_values.extend(mse_batch.cpu().tolist())

            # Batch-level logging
            batch_mse = float(mse_batch.mean().item())
            print(f"[Eval] Batch {batch_idx} - mean MSE: {batch_mse:.6e}")

            # Visualization every 100th slice
            for b in range(B):
                if global_idx % 100 == 0:
                    x_tok_grid = tokens_flat[b].view(H_lat, W_lat).cpu().numpy()
                    x_tok_masked_grid = tokens_masked_flat[b].view(H_lat, W_lat).cpu().numpy()
                    x_tok_pred_grid = tokens_pred_flat[b].view(H_lat, W_lat).cpu().numpy()

                    x_orig_slice = x_orig_batch[b, 0].cpu().numpy()
                    x_tok_recon_slice = x_tok_recon_batch[b, 0].cpu().numpy()
                    x_tok_pred_recon_slice = x_tok_pred_recon_batch[b, 0].cpu().numpy()

                    array_list = [
                        [x_tok_grid, x_tok_masked_grid, x_tok_pred_grid],
                        [x_orig_slice, x_tok_recon_slice, x_tok_pred_recon_slice],
                    ]

                    img_name = f"{split}_{example_counter:02d}.png"
                    img_path = os.path.join(eval_dir, img_name)

                    visualize_eval_grid(array_list, img_path)
                    print(f"Saved example visualization for slice {global_idx} to {img_path}")

                    example_counter += 1

                global_idx += 1

    avg_mse = float(sum(mse_values) / max(len(mse_values), 1))
    print(f"Average MSE over {len(mse_values)} slices: {avg_mse:.6e}")
    return avg_mse


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Transformer + VQVAE on masked MRI slices")
    parser.add_argument(
        "--data_path",
        type=str,
        default="../../data/val/val_data.npy",
        help="Path to MRI data .npy file (e.g., ../../data/train/train_data.npy)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible masking",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    avg_mse = eval(args.data_path, batch_size=args.batch_size)
