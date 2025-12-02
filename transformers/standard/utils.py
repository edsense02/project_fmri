import os
import random
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import MRITokenDataset
from transformer import MRITransformer


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility (Python, NumPy, PyTorch).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """
    Return CUDA device if available, else CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_dataloaders(
    train_tokens_path: str,
    val_tokens_path: str,
    batch_size: int = 16,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, int, int]:
    """
    Create train and validation DataLoaders for MRI token grids.

    Returns:
        train_loader, val_loader, latent_height, latent_width
    """
    train_ds = MRITokenDataset(train_tokens_path)
    val_ds = MRITokenDataset(val_tokens_path)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    H_lat = train_ds.latent_height
    W_lat = train_ds.latent_width

    return train_loader, val_loader, H_lat, W_lat


def mask_tokens(
    input_ids: torch.LongTensor,
    mask_token_id: int,
    p: float = 0.1,
    ignore_index: int = -100,
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """
    Randomly masks out tokens on each slice with probability p per token.

    Parameters
    ----------
    input_ids : LongTensor
        Tensor of shape (batch_size, seq_len) with token IDs (0..vocab_size-1).
    mask_token_id : int
        ID to use as the mask token in the input (e.g. 512).
    p : float
        Probability in [0, 1] of masking each token independently.
    ignore_index : int
        Label value to use for positions that should be ignored by the loss.

    Returns
    -------
    input_ids_masked : LongTensor
        Same shape as input_ids, but some tokens replaced by mask_token_id.
    labels : LongTensor
        Same shape as input_ids, with original token IDs at masked positions
        and ignore_index at unmasked positions.
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must be in [0, 1], got {p}")

    device = input_ids.device
    input_ids_masked = input_ids.clone()
    labels = input_ids.clone()

    # Sample mask positions: mask[i, j] == True means position (i, j) will be masked
    mask = torch.rand(input_ids.shape, device=device) < p  # (B, L) bool

    # Labels: keep original token where masked, ignore elsewhere
    labels[~mask] = ignore_index

    # Input: replace masked positions with mask_token_id
    input_ids_masked[mask] = mask_token_id

    return input_ids_masked, labels


def save_model_and_results(
    model: torch.nn.Module,
    results: dict,
    hyperparameters: dict,
    timestamp: str,
) -> None:
    """
    Save model checkpoint and training results to the top-level results/ folder.

    The structure matches the VQ-VAE save function.
    """
    # Base dir = project root (two levels up from transformers/standard/)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    save_dir = os.path.join(base_dir, "results")
    os.makedirs(save_dir, exist_ok=True)

    results_to_save = {
        "model": model.state_dict(),
        "results": results,
        "hyperparameters": hyperparameters,
    }

    ckpt_path = os.path.join(save_dir, f"transformer_ckpt_{timestamp}.pth")
    torch.save(results_to_save, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")


def load_transformer(checkpoint: str, state: str = "eval") -> MRITransformer:
    """
    Load a trained MRITransformer from checkpoint and set it to train or eval mode.

    Inputs
    ------
    checkpoint : str
        Path to the .pth checkpoint file saved by save_model_and_results.
    state : str
        "eval" or "train". Default is "eval".

    Returns
    -------
    model : MRITransformer
        Model with weights loaded and moved to the current device.
    """
    if state not in ("train", "eval"):
        raise ValueError("state must be 'train' or 'eval'")

    # Load checkpoint to CPU first
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)

    # Extract hyperparameters (may not contain everything, so we also infer from state_dict)
    hparams = ckpt.get("hyperparameters", {})

    vocab_size     = hparams.get("vocab_size", 513)
    d_model        = hparams.get("d_model", 256)
    n_heads        = hparams.get("n_heads", 8)
    num_layers     = hparams.get("num_layers", 4)
    dim_feedforward = hparams.get("dim_feedforward", 1024)
    dropout        = hparams.get("dropout", 0.1)
    pos_encoding   = hparams.get("pos_encoding", "1d")

    # State dict with actual weights
    state_dict = ckpt.get("model", None)
    if state_dict is None:
        raise KeyError(
            "Checkpoint does not contain key 'model'. "
            "Expected a dict with keys: 'model', 'results', 'hyperparameters'."
        )

    # Infer sequence length / latent dims from the embedding weights if needed
    latent_height = hparams.get("latent_height", None)
    latent_width  = hparams.get("latent_width", None)
    max_seq_len   = hparams.get("max_seq_len", None)

    if pos_encoding == "1d":
        # For 1D encoding we only need max_seq_len
        if max_seq_len is None:
            pe_weight = state_dict.get("pos_embed_1d.weight", None)
            if pe_weight is None:
                raise ValueError(
                    "Cannot infer max_seq_len: 'pos_embed_1d.weight' missing in state_dict."
                )
            max_seq_len = pe_weight.shape[0]

        # latent_height / latent_width are not used in 1D mode
        latent_height = None
        latent_width = None

    elif pos_encoding == "2d":
        # For 2D encoding we need latent_height, latent_width; infer from row/col embeddings
        if latent_height is None:
            row_w = state_dict.get("row_embed.weight", None)
            if row_w is None:
                raise ValueError(
                    "Cannot infer latent_height: 'row_embed.weight' missing in state_dict."
                )
            latent_height = row_w.shape[0]

        if latent_width is None:
            col_w = state_dict.get("col_embed.weight", None)
            if col_w is None:
                raise ValueError(
                    "Cannot infer latent_width: 'col_embed.weight' missing in state_dict."
                )
            latent_width = col_w.shape[0]

        if max_seq_len is None:
            max_seq_len = latent_height * latent_width

    else:
        raise ValueError(f"Unknown pos_encoding '{pos_encoding}'. Expected '1d' or '2d'.")

    # Rebuild model with the inferred / stored hyperparameters
    model = MRITransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_seq_len=max_seq_len,
        pos_encoding=pos_encoding,
        latent_height=latent_height,
        latent_width=latent_width,
    )

    # Load weights
    model.load_state_dict(state_dict)

    # Move to device and set mode
    device = get_device()
    model.to(device)

    if state == "train":
        model.train()
    else:
        model.eval()

    return model

