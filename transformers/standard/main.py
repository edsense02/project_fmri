import argparse
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from transformer import MRITransformer
from utils import (
    set_seed,
    get_device,
    create_dataloaders,
    mask_tokens,
    save_model_and_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Transformer on MRI VQ-VAE tokens")

    # Data paths (assuming you run from transformers/standard/)
    parser.add_argument(
        "--train_tokens_path",
        type=str,
        default="../../data/train/train_tokens.npy",
        help="Path to training tokens .npy file",
    )
    parser.add_argument(
        "--val_tokens_path",
        type=str,
        default="../../data/val/val_tokens.npy",
        help="Path to validation tokens .npy file",
    )

    # Model hyperparameters
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=513,  # 512 codebook tokens + 1 MASK token (ID 512)
        help="Vocabulary size (codebook entries + special tokens)",
    )
    parser.add_argument("--d_model", type=int, default=256, help="Transformer hidden size")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument(
        "--dim_feedforward",
        type=int,
        default=1024,
        help="Hidden size of feedforward layers",
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    parser.add_argument(
        "--pos_encoding",
        type=str,
        default="1d",
        choices=["1d", "2d"],
        help="Type of positional encoding to use",
    )

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--mask_prob", type=float, default=0.1, help="Masking probability per token")
    parser.add_argument(
        "--mask_token_id",
        type=int,
        default=512,  # dedicated MASK token id
        help="Token ID used as the [MASK] token in the input",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    return args


def train() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    # Check vocab / mask consistency
    if args.vocab_size <= args.mask_token_id:
        raise ValueError(
            f"vocab_size ({args.vocab_size}) must be > mask_token_id "
            f"({args.mask_token_id}) to have a dedicated MASK token."
        )

    # Data
    if not os.path.exists(args.train_tokens_path):
        raise FileNotFoundError(f"Train tokens not found at {args.train_tokens_path}")
    if not os.path.exists(args.val_tokens_path):
        raise FileNotFoundError(f"Val tokens not found at {args.val_tokens_path}")

    train_loader, val_loader, H_lat, W_lat = create_dataloaders(
        args.train_tokens_path,
        args.val_tokens_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    max_seq_len = H_lat * W_lat

    # Model
    model = MRITransformer(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_seq_len=max_seq_len,
        pos_encoding=args.pos_encoding,
        latent_height=H_lat,
        latent_width=W_lat,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # wandb: always on, like in your VQ-VAE main.py
    wandb.init(
        project="mri-transformer",
        name=f"transformer_encoding_{args.pos_encoding}_epochs{args.epochs}_lr{args.lr}",
        config=vars(args),
    )

    # Logging containers
    global_step = 0
    train_loss_per_step: list[float] = []
    train_loss_per_epoch: list[float] = []
    val_loss_per_epoch: list[float] = []

    LOG_EVERY = 10  # print every N steps

    # Training loop
    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        running_loss = 0.0
        num_train_samples = 0

        for batch in train_loader:
            # batch: (B, H_lat, W_lat)
            batch = batch.to(device)
            B, H, W = batch.shape
            seq_len = H * W

            # Flatten to sequences (B, L)
            input_ids = batch.view(B, seq_len)

            # Apply masking
            input_masked, labels = mask_tokens(
                input_ids,
                mask_token_id=args.mask_token_id,
                p=args.mask_prob,
                ignore_index=-100,
            )

            logits = model(input_masked)  # (B, L, vocab_size)

            loss = loss_fn(
                logits.view(-1, args.vocab_size),
                labels.view(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Step-level logging
            global_step += 1
            loss_value = loss.item()
            train_loss_per_step.append(loss_value)

            if global_step % LOG_EVERY == 0 or global_step == 1:
                print(
                    f"[Epoch {epoch} Step {global_step}] "
                    f"train_loss_step: {loss_value:.4f}"
                )

            wandb.log(
                {
                    "train_loss_step": loss_value,
                    "epoch": epoch,
                    "step": global_step,
                }
            )

            running_loss += loss_value * B
            num_train_samples += B

        avg_train_loss = running_loss / max(num_train_samples, 1)
        train_loss_per_epoch.append(avg_train_loss)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        num_val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                B, H, W = batch.shape
                seq_len = H * W

                input_ids = batch.view(B, seq_len)

                # For validation we still mask, to evaluate prediction performance
                input_masked, labels = mask_tokens(
                    input_ids,
                    mask_token_id=args.mask_token_id,
                    p=args.mask_prob,
                    ignore_index=-100,
                )

                logits = model(input_masked)

                loss = loss_fn(
                    logits.view(-1, args.vocab_size),
                    labels.view(-1),
                )

                loss_value = loss.item()
                val_loss += loss_value * B
                num_val_samples += B

        avg_val_loss = val_loss / max(num_val_samples, 1)
        val_loss_per_epoch.append(avg_val_loss)

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"- train_loss: {avg_train_loss:.4f} "
            f"- val_loss: {avg_val_loss:.4f}"
        )

        wandb.log(
            {
                "epoch": epoch,
                "train_loss_epoch": avg_train_loss,
                "val_loss_epoch": avg_val_loss,
            }
        )

    # Save checkpoint and results (similar to VQ-VAE)
    timestamp = datetime.now().strftime("%a_%b_%d_%H_%M_%S_%Y")
    results = {
        "train_loss_per_step": train_loss_per_step,
        "train_loss_per_epoch": train_loss_per_epoch,
        "val_loss_per_epoch": val_loss_per_epoch,
    }
    hyperparameters = vars(args)
    save_model_and_results(model, results, hyperparameters, timestamp)

    wandb.finish()


if __name__ == "__main__":
    train()
