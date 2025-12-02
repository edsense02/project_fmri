import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import os
from datasets.block import BlockDataset, LatentBlockDataset
import numpy as np
from models.vqvae import VQVAE


def load_block():
    train_file_path = '../data/train/train_data.npy'
    val_file_path = '../data/val/val_data.npy'
    train = BlockDataset(train_file_path)
    val = BlockDataset(val_file_path)
    return train, val

def load_latent_block():
    train_latent_path = '../data/train/train_data.npy'
    val_latent_path = '../data/val/val_data.npy'
    train = LatentBlockDataset(train_latent_path)
    val = LatentBlockDataset(val_latent_path)
    return train, val


def data_loaders(train_data, val_data, batch_size):

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True)
    val_loader = DataLoader(val_data,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True)
    return train_loader, val_loader


def load_data_and_data_loaders(dataset, batch_size):
    if dataset == 'BLOCK':
        training_data, validation_data = load_block()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)

        x_train_var = np.var(training_data.data)
    elif dataset == 'LATENT_BLOCK':
        training_data, validation_data = load_latent_block()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)

        x_train_var = np.var(training_data.data)

    else:
        raise ValueError(
            'Invalid dataset: only BLOCK dataset is supported.')

    return training_data, validation_data, training_loader, validation_loader, x_train_var


def readable_timestamp():
    return time.ctime().replace('  ', ' ').replace(
        ' ', '_').replace(':', '_').lower()


def save_model_and_results(model, results, hyperparameters, timestamp):
    save_dir = '../results'
    os.makedirs(save_dir, exist_ok=True)

    results_to_save = {
        'model': model.state_dict(),
        'results': results,
        'hyperparameters': hyperparameters
    }
    torch.save(results_to_save,
               save_dir + '/vqvae_data_' + timestamp + '.pth')



def load_model(checkpoint, state: str = "eval"):
    """
    returns model with loaded checkpoint in train or eval mode

    Inputs
        checkpoint: str, Path to the .pth checkpoint file.
        state: str, "eval" or "train". Default is "eval".
    Returns
        model: VQVAE model with weights loaded and moved to device.
    """
    # Load checkpoint
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)

    # Rebuild model from saved hyperparameters if available
    hparams       = ckpt.get("hyperparameters", {})
    h_dim         = hparams.get("n_hiddens", 128)
    res_h_dim     = hparams.get("n_residual_hiddens", 32)
    n_res_layers  = hparams.get("n_residual_layers", 2)
    n_embeddings  = hparams.get("n_embeddings", 512)
    embedding_dim = hparams.get("embedding_dim", 64)
    beta          = hparams.get("beta", 0.25)

    model = VQVAE(
        h_dim=h_dim,
        res_h_dim=res_h_dim,
        n_res_layers=n_res_layers,
        n_embeddings=n_embeddings,
        embedding_dim=embedding_dim,
        beta=beta,
    )

    model.load_state_dict(ckpt["model"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if state == "train":
        model.train()
    elif state == "eval":
        model.eval()
    else:
        raise ValueError("state must be 'train' or 'eval'")

    return model


