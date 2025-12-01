import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import os
from datasets.block import BlockDataset, LatentBlockDataset
import numpy as np


def load_block():
    train_file_path = '/home/mingjie/mri230/train_data/train_data.npy'
    val_file_path = '/home/mingjie/mri230/val_data/val_data.npy'
    train = BlockDataset(train_file_path)
    val = BlockDataset(val_file_path)
    return train, val

def load_latent_block():
    train_latent_path = '/home/mingjie/mri230/train_data/train_latent.npy'
    val_latent_path = '/home/mingjie/mri230/val_data/val_latent.npy'
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


def save_model_and_results(model, results, hyperparameters, run_name):
    SAVE_MODEL_PATH = '/home/mingjie/mri230/vqvae_checkpoints/'

    results_to_save = {
        'model': model.state_dict(),
        'results': results,
        'hyperparameters': hyperparameters
    }
    torch.save(results_to_save,
               SAVE_MODEL_PATH + run_name + '.pth')
