import cv2
import numpy as np
from torch.utils.data import Dataset


class BlockDataset(Dataset):
    """
    block dataset 
    """

    def __init__(self, file_path):
        print('Loading block data')
        self.data = np.load(file_path, allow_pickle=True)
        print('Done loading block data')

    def __getitem__(self, index):
        img = self.data[index]
        label = 0
        return img, label

    def __len__(self):
        return len(self.data)


class LatentBlockDataset(Dataset):
    """
    Loads latent block dataset 
    """

    def __init__(self, file_path):
        print('Loading latent block data')
        self.data = np.load(file_path, allow_pickle=True)
        print('Done loading latent block data')

    def __getitem__(self, index):
        img = self.data[index]
        label = 0
        return img, label

    def __len__(self):
        return len(self.data)
