import numpy as np
import torch
from torch.utils.data import Dataset


class MRITokenDataset(Dataset):
    """
    Dataset for MRI VQ-VAE tokens.

    Expects a .npy file with shape (N_slices, H_lat, W_lat),
    containing integer token IDs for each latent position.
    """

    def __init__(self, tokens_path: str):
        super().__init__()
        arr = np.load(tokens_path)  # (N, H_lat, W_lat)
        if arr.ndim != 3:
            raise ValueError(
                f"Expected token array of shape (N, H, W), got shape {arr.shape} from {tokens_path}"
            )

        # Store as torch.LongTensor for indexing
        self.tokens = torch.from_numpy(arr).long()
        self.num_slices, self.height, self.width = self.tokens.shape

    def __len__(self) -> int:
        return self.num_slices

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns:
            token_grid: LongTensor of shape (H_lat, W_lat)
        """
        return self.tokens[idx]

    @property
    def latent_height(self) -> int:
        return self.height

    @property
    def latent_width(self) -> int:
        return self.width
