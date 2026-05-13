
"""

PyTorch Dataset and DataLoaders for preprocessed nuclei segmentation data.

Loads .npz archives produced by Notebook 02 (Preprocessing Pipeline).
Each archive contains images and masks as float32 arrays, already
normalised to [0, 1] and resized to 256x256.

.npz contents per split:
    images     : float32, shape (N, 3, 256, 256)  -- normalised RGB
    masks      : float32, shape (N, 256, 256)      -- binary {0.0, 1.0}
    sample_ids : str array, shape (N,)             -- traceability

"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


PREPROCESSED_DIR = "/content/drive/MyDrive/Medical_Segmentation_Data/preprocessed"


class NucleiDataset(Dataset):
    """
    Dataset wrapper for preprocessed .npz archives.

    Images are already normalised to [0, 1] and shaped (3, 256, 256).
    Masks are binary {0.0, 1.0} and shaped (1, 256, 256) after unsqueeze.

    Args:
        npz_path  : absolute path to the .npz archive
        transform : optional augmentation transforms applied to images only
    """

    def __init__(self, npz_path, transform=None):
        data           = np.load(npz_path)
        self.images    = torch.tensor(data["images"], dtype=torch.float32)
        self.masks     = torch.tensor(
                             data["masks"], dtype=torch.float32
                         ).unsqueeze(1)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]   
        mask  = self.masks[idx]    

        if self.transform:
            image = self.transform(image)

        return image, mask


def get_dataloaders(cfg):
    """
    Build train and validation DataLoaders from preprocessed .npz archives.

    Loads train_preprocessed.npz and val_preprocessed.npz from Drive.
    Split was performed in Notebook 01 (EDA) .

    Args:
        cfg : dict loaded from configs/config.yaml

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_ds = NucleiDataset(f"{PREPROCESSED_DIR}/train_preprocessed.npz")
    val_ds   = NucleiDataset(f"{PREPROCESSED_DIR}/val_preprocessed.npz")

    train_loader = DataLoader(
        train_ds,
        batch_size  = cfg["training"]["batch_size"],
        shuffle     = True,
        num_workers = cfg["data"]["num_workers"],
        pin_memory  = True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = cfg["training"]["batch_size"],
        shuffle     = False,
        num_workers = cfg["data"]["num_workers"],
        pin_memory  = True
    )

    return train_loader, val_loader
