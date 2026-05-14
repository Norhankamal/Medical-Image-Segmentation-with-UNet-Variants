"""
PyTorch Dataset and DataLoaders for preprocessed nuclei segmentation data.

Loads .npz archives produced by Notebook 02 (Preprocessing Pipeline).
Each archive was saved by save_split_npz() in NB02 Sec.8 with the keys:

    images     : float32, shape (N, 3, 256, 256)  -- [0.0, 1.0] normalised RGB
    masks      : float32, shape (N, 256, 256)      -- binary {0.0, 1.0}
    sample_ids : str array, shape (N,)             -- traceability IDs

Preprocessing decisions that produced this format (documented in NB02):
    Step 1  Load image, convert to RGB                    (EDA Sec.6)
    Step 2  Merge all instance mask PNGs -> binary mask   (EDA Sec.8, Sec.10)
    Step 3  Resize image  to 256x256 (bilinear)           (EDA Sec.5)
    Step 4  Resize mask   to 256x256 (nearest-neighbour)  (EDA Sec.5)
    Step 5  Binarise mask: threshold > 127 -> {0,1}       (EDA Sec.11)
    Step 6  Normalise image: /255 -> [0.0, 1.0]           (EDA Sec.7)
    Step 7  Reorder image: (H,W,C) -> (C,H,W)             (PyTorch convention)

Dataset split (stratified by nucleus count, seed=42, defined in NB01 Sec.12):
    Train : ~469 samples (70 %)
    Val   : ~100 samples (15 %)
    Test  : ~101 samples (15 %)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Path to the directory containing the three .npz files produced by NB02.
# Override this constant or pass an explicit path to get_dataloaders()
# when running outside of Google Colab / your default Drive layout.
PREPROCESSED_DIR = "/content/drive/MyDrive/Medical_Segmentation_Data/preprocessed"

# Expected target size — must match TARGET_H / TARGET_W constants in NB02.
TARGET_H = 256
TARGET_W = 256

# NucleiDataset
class NucleiDataset(Dataset):
    """
    PyTorch Dataset wrapper for a single preprocessed .npz archive.

    Images are already normalised to [0.0, 1.0] and stored in channel-first
    format (C, H, W) = (3, 256, 256) by NB02.  Masks are binary {0.0, 1.0}
    in shape (H, W) = (256, 256).  This class adds a channel dimension to
    the mask so that image and mask share the same spatial rank:

        image : Tensor, shape (3, 256, 256),  dtype float32
        mask  : Tensor, shape (1, 256, 256),  dtype float32

    The extra channel dimension is required by PyTorch loss functions that
    expect (batch, channel, H, W), e.g. BCEWithLogitsLoss and the Dice loss
    used in this project.

    Args:
        npz_path  (str)             : Absolute path to the .npz archive.
        transform (callable | None) : Optional augmentation callable applied
                                      jointly to the image-mask pair.  If
                                      provided it must accept (image, mask)
                                      and return (image, mask) as tensors of
                                      the same shapes listed above.
                                      Applied only during training; pass
                                      transform=None for val and test loaders.

    Raises:
        FileNotFoundError : if npz_path does not exist.
        KeyError          : if the archive is missing 'images' or 'masks' keys.
    """

    def __init__(self, npz_path: str, transform=None):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(
                f"Preprocessed archive not found: {npz_path}\n"
                f"Run Notebook 02 (Preprocessing Pipeline) first."
            )

        data = np.load(npz_path)

        # Validate expected keys 
        required_keys = {"images", "masks", "sample_ids"}
        missing = required_keys - set(data.files)
        if missing:
            raise KeyError(
                f"Archive {os.path.basename(npz_path)} is missing keys: {missing}. "
                f"Found: {data.files}. Re-run Notebook 02."
            )

        # Load arrays as tensors 
        # NB02 saves images with shape (N, 3, H, W) — channel-first already.
        self.images = torch.tensor(data["images"], dtype=torch.float32)

        # NB02 saves masks with shape (N, H, W).
        # unsqueeze(1) adds the channel dim -> (N, 1, H, W).
        self.masks = torch.tensor(
            data["masks"], dtype=torch.float32
        ).unsqueeze(1)

        # Sample IDs kept for traceability; not used during forward pass.
        self.sample_ids = data["sample_ids"].tolist()

        self.transform = transform

        # Sanity checks on loaded shapes 
        assert self.images.shape[1:] == (3, TARGET_H, TARGET_W), (
            f"Unexpected image shape in {npz_path}: {self.images.shape}. "
            f"Expected (N, 3, {TARGET_H}, {TARGET_W})."
        )
        assert self.masks.shape[1:] == (1, TARGET_H, TARGET_W), (
            f"Unexpected mask shape in {npz_path}: {self.masks.shape}. "
            f"Expected (N, 1, {TARGET_H}, {TARGET_W})."
        )
        assert len(self.images) == len(self.masks) == len(self.sample_ids), (
            "Length mismatch between images, masks and sample_ids in archive."
        )

    # Required Dataset methods 
    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        """
        Return a single (image, mask) pair at position idx.

        Returns
        image : Tensor, shape (3, 256, 256), float32, values in [0.0, 1.0]
        mask  : Tensor, shape (1, 256, 256), float32, values in {0.0, 1.0}
        """
        image = self.images[idx]   # (3, H, W)
        mask  = self.masks[idx]    # (1, H, W)

        # Apply joint augmentation if provided.
        # transform is expected to accept and return (image, mask) tensors.
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask

    # Utility 
    def __repr__(self) -> str:
        return (
            f"NucleiDataset("
            f"n={len(self)}, "
            f"image_shape={tuple(self.images.shape[1:])}, "
            f"mask_shape={tuple(self.masks.shape[1:])}, "
            f"transform={'yes' if self.transform else 'no'})"
        )


# get_dataloaders
def get_dataloaders(
    cfg,
    train_transform=None,
    preprocessed_dir: str = PREPROCESSED_DIR,
):
    """
    Build train, validation, and test DataLoaders from preprocessed .npz archives.

    The three archives (train_preprocessed.npz, val_preprocessed.npz,
    test_preprocessed.npz) are produced by NB02 Sec.8 (save_split_npz).
    The split itself was defined in NB01 Sec.12 (stratified by nucleus count,
    seed=42): 70% train / 15% val / 15% test.

    Augmentation policy:
        - train_loader : uses train_transform (augmentation enabled)
        - val_loader   : no transform (clean evaluation)
        - test_loader  : no transform (held-out evaluation only)

    Args:
        cfg             (dict)          : Config dict loaded from
                                         configs/config.yaml.  Expected keys:
                                             cfg["training"]["batch_size"] : int
                                             cfg["data"]["num_workers"]    : int
        train_transform (callable|None) : Augmentation callable for training set.
                                         Must accept (image, mask) tensors and
                                         return (image, mask). Pass None to
                                         disable augmentation.
        preprocessed_dir (str)          : Override for the directory containing
                                         the .npz files.  Defaults to
                                         PREPROCESSED_DIR at module level.

    Returns:
        train_loader : DataLoader  (shuffle=True)
        val_loader   : DataLoader  (shuffle=False)
        test_loader  : DataLoader  (shuffle=False)

    Raises:
        FileNotFoundError : if any of the three .npz archives is missing.
        KeyError          : if cfg is missing expected keys.
    """
    batch_size  = cfg["training"]["batch_size"]
    num_workers = cfg["data"]["num_workers"]

    # Build datasets 
    train_ds = NucleiDataset(
        npz_path  = os.path.join(preprocessed_dir, "train_preprocessed.npz"),
        transform = train_transform,    # augmentation on train only
    )
    val_ds = NucleiDataset(
        npz_path  = os.path.join(preprocessed_dir, "val_preprocessed.npz"),
        transform = None,               # no augmentation on val
    )
    test_ds = NucleiDataset(
        npz_path  = os.path.join(preprocessed_dir, "test_preprocessed.npz"),
        transform = None,               # no augmentation on test
    )

    # Build DataLoaders 
    # pin_memory=True speeds up CPU->GPU transfer when a CUDA GPU is available.
    # drop_last=True on train avoids a final incomplete batch that can destabilise
    # batch normalisation layers in UNet encoder blocks.
    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = False,
    )

    return train_loader, val_loader, test_loader

# Quick smoke test — run this file directly to verify archives are loadable
#   python src/dataset.py
if __name__ == "__main__":
    import sys

    print("dataset.py — smoke test")
    print(f"Looking for archives in: {PREPROCESSED_DIR}\n")

    for split in ("train", "val", "test"):
        path = os.path.join(PREPROCESSED_DIR, f"{split}_preprocessed.npz")
        try:
            ds = NucleiDataset(path)
            img, mask = ds[0]
            print(f"  {split:<6}: {ds}")
            print(f"         image[0] shape={tuple(img.shape)}  "
                  f"range=[{img.min():.3f}, {img.max():.3f}]")
            print(f"         mask[0]  shape={tuple(mask.shape)}  "
                  f"unique={torch.unique(mask).tolist()}")
        except FileNotFoundError as e:
            print(f"  {split:<6}: SKIPPED — {e}")
            sys.exit(1)

    print("\nAll archives loaded successfully.")

