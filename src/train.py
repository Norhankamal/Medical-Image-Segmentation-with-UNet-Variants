"""
train.py
--------
Training loop for UNet binary segmentation model.

Dataset  : 2018 Data Science Bowl (nuclei segmentation)
Loss     : Combined BCE + Dice
Optimizer: Adam with ReduceLROnPlateau scheduler
Platform : Google Colab (GPU)

"""

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.model    import UNet
from src.dataset  import get_dataloaders
from src.evaluate import dice_score, iou_score
from src.utils    import (plot_training_curves,
                           save_predictions,
                           save_metrics)


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


class BCEDiceLoss(nn.Module):
    """
    Combined BCE + Dice loss for binary segmentation.

    BCE  : stable gradients, handles pixel-wise classification.
    Dice : handles class imbalance (13.9% foreground pixels).

    Total = bce_weight * BCE + dice_weight * (1 - Dice)

    Args:
        bce_weight  : scalar weight for BCE term (from config)
        dice_weight : scalar weight for Dice term (from config)
        smooth      : epsilon to prevent division by zero
    """

    def __init__(self, bce_weight=1.0, dice_weight=1.0, smooth=1.0):
        super().__init__()
        self.bce_fn      = nn.BCELoss()
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight
        self.smooth      = smooth

    def forward(self, pred, target):
        bce_loss  = self.bce_fn(pred, target)

        pred_f    = pred.view(-1)
        tgt_f     = target.view(-1)
        inter     = (pred_f * tgt_f).sum()
        dice_loss = 1.0 - (2.0 * inter + self.smooth) / (
                    pred_f.sum() + tgt_f.sum() + self.smooth)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    One full pass over the training set.

    Returns:
        Average training loss for this epoch.
    """
    model.train()
    total_loss = 0.0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        preds = model(images)
        loss  = criterion(preds, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    """
    Evaluate model on validation set.

    Returns:
        Tuple of (avg_loss, avg_dice, avg_iou)
    """
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou  = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)

            preds        = model(images)
            total_loss  += criterion(preds, masks).item()
            total_dice  += dice_score(preds, masks).item()
            total_iou   += iou_score(preds, masks).item()

    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


def train():
    """
    Full training pipeline.

    1. Load config
    2. Load data via get_dataloaders()
    3. Build UNet model
    4. Train for N epochs, validate each epoch
    5. Save best model checkpoint by Dice score
    6. Save training curves, predictions, and metrics
    """

    # ── Config & Device ───────────────────────────────────────
    cfg    = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────
    train_loader, val_loader = get_dataloaders(cfg)

    # ── Model ─────────────────────────────────────────────────
    model = UNet(
        in_channels  = cfg["model"]["in_channels"],
        out_channels = cfg["model"]["out_channels"]
    ).to(device)

    # ── Loss, Optimizer, Scheduler ────────────────────────────
    criterion = BCEDiceLoss(
        bce_weight  = cfg["loss"]["bce_weight"],
        dice_weight = cfg["loss"]["dice_weight"]
    )
    optimizer = optim.Adam(
        model.parameters(),
        lr = cfg["training"]["lr"]
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience = cfg["training"]["lr_patience"],
        factor   = cfg["training"]["lr_factor"]
    )

    # ── Training Loop ─────────────────────────────────────────
    history = {
        "train_loss" : [],
        "val_loss"   : [],
        "val_dice"   : [],
        "val_iou"    : []
    }

    best_dice = 0.0
    epochs    = cfg["training"]["epochs"]

    print(f"\nTraining UNet for {epochs} epochs\n")
    print(f"{'Epoch':>6} {'Train Loss':>12} "
          f"{'Val Loss':>10} {'Dice':>8} {'IoU':>8}")
    print("-" * 50)

    for epoch in range(1, epochs + 1):

        trn_loss                    = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_dice, val_iou = validate(
            model, val_loader, criterion, device
        )
        scheduler.step(val_loss)

        history["train_loss"].append(trn_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)

        print(f"{epoch:>6} {trn_loss:>12.4f} "
              f"{val_loss:>10.4f} {val_dice:>8.4f} {val_iou:>8.4f}")

        # Save best model by Dice score
        if val_dice > best_dice:
            best_dice = val_dice
            os.makedirs(cfg["paths"]["checkpoint_dir"], exist_ok=True)
            torch.save(model.state_dict(), cfg["paths"]["best_model"])
            print(f"         -> best model saved (Dice {best_dice:.4f})")

    # ── Save Results ──────────────────────────────────────────
    plot_training_curves(history,
                         cfg["paths"]["figures_dir"])
    save_predictions(model, val_loader, device,
                     save_dir=cfg["paths"]["figures_dir"])
    save_metrics(
        {
            "val_dice" : max(history["val_dice"]),
            "val_iou"  : max(history["val_iou"]),
            "val_loss" : min(history["val_loss"])
        },
        cfg["paths"]["figures_dir"]
    )

    print(f"\nBest Dice : {max(history['val_dice']):.4f}")
    print(f"Best IoU  : {max(history['val_iou']):.4f}")


if __name__ == "__main__":
    train()