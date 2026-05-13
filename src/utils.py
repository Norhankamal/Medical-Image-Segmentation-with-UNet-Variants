"""

Utility functions for saving training results and visualizations.

Functions:
    - plot_training_curves : saves loss and metric plots
    - save_predictions     : saves visual prediction grid
    - save_metrics         : saves final scores to text file

"""

import os
import torch
import matplotlib.pyplot as plt


def plot_training_curves(history, save_dir="reports/figures/"):
    """
    Plot and save training and validation curves.

    Generates three subplots:
        - Loss curve (train vs validation)
        - Dice score curve (validation)
        - IoU curve (validation)

    Args:
        history  : dict with keys train_loss, val_loss,
                   val_dice, val_iou
        save_dir : folder to save the figure
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss
    axes[0].plot(history["train_loss"],
                 label="Train", color="#E24B4A")
    axes[0].plot(history["val_loss"],
                 label="Val",   color="#185FA5")
    axes[0].set_title("Loss over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Dice
    axes[1].plot(history["val_dice"], color="#0F6E56")
    axes[1].set_title("Validation Dice Score")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    # IoU
    axes[2].plot(history["val_iou"], color="#534AB7")
    axes[2].set_title("Validation IoU Score")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("IoU")
    axes[2].set_ylim(0, 1)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Training curves saved to {path}")


def save_predictions(model, loader, device,
                     num_samples=4,
                     save_dir="reports/figures/"):
    """
    Save a visual grid of model predictions.

    Each row: Input Image | Ground Truth Mask | Model Prediction
    Includes both successful and failed predictions for analysis.

    Args:
        model       : trained UNet model
        loader      : validation DataLoader
        device      : cpu or cuda
        num_samples : number of examples to visualize
        save_dir    : folder to save the figure
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    images_list, masks_list, preds_list = [], [], []

    with torch.no_grad():
        for images, masks in loader:
            preds = model(images.to(device)).cpu()
            images_list.append(images)
            masks_list.append(masks)
            preds_list.append(preds)
            if len(images_list) * images.size(0) >= num_samples:
                break

    images = torch.cat(images_list)[:num_samples]
    masks  = torch.cat(masks_list)[:num_samples]
    preds  = torch.cat(preds_list)[:num_samples]

    fig, axes = plt.subplots(
        num_samples, 3,
        figsize=(10, num_samples * 3)
    )
    titles = ["Input Image", "Ground Truth", "Prediction"]

    for i in range(num_samples):
        img  = images[i].permute(1, 2, 0).numpy()
        img  = (img - img.min()) / (img.max() - img.min() + 1e-8)
        mask = masks[i].squeeze().numpy()
        pred = (preds[i].squeeze().numpy() > 0.5).astype(float)

        for j, (data, title) in enumerate(
            zip([img, mask, pred], titles)
        ):
            axes[i][j].imshow(
                data, cmap="gray" if j > 0 else None
            )
            if i == 0:
                axes[i][j].set_title(title, fontsize=12)
            axes[i][j].axis("off")

    plt.suptitle(
        "UNet Predictions vs Ground Truth",
        fontsize=14, y=1.02
    )
    plt.tight_layout()
    path = os.path.join(save_dir, "predictions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Predictions saved to {path}")


def save_metrics(metrics, save_dir="reports/figures/"):
    """
    Save final evaluation metrics to a text file.

    Args:
        metrics  : dict with metric names and values
        save_dir : folder to save metrics.txt
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "metrics.txt")

    with open(path, "w") as f:
        f.write("=" * 40 + "\n")
        f.write("Final Evaluation Results\n")
        f.write("Model    : UNet\n")
        f.write("Dataset  : 2018 Data Science Bowl\n")
        f.write("=" * 40 + "\n\n")
        for key, value in metrics.items():
            f.write(f"{key:20s}: {value:.4f}\n")

    print(f"Metrics saved to {path}")