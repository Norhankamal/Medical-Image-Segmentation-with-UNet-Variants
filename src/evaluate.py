"""

Evaluation metrics and baseline for binary image segmentation.

Metrics:
    - Dice Similarity Coefficient
    - Intersection over Union (IoU)

Baseline:
    - Otsu Thresholding (classical intensity-based method)
      Used as comparison against UNet to quantify the
      advantage of learned features over classical methods.

"""

import cv2
import numpy as np
import torch


def dice_score(pred, target, threshold=0.5, smooth=1.0):
    """
    Dice Similarity Coefficient for binary segmentation.

        Dice = 2|P ∩ G| / (|P| + |G|)

    Ranges from 0 (no overlap) to 1 (perfect match).
    Preferred over pixel accuracy for this task due to class
    imbalance — 86.1% of pixels are background, so accuracy
    is not a meaningful measure of segmentation quality.

    Args:
        pred      : model output tensor, values in [0, 1]
        target    : ground truth binary mask
        threshold : probability cutoff for binary decision
        smooth    : epsilon to prevent division by zero

    Returns:
        Dice score as scalar tensor
    """
    pred   = (pred > threshold).float()
    pred_f = pred.view(-1)
    tgt_f  = target.view(-1)

    intersection = (pred_f * tgt_f).sum()
    return (2.0 * intersection + smooth) / (
        pred_f.sum() + tgt_f.sum() + smooth
    )


def iou_score(pred, target, threshold=0.5, smooth=1.0):
    """
    Intersection over Union for binary segmentation.

        IoU = |P ∩ G| / |P ∪ G|

    Stricter than Dice — penalizes false positives more heavily.

    Args:
        pred      : model output tensor, values in [0, 1]
        target    : ground truth binary mask
        threshold : probability cutoff for binary decision
        smooth    : epsilon to prevent division by zero

    Returns:
        IoU score as scalar tensor
    """
    pred   = (pred > threshold).float()
    pred_f = pred.view(-1)
    tgt_f  = target.view(-1)

    intersection = (pred_f * tgt_f).sum()
    union        = pred_f.sum() + tgt_f.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def otsu_predict(image_path):
    """
    Classical Otsu thresholding baseline.

    Otsu's method finds the optimal intensity threshold by
    minimizing intra-class variance between foreground and
    background pixels. Requires no training data or learned
    weights — purely intensity-based.

    Limitation: fails under varying illumination, low contrast,
    or overlapping nuclei — conditions common in this dataset.
    This is why we compare it against UNet.

    Args:
        image_path : path to input image file

    Returns:
        Binary mask as numpy array (0 or 1, same size as input)
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (256, 256))

    _, mask = cv2.threshold(
        img, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return (mask / 255.0).astype(np.float32)


def evaluate_otsu(image_paths, mask_paths, smooth=1.0):
    """
    Run Otsu baseline on a list of images and compute
    average Dice and IoU against ground truth masks.

    Args:
        image_paths : list of paths to input images
        mask_paths  : list of paths to ground truth masks
        smooth      : epsilon to prevent division by zero

    Returns:
        Dict with average Dice and IoU scores
    """
    total_dice = 0.0
    total_iou  = 0.0
    n          = len(image_paths)

    for img_path, mask_path in zip(image_paths, mask_paths):
        # Otsu prediction
        pred = otsu_predict(img_path)

        # Ground truth
        gt   = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt   = cv2.resize(gt, (256, 256))
        gt   = (gt > 127).astype(np.float32)

        # Convert to tensors for metric functions
        pred_t = torch.tensor(pred).unsqueeze(0).unsqueeze(0)
        gt_t   = torch.tensor(gt).unsqueeze(0).unsqueeze(0)

        total_dice += dice_score(pred_t, gt_t,
                                  threshold=0.5).item()
        total_iou  += iou_score(pred_t, gt_t,
                                 threshold=0.5).item()

    return {
        "otsu_dice" : total_dice / n,
        "otsu_iou"  : total_iou  / n
    }
