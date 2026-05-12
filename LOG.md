# Development Log
> **Project:** UNet-Based Segmentation for Small Medical Datasets  
> **Dataset:** 2018 Kaggle Data Science Bowl — Nuclei Segmentation  
---

## Day 1 — May 11, 2025 | Repository Setup & Project Structure

**Responsible:** Student A  

### Completed
- Created GitHub repository: `Medical-Image-Segmentation-with-UNet-Variants`
- Set up full project folder structure:
  `src/`, `notebooks/`, `data/`, `configs/`, `reports/figures/`, `checkpoints/`
- Wrote `.gitignore` (excludes raw data, model weights, pycache)
- Wrote `requirements.txt` with all dependencies and versions
- Wrote `README.md` with project overview, team roles, and setup instructions
- Wrote `data/README.md` with dataset download instructions
- Wrote `configs/config.yaml` with all hyperparameters documented
- Downloaded 2018 Kaggle Data Science Bowl dataset to local machine
- Extracted `stage1_train.zip` — confirmed 670 training samples

### Key Decisions
| Decision | Reason |
|----------|--------|
| Public GitHub repo | Required for instructor review |
| Separate `configs/config.yaml` | Centralizes all hyperparameters for reproducibility |
| Raw data excluded via `.gitignore` | Dataset too large for GitHub (>1GB) |
| `data/README.md` instead | Documents source and download steps clearly |

### Issues Encountered
| Issue | Solution |
|-------|----------|
| None on this day | — |

---

## Day 2 — May 12, 2025 | EDA + Preprocessing + Augmentation

**Responsible:** Student A  
**Commits:** `Add EDA notebook`, `Add preprocessing pipeline`, `Add augmentation pipeline`, `Add NucleiDataset class`, `Update LOG.md`

### ✅ Completed

#### Exploratory Data Analysis (EDA)
- Scanned all 670 training samples for image statistics
- Found image sizes vary widely — from 256×256 up to 1040×1040
- Found 3 channel types: RGB (3ch), Grayscale (1ch), RGBA (4ch)
- Computed nucleus count per image: mean ≈ X, min = X, max = X
- Visualized 4 sample images with their ground truth masks and overlays
- Saved all EDA figures to `reports/figures/`

#### Preprocessing Pipeline
- Implemented `load_image()`: convert to RGB → resize 256×256 → normalize
- Implemented `load_mask()`: merge all instance masks → single binary mask
- Applied threshold > 127 to ensure clean binary mask values (0 or 1)
- Verified pipeline output: image shape (256, 256, 3), mask shape (256, 256)
- Visualized before/after preprocessing for 3 samples

#### Data Augmentation
- Implemented augmentation pipeline using `albumentations` library
- Augmentations applied: HorizontalFlip, VerticalFlip, RandomRotate90, ElasticTransform, RandomBrightnessContrast
- Verified augmentations apply consistently to BOTH image and mask
- Visualized 3 augmented versions of one sample (image + mask side by side)

#### Dataset Class
- Implemented `NucleiDataset` (PyTorch Dataset class) in `src/dataset.py`
- Implemented `get_dataloaders()` using config.yaml
- Train/Val split: 80% / 20% (seed=42 for reproducibility)
- Confirmed: Train = 536 samples, Val = 134 samples
- Verified DataLoader output shapes: image (8, 3, 256, 256), mask (8, 1, 256, 256)

### 🔑 Key Decisions
| Decision | Reason |
|----------|--------|
| Resize to 256×256 | Balances spatial detail and GPU memory usage |
| Convert all images to RGB | Handles grayscale/RGBA inconsistency uniformly |
| Normalize with ImageNet mean/std | Compatible with potential transfer learning |
| Threshold mask at > 127 | Handles non-binary pixel artifacts in some masks |
| Merge instance masks → binary | Task is binary segmentation (nucleus vs background) |
| ElasticTransform p=0.3 | Mimics realistic biological tissue deformation |
| Seed = 42 | Ensures reproducible train/val split across runs |
| `albumentations` library | Applies same transform to image AND mask correctly |

### ⚠️ Issues Encountered
| Issue | Solution |
|-------|----------|
| Images have mixed channel types (1ch, 3ch, 4ch) | Applied `.convert("RGB")` to all images |
| Some mask pixel values not strictly 0 or 255 | Applied threshold > 127 to binarize |
| Image sizes vary — can't batch directly | Resize all to fixed 256×256 before batching |
| Augmentation must match image and mask exactly | Used `albumentations.Compose` with shared seed |

### 📊 EDA Key Findings
| Finding | Value |
|---------|-------|
| Total training samples | 670 |
| Unique image sizes | [انتي هتملي من النتائج الحقيقية] |
| Grayscale images | [X] |
| RGB images | [X] |
| Average nuclei per image | [X] |
| Min / Max nuclei | [X] / [X] |

---