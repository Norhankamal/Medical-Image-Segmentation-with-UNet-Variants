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

## Day 2 — May 13, 2025 | EDA + Preprocessing + Augmentation

**Responsible:** Student A  
**Commits:** `Add EDA notebook`, `Add preprocessing pipeline`, `Add augmentation pipeline`, `Add NucleiDataset class`, `Update LOG.md`

### Completed

#### Exploratory Data Analysis (EDA)
- Scanned all 670 training samples for image statistics
- Found image sizes vary widely — 9 distinct sizes from 256×256 up to 1388×1040
- Most common size: 256×256 (334 images, 49.9%)
- Found ALL 670 images are RGBA (4ch) — no grayscale or pure RGB images
- Computed nucleus count per image: mean=44.0 ±48.0, median=27, min=1, max=375
- Identified 35 high-density images (≥136 nuclei, top 5%) for separate evaluation
- Mask coverage: mean foreground = 13.9% ±11.1% — class imbalance confirmed
- Checked 29,461 instance masks: 0 non-binary, 0 empty, 0 size mismatches
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
- Train/Val/Test split: 70% / 15% / 15% (stratified by nucleus count quartile, seed=42)
- Confirmed: Train = 469, Val = 100, Test = 101 samples
- Split IDs saved to `data/splits/` — loaded by all downstream notebooks
- Verified DataLoader output shapes: image (8, 3, 256, 256), mask (8, 1, 256, 256)

### Key Decisions
| Decision | Reason |
|----------|--------|
| Resize to 256×256 | Balances spatial detail and GPU memory usage |
| Convert RGBA → RGB | All 670 images are RGBA — .convert("RGB") drops alpha channel cleanly |
| Normalize with ImageNet mean/std | Compatible with potential transfer learning |
| Threshold mask at > 127 | Handles non-binary pixel artifacts in some masks |
| Merge instance masks → binary | Task is binary segmentation (nucleus vs background) |
| ElasticTransform p=0.3 | Mimics realistic biological tissue deformation |
| Seed = 42 | Ensures reproducible train/val split across runs |
| `albumentations` library | Applies same transform to image AND mask correctly |
| Split 70/15/15 stratified | Balanced nucleus count distribution across all subsets |
| Save split IDs to .txt files | Prevents re-splitting and data leakage in downstream notebooks |
| Dice Loss over BCE | Class imbalance confirmed: 13.9% fg vs 86.1% bg |
| Nearest-neighbour for mask resize | Preserves binary values — no sub-pixel artefacts |

### Issues Encountered
| Issue | Solution |
|-------|----------|
| All 670 images are RGBA (4ch) — not mixed as initially assumed | Applied `.convert("RGB")` — drops alpha channel cleanly |
| Masks passed all quality checks (all binary) | Applied threshold > 127 as defensive measure anyway |
| Image sizes vary — can't batch directly | Resize all to fixed 256×256 before batching |
| Augmentation must match image and mask exactly | Used `albumentations.Compose` with shared seed |

### EDA Key Findings
| Finding | Value |
|---------|-------|
| Total training samples | 670 |
| Unique image sizes | 9 distinct (256×256 to 1388×1040) |
| Most common size | 256×256 — 334 images (49.9%) |
| Channel type | ALL RGBA (4ch) — 100% of dataset |
| Average nuclei per image | 44.0 ± 48.0 (median = 27) |
| Min / Max nuclei | 1 / 375 |
| Mask foreground coverage | 13.9% ± 11.1% (class imbalance confirmed) |
| Total masks quality-checked | 29,461 — 0 errors found |
| High-density images (top 5%) | 35 images (≥136 nuclei) |
| Train / Val / Test split | 469 / 100 / 101 (70/15/15) |

---