# Development Log
> **Project:** UNet-Based Segmentation for Small Medical Datasets
> **Dataset:** 2018 Kaggle Data Science Bowl — Nuclei Segmentation

---

## Day 1 — May 11, 2026 | Repository Setup & Project Structure

**Contributor: Student A**

### Completed
- Created GitHub repository: `Medical-Image-Segmentation-with-UNet-Variants`
- Set up full project folder structure:
  `src/`, `notebooks/`, `data/`, `configs/`, `reports/figures/`, `checkpoints/`
- Wrote `.gitignore` (excludes raw data, model weights, pycache)
- Wrote `requirements.txt` with all dependencies and versions
- Wrote `README.md` with project overview, team roles, and setup instructions
- Wrote `data/README.md` with dataset download instructions
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

## Day 2 — May 13, 2026 | EDA + Preprocessing + Model + Training Pipeline

---

### Student A — EDA + Preprocessing Pipeline

#### Exploratory Data Analysis — completed locally, uploaded Day 3

- Scanned all 670 training samples in a single pass to collect all statistics
- Found image sizes vary — 9 distinct sizes from 256×256 up to 1388×1040
- Most common size: 256×256 (334 images, 49.9%)
- Channel type distribution determined dynamically from full scan; DSB 2018 contains a mix of Grayscale, RGB, and RGBA images — all standardised to RGB via `.convert("RGB")`
- Computed nucleus count per image: mean=44.0 ±48.0, median=27, min=1, max=375
- Identified high-density images (nucleus count strictly above P95 threshold) for separate evaluation
- Mask coverage: mean foreground = 13.9% ±11.1% — class imbalance confirmed
- Spearman correlation (nucleus count vs foreground coverage) computed — moderate positive correlation confirmed
- Checked all instance masks: 0 non-binary, 0 empty, 0 size mismatches
- Visualised 6 stratified sample images with ground truth masks and overlays
- 8 EDA figures generated and saved to `reports/figures/`

#### Preprocessing Pipeline — `notebooks/02_Preprocessing.ipynb`

- Implemented `load_image_as_rgb()`: opens PNG, converts to RGB via PIL `.convert("RGB")`
- Implemented `merge_instance_masks()`: merges all per-nucleus instance masks into one binary mask using `np.maximum()`; mismatched masks resized with NEAREST interpolation before merging
- Implemented `resize_image()`: PIL BILINEAR → (256, 256)
- Implemented `resize_mask()`: PIL NEAREST → (256, 256) — preserves binary values
- Implemented `binarise_mask()`: threshold > 127 → float32 output ∈ {0.0, 1.0}
- Implemented `normalise_image()`: divide by 255.0 → float32 output ∈ [0.0, 1.0]
- Implemented `preprocess_sample()`: full pipeline returning image shape (3, 256, 256) float32 and mask shape (256, 256) float32
- Applied full preprocessing pass over all 670 samples with per-sample error handling
- Saved preprocessed arrays as compressed NPZ files:
  - `train_data.npz`: images (469, 3, 256, 256) float32 + masks (469, 256, 256) float32 + sample_ids
  - `val_data.npz`: images (100, 3, 256, 256) float32 + masks (100, 256, 256) float32 + sample_ids
  - `test_data.npz`: images (101, 3, 256, 256) float32 + masks (101, 256, 256) float32 + sample_ids
- Implemented `NucleiDataset` (PyTorch Dataset class) in `src/dataset.py`
- Implemented `get_dataloaders()` using `configs/config.yaml`
- Train/Val/Test split: 70% / 15% / 15% (stratified by nucleus count quartile, seed=42)
- Confirmed: Train = 469, Val = 100, Test = 101 samples
- Split IDs saved to `data/splits/train_ids.txt`, `val_ids.txt`, `test_ids.txt`

### Key Decisions (Student A)
| Decision | Reason |
|----------|--------|
| Resize to 256×256 | Most common native size; divisible by 2⁴=16 (UNet depth-4 requirement) |
| Convert all channels → RGB | Dataset contains mixed channel types; `.convert("RGB")` handles all three without branching |
| Normalise by dividing by 255 | UNet trained from scratch — ImageNet mean/std not meaningful for microscopy data |
| Threshold mask at > 127 | Defensive binarisation step |
| Merge instance masks → binary | Task is binary segmentation; masks merged via `np.maximum()` |
| Split 70/15/15 stratified by nucleus count quartile | Balanced distribution across all three subsets |
| Save split IDs to `.txt` files | Prevents re-splitting and data leakage in downstream notebooks |
| Nearest-neighbour for mask resize | Preserves binary {0, 255} values |

### Issues Encountered (Student A)
| Issue | Solution |
|-------|----------|
| Dataset contains mixed channel types | `.convert("RGB")` applied uniformly regardless of actual channel count |
| Image sizes vary — cannot batch directly | Resize all to fixed 256×256 before batching |
| Mask channel dimension (H, W) vs DataLoader expectation (1, H, W) | NPZ stores masks as (N, H, W); DataLoader adds channel dim at collation |

---

### Student B — Environment Setup + Config + Model + Utilities + Training Loop

#### Environment Setup
- Cloned repository and set up local development environment
- Installed all libraries and verified versions
- Set up VS Code for development
- Fixed `requirements.txt` — added `torchsummary`

#### Config — `configs/config.yaml`
- Wrote full config file with all hyperparameters documented
- Sections: data paths, model settings, training settings, loss weights, paths, evaluation threshold

#### Model — `src/model.py`
- Implemented `DoubleConv` block: Conv2d → BatchNorm → ReLU × 2
- Implemented `UNet`: 4-level encoder (64→128→256→512), bottleneck (1024), 4-level decoder with skip connections
- Output: 1×1 Conv + Sigmoid → binary probability mask
- Verified with sanity check: input (2, 3, 256, 256) → output (2, 1, 256, 256)
- Total parameters: ~31M

#### Utilities — `src/utils.py`
- Implemented `plot_training_curves(history, figures_dir)` — saves training/validation curves
- Implemented `save_predictions(model, loader, device, save_dir)` — saves visual predictions
- Implemented `save_metrics(metrics_dict, figures_dir)` — saves metrics to `metrics.txt`

#### Training Loop — `src/train.py`
- Implemented `BCEDiceLoss`: combined BCE + Dice loss with configurable weights from config
- Implemented `train_one_epoch()` — runs one full pass over the training set, returns average loss
- Implemented `validate()` — returns average loss, Dice, IoU on validation set
- Implemented full `train()` pipeline — saves best model checkpoint by Dice score
- Optimizer: Adam (lr=0.0001), Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)

### Key Decisions (Student B)
| Decision | Reason |
|----------|--------|
| ConvTranspose2d for upsampling | Learnable upsampling — better boundary recovery than bilinear |
| BatchNorm after every Conv2d | Stabilises training, allows higher learning rate |
| Save best model by Dice score | Dice directly measures segmentation quality |
| BCEDiceLoss | BCE handles stable gradients; Dice handles 13.9% foreground class imbalance |
| ReduceLROnPlateau on val_loss | Reduces LR automatically when validation stops improving |

### Issues Encountered (Student B)
| Issue | Solution |
|-------|----------|
| `torchsummary` missing from requirements.txt | Added to requirements.txt and pushed fix |

---

### Student C — Evaluation Metrics & Baseline — `src/evaluate.py`

- Implemented `dice_score(pred, target)` — Dice Similarity Coefficient
- Implemented `iou_score(pred, target)` — Intersection over Union
- Implemented `otsu_predict(image_path)` — classical Otsu thresholding baseline
- Implemented `evaluate_otsu(image_paths, mask_paths)` — evaluation function ready to run against ground truth masks

### Key Decisions (Student C)
| Decision | Reason |
|----------|--------|
| Otsu as baseline | Classical intensity-based method — no training required; quantifies UNet advantage |
| threshold=0.5 for metrics | Standard binary decision boundary for segmentation evaluation |
| smooth=1.0 in metrics | Prevents division by zero on empty predictions |

---

## Day 3 — May 14, 2026 | EDA Upload + Training Run + Results

---

### Student A — EDA Notebook Corrections & Upload

> The EDA notebook (`01_EDA_Final_v6.ipynb`) was completed locally on Day 2 but intentionally held back from GitHub after identifying several issues during self-review.

#### Corrections Applied Before Upload
| Issue identified | Correction made |
|-----------------|----------------|
| Initial scan claimed all 670 images are RGBA — contradicted by DSB 2018 known composition | Removed hardcoded claim; Cell 14 now reports channel distribution dynamically |
| Normalisation decision said "ImageNet mean/std" — contradicted preprocessing notebook | Changed to divide by 255.0 |
| Outlier threshold comment said "exactly 5%" — incorrect with exclusive `>` operator | Removed "exactly"; comment now reads "strictly above P95" |
| `channel_freq` variable used without guard — raises `NameError` if Section 6 not run | Added `if 'channel_freq' in vars()` guard |
| `_target_str` re-defined locally in Cell 12 — redundant after Cell 5 | Removed local re-definition |

- Applied all 5 corrections to `01_EDA_Final_v6.ipynb`
- Re-ran all cells end-to-end — no errors
- Confirmed all 8 figures saved to `reports/figures/`:
  1. `eda_size_distribution.png`
  2. `eda_channel_types.png`
  3. `eda_intensity_analysis.png`
  4. `eda_nucleus_counts.png`
  5. `eda_mask_coverage.png`
  6. `eda_visual_samples.png`
  7. `eda_instance_masks.png`
  8. `eda_split_distribution.png`
- Confirmed split files present: `train_ids.txt` (469), `val_ids.txt` (100), `test_ids.txt` (101)
- Pushed `01_EDA_Final_v6.ipynb` to GitHub

### Issues Encountered (Student A)
| Issue | Solution |
|-------|----------|
| Notebook held back on Day 2 due to multiple inconsistencies | Reviewed and corrected all issues before upload |

---

### Student B — Full Training Run & Results

- Ran full training pipeline via `04_Training_Pipeline.ipynb` on Google Colab GPU
- Mounted Google Drive, loaded `train_data.npz` and `val_data.npz`
- Trained UNet for 50 epochs — logged Train Loss / Val Loss / Dice / IoU per epoch
- Best model saved to `checkpoints/best_model.pth`
- Pushed results to `reports/figures/`:
  - `training_curves.png`
  - `predictions.png`
  - `metrics.txt`
- Fixed `config.yaml` splits: replaced `val_split: 0.2` with correct `train_split: 0.70`, `val_split: 0.15`, `test_split: 0.15`
- Pushed `04_Training_Pipeline.ipynb` to GitHub

### Final Training Results
| Metric   | Value  |
|----------|--------|
| val_dice | 0.9180 |
| val_iou  | 0.8492 |
| val_loss | 0.1679 |

### Issues Encountered (Student B)
| Issue | Solution |
|-------|----------|
| `evaluate.py` had non-breaking spaces (U+00A0) causing `IndentationError` | Added fix cell in `04_Training_Pipeline.ipynb` to replace with regular spaces before training |
| NPZ filenames in `dataset.py` did not match files saved by NB02 | Updated to `train_data.npz`, `val_data.npz`, `test_data.npz` |
| `val_split: 0.2` in config did not match actual 70/15/15 split | Fixed config to correct split values |

---

## Day 4 — May 16, 2026 | Repository Cleanup & Documentation Fixes

**Contributor: Student B**

### Completed
- Fixed `.gitignore` — removed global `*.png` block; now only ignores `eda_*.png` and `prep_*.png`; training results tracked explicitly
- Fixed `README.md`:
  - Removed incorrect UNet++ reference from overview
  - Added missing notebooks to project structure
  - Fixed train command from `python src/train.py` to `python -m src.train`
  - Filled results table with real training results (Dice 0.9180, IoU 0.8492)
  - Added team contributions table with Student A, B, C roles
- Fixed `LOG.md`:
  - Fixed character encoding (UTF-8)
  - Separated contributions clearly by student and day

### Issues Encountered
| Issue | Solution |
|-------|----------|
| LOG.md had broken encoding (UTF-8 corruption) | Rewrote file directly on GitHub with correct encoding |
| README referenced UNet++ which is not implemented | Corrected to reflect actual UNet-only implementation |

## Day 5 — June 6, 2026 | Data Augmentation

**Contributor: Student A**

### Completed
- Implemented full augmentation pipeline in `notebooks/03_Augmentation.ipynb`
- Defined 12-transform Albumentations pipeline covering geometric, photometric, and regularisation augmentations
- Geometric transforms (applied to image + mask): HorizontalFlip, VerticalFlip, RandomRotate90, ShiftScaleRotate, ElasticTransform, GridDistortion
- Photometric transforms (image only): RandomBrightnessContrast, RandomGamma, GaussNoise, GaussianBlur, CLAHE
- Regularisation: CoarseDropout with mask_fill_value=0
- Excluded colour jitter and channel shuffle — staining colour carries diagnostic meaning in fluorescence microscopy
- Augmented training set: 469 → 2,814 samples (N_AUG=5 copies + originals)
- Saved augmented arrays to `data/train_aug.npz`

### Key Decisions
| Decision | Reason |
|----------|--------|
| Albumentations over torchvision transforms | Native mask propagation via `additional_targets` — no manual sync needed |
| N_AUG=5 (×6 total) | Balances dataset size vs training time for UNet from scratch on 469 samples |
| Geometric transforms applied to both image and mask | Pixel-level alignment must be preserved for segmentation |
| Colour jitter excluded | Staining colour carries diagnostic meaning in fluorescence and brightfield microscopy |
| CoarseDropout mask_fill_value=0 | Bounded label noise ≤0.8% per sample; accepted cost for occlusion-robustness |
| Post-aug mask binarisation ≥0.5 | ElasticTransform/GridDistortion bilinear interpolation introduces sub-integer values |

### Issues Encountered
| Issue | Solution |
|-------|----------|
| A.CLAHE raises TypeError on float32 input | Wrapped in A.Lambda with custom float32↔uint8 converter |
| Notebook rendered as "Invalid Notebook" on GitHub | Fixed missing `state` key in `metadata.widgets` and set `nbformat_minor=4` |
---

## Day 6 — June 8, 2026 | Ablation Experiments Setup

**Contributor: Student B**

### Completed

- Added experiments section to `04_Training_Pipeline.ipynb`
- Set up shared experiment infrastructure (imports, EXPERIMENT_EPOCHS=30, shared dataloaders)
- Ran Experiment 1 — Learning Rate Comparison (lr: 0.001, 0.0001, 0.00001)
- Ran Experiment 2 — Loss Function Comparison (BCE only vs BCE + Dice)

### Experiment Results (Day 6)

| Experiment | Settings Tested | Best Setting | Best Dice |
|---|---|---|---|
| 1 — Learning Rate | 0.001, 0.0001, 0.00001 | lr=0.0001 | 0.9123 |
| 2 — Loss Function | BCE only, BCE + Dice | BCE + Dice | 0.9134 |

### Key Decisions

| Decision | Reason |
|---|---|
| 30 epochs per experiment | Enough to compare settings without wasting GPU time |
| One variable changed per experiment | Clean ablation — isolates effect of each decision |

### Issues Encountered

| Issue | Solution |
|---|---|
| Colab session ended before all experiments completed | Re-ran the full notebook from the beginning in a new session |

---

## Day 7 — June 9, 2026 | Augmentation Experiment & Final Test Evaluation

**Contributor: Student B**

### Completed

- Ran Experiment 3 — Augmentation Comparison (without vs with augmentation from NB03)
- Loaded Student A's augmented data from Drive: 469 → 2,814 training samples
- Evaluated final model on held-out test set (101 images): Dice 0.9235, IoU 0.8591
- Saved all experiment charts to `reports/figures/`
- Updated `04_Training_Pipeline.ipynb` with full experiments section and final evaluation
- Updated `README.md` results table with final test numbers
- Updated `LOG.md` with all Student B entries

### Experiment Results (Day 7)

| Experiment | Settings Tested | Best Setting | Best Dice |
|---|---|---|---|
| 3 — Augmentation | Without, With (2,814 samples) | With augmentation | 0.9185 |

### Final Test Set Results

| Metric | Value |
|---|---|
| Test Dice | 0.9235 |
| Test IoU | 0.8591 |

### Key Decisions

| Decision | Reason |
|---|---|
| Augmented data from Student A's NB03 | 469 → 2,814 samples improves generalization |
| Final evaluation on held-out test set | Unbiased measure of model performance |

### Issues Encountered

| Issue | Solution |
|---|---|
| Some output files not persisted after session restart | Re-ran the notebook and re-uploaded results to GitHub |
