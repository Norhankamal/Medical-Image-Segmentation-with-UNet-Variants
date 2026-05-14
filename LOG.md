# Development Log
> **Project:** UNet-Based Segmentation for Small Medical Datasets
> **Dataset:** 2018 Kaggle Data Science Bowl — Nuclei Segmentation

---

## Day 1 — May 11, 2025 | Repository Setup & Project Structure

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

## Day 2 — May 13, 2025 | EDA (local, not yet pushed) + Preprocessing

**Commits:** `Add preprocessing pipeline`, `Add NucleiDataset class`, `Update LOG.md`

> **Note — EDA notebook:** The EDA notebook (`01_EDA_Final_v6.ipynb`) was completed locally on this day but **was NOT pushed to GitHub**. Several corrections were identified after the first run (channel type claim, normalisation decision, outlier threshold direction, and `channel_freq` dependency guard) and the notebook was kept locally for revision before upload. The corrected final version was uploaded on Day 3 (May 14, 2026). All EDA findings listed below reflect the final corrected notebook.

### Completed

#### Exploratory Data Analysis (EDA) — completed locally, uploaded Day 3

- Scanned all 670 training samples in a single pass to collect all statistics
- Found image sizes vary — 9 distinct sizes from 256×256 up to 1388×1040
- Most common size: 256×256 (334 images, 49.9%)
- Channel type distribution determined dynamically from full scan (see notebook Cell 14 output); DSB 2018 contains a mix of Grayscale, RGB, and RGBA images — all standardised to RGB via `.convert("RGB")`
- Computed nucleus count per image: mean=44.0 ±48.0, median=27, min=1, max=375
- Identified high-density images (nucleus count strictly above P95 threshold) for separate evaluation — exact count from scan output (Cell 18)
- Mask coverage: mean foreground = 13.9% ±11.1% — class imbalance confirmed
- Spearman correlation (nucleus count vs foreground coverage) computed — moderate positive correlation confirmed
- Checked all instance masks from scan: 0 non-binary, 0 empty, 0 size mismatches — all counts from Cell 26 output
- Visualised 6 stratified sample images (low / medium / high nucleus density) with ground truth masks and overlays
- 8 EDA figures generated and saved to `reports/figures/` (full list in Day 3 entry)

#### Preprocessing Pipeline — `notebooks/02_Preprocessing_Final_v3.ipynb`

- Implemented `load_image_as_rgb()`: opens PNG, converts to RGB via PIL `.convert("RGB")`
- Implemented `merge_instance_masks()`: merges all per-nucleus instance masks into one binary mask using `np.maximum()`; mismatched masks resized with NEAREST interpolation before merging
- Implemented `resize_image()`: PIL BILINEAR → (256, 256)
- Implemented `resize_mask()`: PIL NEAREST → (256, 256) — preserves binary values
- Implemented `binarise_mask()`: threshold > 127 → float32 output ∈ {0.0, 1.0}
- Implemented `normalise_image()`: divide by 255.0 → float32 output ∈ [0.0, 1.0]
- Implemented `preprocess_sample()`: full pipeline returning image shape (3, 256, 256) float32 and mask shape (256, 256) float32
- Verified pipeline with dry-run assertions: shape, dtype, value range, binary mask
- Applied full preprocessing pass over all 670 samples with per-sample error handling
- Saved preprocessed arrays as compressed NPZ files:
  - `train_data.npz`: images (469, 3, 256, 256) float32 + masks (469, 256, 256) float32 + sample_ids
  - `val_data.npz`:   images (100, 3, 256, 256) float32 + masks (100, 256, 256) float32 + sample_ids
  - `test_data.npz`:  images (101, 3, 256, 256) float32 + masks (101, 256, 256) float32 + sample_ids
- Verified reload: shapes, dtypes, value ranges, binary mask values confirmed post-load

#### Dataset Class

- Implemented `NucleiDataset` (PyTorch Dataset class) in `src/dataset.py`
- Implemented `get_dataloaders()` using `configs/config.yaml`
- Train/Val/Test split: 70% / 15% / 15% (stratified by nucleus count quartile, seed=42)
- Confirmed: Train = 469, Val = 100, Test = 101 samples
- Split IDs saved to `data/splits/train_ids.txt`, `val_ids.txt`, `test_ids.txt` — loaded (not re-generated) by all downstream notebooks to prevent data leakage
- Verified DataLoader output shapes: image `(8, 3, 256, 256)`, mask `(8, 1, 256, 256)` — the channel dimension is added by the DataLoader collation, not stored in the NPZ files

### Key Decisions
| Decision | Reason |
|----------|--------|
| Resize to 256×256 | Most common native size; divisible by 2⁴=16 (UNet depth-4 requirement); balances spatial detail and GPU memory |
| Convert all channels → RGB | Dataset contains mixed channel types (Grayscale/RGB/RGBA); `.convert("RGB")` handles all three without branching |
| Normalise by dividing by 255 (Option B) | UNet trained from scratch — no pre-trained encoder. ImageNet mean/std (Option A) not meaningful for fluorescence/brightfield microscopy data |
| Threshold mask at > 127 | Defensive binarisation step; all masks passed quality checks but threshold applied as a precaution |
| Merge instance masks → binary | Task is binary segmentation (nucleus vs background); individual instance masks merged via `np.maximum()` |
| Split 70/15/15 stratified by nucleus count quartile | Balanced nucleus count distribution across all three subsets — verified by similar means and coverage ratios |
| Save split IDs to `.txt` files | Prevents re-splitting and data leakage in downstream notebooks |
| Dice Loss over BCE | Class imbalance confirmed: mean foreground = 13.9%, background = 86.1% |
| Nearest-neighbour for mask resize | Preserves binary {0, 255} values — avoids sub-pixel grey artefacts from bilinear interpolation |
| Save preprocessed arrays as NPZ | Avoids repeated disk reads during training; format: images (N, 3, H, W) float32, masks (N, H, W) float32 |

### Issues Encountered
| Issue | Solution |
|-------|----------|
| Dataset contains mixed channel types (not uniformly RGBA as initially assumed) | Channel distribution determined dynamically from full scan; `.convert("RGB")` applied uniformly regardless of actual channel count |
| Masks passed all quality checks (all binary {0, 255}) | Applied threshold > 127 as defensive measure anyway |
| Image sizes vary — cannot batch directly | Resize all to fixed 256×256 before batching |
| Mask channel dimension (H, W) vs DataLoader expectation (1, H, W) | NPZ stores masks as (N, H, W); DataLoader adds channel dim at collation — documented in `NucleiDataset.__getitem__` |

### EDA Key Findings
| Finding | Value |
|---------|-------|
| Total training samples | 670 |
| Unique image sizes | 9 distinct (256×256 to 1388×1040) |
| Most common size | 256×256 — 334 images (49.9%) |
| Channel type | Mixed (Grayscale / RGB / RGBA) — exact distribution from scan output (NB01 Cell 14) |
| Average nuclei per image | 44.0 ± 48.0 (median = 27) |
| Min / Max nuclei | 1 / 375 |
| Mask foreground coverage | 13.9% ± 11.1% (class imbalance confirmed) |
| Total instance masks quality-checked | From scan output (NB01 Cell 26) — 0 non-binary, 0 empty, 0 size mismatches |
| High-density images (strictly above P95) | From scan output (NB01 Cell 18) — exact count and threshold value printed at runtime |
| Train / Val / Test split | 469 / 100 / 101 (70% / 14.9% / 15.1%) |

---

## Day 3 — May 14, 2026 | EDA Notebook — Corrections & Upload

**Commits:** `Upload corrected EDA notebook (01_EDA_Final_v6)`, `Update LOG.md`

### Context

The EDA notebook was completed locally on Day 2 but intentionally held back from GitHub after identifying several issues during self-review. The corrections listed below were applied before upload to ensure the notebook is accurate, internally consistent, and fully consistent with the preprocessing notebook already on GitHub.

### Corrections Applied Before Upload

| Issue identified | Correction made |
|-----------------|----------------|
| Initial scan claimed all 670 images are RGBA — contradicted by DSB 2018 known composition | Removed hardcoded "ALL RGBA" claim; Cell 14 now reports channel distribution dynamically from scan output |
| Normalisation decision said "ImageNet mean/std" — contradicted by preprocessing notebook which implements `/255.0` | Changed to Option B (divide by 255.0); rationale: UNet trained from scratch, ImageNet stats not applicable to fluorescence/brightfield microscopy |
| Outlier threshold comment said "exactly 5%" — incorrect with exclusive `>` operator on discrete data | Removed "exactly"; comment now reads "images strictly above P95 are flagged" |
| `channel_freq` variable used in Section 13 summary table without guard — raises `NameError` if Section 6 not run first | Added `if 'channel_freq' in vars()` guard with a clear fallback message |
| `_target_str` and `_target_w/_target_h` re-defined locally in Cell 12 — redundant after being declared as top-level constant in Cell 5 | Removed local re-definition; Cell 12 now reads `TARGET_SIZE` from Cell 5 directly via `_tw, _th = TARGET_SIZE` aliases |

### Completed on Day 3

- Applied all 5 corrections above to `01_EDA_Final_v6.ipynb`
- Re-ran all cells end-to-end — no errors, all outputs consistent with preprocessing notebook
- Confirmed all 8 figures saved correctly to `reports/figures/`:
  1. `eda_size_distribution.png`
  2. `eda_channel_types.png`
  3. `eda_intensity_analysis.png`
  4. `eda_nucleus_counts.png`
  5. `eda_mask_coverage.png`
  6. `eda_visual_samples.png`
  7. `eda_instance_masks.png`
  8. `eda_split_distribution.png`
- Confirmed split files present in `data/splits/`: `train_ids.txt` (469 IDs), `val_ids.txt` (100 IDs), `test_ids.txt` (101 IDs)
- Pushed `01_EDA_Final_v6.ipynb` to GitHub
- Updated `LOG.md` to reflect upload and document all corrections made since Day 2

### Issues Encountered
| Issue | Solution |
|-------|----------|
| Notebook held back on Day 2 due to channel type claim and normalisation conflict with preprocessing notebook | Reviewed both notebooks side by side; corrected all inconsistencies before upload — no conflicts remain |

---

