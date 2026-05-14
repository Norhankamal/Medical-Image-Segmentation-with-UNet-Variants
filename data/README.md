# Dataset — 2018 Kaggle Data Science Bowl

## Source

**Competition:** 2018 Kaggle Data Science Bowl — Nucleus Segmentation  
**Link:** <https://www.kaggle.com/competitions/data-science-bowl-2018/data>

---

## Download Instructions

1. Create a [Kaggle account](https://www.kaggle.com) and accept the competition rules
2. Download `stage1_train.zip`
3. Extract and place under `data/raw/`:

```
data/
└── raw/
    └── stage1_train/
        ├── {image_id}/
        │   ├── images/
        │   │   └── {image_id}.png
        │   └── masks/
        │       ├── {mask_id_1}.png
        │       └── {mask_id_2}.png
        └── ...
```

> Raw data is excluded from this repository via `.gitignore` (≈ 1 GB).  
> To reproduce results, download from the Kaggle link above and place in `data/raw/`.

---

## Dataset Statistics

> **Note:** All values below are printed dynamically by `notebooks/01_EDA_Final_v6.ipynb`
> (Section 4 scan + Section 5–11 analysis). The numbers here reflect the
> stage1\_train split of DSB 2018. Re-run NB01 to confirm exact values.

| Property | Value |
|:---|:---|
| Training samples | 670 |
| Image sizes | Multiple distinct sizes — most common: 256×256. Full distribution in NB01 §5 |
| Channel types | Mixed (Grayscale / RGB / RGBA) — exact counts in NB01 §6 scan output |
| Nuclei per image | Min=1, Max=375 — mean, median, std in NB01 §8 output |
| Mask coverage (foreground %) | Mean ≈ 13–14% — exact value in NB01 §9 output |
| Mask quality | All instance masks pass binary / empty / size-mismatch checks (NB01 §11) |
| Annotation type | Instance segmentation — one PNG per nucleus |
| Task | Binary segmentation (nucleus vs. background) |

---

## Preprocessing Applied

Full details and visual verification in `notebooks/02_Preprocessing.ipynb`.

| Step | Operation | Detail |
|:---:|:---|:---|
| 1 | Convert to RGB | `PIL .convert("RGB")` — handles Grayscale / RGB / RGBA uniformly |
| 2 | Merge instance masks | `np.maximum()` across all per-nucleus PNGs → single binary mask |
| 3 | Resize image | 256×256, bilinear interpolation |
| 4 | Resize mask | 256×256, nearest-neighbour (preserves binary {0, 255} values) |
| 5 | Binarise mask | threshold > 127 → float32 {0.0, 1.0} |
| 6 | Normalise image | divide by 255 → float32 [0.0, 1.0] (UNet trained from scratch) |
| 7 | Reorder axes | `np.transpose (H,W,C) → (C,H,W)` — PyTorch convention |

---

## Dataset Split

| Subset | Count | Ratio | Purpose |
|:---|:---:|:---:|:---|
| Train | ~469 | 70 % | Model training |
| Validation | ~100 | 15 % | Hyperparameter tuning |
| Test | ~101 | 15 % | Final evaluation only — not seen during training |

**Method:** Stratified two-step `train_test_split` (scikit-learn), seed = 42,
stratified by nucleus-count quartile (`pd.qcut(n_nuclei, q=4)`).

**Split ID files** are saved in `data/splits/` by `notebooks/01_EDA.ipynb`.  

> **Do not re-run `train_test_split`.**  
> Loading from the fixed `*_ids.txt` files guarantees that train / val / test
> sets are identical across all notebooks and scripts. Re-splitting constitutes
> data leakage and invalidates reported metrics.
