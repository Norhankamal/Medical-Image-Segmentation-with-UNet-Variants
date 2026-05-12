# Dataset — 2018 Kaggle Data Science Bowl

## Source
**Competition:** 2018 Kaggle Data Science Bowl — Nucleus Segmentation  
**Link:** https://www.kaggle.com/competitions/data-science-bowl-2018/data

---

## Download Instructions

1. Create a [Kaggle account](https://www.kaggle.com)
2. Go to the competition page and accept the rules
3. Download `stage1_train.zip`
4. Extract and place under `data/raw/`:

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

> Raw data is excluded from this repository via `.gitignore` due to size (~1GB).  
> Dataset is stored locally and on Google Drive for Colab access.  
> To reproduce results, download from the Kaggle link above and place in `data/raw/`.

---

## Dataset Statistics

| Property | Value |
|----------|-------|
| Training samples | 670 |
| Image sizes | Variable (256×256 to 1040×1040) |
| Channel types | RGB, Grayscale, RGBA (mixed) |
| Annotation type | Instance segmentation masks |
| Task | Binary segmentation (nucleus vs background) |

---

## Preprocessing Applied

See `notebooks/02_Preprocessing.ipynb` for full details.

1. Convert all images to RGB
2. Resize to 256×256
3. Normalize using ImageNet mean/std
4. Merge all instance masks → single binary mask (threshold > 127)
