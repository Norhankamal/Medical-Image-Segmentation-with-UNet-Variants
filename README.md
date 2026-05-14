# UNet-Based Segmentation for Small Medical Datasets

> Binary segmentation of cell nuclei using UNet variants on the 2018 Kaggle Data Science Bowl dataset.

---

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Dataset](#dataset)
- [Usage](#usage)
- [Results](#results)
- [References](#references)

---

## Overview

This project implements UNet-based binary segmentation of cell nuclei in microscopy images.
We compare a classical Otsu thresholding baseline against UNet and UNet++ architectures,
evaluated using Dice coefficient and Intersection over Union (IoU).

**Dataset:** 2018 Kaggle Data Science Bowl (670 labeled microscopy images)  
**Task:** Binary segmentation (nucleus vs. background)  
**Out of scope:** 3D volumetric segmentation

---

## Project Structure

```
medical-segmentation/
├── README.md
├── LOG.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   └── config.yaml                     ← All hyperparameters in one place
│
├── src/
│   ├── dataset.py                      ← PyTorch Dataset class + DataLoaders
│   ├── model.py                        ← UNet architecture
│   ├── train.py                        ← Training loop
│   ├── evaluate.py                     ← Metrics + Otsu baseline
│   └── utils.py                        ← Shared helper functions
│
├── notebooks/
│   ├── 01_EDA_Final_v6.ipynb           ← Exploratory Data Analysis
│   └── 02_Preprocessing.ipynb          ← Preprocessing pipeline
│
├── data/
│   ├── splits/                         ← train/val/test split ID files (.txt)
│   └── README.md                       ← Download instructions (no raw data here)
│
├── reports/
│   └── figures/                        ← All plots used in reports
│
└── checkpoints/
    └── .gitkeep                        ← Folder tracked, weights ignored
```

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/Norhankamal/Medical-Image-Segmentation-with-UNet-Variants.git
cd Medical-Image-Segmentation-with-UNet-Variants
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download dataset
See [data/README.md](data/README.md) for full instructions.

---

## Dataset

**Source:** [2018 Kaggle Data Science Bowl](https://www.kaggle.com/competitions/data-science-bowl-2018/data)

- 670 training samples
- 9 distinct image sizes (256×256 to 1388×1040); most common: 256×256 (49.9%)
- Mixed channel types (Grayscale / RGB / RGBA) — all converted to RGB in preprocessing
- Nuclei per image: mean=44, median=27, range=[1, 375]
- Class imbalance: ~13.9% foreground / ~86.1% background (Dice Loss used to address this)
- Instance-level nucleus masks provided per sample — merged to a single binary mask for this task
- Train / Val / Test split: 469 / 100 / 101 (70/15/15, stratified by nucleus count, seed=42)
- Preprocessed arrays saved as compressed NPZ: images `(N, 3, 256, 256)` float32, masks `(N, 256, 256)` float32

---

## Usage

### Run Otsu baseline
```bash
python src/evaluate.py
```

### Train UNet
```bash
python src/train.py
```

All hyperparameters are in `configs/config.yaml`.

---

## Results (Updated Incrementally)

| Method | Subset | Dice ↑ | IoU ↑ |
|--------|--------|--------|-------|
| Otsu Thresholding (Baseline) | Test (101) | TBD | TBD |
| UNet | Test (101) | TBD | TBD |
| UNet++ | Test (101) | TBD | TBD |

---

## References

1. Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation," MICCAI 2015. https://arxiv.org/abs/1505.04597  
2. Zhou et al., "UNet++: A Nested U-Net Architecture for Medical Image Segmentation," DLMIA 2018. https://arxiv.org/abs/1807.10165
