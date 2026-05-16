# UNet-Based Segmentation for Small Medical Datasets
> Binary segmentation of cell nuclei using UNet on the 2018 Kaggle Data Science Bowl dataset.

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
We compare a classical Otsu thresholding baseline against a UNet architecture,
evaluated using Dice coefficient and Intersection over Union (IoU).

**Dataset:** 2018 Kaggle Data Science Bowl (670 labeled microscopy images)  
**Task:** Binary segmentation (nucleus vs. background)  
**Out of scope:** 3D volumetric segmentation

---

## Project Structure
