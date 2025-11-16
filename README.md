# Magnetic Tile Defect Segmentation

This project focuses on pixel-level segmentation of surface defects on magnetic tiles using a UNet-based deep learning model.  
The objective is to automatically identify cracks, blowholes, frays, uneven surfaces, and related defects to support reliable visual inspection.

## 1. Project Overview

Each sample in the dataset includes:

- a **grayscale magnetic-tile image**, and  
- a **ground-truth mask** marking the defect region.

The model learns this relationship and predicts a binary mask for new images, helping automate defect inspection and reduce manual verification.

## 2. Dataset Pipeline

The dataset is organized by defect category:

data/raw/images/

MT_Blowhole/

MT_Crack/

MT_Fray/

MT_Uneven/


Inside each folder:

- **.jpg** - grayscale tile surface  
- **.png** - binary defect mask  

### Pipeline Steps

1. Scan all defect folders  
2. Pair each `.jpg` with its matching `.png`  
3. Convert images to tensors and resize them  
4. Feed them into the UNet model for training and evaluation  

This ensures consistent preprocessing across all defect classes.

## 3. Model: UNet

A UNet architecture is used because it:

- captures fine-grained surface texture  
- preserves spatial information through skip connections  
- performs well for pixel-level segmentation  
- is widely adopted in industrial defect detection tasks  

UNet serves as a suitable model for grayscale manufacturing inspection images.

## 4. Sample Predictions

For each tested image, the model generates:

- **Raw Input** – grayscale tile  
- **Predicted Mask** – binary defect region  
- **Overlay Output** – mask highlighted in red  
- **Side-by-Side View** – comparison of input, mask, and overlay  

All samples are available in:

sample_results/

These examples make it easy to evaluate segmentation quality.


## 5. Summary

This project demonstrates:

- a clean dataset-pairing pipeline  
- a UNet segmentation model tailored for magnetic-tile defects  
- consistent prediction outputs for quality inspection  

It provides a practical starting point for transitioning from classical CV to ML-based segmentation in manufacturing workflows.


