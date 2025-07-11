# 📦 Deep Learning Ensemble for Automated Pallet Counting

This repository provides a real-time solution for **automated pallet counting** in operational warehouses using deep learning ensemble models. Our system predicts the **total**, **CHEP**, and **EPAL** pallet counts from RGB images through a **multi-target regression** approach with **explainable AI**.


## 🔍 Overview

Manual pallet counting is still widely used despite being error-prone and labor-intensive. We propose an **ensemble of CNN backbones** to replace this with a vision-based system that is:

- **Accurate**: MAE as low as 1.15 pallets
- **Fast**: Inference in <75ms on commodity hardware
- **Explainable**: Grad-CAM for model transparency

## 📊 Key Results

| Model           | MAE (Total) | MAE (CHEP) | MAE (EPAL) | R²       |
|----------------|-------------|------------|------------|-----------|
| EfficientNet-B3| 1.401       | 0.991      | 1.009      | 0.840     |
| ResNet-50      | 1.918       | 1.682      | 1.475      | 0.729     |
| ConvNeXt-Tiny  | 1.696       | **0.721**  | 1.426      | 0.702     |
| **Ensemble**   | **1.151**   | 0.918      | **0.857**  | **0.875** |

## 🧠 Model Architecture

- **Backbones**: EfficientNet-B3, ResNet-50, ConvNeXt-Tiny
- **Task**: Multi-target regression for `[CHEP, EPAL]` with total = CHEP + EPAL
- **Loss**: Smooth L1
- **Explainability**: Grad-CAM visualizations on final conv layers

## 🛠 Training & Augmentation

- Framework: PyTorch
- Image size: 224×224
- Augmentations: Crop, flip, affine, color jitter, blur, compression, dropout
- Optimizer: AdamW with One-Cycle LR

## 📁 Dataset

A proprietary dataset of **112 real-world warehouse images** with manual annotations. Due to privacy concerns, the dataset is **not publicly available**. Contact for potential access or collaboration.

## 📂 Repository Structure
├── CNN-Pallet.ipynb 
├── dataset/ 
│   ├── images/
│   └── labels.csv
└── README.md 


## 📈 Explainable AI

Grad-CAM visualizations show attention regions for each model:
- EfficientNet focuses on stack edges and shadows
- ResNet highlights CHEP markings and textures
- ConvNeXt captures broader stack areas with robustness to lighting

## 💡 Future Work

- Expand dataset across multiple facilities
- Integrate RGB-D or stereo vision for depth-aware counting
- Explore temporal modeling with video streams




