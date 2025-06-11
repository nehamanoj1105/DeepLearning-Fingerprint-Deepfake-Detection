# Biometric Spoof Detection: A Deep Learning Approach to Fingerprint Deepfakes

## Problem Statement

Fingerprint recognition systems are increasingly targeted by deepfake and spoofing attacks using silicone molds, 3D prints, or AI-generated images. Detecting these attacks is challenging due to their realistic nature. Traditional machine learning models relying on handcrafted features often overfit and fail to generalize. This project focuses on deep learning approaches that can learn complex patterns directly from fingerprint images for more accurate and robust spoof detection.



## Dataset Collection

The fingerprint datasets used are too large to host directly on GitHub. Please download them from the Google Drive link below:

**Download Datasets:** [Datasets (Google Drive)](https://drive.google.com/drive/folders/19QanCCAnysk-3JebsYqcXtaaZCz-RJcj?usp=drive_link)

### Dataset Contents

- `original_dataset.zip` – Raw fingerprint images
- `resized(224x224).zip` – Resized images for deep learning models
- `normalised.zip` – Normalized images for ML models
- `Generated_Dataset.zip` – GAN-generated fingerprint spoof images
- `Generated_Resized_Normalised/` – Final processed dataset used for training

## Preprocessing

Fingerprint images were processed using the following steps:

1. **Resizing**: All images resized to 224x224 using `resize.py`
2. **Normalization**: Pixel values normalized using `normalise.py`
3. **Feature Extraction**: Gabor, LBP, or texture features extracted using `extract_fingerprint_features.py` for ML models

## Traditional Machine Learning Models

We used traditional ML classifiers :

- `svm.py`: Implements Support Vector Machine classifier
- `xgboost_fingerprint.py`: Implements XGBoost classifier

Limitations:
- Overfitting observed due to low feature diversity
- Poor generalization to unseen spoofing techniques

## Deep Learning Models

Given the complexity of biometric spoof detection, deep learning is more effective. The following models were implemented:

| Model             | Strengths                             | Type               |
|------------------|----------------------------------------|--------------------|
| SimpleCNN         | Lightweight, fast                      | Shallow CNN        |
| ResNet-18         | Robust deep feature extraction         | Residual CNN       |
| EfficientNet-B0   | High accuracy with efficient scaling   | Optimized CNN      |
| DEIT              | Captures global context and structure  | Transformer Model  |

## Ensemble Strategy

Extensive ensemble combinations between ResNet18, EfficientNetB0, and DeiT-Small were implemented:

**Pairwise Ensembles:**

- ResNet18 + EfficientNetB0  
- ResNet18 + DeiT  
- EfficientNetB0 + DeiT  

**Final Ensemble:**  
Combines outputs of all three models (ResNet18, EfficientNetB0, DeiT) using softmaxing for improved spoof detection accuracy.


## Challenges Faced

- Overfitting in ML models due to limited feature representation
- Dataset-specific cues affecting model generalization
- Need for models that can detect evolving spoofing methods
- DL models still risk learning superficial patterns without diverse data

## Key Takeaways

1. Fingerprint spoof detection is a complex, high-stakes problem in biometric security.
2. Classical ML models struggle to generalize across spoof types.
3. Deep learning models perform better due to their ability to learn discriminative patterns.
4. Combining CNNs and ViTs provides both local and global feature understanding.
5. Dataset diversity and augmentation are critical for robust spoof detection.



