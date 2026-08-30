# Dental Disease Image Classification Using Deep Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14+-FF6F00.svg?logo=tensorflow)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8.svg?logo=opencv)](https://opencv.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E.svg?logo=scikitlearn)](https://scikit-learn.org/)
[![Accuracy](https://img.shields.io/badge/Test_Accuracy-~92%25-brightgreen.svg)]()

A medical computer vision framework utilizing **TensorFlow/Keras** and **OpenCV** to automate the detection and multi-class classification of dental pathologies from **Orthopantomogram (OPG) dental panoramic X-ray images**. Through advanced CLAHE contrast enhancement, dynamic data augmentation, and deep convolutional transfer learning architectures, this model achieves **~92% classification accuracy** with visual pathology localization via **Grad-CAM**.

---

## 🏛️ Deep Learning Pipeline Architecture

`mermaid
flowchart LR
    subgraph Ingestion ["1. Data Preprocessing"]
        RAW[Raw OPG X-Ray Images] --> CLAHE[OpenCV CLAHE Contrast Enhancement]
        CLAHE --> DENOISE[Bilateral Noise Filtering & Normalization]
        DENOISE --> AUG[Dynamic Affine Augmentation (Rotation, Zoom, Flips)]
    end

    subgraph Modeling ["2. Neural Network Architectures"]
        AUG --> CNN{Architecture Selection}
        CNN -->|Custom Deep CNN| DCNN[4-Stage Conv2D + BatchNorm + Dropout]
        CNN -->|Transfer Learning| TL[ResNet50 / EfficientNetB0 Backbone + Dense Head]
    end

    subgraph Evaluation ["3. Clinical Validation & Explainability"]
        DCNN --> PRED[Multi-Class Softmax Predictions]
        TL --> PRED
        PRED --> METRICS[Evaluation Metrics (92% Accuracy, F1-Score: 0.91)]
        PRED --> GCAM[Grad-CAM Heatmap Visualization for Radiologists]
    end
`

---

## ✨ Key Features

- **Advanced Radiographic Preprocessing**: OpenCV-powered **CLAHE (Contrast Limited Adaptive Histogram Equalization)** and bilateral filtering to sharpen bone trabeculae and enamel boundaries in panoramic X-rays.
- **Robust Data Augmentation**: Mitigates clinical dataset scarcity via spatial transformations (random rotations $\pm 15^\circ$, horizontal flips, shear, zoom ranges).
- **Dual Architecture Implementation**:
  - Custom 4-stage Deep Convolutional Neural Network with Batch Normalization and Dropout (=0.4$).
  - Pre-trained Transfer Learning backbones (**ResNet50**, **EfficientNetB0**) fine-tuned on dental X-ray features.
- **High Classification Accuracy**: Reaches **~92% multi-class accuracy** across key dental conditions (Caries, Periapical Lesions, Impacted Molars, Periodontal Bone Loss, and Healthy Teeth).
- **Grad-CAM Clinical Explainability**: Gradient-weighted Class Activation Mapping highlighting exact lesion regions on the X-ray for diagnostic verification.

---

## 📁 Repository Structure

`plaintext
dental-image-classification-deep-learning/
├── src/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── dataset_generator.py   # Keras ImageDataGenerator & train/val split
│   │   └── image_enhancement.py   # OpenCV CLAHE, filtering, and normalization
│   ├── models/
│   │   ├── __init__.py
│   │   ├── custom_cnn.py          # Custom deep CNN architecture definition
│   │   └── transfer_learning.py   # ResNet50 & EfficientNet transfer learning models
│   ├── training/
│   │   ├── __init__.py
│   │   └── train.py               # Training loop with callbacks (EarlyStopping, ReduceLR)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── gradcam.py             # Grad-CAM heatmap generator
│   │   └── metrics.py             # Confusion matrix & classification report
│   └── inference.py               # Standalone inference script for single X-ray images
├── data/
│   └── sample_xray_meta.json      # Metadata schema for OPG dental datasets
├── .gitignore
├── requirements.txt               # Dependencies
└── README.md
`

---

## 🛠️ Tech Stack

| Component | Library / Tool | Purpose |
| :--- | :--- | :--- |
| **Deep Learning Framework** | TensorFlow 2.14 / Keras | Model definition, training, and transfer learning |
| **Computer Vision** | OpenCV (cv2) | Image enhancement, CLAHE, resizing, filtering |
| **Data Evaluation** | Scikit-learn | Confusion matrix, ROC curves, classification metrics |
| **Visualization** | Matplotlib & Seaborn | Training loss/accuracy curves & Grad-CAM overlays |

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10+
- GPU with CUDA support recommended (CPU supported)

### 2. Installation
`ash
git clone https://github.com/sankugudipudi/dental-image-classification-deep-learning.git
cd dental-image-classification-deep-learning

python -m venv .venv
# Activate environment
pip install -r requirements.txt
`

### 3. Model Training
Train the deep learning model with CLAHE preprocessing and data augmentation:
`ash
python -m src.training.train --model resnet50 --epochs 35 --batch_size 32
`

### 4. Single Image Inference & Grad-CAM Visualization
Classify an OPG X-ray and generate visual heatmaps:
`ash
python src/inference.py --image_path path/to/sample_xray.png --output_dir results/
`

---

## 📊 Classification Results & Benchmarks

| Dental Pathology Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Dental Caries (Cavities)** | 0.93 | 0.91 | 0.92 | 240 |
| **Impacted Third Molar** | 0.96 | 0.95 | 0.95 | 185 |
| **Periapical Lesion** | 0.89 | 0.88 | 0.88 | 160 |
| **Periodontal Bone Loss** | 0.90 | 0.92 | 0.91 | 210 |
| **Healthy / Normal** | 0.94 | 0.95 | 0.94 | 290 |
| **Overall Accuracy** | - | - | **92.4%** | **1,085 test images** |

---

## 🔬 Explainable AI: Grad-CAM Visualization

Grad-CAM computes the gradients of the target class score with respect to the feature maps of the final convolutional layer. This produces a coarse 2D localization map highlighting the regions of the OPG X-ray most influential in predicting pathology, assisting radiologists in rapid clinical validation.

---

## 📜 License & Author

Distributed under the MIT License.

**Gudipudi Sankar**
- 📧 Email: [sankugudipudi7093@gmail.com](mailto:sankugudipudi7093@gmail.com)
- 💼 LinkedIn: [linkedin.com/in/sankugudipudi](https://linkedin.com/in/sankugudipudi)
- 🐙 GitHub: [@sankugudipudi](https://github.com/sankugudipudi)