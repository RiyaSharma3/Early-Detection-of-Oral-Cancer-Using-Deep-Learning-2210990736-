# Early Detection of Oral Cancer Using Deep Learning on Intraoral Images

> **Riya Sharma** — Department of Computer Science and Engineering, Chitkara University, Rajpura, India
> `riya736.be22@chitkara.edu.in`

---

## 📋 Project Overview

This repository contains the complete source code, preprocessing pipeline, and model training scripts for automated binary classification of oral lesion images into **cancerous** and **non-cancerous** categories using deep convolutional neural networks (CNNs).

Three CNN architectures — **ResNet50**, **DenseNet121**, and **EfficientNetB0** — were evaluated under a standardised two-phase transfer learning protocol. Three ensemble strategies were further applied on top of the trained backbones. The best ensemble (Weighted Average & Logistic Regression Meta-Learner) achieved a test AUC of **0.9194**, outperforming all individual models.

> 📄 Full methodology and results are documented in the accompanying research paper:
> **"Early Detection of Oral Cancer Using Deep Learning on Intraoral Images"**

---

## 👥 Team Details

| Name | Roll Number | Email |
|------|-------------|-------|
| Riya Sharma | *(Roll No.)* | riya736.be22@chitkara.edu.in |

**Project Type:** Research Paper
**Current Status:** Submitted for IPR / Conference Publication

---

## 📁 Repository Structure

```
Project Title (Roll No of Team Members)/
│
├── IPR Submission Proof/
│   ├── research_paper.pdf              # Full research paper
│   └── submission_screenshot.png       # Screenshot of submission
│
├── Report and PPT/
│   ├── Project_Report.pdf              # Detailed project report
│   └── Project_Presentation.pptx      # Presentation slides
│
├── Source Code/
│   ├── ResNet50.ipynb                  # ResNet50 training & evaluation
│   ├── DenseNet121.ipynb               # DenseNet121 training & evaluation
│   ├── EfficientNetB0.ipynb            # EfficientNetB0 training & evaluation
│   └── Ensemble.ipynb                  # Ensemble methods (Weighted Avg, LR Meta-Learner, Keras Meta-Head)
│
└── README.md                           # This file
```

---

## 🧠 Methodology Summary

### Dataset
| Partition | Cancer | Non-Cancer | Total |
|-----------|--------|------------|-------|
| Training (pre-augmentation) | 355 | 360 | 715 |
| Validation | 89 | 90 | 179 |
| Test (external, held-out) | 87 | 44 | 131 |
| Training (post-augmentation) | 1,775 | 1,800 | 3,575 |

- **Training source:** [Oral Cancer Dataset v2.0 — Zaid, Kaggle](https://www.kaggle.com/datasets/zaidpy/oral-cancer-dataset)
- **Test source:** [Oral Cancer Lips and Tongue Images — Shivam, Kaggle](https://www.kaggle.com/datasets/shivam17299/oral-cancer-lips-and-tongue-images)
- Class balancing: 300 cancer images randomly removed → perfectly balanced 700/700 corpus

### Preprocessing Pipeline
1. **BGR → RGB** colour space conversion
2. **CLAHE** on the luminance channel (clip limit 2.0, tile grid 8×8)
3. **Gaussian Sharpening** via unsharp masking (σ = 2.0)
4. **Centre-Crop Resizing** to 224×224 pixels
5. **Architecture-specific normalisation** (ImageNet statistics)

### Data Augmentation (Training Only)
Offline augmentation via [Albumentations](https://albumentations.ai/) — 4 additional copies per training image:
- Horizontal & Vertical Flip
- Random 90° and Continuous Rotation (±25°)
- Random Brightness/Contrast (±0.25)
- Hue, Saturation, Value Adjustment
- Gaussian Noise, CLAHE, Grid Distortion

### Transfer Learning Protocol
| Phase | Description | Epochs | LR |
|-------|-------------|--------|----|
| Phase 1 | Head-only training (backbone frozen) | Up to 20 | 1×10⁻³ |
| Phase 2 | Fine-tuning (top layers unfrozen) | Up to 30 | 1×10⁻⁵ |

Model selection criterion: `Score = 0.6 × AUC + 0.4 × Accuracy`

---

## 📊 Results

### Individual Model Performance (External Test Set, Optimal Threshold)

| Metric | DenseNet121 | EfficientNetB0 | ResNet50 |
|--------|-------------|----------------|----------|
| Optimal Threshold | 0.90 | 0.27 | 0.84 |
| Accuracy | 83% | 88% | **90%** |
| Cancer Precision | 0.90 | 0.93 | 0.92 |
| Cancer Recall | 0.84 | 0.89 | **0.93** |
| Cancer F1 | 0.87 | 0.91 | **0.93** |
| Non-Cancer F1 | 0.77 | 0.83 | **0.85** |
| AUC | 0.8685 | 0.8977 | **0.9079** |
| Parameters | ~7.3M | ~5.3M | ~24.2M |

### Ensemble Methods Performance (External Test Set)

| Model | Type | Test AUC |
|-------|------|----------|
| DenseNet121 | Individual backbone | 0.8685 |
| EfficientNetB0 | Individual backbone | 0.9050 |
| ResNet50 | Individual backbone | 0.9079 |
| Keras Meta-Head | Ensemble (Method C) | 0.8823 |
| **Weighted Average** | **Ensemble (Method A)** | **0.9194** |
| **LR Meta-Learner** | **Ensemble (Method B)** | **0.9194** |

---

## 🛠️ Tech Stack

| Component | Library / Tool |
|-----------|---------------|
| Deep Learning Framework | TensorFlow 2.x / Keras |
| Image Processing | OpenCV |
| Augmentation | Albumentations |
| Evaluation & ML | scikit-learn |
| Visualisation | Matplotlib, Seaborn |
| Interpretability | GradCAM (manual implementation) |
| Environment | Google Colab (GPU) |
| Data Access | kagglehub |

---

## ⚙️ Setup & Usage

### Prerequisites

```bash
pip install tensorflow opencv-python albumentations scikit-learn matplotlib seaborn tqdm kagglehub
```

### Running the Notebooks

All notebooks are designed to run on **Google Colab** with GPU acceleration. Open the relevant `.ipynb` file and run cells sequentially.

**Step-by-step workflow in each notebook:**
1. Install dependencies and import libraries
2. Download datasets via `kagglehub`
3. Balance and split the dataset (80:20 stratified split)
4. Apply CLAHE → Sharpening → Centre-Crop preprocessing and save to disk
5. Perform offline augmentation on the training partition (×4 copies)
6. Train Phase 1 (frozen backbone) → save best checkpoint
7. Train Phase 2 (fine-tune top layers) → save best checkpoint
8. Evaluate on the external test set at default and optimal Youden-index thresholds
9. Generate GradCAM visualisations (DenseNet121 notebook)
10. Run ensemble methods (Ensemble notebook)

### Key Hyperparameters

| Hyperparameter | Value |
|----------------|-------|
| Input Resolution | 224 × 224 × 3 |
| Batch Size | 32 |
| Phase 1 LR | 1×10⁻³ |
| Phase 2 LR | 1×10⁻⁵ |
| Early Stopping Patience | 12 epochs |
| LR Reduction Patience | 5 epochs |
| Augmentation Copies | 4 per image |
| Optimiser | Adam |
| Loss | Binary Cross-Entropy |

---

## 📖 Citation

If you use this code or research in your work, please cite:

```
Riya Sharma, "Early Detection of Oral Cancer Using Deep Learning on Intraoral Images,"
Department of Computer Science and Engineering, Chitkara University, Rajpura, India.
```

---

## 🙏 Acknowledgements

The authors gratefully acknowledge the contributors of the publicly available oral cancer datasets hosted on the Kaggle platform — [Zaid](https://www.kaggle.com/datasets/zaidpy/oral-cancer-dataset) and [Shivam](https://www.kaggle.com/datasets/shivam17299/oral-cancer-lips-and-tongue-images) — which made this research possible.

---

## 📬 Contact

For queries related to this project, reach out via:
- **Email:** riya736.be22@chitkara.edu.in
- **Institution:** Chitkara University, Rajpura, Punjab, India
