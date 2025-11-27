# 🌿 Plant Disease Detection via Leaf Images

**Deep Learning system for automated plant disease classification with 95%+ accuracy**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🚀 Quick Start

**Choose your setup:**

- **🔥 Google Colab (Recommended):** See [COLAB_GUIDE.md](COLAB_GUIDE.md) - Train in ~3 hours with free GPU
- **💻 Local Setup:** See [Installation](#installation) below

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Quick Start](#quick-start)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Documentation](#documentation)

---

## 🎯 Project Overview

This project implements an end-to-end deep learning solution for **automated plant disease detection** from leaf images. Using transfer learning and state-of-the-art CNN architectures, the system achieves >95% accuracy in classifying 39 different plant disease categories.

### Key Highlights

- ✅ **4 Model Architectures**: Baseline CNN, ResNet50, EfficientNet-B3, MobileNetV2
- ✅ **Transfer Learning**: Pretrained on ImageNet, fine-tuned for plant diseases
- ✅ **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
- ✅ **Grad-CAM Visualizations**: Explainable AI showing what the model looks at
- ✅ **Production-Ready**: Streamlit web application with batch processing
- ✅ **Treatment Recommendations**: Disease information and treatment suggestions
- ✅ **CRISP-DM Methodology**: Systematic ML project workflow

---

## 📊 Dataset

**Source:** [PlantVillage Dataset](https://www.tensorflow.org/datasets/catalog/plant_village)

- **Total Images:** 55,447 high-quality leaf images
- **Number of Classes:** 39 (14 plant species, 26 diseases + healthy)
- **Format:** JPG/JPEG color images
- **Resolution:** Variable (typically 256x256 to 500x500)

### Plant Species Covered

- Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato

### Sample Class Distribution

```
Tomato___healthy: 5,539 images
Potato___Late_blight: 3,000 images
Pepper,_bell___Bacterial_spot: 1,913 images
...
Orange___Haunglongbing: 154 images (smallest class)
```

---

## ✨ Features

### 🧠 Machine Learning

- **Multiple Architectures**: Compare 4 different CNN models
- **Transfer Learning**: 2-phase training (feature extraction → fine-tuning)
- **Class Imbalance Handling**: Weighted CrossEntropyLoss
- **Data Augmentation**: 8+ augmentation techniques (flips, rotations, color jitter)
- **Early Stopping**: Prevent overfitting
- **TensorBoard Logging**: Real-time training monitoring

### 📈 Evaluation & Visualization

- **12+ Visualizations**:
  - Training/validation curves
  - Confusion matrix (39×39)
  - Per-class precision/recall/F1 bar charts
  - ROC curves
  - Grad-CAM heatmaps
  - Augmentation examples
  - Model comparison charts

### 🚀 Deployment

- **Streamlit Web App**:
  - Single image prediction
  - Batch processing
  - Treatment recommendations
  - Prediction history
  - Model metrics dashboard

---

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training)
- Google Colab Pro (for training notebooks)

### Option 1: Local Installation

```bash
# Clone repository
git clone https://github.com/yourusername/plant-disease-detection.git
cd plant-disease-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements/streamlit_requirements.txt
```

### Option 2: Google Colab

1. Upload notebooks to Google Drive
2. Upload dataset to Google Drive
3. Open notebooks in Google Colab
4. Run setup cells

---

## 📖 Usage

### 1. Data Preprocessing

**Notebook:** `notebooks/01_data_preprocessing_eda.ipynb`

```python
# Run in Google Colab or locally
# Creates:
# - Train/Val/Test splits (70/15/15)
# - Class mapping JSON
# - Stratified samples for ablation studies
# - Initial visualizations
```

**Outputs:**
- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`
- `data/processed/class_mapping.json`

### 2. Model Training

**Notebook:** `notebooks/02_model_training.ipynb`

```python
# Train models with different configurations
# Options:
# - Model: baseline, resnet50, mobilenet, efficientnet
# - Augmentation: yes/no
# - Optimizer: Adam, SGD
# - Batch size: 32, 64
```

**Example:**

```python
from src.models import get_resnet50
from src.training.trainer import Trainer

# Create model
model = get_resnet50(num_classes=39, pretrained=True)

# 2-Phase Training
# Phase 1: Freeze backbone (5 epochs)
model.freeze_backbone()
# ... train classifier only

# Phase 2: Unfreeze all (15-20 epochs)
model.unfreeze_backbone()
# ... fine-tune entire model
```

**Outputs:**
- `models/resnet50/best_model.pth`
- `tensorboard_logs/resnet50/`

### 3. Evaluation

**Notebook:** `notebooks/03_model_evaluation_viz.ipynb`

```python
from src.evaluation.metrics import evaluate_model
from src.evaluation.visualizations import plot_confusion_matrix

# Evaluate model
metrics = evaluate_model(model, test_loader, device)

# Generate visualizations
plot_confusion_matrix(metrics['confusion_matrix'], class_names)
# ... 12+ visualizations
```

**Outputs:**
- `results/visualizations/` (all plots)
- `results/metrics/classification_report.txt`

### 4. Streamlit App

```bash
# Run Streamlit app
streamlit run src/app/main.py
```

**Features:**
- Upload leaf image → Get disease prediction
- View top-3 predictions with confidence scores
- Read treatment recommendations
- Batch process multiple images
- View model performance metrics

---

## 📁 Project Structure

```
plant-disease-detection/
├── data/
│   ├── processed/              # Train/val/test CSVs
│   └── stratified_samples/     # Samples for ablation
├── notebooks/
│   ├── 01_data_preprocessing_eda.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_model_evaluation_viz.ipynb
│   └── 04_autogluon_comparison.ipynb
├── src/
│   ├── data/                   # Dataset, DataLoader, transforms
│   ├── models/                 # All model architectures
│   ├── training/               # Trainer, losses, early stopping
│   ├── evaluation/             # Metrics, Grad-CAM, visualizations
│   ├── utils/                  # Logger, checkpoint utilities
│   └── app/                    # Streamlit application
├── models/                     # Saved model checkpoints
├── results/                    # Visualizations, metrics
├── docs/                       # Documentation
├── database/                   # Treatment recommendations
└── requirements/               # Dependency files
```

---

## 🏆 Results

### Model Performance (Test Set)

| Model | Accuracy | Weighted F1 | Parameters | Inference Time |
|-------|----------|-------------|------------|----------------|
| **ResNet50** | **95.2%** | **94.8%** | 23M | 18 ms/image |
| EfficientNet-B3 | 94.8% | 94.3% | 10M | 22 ms/image |
| MobileNetV2 | 93.1% | 92.5% | 3.5M | 10 ms/image |
| Baseline CNN | 85.4% | 84.2% | 2M | 12 ms/image |

### Ablation Study: Augmentation Impact

| Experiment | Augmentation | Test Accuracy | Improvement |
|------------|--------------|---------------|-------------|
| With Augmentation | ✅ | 95.2% | +4.5% |
| Without Augmentation | ❌ | 90.7% | - |

### Key Findings

- ✅ Transfer learning dramatically outperforms training from scratch
- ✅ Data augmentation improves accuracy by ~4.5%
- ✅ ResNet50 achieves best accuracy; MobileNetV2 offers best speed/accuracy trade-off
- ✅ Model focuses on diseased leaf areas (validated via Grad-CAM)

---

## 🔄 CRISP-DM Methodology

This project follows the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** methodology:

### 1. Business Understanding
- **Goal**: Automate plant disease detection for early treatment
- **Success Criteria**: >90% accuracy, <2s inference time

### 2. Data Understanding
- **Dataset**: PlantVillage (55K images, 39 classes)
- **Challenges**: Class imbalance (154 to 5,539 images per class)
- **EDA**: Distribution analysis, image quality assessment

### 3. Data Preparation
- **Cleaning**: Verified image integrity
- **Splitting**: 70/15/15 stratified split
- **Augmentation**: Flips, rotations, color jitter, noise
- **Normalization**: ImageNet statistics

### 4. Modeling
- **Architectures**: 4 CNN models + AutoML baseline
- **Approach**: Transfer learning with 2-phase training
- **Hyperparameters**: Adam optimizer, ReduceLROnPlateau scheduler
- **Class Imbalance**: Weighted loss function

### 5. Evaluation
- **Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC
- **Validation**: Stratified splits, confusion matrix analysis
- **Interpretability**: Grad-CAM visualizations
- **Ablation Studies**: Augmentation, architecture, optimizer comparisons

### 6. Deployment
- **Application**: Streamlit web app
- **Features**: Single/batch prediction, treatment recommendations
- **Maintenance**: Model retraining procedures documented

📄 **Full CRISP-DM Documentation**: See `docs/crisp_dm_artifacts.md`

---

## 📚 Documentation

- **[CRISP-DM Artifacts](docs/crisp_dm_artifacts.md)** - Complete methodology documentation
- **[Model Documentation](docs/model_documentation.md)** - Architecture details, design choices
- **[Presentation Slides](docs/presentation.pptx)** - Project presentation (15-20 slides)
- **[Demo Video](docs/demo_video.md)** - Link to demo video (5-15 minutes)

---

## 🔬 Technologies Used

### Deep Learning
- **PyTorch 2.0** - Deep learning framework
- **torchvision** - Pretrained models
- **Albumentations** - Image augmentation

### Data Science
- **NumPy, Pandas** - Data manipulation
- **scikit-learn** - Metrics, data splitting
- **Matplotlib, Seaborn, Plotly** - Visualization

### Deployment
- **Streamlit** - Web application framework
- **SQLite** - Prediction history database

### Development
- **Jupyter Notebooks** - Experimentation
- **TensorBoard** - Training monitoring
- **Git** - Version control

---

## 👥 Contributors

- **Pratham Rajesh** - Model development, training pipeline, evaluation
- **Shreram Palanisamy** - Data preprocessing, Streamlit app, documentation

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **PlantVillage Dataset** - For providing high-quality labeled data
- **PyTorch Team** - For excellent deep learning framework
- **Course Instructors** - For guidance and support

---

## 📧 Contact

For questions or collaboration:
- Email: prathamr@example.com
- GitHub: [@prathamr](https://github.com/prathamr)

---

**⭐ If you find this project helpful, please give it a star!**

---

## 🚀 Quick Start

```bash
# 1. Clone and install
git clone https://github.com/yourusername/plant-disease-detection.git
cd plant-disease-detection
pip install -r requirements/streamlit_requirements.txt

# 2. Run Streamlit app (using pretrained model)
streamlit run src/app/main.py

# 3. Open browser to http://localhost:8501
# 4. Upload a leaf image and get instant disease prediction!
```

