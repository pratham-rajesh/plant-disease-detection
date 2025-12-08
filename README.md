# 🌿 Plant Disease Detection and Management System

**CS 255: Data Mining - Section 49**  
**Final Project Submission**

---

## 👥 Team Members

- **Pratham Rajesh**
- **Shreram Palanisamy**

---

## 📋 Table of Contents

- [Abstract](#abstract)
- [Project Overview](#project-overview)
- [Methodology](#methodology)
- [Key Features](#key-features)
- [Project Deliverables](#project-deliverables)
- [Repository Structure](#repository-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Results & Metrics](#results--metrics)
- [Acknowledgments](#acknowledgments)

---

## 📝 Abstract

Agriculture faces significant crop losses (20-40% globally) due to delayed disease identification. This project presents a comprehensive plant disease detection system using deep learning to enable early diagnosis from leaf images. We developed a CNN-based model using MobileNetV2 architecture with transfer learning on the PlantVillage dataset containing 54,305 images across 38 disease classes spanning 14 plant species. Our model achieved **96.8% validation accuracy without data augmentation** during training, demonstrating robust feature learning. The system includes multiple innovative features: a Grad-CAM visualization module for model explainability, a comprehensive disease management database with treatment protocols, an LLM-powered RAG chatbot using Groq API, and Semantic Scholar API integration for research papers. The complete solution is deployed via a Gradio web interface, providing an accessible, interpretable, and actionable diagnostic tool.

---

## 🎯 Project Overview

### Problem Statement

Plant diseases cause massive agricultural losses globally. Traditional identification methods are time-consuming, expensive, and inaccessible to small farmers. Our system provides automated, accurate, and accessible disease detection through smartphone images.

### Solution

An end-to-end AI system that:
- Detects plant diseases from leaf images with 96.8% accuracy
- Provides visual explanations via Grad-CAM
- Offers treatment recommendations and prevention strategies
- Answers user questions via intelligent chatbot
- Connects users to scientific research

### Key Innovation

**Training without data augmentation** - Unlike most approaches, we achieved high accuracy without augmentation, demonstrating the model's fundamental learning capacity.

---

## 🔬 Methodology

### CRISP-DM Framework

This project follows the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** methodology:

#### 1. Business Understanding
- **Problem**: 20-40% global crop losses due to late disease detection
- **Objective**: Build automated disease detection system accessible to farmers
- **Success Criteria**: >95% accuracy, <1 second inference, mobile-ready

#### 2. Data Understanding
- **Dataset**: PlantVillage (54,305 images, 38 classes, 14 plant species)
- **Source**: TensorFlow Datasets / Penn State & EPFL collaboration
- **Analysis**: Balanced distribution (952-2,127 images per class)
- **Visualizations**: Class distribution, sample images, data statistics

#### 3. Data Preparation
- **Preprocessing**: Resize to 224×224, normalize [0,1], stratified split (85/15)
- **No Augmentation**: Deliberately avoided augmentation to test base model capacity
- **Quality Check**: Identified and documented data quality issues

#### 4. Modeling
- **Architecture**: MobileNetV2 (pre-trained ImageNet) + custom classification head
- **Transfer Learning**: Froze base, trained top layers only
- **Training**: Adam optimizer (lr=0.001), 15 epochs, batch size 32
- **Callbacks**: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

#### 5. Evaluation
- **Metrics**: Accuracy, Precision, Recall, F1-score, Confusion Matrix
- **Validation Accuracy**: 96.8%
- **Analysis**: Per-class performance, misclassification patterns, confidence distribution
- **Explainability**: Grad-CAM visualizations

#### 6. Deployment
- **Interface**: Gradio web application
- **Features**: Image upload, real-time prediction, Grad-CAM overlay, chatbot, research papers
- **Accessibility**: Browser-based, no installation required

---

## ✨ Key Features

### 1. **High-Accuracy Disease Detection**
- MobileNetV2-based CNN achieving 96.8% validation accuracy
- Recognizes 38 disease classes across 14 plant species
- Efficient model size (~14 MB) suitable for mobile deployment

### 2. **Model Explainability (Grad-CAM)**
- Visual heatmaps showing which leaf regions influenced predictions
- Builds user trust and validates model learned disease-relevant features

### 3. **Comprehensive Disease Management**
- Treatment protocols for all 38 diseases
- Prevention strategies and severity indicators
- Contagion information for quarantine decisions

### 4. **Intelligent Chatbot (RAG)**
- Powered by Groq API with Llama 3.3 70B
- Sub-second response times (<1s average)
- Contextually relevant advice based on detected disease

### 5. **Research Paper Integration**
- Semantic Scholar API for current peer-reviewed literature
- 8-12 relevant papers per disease query
- Direct links to full papers with abstracts and citations

---

## 📦 Project Deliverables

### 🎥 Video Presentation
**Duration**: 12 minutes  
**Content**: Project demo, methodology explanation, results showcase, deployment walkthrough

📹 **[Watch Video Presentation](https://youtu.be/-XUKU6jTzc0)**

### 📊 Presentation Deck
PowerPoint presentation covering all project aspects

📑 **[View Presentation Slides](https://docs.google.com/presentation/d/1p1nDEBM0EXoQZKupcWf8gTrVdJUVkbCmnag_07IO3iM/edit?usp=sharing)**

### 📄 Project Report
Comprehensive 25-page academic report with 10 figures

📖 **[Read Full Report](https://docs.google.com/document/d/1sHKokJdhD-La5g9CH-0c5yVa6yNR-5u3WRUqFZoyduY/edit?usp=sharing)**  

### 💻 Source Code & Notebooks

#### Main Training Notebook
Complete model training with CRISP-DM methodology
- Data exploration and visualization
- Model architecture and training
- Comprehensive evaluation and metrics
- All visualizations (10+ figures)

📓 **[Training Notebook (Colab)](https://colab.research.google.com/drive/1ptIRrEyHg90OsnhJgfq8TYRU0ggWeROe?usp=sharing)**

#### Demo Application Notebook
Production-ready Gradio interface with all features
- Pre-trained model loading
- Disease detection with Grad-CAM
- Treatment recommendations
- LLM chatbot integration
- Research paper retrieval

📓 **[Demo Application (Colab)](https://colab.research.google.com/drive/1OGbG4cjQQ8QgIagJeEODbGBj4EPoMtFp?usp=sharing)**

### 🤖 Trained Model
Pre-trained MobileNetV2 model (H5 format, ~14 MB)


### 📊 CRISP-DM Artifacts(please look at the files for the arifact readme)

#### Business Understanding
- Problem definition and objectives
- Success criteria and constraints

📄 **[Business Understanding Document](LINK_TO_BUSINESS_DOC)**

#### Data Understanding
- Dataset analysis and statistics
- Exploratory data analysis visualizations
- Data quality assessment

📊 **[Data Analysis Report](LINK_TO_DATA_ANALYSIS)**

#### Evaluation Results
- Confusion matrix (38×38)
- Per-class precision, recall, F1-scores
- Training/validation curves
- Confidence distribution
- Sample predictions with Grad-CAM
- Misclassification analysis

📈 **[Evaluation Dashboard](LINK_TO_EVAL_DASHBOARD)**

---

## 📁 Repository Structure

```
plant-disease-detection/
│
├── README.md                          # This file
├── LICENSE                            # Project license
│
├── notebooks/
│   ├── Plant_Disease_Training.ipynb  # Complete training pipeline
│   └── Plant_Disease_Demo.ipynb      # Demo application with Gradio
│
├── models/
│   └── plant_disease_model_final.h5  # Trained model weights
│
├── data/
│   └── README.md                      # Dataset information and download links
│
├── docs/
│   ├── Project_Report.pdf             # Academic report
│   ├── Presentation.pptx              # Presentation slides
│   ├── CRISP_DM_Artifacts/            # Methodology documentation
│   │   ├── Business_Understanding.md
│   │   ├── Data_Understanding.md
│   │   ├── Data_Preparation.md
│   │   ├── Modeling.md
│   │   ├── Evaluation.md
│   │   └── Deployment.md
│   └── figures/                       # All visualization figures
│
├── src/
│   ├── __init__.py
│   ├── model.py                       # Model architecture
│   ├── train.py                       # Training utilities
│   ├── predict.py                     # Inference functions
│   ├── gradcam.py                     # Grad-CAM implementation
│   ├── chatbot.py                     # RAG chatbot
│   └── utils.py                       # Helper functions
│
├── deployment/
│   ├── app.py                         # Gradio application
│   ├── requirements.txt               # Python dependencies
│   └── disease_database.json          # Disease management information
│
├── results/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── per_class_metrics.png
│   ├── gradcam_examples.png
│   └── evaluation_summary.json
│
└── videos/
    └── project_demo.mp4               # Project demonstration video
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- Google Colab (recommended) or local Jupyter environment
- GPU recommended (but not required)

### Quick Start (Google Colab)

1. **Training Pipeline**:
   ```
   Open: notebooks/Plant_Disease_Training.ipynb in Google Colab
   Runtime → Run all
   ```

2. **Demo Application**:
   ```
   Open: notebooks/Plant_Disease_Demo.ipynb in Google Colab
   Upload plant_disease_model_final.h5 to Google Drive
   Runtime → Run all
   ```

### Local Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/plant-disease-detection.git
cd plant-disease-detection

# Install dependencies
pip install -r deployment/requirements.txt

# Run Gradio app
python deployment/app.py
```

### Required Dependencies
```
tensorflow==2.15.0
gradio==4.0.0
pillow==10.0.0
numpy==1.24.3
matplotlib==3.7.1
seaborn==0.12.2
scikit-learn==1.3.0
groq==0.4.0
requests==2.31.0
```

---

## 💡 Usage

### Training a New Model

```python
# In Colab or Jupyter
from src.model import create_model
from src.train import train_model

# Create model
model = create_model(num_classes=38)

# Train model
history = train_model(
    model=model,
    train_data=train_generator,
    val_data=validation_generator,
    epochs=15,
    batch_size=32
)
```

### Making Predictions

```python
from src.predict import predict_disease
from src.gradcam import generate_gradcam

# Predict disease
prediction, confidence = predict_disease(image_path, model)

# Generate Grad-CAM
heatmap = generate_gradcam(image_path, model, class_index)
```

### Running Demo Application

```python
# Launch Gradio interface
python deployment/app.py

# Or in notebook
import gradio as gr
from deployment.app import create_interface

demo = create_interface()
demo.launch(share=True)
```

---

## 📊 Results & Metrics

### Model Performance

| Metric | Training Set | Validation Set |
|--------|-------------|----------------|
| **Accuracy** | 98.2% | **96.8%** |
| **Loss** | 0.067 | 0.124 |
| **Training Time** | 45 minutes (T4 GPU) | - |

### Per-Class Performance Highlights

**Top Performing Classes:**
- Tomato healthy: 99.2% F1-score
- Potato healthy: 98.9% F1-score
- Grape Black rot: 98.7% F1-score
- Corn Common rust: 98.5% F1-score

**Challenging Classes:**
- Apple Cedar apple rust: 94.1% F1-score
- Pepper Bacterial spot: 94.8% F1-score

### Visualizations (20% Project Requirement)

Our project includes extensive visualizations:

1. **Class Distribution** - Dataset balance analysis
2. **Sample Images** - Representative examples from each class
3. **Data Augmentation Examples** - Techniques (not used in training)
4. **Training Curves** - Accuracy and loss over epochs
5. **Confusion Matrix** - 38×38 classification performance
6. **Per-Class Metrics** - Precision, recall, F1-scores
7. **Misclassification Examples** - Common error patterns
8. **Confidence Distribution** - Model certainty analysis
9. **Performance Summary** - Comprehensive metrics dashboard
10. **Grad-CAM Visualizations** - Model explainability examples

📊 **[View All Visualizations](LINK_TO_FIGURES_FOLDER)**

### Comparison with State-of-the-Art

| Study | Architecture | Accuracy | Model Size |
|-------|-------------|----------|------------|
| Mohanty et al. (2016) | AlexNet | 99.35% | ~200 MB |
| Ferentinos (2018) | VGGNet | 99.53% | ~500 MB |
| Too et al. (2019) | DenseNet | 99.75% | ~30 MB |
| **Our Work (2024)** | **MobileNetV2** | **96.8%*** | **~14 MB** |

*Without data augmentation; lightweight and mobile-ready

### Model Design Rationale

#### Architecture Choices
- **MobileNetV2**: Chosen for efficiency (14 MB vs 500+ MB VGGNet)
- **Transfer Learning**: Pre-trained ImageNet features reduce training time
- **Custom Head**: 128-unit dense layer + 30% dropout for task adaptation

#### Training Configuration
- **Optimizer**: Adam (adaptive learning rates, proven for image tasks)
- **Learning Rate**: 0.001 (standard for transfer learning)
- **Loss Function**: Categorical crossentropy (multi-class classification)
- **Batch Size**: 32 (balance between memory and gradient stability)
- **Epochs**: 15 (early stopping at convergence)

#### Why No Augmentation?
- Test model's fundamental learning capacity
- Dataset size (54K images) sufficient for robustness
- Demonstrates strong generalization on original distribution
- Note: Production deployment would benefit from augmentation for real-world variance

#### Evaluation Metrics
- **Accuracy**: Overall correctness
- **Precision**: Avoid false disease alarms
- **Recall**: Don't miss actual diseases
- **F1-Score**: Balanced performance measure
- **Confusion Matrix**: Inter-disease confusion patterns

---

## 🎓 Key Learnings & Future Work

### What We Learned

1. **Transfer learning is highly effective** for agricultural image classification
2. **Large datasets can succeed without augmentation** when properly configured
3. **Explainability (Grad-CAM) significantly increases user trust**
4. **End-to-end systems are more valuable** than isolated models
5. **Modern LLM infrastructure (Groq) transforms UX** with sub-second responses

### Future Extensions

1. **Mobile Application**: Deploy as native app with TensorFlow Lite
2. **Domain Adaptation**: Fine-tune on real-world field images
3. **Multi-disease Detection**: Handle multiple simultaneous infections
4. **Early Detection**: Train on earlier disease stages
5. **Temporal Monitoring**: Track disease progression over time
6. **Geographic Customization**: Adapt to local conditions
7. **Integrated Pest Management**: Expand to insects, weeds, nutrients
8. **Community Data**: Crowdsource field images for diversity
9. **Economic Analysis**: Estimate crop loss and treatment ROI
10. **Language Localization**: Support multiple languages globally

---

## 📚 References

1. Hughes, D. P., & Salathé, M. (2015). An open access repository of images on plant health. arXiv:1511.08060.

2. Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using deep learning for image-based plant disease detection. Frontiers in Plant Science, 7, 1419.

3. Howard, A. G., et al. (2017). MobileNets: Efficient convolutional neural networks for mobile vision applications. arXiv:1704.04861.

4. Sandler, M., et al. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. CVPR 2018.

5. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks. ICCV 2017.

6. Lewis, P., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. NeurIPS 2020.

---

## 🏆 Acknowledgments

- **PlantVillage Dataset**: Penn State University & EPFL
- **TensorFlow Team**: Pre-trained MobileNetV2 models
- **Groq**: Ultra-fast LLM inference API
- **Semantic Scholar**: Research paper API access
- **Course Instructors**: CS 255 Data Mining teaching team

---

## 📧 Contact

For questions or collaboration:

- **Pratham Rajesh**: [email@example.com](mailto:email@example.com)
- **Shreram Palanisamy**: [email@example.com](mailto:email@example.com)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Academic Integrity Statement

This project was developed entirely from scratch by our team following the CRISP-DM methodology. All code, documentation, and analysis are original work. We have properly cited all external resources, datasets, and prior research. The notebooks are heavily documented to demonstrate our understanding and decision-making process throughout the project lifecycle.

---

## 🌟 Project Highlights

✅ **Complete CRISP-DM Implementation**  
✅ **96.8% Accuracy without Augmentation**  
✅ **10+ Comprehensive Visualizations**  
✅ **Model Explainability (Grad-CAM)**  
✅ **End-to-End Deployment (Gradio)**  
✅ **LLM Integration (Groq + RAG)**  
✅ **Research Integration (Semantic Scholar)**  
✅ **Production-Ready Application**  
✅ **Heavily Documented Code**  
✅ **Mobile-Ready Architecture (~14 MB)**

---

**Made with ❤️ for sustainable agriculture and food security**
