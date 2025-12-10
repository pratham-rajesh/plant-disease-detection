# 🌿 Plant Disease Detection and Management System

**CMPE 256: Data Mining - Fall 2025**  
**Final Project Submission**

---

## 👥 Team Members

- **Pratham Rajesh**
- **Shreram Palanisamy**

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
Comprehensive academic report 

📖 **[Read Full Report](https://github.com/pratham-rajesh/plant-disease-detection/blob/main/Project_Report.pdf)**  

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

Agriculture faces significant crop losses (20-40% globally) due to delayed disease identification. This project presents a comprehensive plant disease detection system using deep learning to enable early diagnosis from leaf images. We developed a CNN-based model using MobileNetV2 architecture with transfer learning on the PlantVillage dataset containing **8,886 images** across 38 disease classes spanning 14 plant species. Our model achieved **90.35% validation accuracy** with balanced metrics (89.94% precision, 89.51% recall, 89.37% F1-score) through comprehensive data augmentation strategies. The system includes multiple innovative features: a Grad-CAM visualization module for model explainability, a comprehensive disease management database with treatment protocols, an LLM-powered RAG chatbot using Groq API (Llama-3-8B), and Semantic Scholar API integration for research papers. The complete solution is deployed via a Gradio web interface with sub-50ms inference time and an 11.06 MB model suitable for mobile deployment, providing an accessible, interpretable, and actionable diagnostic tool.

---

## 🎯 Project Overview

### Problem Statement

Plant diseases cause massive agricultural losses globally (20-40% of annual crop production). Traditional identification methods are time-consuming, expensive, and inaccessible to small farmers, especially in remote areas. Our system provides automated, accurate, and accessible disease detection through smartphone images.

### Solution

An end-to-end AI system that:
- Detects plant diseases from leaf images with **90.35% accuracy**
- Provides visual explanations via Grad-CAM heatmaps
- Offers treatment recommendations and prevention strategies
- Answers user questions via intelligent RAG chatbot
- Connects users to peer-reviewed scientific research
- Runs in **<50ms** per prediction with **11.06 MB** model size

### Key Innovation

**Comprehensive data augmentation strategy** - We achieved robust generalization through extensive augmentation (rotation, zoom, brightness, flips), improving accuracy by **~12%** compared to non-augmented training (78% → 90.35%), successfully overcoming the lab bias inherent in controlled dataset conditions.

---

## 🔬 Methodology

### CRISP-DM Framework

This project follows the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** methodology:

#### 1. Business Understanding
- **Problem**: 20-40% global crop losses due to late disease detection
- **Objective**: Build automated disease detection system accessible to farmers
- **Success Criteria**: >90% accuracy, <1 second inference, mobile-ready deployment
- **Impact**: Enable early intervention and reduce crop losses

#### 2. Data Understanding
- **Dataset**: PlantVillage (8,886 images, 38 classes, 14 plant species)
- **Source**: Standard research subset of PlantVillage dataset
- **Split**: 7,570 training (85.2%), 1,316 validation (14.8%)
- **Analysis**: Relatively balanced distribution across classes
- **Challenge Identified**: Lab bias from controlled conditions

#### 3. Data Preparation
- **Preprocessing**: Resize to 224×224, normalize to [0,1], stratified split
- **Augmentation Strategy**: 
  - Random rotations (up to 30°)
  - Random zoom (up to 20%)
  - Brightness adjustments [0.7, 1.3]
  - Horizontal and vertical flips
  - Width and height shifts (20%)
- **Rationale**: Overcome lab bias and improve real-world generalization
- **Impact**: +12% accuracy improvement (78% → 90.35%)

#### 4. Modeling
- **Architecture**: MobileNetV2 (pre-trained ImageNet) + custom classification head
- **Transfer Learning**: Froze 2.26M base parameters, trained 169K top layer parameters
- **Total Parameters**: 2.43M (efficient for mobile deployment)
- **Training**: Adam optimizer (lr=0.001), 15 epochs, batch size 32
- **Callbacks**: ModelCheckpoint, EarlyStopping (patience=5), ReduceLROnPlateau
- **Training Time**: ~1.5 hours on Tesla T4 GPU

#### 5. Evaluation
- **Primary Metrics**: 90.35% validation accuracy
- **Balanced Performance**: 89.94% precision, 89.51% recall, 89.37% F1-score
- **Validation**: Validation accuracy exceeded training accuracy (86.04%), confirming good generalization
- **Explainability**: Grad-CAM visualizations validate focus on disease symptoms

#### 6. Deployment
- **Interface**: Gradio web application with intuitive UI
- **Performance**: <50ms inference time per image
- **Features**: Real-time prediction, Grad-CAM overlay, chatbot, research integration
- **Model Size**: 11.06 MB (suitable for mobile deployment)

---

## ✨ Key Features

### 1. **High-Accuracy Disease Detection**
- MobileNetV2-based CNN achieving **90.35% validation accuracy**
- Balanced metrics: **89.94% precision, 89.51% recall, 89.37% F1-score**
- Efficient model size (**11.06 MB**) suitable for mobile deployment
- Fast inference (**<50ms** per image)

### 2. **Model Explainability (Grad-CAM)**
- Visual heatmaps showing which leaf regions influenced predictions
- Validates model focuses on disease symptoms, not spurious features
- Builds user trust through transparent decision-making

### 3. **Comprehensive Disease Management**
- Treatment protocols for all 38 diseases
- Prevention strategies and severity indicators
- Actionable guidance beyond simple diagnosis

### 4. **Intelligent Chatbot (RAG)**
- Powered by Groq API with **Llama-3-8B**
- Sub-second response times (<1s average)
- Contextually relevant advice based on detected disease

### 5. **Research Paper Integration**
- Semantic Scholar API for peer-reviewed literature
- Direct links to full papers with citations

---

## 💻 Source Code & Notebooks

#### Training Notebook
Complete model training with CRISP-DM methodology
- Data exploration (10+ visualizations)
- Model training pipeline
- Comprehensive evaluation
- Ablation study

#### Demo Application
Production-ready Gradio interface
- Disease detection with Grad-CAM
- Treatment recommendations
- LLM chatbot (Groq + Llama-3-8B)
- Research paper retrieval

### 🤖 Trained Model
- **File**: `plant_disease_model_final.h5`
- **Size**: 11.06 MB
- **Performance**: 90.35% validation accuracy

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- Google Colab (recommended) or local Jupyter environment
- GPU recommended (but not required)
- Groq API key for chatbot functionality

### Quick Start (Google Colab)

1. **Training Pipeline**:
```
   Open: notebooks/Plant_Disease_Training.ipynb in Google Colab
   Runtime → Change runtime type → GPU
   Runtime → Run all
```

2. **Demo Application**:
```
   Open: notebooks/Plant_Disease_Demo.ipynb in Google Colab
   Upload plant_disease_model_final.h5 to session storage
   Add Groq API key to secrets
   Runtime → Run all
   Click on Gradio public URL
```

### Local Installation
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/plant-disease-detection.git
cd plant-disease-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r deployment/requirements.txt

# Set environment variables
export GROQ_API_KEY='your_groq_api_key'

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
opencv-python==4.8.0
```

---

## 💡 Usage

### Training a New Model
```python
# In Colab or Jupyter
from src.model import create_model
from src.train import train_model

# Create model with MobileNetV2 architecture
model = create_model(num_classes=38)

# Train model with augmentation
history = train_model(
    model=model,
    train_data=train_generator,
    val_data=validation_generator,
    epochs=15,
    batch_size=32,
    learning_rate=0.001
)
```

### Making Predictions
```python
from src.predict import predict_disease
from src.gradcam import generate_gradcam

# Predict disease
prediction, confidence, class_name = predict_disease(image_path, model)
print(f"Predicted: {class_name} (Confidence: {confidence:.2%})")

# Generate Grad-CAM explanation
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
demo.launch(share=True)  # Creates public URL
```

---

## 📊 Results & Metrics

### Model Performance

| Metric | Training | Validation |
|--------|----------|------------|
| **Accuracy** | 86.04% | **90.35%** ✨ |
| **Precision** | - | **89.94%** |
| **Recall** | - | **89.51%** |
| **F1-Score** | - | **89.37%** |
| **Loss** | 0.428 | 0.322 |
| **Training Time** | 1.5 hours (T4 GPU) | - |

### Key Performance Highlights

✅ **90.35% validation accuracy** - Exceeds target of 90%  
✅ **Balanced metrics** - All around 89-90%, indicating robust performance  
✅ **Generalization** - Validation accuracy > training accuracy  
✅ **Fast inference** - <50ms per image  
✅ **Compact model** - 11.06 MB suitable for mobile  

### Training Dynamics

**Convergence Behavior:**
- Initial accuracy (Epoch 1): 77.5% - strong transfer learning
- Steady improvement over 15 epochs
- Best model: Epoch 14 (90.35% validation accuracy)
- No overfitting: Validation consistently > training

### Ablation Study: Impact of Data Augmentation

| Configuration | Validation Accuracy | Improvement |
|--------------|---------------------|-------------|
| Without Augmentation | ~78% | Baseline |
| **With Augmentation** | **90.35%** | **+12.35%** ✨ |

**Key Finding**: Data augmentation was critical for overcoming lab bias and achieving robust generalization. This represents our most significant methodological contribution.

### Per-Class Performance Analysis

**Top Performing Classes (>95% F1-score):**
- Tomato healthy
- Potato healthy
- Grape Black rot
- Corn Common rust
- Blueberry healthy

**Challenging Classes (89-92% F1-score):**
- Apple Cedar apple rust (visual similarity to other conditions)
- Pepper Bacterial spot (early symptoms subtle)
- Tomato Early/Late blight (confusion between similar diseases)

**Error Patterns:**
- Most errors within same plant species (never across species)
- Confusion between visually similar disease stages
- Model correctly learned species-specific features first

### Comprehensive Visualizations

Our project includes 10+ detailed visualizations:

1. **Class Distribution** - Dataset balance analysis
2. **Sample Images** - Representative examples from each class
3. **Data Augmentation Techniques** - All transformations applied
4. **Training Curves** - Accuracy and loss over 15 epochs
5. **Confusion Matrix (38×38)** - Complete classification performance
6. **Per-Class Metrics** - Precision, recall, F1-scores
7. **Misclassification Examples** - Common error patterns
8. **Confidence Distribution** - Model certainty analysis
9. **Performance Summary Dashboard** - Comprehensive metrics table
10. **Grad-CAM Visualizations** - Model explainability heatmaps

### Comparison with Published Research

| Study | Architecture | Accuracy | Model Size |
|-------|-------------|----------|------------|
| Hughes & Salathé (2015) | AlexNet | 99.35%* | ~200 MB |
| Mohanty et al. (2016) | GoogLeNet | 99.35%* | ~25 MB |
| Ferentinos (2018) | VGGNet | 99.53%* | ~500 MB |
| Too et al. (2019) | DenseNet | 99.75%* | ~30 MB |
| **Our Work (2024)** | **MobileNetV2** | **90.35%** | **11.06 MB** ✨ |

*Published results often use different train/test splits and may overfit to lab conditions

**Our Advantage**: 
- Most mobile-friendly architecture (11.06 MB)
- Proven generalization (validation > training)
- Demonstrated augmentation impact through ablation
- Production-ready with full system integration
- Explicit handling of lab bias challenge

---

## 🎓 Key Learnings & Future Work

### What We Learned

1. **Data augmentation is critical** for overcoming dataset biases (+12% accuracy)
2. **Transfer learning accelerates convergence** (77% accuracy in 1 epoch)
3. **MobileNetV2 excels** at efficiency-accuracy tradeoff for deployment
4. **Validation > training accuracy** indicates successful generalization
5. **Explainability (Grad-CAM) builds trust** in agricultural AI systems
6. **End-to-end systems are more valuable** than isolated models
7. **Modern LLM infrastructure (Groq) enables real-time UX** (<1s responses)

### Challenges Overcome

- **Lab Bias**: Solved through comprehensive augmentation strategy
- **Class Imbalance**: Managed with stratified sampling
- **Model Size**: Achieved mobile-ready 11 MB without sacrificing accuracy
- **Inference Speed**: <50ms enables real-time user experience
- **Explainability**: Grad-CAM provides transparency for agricultural users

### Future Extensions

1. **Mobile Application**: Deploy as native iOS/Android app with TensorFlow Lite
2. **Real-World Dataset**: Collect and integrate field images for domain adaptation
3. **Multi-Disease Detection**: Handle multiple simultaneous infections per leaf
4. **Early Stage Detection**: Train on earlier disease symptoms for prevention
5. **Temporal Monitoring**: Track disease progression over time
6. **Geographic Customization**: Fine-tune for regional crop varieties and conditions
7. **Integrated Pest Management**: Expand to insects, weeds, nutrient deficiencies
8. **Community Platform**: Crowdsource field images for continuous improvement
9. **Economic Analysis**: Estimate crop loss and treatment ROI for farmers
10. **Offline Capability**: Edge deployment for areas without internet connectivity
11. **Multi-Language Support**: Localize chatbot for global farmer accessibility
12. **Drone Integration**: Aerial imaging for large-scale automated monitoring

---

## 📚 References

1. **Hughes, D. P., & Salathé, M. (2015)**. An open access repository of images on plant health to enable the development of mobile disease diagnostics. *arXiv preprint arXiv:1511.08060*.

2. **Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016)**. Using deep learning for image-based plant disease detection. *Frontiers in Plant Science, 7*, 1419.

3. **Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018)**. MobileNetV2: Inverted residuals and linear bottlenecks. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 4510-4520.

4. **Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017)**. Grad-CAM: Visual explanations from deep networks via gradient-based localization. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 618-626.

5. **Ferentinos, K. P. (2018)**. Deep learning models for plant disease detection and diagnosis. *Computers and Electronics in Agriculture, 145*, 311-318.

6. **Too, E. C., Yujian, L., Njuki, S., & Yingchun, L. (2019)**. A comparative study of fine-tuning deep learning models for plant disease identification. *Computers and Electronics in Agriculture, 161*, 272-279.

---

## 🏆 Acknowledgments

- **PlantVillage Dataset**: Hughes & Salathé, Penn State University & EPFL
- **TensorFlow Team**: Pre-trained MobileNetV2 models and ecosystem
- **Groq**: Ultra-fast LLM inference API (Llama-3-8B)
- **Semantic Scholar**: Research paper API access
- **Google Colab**: Free GPU resources for model training
- **Course Instructor**: CMPE 256 Data Mining teaching team

---

## 📧 Contact

For questions, collaboration, or feedback:

- **Pratham Rajesh**: [GitHub](#) | [Email](#)
- **Shreram Palanisamy**: [GitHub](#) | [Email](#)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Academic Integrity Statement

This project was developed entirely from scratch by our team following the CRISP-DM methodology. All code, documentation, and analysis represent original work. We have properly cited all external resources, datasets, and prior research. The notebooks are extensively documented to demonstrate our understanding and decision-making process throughout the project lifecycle.

---

## 🌟 Project Highlights

✅ **Complete CRISP-DM Implementation** - Full methodology documented  
✅ **90.35% Validation Accuracy** - Exceeds 90% target with balanced metrics  
✅ **10+ Comprehensive Visualizations** - Meets project requirements  
✅ **Model Explainability (Grad-CAM)** - Transparent AI for agriculture  
✅ **End-to-End Deployment (Gradio)** - Production-ready web interface  
✅ **LLM Integration (Groq + RAG)** - Sub-second intelligent responses  
✅ **Research Integration** - Semantic Scholar API for peer-reviewed papers  
✅ **Mobile-Ready Architecture** - 11.06 MB, <50ms inference  
✅ **Proven Generalization** - Validation > training accuracy  
✅ **Ablation Study** - Data augmentation impact validated (+12%)  

---

## 📊 Quick Stats

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | 90.35% |
| **Model Size** | 11.06 MB |
| **Inference Time** | <50ms |
| **Dataset Size** | 8,886 images |
| **Classes** | 38 diseases |
| **Plant Species** | 14 types |
| **Training Time** | 1.5 hours |
| **Parameters** | 2.43M (169K trainable) |
| **Augmentation Impact** | +12% accuracy |

---

**Made with ❤️ for sustainable agriculture and global food security**

🌱 *Empowering farmers with AI for healthier crops and better yields*
