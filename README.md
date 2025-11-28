# Plant Disease Detection via Leaf Images

AI-powered plant disease detection system using deep learning and computer vision.

## Project Overview

This project implements an end-to-end machine learning system to detect plant diseases from leaf images using transfer learning with ResNet50. The system is trained on the PlantVillage dataset and deployed via a Streamlit web application.

**Key Features:**
- **Dataset:** PlantVillage (55,448 images across 39 disease classes)
- **Model:** ResNet50 Transfer Learning (TensorFlow/Keras)
- **Baseline:** AutoGluon Vision for comparison
- **Target Accuracy:** ≥95% on test set
- **Deployment:** Streamlit web app (local)
- **Methodology:** CRISP-DM framework

## Team

- **Pratham Rajesh**
- **Shreram Palanisamy**

**Project Date:** November 2025

## Project Structure

```
plant-disease/
├── notebooks/
│   ├── plant_disease_training.ipynb          # Main training notebook (Colab)
│   └── exploratory_data_analysis.ipynb       # EDA & visualizations
├── app/
│   └── streamlit_app.py                      # Streamlit deployment app
├── models/
│   ├── resnet50_best.h5                      # Best ResNet50 model
│   ├── resnet50_no_aug.h5                    # Ablation: no augmentation
│   └── class_names.json                      # Class label mapping
├── reports/
│   ├── figures/                              # All visualization outputs
│   ├── final_report.md                       # Complete project report
│   └── crisp_dm_artifacts.md                 # CRISP-DM documentation
├── slides/
│   └── presentation.pptx                     # PowerPoint presentation
├── Plant_leave_diseases_dataset_without_augmentation/  # Dataset
├── requirements.txt                           # Python dependencies
├── README.md                                  # This file
└── .gitignore                                # Git ignore file
```

## Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/[username]/plant-disease-detection.git
cd plant-disease-detection
```

### 2. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. Dataset

The dataset is already included in the repository at:
```
Plant_leave_diseases_dataset_without_augmentation/
```

**Dataset Details:**
- **Source:** PlantVillage
- **Total Images:** 55,448
- **Classes:** 39 (including healthy and diseased variants)
- **Format:** 256x256 RGB JPEG images
- **Plants Covered:** Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato

### 4. Run Training Notebook

**Option A: Google Colab (Recommended)**
1. Upload `notebooks/plant_disease_training.ipynb` to Google Colab
2. Upload the dataset to Google Drive (or use Colab's file upload)
3. Run all cells to train the model
4. Download trained models to `models/` directory

**Option B: Local Jupyter**
```bash
jupyter notebook notebooks/plant_disease_training.ipynb
```

### 5. Run Streamlit App

```bash
cd app
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

### Training

1. Open the training notebook in Google Colab or Jupyter
2. Follow the step-by-step instructions in the notebook
3. The notebook includes:
   - Data loading and preprocessing
   - Data augmentation visualization
   - ResNet50 transfer learning model
   - Ablation study (with/without augmentation)
   - AutoGluon baseline comparison
   - Comprehensive evaluation metrics
   - Grad-CAM visualizations

### Inference (Streamlit App)

1. Launch the Streamlit app
2. Upload a leaf image (JPG, JPEG, or PNG)
3. Click "Detect Disease"
4. View prediction results with confidence scores

## Model Performance

### Results Summary

| Model | Test Accuracy | Test F1 | Training Time | Inference Time |
|-------|---------------|---------|---------------|----------------|
| ResNet50 + Augmentation | TBD | TBD | TBD | TBD |
| ResNet50 (No Augmentation) | TBD | TBD | TBD | TBD |
| AutoGluon Baseline | TBD | TBD | TBD | TBD |

*Results will be updated after training completion*

### Best Performing Classes
- TBD (will be updated after training)

### Most Confused Classes
- TBD (will be updated after training)

## Methodology

This project follows the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** methodology:

1. **Business Understanding:** Address agricultural losses from late disease detection
2. **Data Understanding:** Explore PlantVillage dataset structure and characteristics
3. **Data Preparation:** Preprocess images, apply augmentation, create train/val/test splits
4. **Modeling:** Implement ResNet50 transfer learning and AutoGluon baseline
5. **Evaluation:** Comprehensive metrics, confusion matrix, per-class analysis
6. **Deployment:** Streamlit web application for inference

## Key Technologies

- **Framework:** TensorFlow/Keras 2.13+
- **Architecture:** ResNet50 (Transfer Learning from ImageNet)
- **AutoML:** AutoGluon Vision
- **Deployment:** Streamlit
- **Visualization:** Matplotlib, Seaborn, Grad-CAM
- **Data Processing:** NumPy, Pandas, OpenCV, Pillow

## Visualizations

The project includes comprehensive visualizations (20% of project requirements):

- Class distribution analysis
- Sample images (healthy vs diseased)
- Data augmentation examples
- Training/validation curves (accuracy & loss)
- Confusion matrix (39x39 heatmap)
- Per-class precision, recall, F1 bar charts
- Grad-CAM attention maps
- Error analysis with misclassified examples

## Deliverables

- [x] Training notebooks (EDA + Main Training)
- [x] Streamlit deployment app
- [ ] Trained models (ResNet50, AutoGluon)
- [ ] Final report (Markdown)
- [ ] CRISP-DM artifacts documentation
- [ ] PowerPoint presentation
- [ ] Demo video (5-15 minutes)

## Future Work

- Deploy to Streamlit Community Cloud for public access
- Implement mobile app (iOS/Android) using TensorFlow Lite
- Add real-time inference via webcam
- Expand dataset with field-collected images (vs lab images)
- Multi-crop disease detection in single image
- Integration with agricultural advisory systems
- Support for more plant species and diseases

## References

- Hughes, D. P., & Salathé, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics. *arXiv preprint arXiv:1511.08060*.
- PlantVillage Dataset: https://github.com/spMohanty/PlantVillage-Dataset
- TensorFlow: https://www.tensorflow.org/
- AutoGluon: https://auto.gluon.ai/
- Streamlit: https://streamlit.io/

## License

MIT License - See LICENSE file for details

## Acknowledgments

- PlantVillage project for the dataset
- TensorFlow and Keras teams
- AutoGluon developers
- Streamlit community

---

**For questions or issues, please contact:**
- Pratham Rajesh: [email]
- Shreram Palanisamy: [email]
