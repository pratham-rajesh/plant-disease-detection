# 📁 Project Files Guide

**Simple reference for what each file/folder does**

---

## 📄 Documentation (READ THESE)

| File | Purpose | When to Use |
|------|---------|-------------|
| **README.md** | Main project overview | Start here for project info |
| **COLAB_GUIDE.md** | How to run in Google Colab | Training the model in Colab |
| **PROJECT_FILES.md** | This file - explains all files | Finding your way around |

---

## 📓 Notebooks (RUN THESE IN ORDER)

| Notebook | What It Does | Duration |
|----------|--------------|----------|
| `01_data_preprocessing_eda.ipynb` | Split data into train/val/test | ~15 min |
| `02_train_resnet50_demo.ipynb` | Train ResNet50 model | ~2-3 hours |

---

## 🐍 Source Code (`src/`)

| Folder | Contains |
|--------|----------|
| `src/data/` | Dataset loading, data augmentation |
| `src/models/` | Model architectures (ResNet50, etc.) |
| `src/training/` | Training loop, losses, early stopping |
| `src/evaluation/` | Metrics, visualizations, Grad-CAM |

---

## 📦 Key Folders

| Folder | Contains |
|--------|----------|
| `Plant_leave_diseases_dataset_without_augmentation/` | Raw dataset (55K images) |
| `data/processed/` | Train/val/test splits (created by notebook 01) |
| `models/resnet50/` | Trained model files (created by notebook 02) |
| `database/` | Treatment recommendations for diseases |
| `notebooks/` | Jupyter notebooks for training |
| `scripts/` | Command-line prediction tools |

---

## 🎨 Web App

| File | Purpose |
|------|---------|
| `app.py` | Streamlit web application |

Run with: `streamlit run app.py`

---

## 🔧 Scripts

| Script | What It Does |
|--------|--------------|
| `scripts/predict_single.py` | Predict one image from command line |
| `scripts/predict_batch.py` | Predict multiple images, save to CSV |

---

## 📋 Requirements

| File | For |
|------|-----|
| `requirements/local_requirements.txt` | Local development |
| `requirements/colab_requirements.txt` | Google Colab |
| `requirements/streamlit_requirements.txt` | Streamlit app only |

---

## 🗂️ Typical Workflow

```
1. Read README.md
   ↓
2. Follow COLAB_GUIDE.md (or local setup)
   ↓
3. Run notebook 01 (preprocessing)
   ↓
4. Run notebook 02 (training)
   ↓
5. Run app.py (web demo)
```

---

## 💡 Quick Reference

**Training in Colab?** → `COLAB_GUIDE.md`

**What's this project?** → `README.md`

**Find a file?** → This file (`PROJECT_FILES.md`)

**Run the app?** → `streamlit run app.py`

**Predict from CLI?** → `python scripts/predict_single.py --image test.jpg`

---

That's it! Everything else is generated during training or is supporting code.
