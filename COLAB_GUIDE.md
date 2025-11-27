# 🚀 Google Colab Setup Guide

**Complete guide to run this project in Google Colab**

---

## ⚡ Quick Start (Copy-Paste This)

Open Google Colab, paste this in a cell, and run:

```python
# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Clone project (replace with your GitHub URL or use zip from Drive)
!git clone https://github.com/YOUR_USERNAME/plant-disease-detection.git
import os
os.chdir('/content/plant-disease-detection')

# 3. Install dependencies
!pip install -q albumentations opencv-python-headless

# 4. Verify GPU
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

---

## 📋 Step-by-Step Instructions

### Step 1: Setup Colab with GPU

1. Go to https://colab.research.google.com
2. File → New Notebook
3. Runtime → Change runtime type → **GPU (T4)**

### Step 2: Mount Drive & Get Project

**Option A - From GitHub:**
```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/YOUR_USERNAME/plant-disease-detection.git
import os
os.chdir('/content/plant-disease-detection')
```

**Option B - From Drive (if you uploaded zip):**
```python
from google.colab import drive
drive.mount('/content/drive')

import zipfile
with zipfile.ZipFile('/content/drive/MyDrive/plant-disease-detection.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/')

import os
os.chdir('/content/plant-disease-detection')
```

### Step 3: Install Dependencies

```python
!pip install -q albumentations opencv-python-headless torch torchvision torchaudio
```

### Step 4: Get Dataset

**Download from Kaggle:**
```python
# Upload your kaggle.json to Colab first
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download PlantVillage dataset
!kaggle datasets download -d vipoooool/new-plant-diseases-dataset
!unzip -q new-plant-diseases-dataset.zip -d data/raw/
```

**Or copy from Drive:**
```python
!mkdir -p data/raw
!cp -r /content/drive/MyDrive/plant_disease_data/raw/* data/raw/
```

### Step 5: Run Preprocessing

```python
!jupyter nbconvert --to notebook --execute \
  notebooks/01_data_preprocessing_eda.ipynb \
  --output 01_output.ipynb
```

**Duration:** ~15 minutes

### Step 6: Train Model

```python
!jupyter nbconvert --to notebook --execute \
  notebooks/02_train_resnet50_demo.ipynb \
  --output 02_output.ipynb
```

**Duration:** ~2-3 hours
**What it does:**
- Phase 1: Feature extraction (5 epochs)
- Phase 2: Fine-tuning (20 epochs)
- Achieves 93-96% accuracy

### Step 7: Save Model to Drive

```python
import shutil
import os

# Save trained model
os.makedirs('/content/drive/MyDrive/plant_disease_models', exist_ok=True)
shutil.copy('models/resnet50/best_model.pth',
            '/content/drive/MyDrive/plant_disease_models/best_model.pth')
shutil.copy('data/processed/class_mapping.json',
            '/content/drive/MyDrive/plant_disease_models/class_mapping.json')

print("✓ Model saved to Google Drive!")
```

---

## 🧪 Test Your Model

```python
import torch
from PIL import Image
import json
import sys
import numpy as np
import glob

sys.path.append('src')
from models.resnet50 import get_resnet50
from data.transforms import get_val_transforms

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_resnet50(num_classes=39, pretrained=False)
checkpoint = torch.load('models/resnet50/best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Load class mapping
with open('data/processed/class_mapping.json', 'r') as f:
    idx_to_class = json.load(f)['idx_to_class']

# Test prediction
transform = get_val_transforms(image_size=224)
test_img = glob.glob('data/processed/test/*/*')[0]
image = Image.open(test_img).convert('RGB')
img_tensor = transform(image=np.array(image))['image'].unsqueeze(0).to(device)

with torch.no_grad():
    output = model(img_tensor)
    probs = torch.nn.functional.softmax(output, dim=1)[0]
    top3_probs, top3_idx = torch.topk(probs, 3)

print(f"\n✓ Prediction on: {test_img}")
for i, (prob, idx) in enumerate(zip(top3_probs, top3_idx), 1):
    print(f"  {i}. {idx_to_class[str(idx.item())]}: {prob.item()*100:.2f}%")
```

---

## 🔧 Troubleshooting

**CUDA out of memory:**
```python
# In notebook, change:
BATCH_SIZE = 16  # instead of 32
```

**Module not found:**
```python
import sys
sys.path.append('/content/plant-disease-detection/src')
```

**Colab disconnected:**
- Checkpoints are saved in `models/resnet50/phase1/` and `phase2/`
- You can resume from last checkpoint

---

## ⏱️ Time Estimates

| Task | Duration |
|------|----------|
| Setup | 10 min |
| Preprocessing | 15 min |
| Training | 2-3 hours |
| **Total** | **~3 hours** |

---

## ✅ Expected Results

After training:
- Model: `models/resnet50/best_model.pth` (~95 MB)
- Accuracy: 93-96%
- Saved to Google Drive for persistence

---

## 📥 Download Model to Local

```python
from google.colab import files
files.download('models/resnet50/best_model.pth')
files.download('data/processed/class_mapping.json')
```

Then run locally:
```bash
streamlit run app.py
```

---

## 💡 Tips

1. **Use GPU runtime** - Makes training 10x faster
2. **Save to Drive** - Colab sessions timeout after 12 hours
3. **Monitor progress** - Watch loss decrease and accuracy increase
4. **Batch size** - Reduce if you get CUDA errors

---

For detailed project info, see `README.md`
