"""
Colab Environment Setup Script
This script sets up the Google Colab environment for plant disease detection project.
"""

import os
import sys
from pathlib import Path

def mount_google_drive():
    """Mount Google Drive in Colab."""
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✓ Google Drive mounted successfully")
        return True
    except ImportError:
        print("✗ Not running in Colab environment")
        return False
    except Exception as e:
        print(f"✗ Error mounting Google Drive: {e}")
        return False

def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    os.system('pip install -q -r /content/drive/MyDrive/plant_disease_data/requirements/colab_requirements.txt')
    print("✓ Packages installed")

def setup_directories():
    """Create necessary directories."""
    directories = [
        '/content/data/raw',
        '/content/data/processed/train',
        '/content/data/processed/val',
        '/content/data/processed/test',
        '/content/models',
        '/content/results',
        '/content/tensorboard_logs'
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    print(f"✓ Created {len(directories)} directories")

def copy_dataset_to_local():
    """Copy dataset from Drive to local Colab storage for faster I/O."""
    print("Copying dataset to local storage (this may take a few minutes)...")

    drive_path = '/content/drive/MyDrive/plant_disease_data/Plant_leave_diseases_dataset_without_augmentation'
    local_path = '/content/data/raw/Plant_leave_diseases_dataset_without_augmentation'

    if os.path.exists(drive_path):
        os.system(f'cp -r "{drive_path}" "{local_path}"')
        print(f"✓ Dataset copied to {local_path}")
    else:
        print(f"✗ Dataset not found at {drive_path}")
        print("  Please upload the dataset to Google Drive first")

def check_gpu():
    """Check if GPU is available."""
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✓ GPU available: {gpu_name}")
        print(f"  GPU Memory: {gpu_memory:.2f} GB")
        return True
    else:
        print("✗ No GPU available. Please enable GPU in Runtime > Change runtime type")
        return False

def main():
    """Main setup function."""
    print("=" * 60)
    print("PLANT DISEASE DETECTION - Colab Environment Setup")
    print("=" * 60)
    print()

    # Step 1: Mount Google Drive
    if not mount_google_drive():
        return False

    # Step 2: Check GPU
    check_gpu()

    # Step 3: Install requirements
    install_requirements()

    # Step 4: Setup directories
    setup_directories()

    # Step 5: Copy dataset (optional, comment out if dataset is already local)
    # copy_dataset_to_local()

    print()
    print("=" * 60)
    print("✓ Setup complete! Ready to start training.")
    print("=" * 60)

    return True

if __name__ == "__main__":
    main()
