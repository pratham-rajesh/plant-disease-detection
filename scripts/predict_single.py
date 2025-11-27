#!/usr/bin/env python
"""
Single Image Prediction Script
Quick command-line tool to test the trained model on a single image.

Usage:
    python scripts/predict_single.py --image path/to/image.jpg
    python scripts/predict_single.py --image path/to/image.jpg --top-k 5
"""

import sys
import os
import argparse
import json
from pathlib import Path

# Add src to path
sys.path.append('src')

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from models.resnet50 import get_resnet50
from data.transforms import get_val_transforms


def load_model(model_path, num_classes=39):
    """Load trained model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_resnet50(num_classes=num_classes, pretrained=False)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, device


def load_class_mapping(mapping_path):
    """Load class index to name mapping."""
    with open(mapping_path, 'r') as f:
        data = json.load(f)
    return data['idx_to_class']


def predict_image(model, image_path, transform, device, idx_to_class, top_k=3):
    """Make prediction on an image."""
    # Load image
    image = Image.open(image_path).convert('RGB')

    # Preprocess
    image_array = np.array(image)
    transformed = transform(image=image_array)
    image_tensor = transformed['image'].unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]

    # Get top-k predictions
    topk_probs, topk_indices = torch.topk(probabilities, top_k)

    predictions = []
    for prob, idx in zip(topk_probs, topk_indices):
        class_name = idx_to_class[str(idx.item())]
        predictions.append({
            'class': class_name,
            'confidence': float(prob)
        })

    return predictions


def main():
    parser = argparse.ArgumentParser(
        description='Predict plant disease from a single image',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/predict_single.py --image test.jpg
  python scripts/predict_single.py --image test.jpg --top-k 5
  python scripts/predict_single.py --image test.jpg --model models/resnet50/best_model.pth
        """
    )

    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to input image'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='models/resnet50/best_model.pth',
        help='Path to trained model (default: models/resnet50/best_model.pth)'
    )

    parser.add_argument(
        '--class-mapping',
        type=str,
        default='data/processed/class_mapping.json',
        help='Path to class mapping file (default: data/processed/class_mapping.json)'
    )

    parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='Number of top predictions to show (default: 3)'
    )

    parser.add_argument(
        '--image-size',
        type=int,
        default=224,
        help='Input image size (default: 224)'
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.image):
        print(f"❌ Error: Image not found: {args.image}")
        sys.exit(1)

    if not os.path.exists(args.model):
        print(f"❌ Error: Model not found: {args.model}")
        print("   Please train the model first using notebooks/02_train_resnet50_demo.ipynb")
        sys.exit(1)

    if not os.path.exists(args.class_mapping):
        print(f"❌ Error: Class mapping not found: {args.class_mapping}")
        print("   Please run preprocessing first using notebooks/01_data_preprocessing_eda.ipynb")
        sys.exit(1)

    # Load model and mapping
    print("🔧 Loading model...")
    model, device = load_model(args.model)
    print(f"✓ Model loaded on {device}")

    print("📋 Loading class mapping...")
    idx_to_class = load_class_mapping(args.class_mapping)
    print(f"✓ {len(idx_to_class)} classes loaded")

    # Load transforms
    transform = get_val_transforms(image_size=args.image_size)

    # Make prediction
    print(f"\n🔍 Analyzing image: {args.image}")
    predictions = predict_image(
        model, args.image, transform, device, idx_to_class, top_k=args.top_k
    )

    # Display results
    print("\n" + "="*70)
    print("🎯 PREDICTION RESULTS")
    print("="*70)

    for i, pred in enumerate(predictions, 1):
        class_name = pred['class'].replace('___', ' - ')
        confidence = pred['confidence'] * 100

        # Create progress bar
        bar_length = 40
        filled_length = int(bar_length * pred['confidence'])
        bar = '█' * filled_length + '░' * (bar_length - filled_length)

        if i == 1:
            print(f"\n🏆 TOP PREDICTION:")
            print(f"   {class_name}")
            print(f"   Confidence: {confidence:.2f}%")
            print(f"   [{bar}] {confidence:.1f}%")
        else:
            print(f"\n{i}. {class_name}")
            print(f"   [{bar}] {confidence:.1f}%")

    print("\n" + "="*70)

    # Additional info
    top_pred = predictions[0]
    if top_pred['confidence'] > 0.9:
        print("✅ High confidence prediction (>90%)")
    elif top_pred['confidence'] > 0.7:
        print("⚠️  Moderate confidence prediction (70-90%)")
    else:
        print("⚠️  Low confidence prediction (<70%)")
        print("   Consider:")
        print("   - Retaking the image with better lighting")
        print("   - Using a clearer close-up of the leaf")
        print("   - Checking if the plant species is in the supported list")

    print(f"\n💡 For detailed treatment recommendations, use the web app:")
    print(f"   streamlit run app.py")
    print()


if __name__ == '__main__':
    main()
