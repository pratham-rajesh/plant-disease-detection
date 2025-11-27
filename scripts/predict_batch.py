#!/usr/bin/env python
"""
Batch Image Prediction Script
Process multiple images and save results to CSV.

Usage:
    python scripts/predict_batch.py --input-dir path/to/images --output results.csv
    python scripts/predict_batch.py --input-dir test_images/ --output predictions.csv --pattern "*.jpg"
"""

import sys
import os
import argparse
import json
from pathlib import Path
import glob

# Add src to path
sys.path.append('src')

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm

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


def predict_image(model, image_path, transform, device, idx_to_class):
    """Make prediction on an image."""
    try:
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

        # Get top-3 predictions
        top3_probs, top3_indices = torch.topk(probabilities, 3)

        result = {
            'filename': os.path.basename(image_path),
            'success': True,
            'predicted_class': idx_to_class[str(top3_indices[0].item())],
            'confidence': float(top3_probs[0]),
            'top2_class': idx_to_class[str(top3_indices[1].item())],
            'top2_confidence': float(top3_probs[1]),
            'top3_class': idx_to_class[str(top3_indices[2].item())],
            'top3_confidence': float(top3_probs[2])
        }

        return result

    except Exception as e:
        return {
            'filename': os.path.basename(image_path),
            'success': False,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(
        description='Batch predict plant diseases from multiple images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/predict_batch.py --input-dir test_images/ --output results.csv
  python scripts/predict_batch.py --input-dir data/processed/test/Tomato___Late_blight/ --output tomato_predictions.csv
  python scripts/predict_batch.py --input-dir images/ --output results.csv --pattern "*.png"
        """
    )

    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Directory containing images to predict'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output CSV file path'
    )

    parser.add_argument(
        '--pattern',
        type=str,
        default='*.jpg',
        help='File pattern to match (default: *.jpg). Use "*.{jpg,jpeg,png}" for multiple types'
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
        '--image-size',
        type=int,
        default=224,
        help='Input image size (default: 224)'
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.input_dir):
        print(f"❌ Error: Input directory not found: {args.input_dir}")
        sys.exit(1)

    if not os.path.exists(args.model):
        print(f"❌ Error: Model not found: {args.model}")
        print("   Please train the model first using notebooks/02_train_resnet50_demo.ipynb")
        sys.exit(1)

    if not os.path.exists(args.class_mapping):
        print(f"❌ Error: Class mapping not found: {args.class_mapping}")
        print("   Please run preprocessing first using notebooks/01_data_preprocessing_eda.ipynb")
        sys.exit(1)

    # Find images
    print(f"🔍 Finding images in {args.input_dir}")

    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []

    for ext in image_extensions:
        pattern = os.path.join(args.input_dir, '**', ext) if '**' not in args.input_dir else os.path.join(args.input_dir, ext)
        image_files.extend(glob.glob(pattern, recursive=True))

    if not image_files:
        print(f"❌ No images found in {args.input_dir}")
        print(f"   Searched for: {', '.join(image_extensions)}")
        sys.exit(1)

    print(f"✓ Found {len(image_files)} images")

    # Load model and mapping
    print("\n🔧 Loading model...")
    model, device = load_model(args.model)
    print(f"✓ Model loaded on {device}")

    print("📋 Loading class mapping...")
    idx_to_class = load_class_mapping(args.class_mapping)
    print(f"✓ {len(idx_to_class)} classes loaded")

    # Load transforms
    transform = get_val_transforms(image_size=args.image_size)

    # Process images
    print(f"\n🚀 Processing {len(image_files)} images...")
    results = []

    for image_path in tqdm(image_files, desc="Predicting"):
        result = predict_image(model, image_path, transform, device, idx_to_class)
        results.append(result)

    # Create DataFrame
    df = pd.DataFrame(results)

    # Format results
    if 'success' in df.columns and df['success'].all():
        # All predictions successful
        df_output = pd.DataFrame({
            'Filename': df['filename'],
            'Predicted Disease': df['predicted_class'].str.replace('___', ' - '),
            'Confidence (%)': (df['confidence'] * 100).round(2),
            'Top-2 Disease': df['top2_class'].str.replace('___', ' - '),
            'Top-2 Confidence (%)': (df['top2_confidence'] * 100).round(2),
            'Top-3 Disease': df['top3_class'].str.replace('___', ' - '),
            'Top-3 Confidence (%)': (df['top3_confidence'] * 100).round(2)
        })
    else:
        # Some predictions failed
        df_output = df

    # Save to CSV
    df_output.to_csv(args.output, index=False)
    print(f"\n✅ Results saved to: {args.output}")

    # Summary statistics
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)

    if 'success' in df.columns:
        successful = df['success'].sum()
        failed = len(df) - successful
        print(f"Total images processed: {len(df)}")
        print(f"Successful predictions: {successful}")
        if failed > 0:
            print(f"Failed predictions: {failed}")

    if 'confidence' in df.columns and df['confidence'].notna().any():
        print(f"\nConfidence Statistics:")
        print(f"  Mean: {df['confidence'].mean()*100:.2f}%")
        print(f"  Median: {df['confidence'].median()*100:.2f}%")
        print(f"  Min: {df['confidence'].min()*100:.2f}%")
        print(f"  Max: {df['confidence'].max()*100:.2f}%")

        high_conf = (df['confidence'] > 0.9).sum()
        med_conf = ((df['confidence'] >= 0.7) & (df['confidence'] <= 0.9)).sum()
        low_conf = (df['confidence'] < 0.7).sum()

        print(f"\nConfidence Distribution:")
        print(f"  High (>90%): {high_conf} images ({high_conf/len(df)*100:.1f}%)")
        print(f"  Medium (70-90%): {med_conf} images ({med_conf/len(df)*100:.1f}%)")
        print(f"  Low (<70%): {low_conf} images ({low_conf/len(df)*100:.1f}%)")

    if 'predicted_class' in df.columns:
        print(f"\nTop 5 Most Common Predictions:")
        top_diseases = df['predicted_class'].value_counts().head(5)
        for disease, count in top_diseases.items():
            print(f"  {disease.replace('___', ' - ')}: {count} images")

    print("\n" + "="*70)
    print(f"\n💡 Open {args.output} in Excel or any spreadsheet software to view results")
    print()


if __name__ == '__main__':
    main()
