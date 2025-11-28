"""
Plant Disease Detection - Streamlit Web Application

Authors: Pratham Rajesh, Shreram Palanisamy
Date: November 2025

This Streamlit app provides a user-friendly interface for plant disease detection
using the trained ResNet50 model.
"""

import streamlit as st
from PIL import Image
import numpy as np
import json
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Configuration
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #616161;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disease-result {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .healthy {
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
    }
    .diseased {
        background-color: #FFEBEE;
        border-left: 5px solid #F44336;
    }
    .metric-card {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 5px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# Load model and class names
@st.cache_resource
def load_model():
    """Load the trained ResNet50 model."""
    try:
        import tensorflow as tf
        model_path = Path(__file__).parent.parent / "models" / "resnet50_best.h5"

        if not model_path.exists():
            st.error(f"Model not found at {model_path}")
            st.info("Please train the model first by running the training notebook.")
            st.stop()

        model = tf.keras.models.load_model(str(model_path))
        return model
    except ImportError:
        st.error("TensorFlow not installed. Please install dependencies: pip install -r requirements.txt")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()


@st.cache_data
def load_class_names():
    """Load class names from JSON file."""
    try:
        class_names_path = Path(__file__).parent.parent / "models" / "class_names.json"

        if not class_names_path.exists():
            st.error(f"Class names not found at {class_names_path}")
            st.info("Please run the EDA notebook first to generate class names.")
            st.stop()

        with open(class_names_path, 'r') as f:
            class_names = json.load(f)
        return class_names
    except Exception as e:
        st.error(f"Error loading class names: {e}")
        st.stop()


def preprocess_image(image: Image.Image, target_size=(224, 224)):
    """
    Preprocess image for model prediction.

    Args:
        image: PIL Image object
        target_size: Target size for resizing (default: 224x224)

    Returns:
        Preprocessed image array ready for prediction
    """
    # Resize image
    image = image.resize(target_size)

    # Convert to array
    img_array = np.array(image) / 255.0

    # Apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def parse_class_name(class_name):
    """
    Parse class name to extract plant type and disease.

    Args:
        class_name: Full class name (e.g., "Tomato___Early_blight")

    Returns:
        Tuple of (plant_type, disease_name)
    """
    if "___" in class_name:
        plant, disease = class_name.split("___", 1)
        disease = disease.replace("_", " ")
    else:
        plant = class_name
        disease = "Unknown"

    return plant, disease


def display_prediction_results(predictions, class_names, top_k=5):
    """
    Display prediction results in a formatted way.

    Args:
        predictions: Model predictions (probabilities)
        class_names: List of class names
        top_k: Number of top predictions to show
    """
    # Get top-k predictions
    top_k_indices = np.argsort(predictions[0])[-top_k:][::-1]

    # Top prediction
    top_idx = top_k_indices[0]
    top_class = class_names[top_idx]
    top_prob = predictions[0][top_idx]

    # Parse class name
    plant, disease = parse_class_name(top_class)

    # Display main result
    st.markdown("### 🔍 Diagnosis Results")

    is_healthy = "healthy" in disease.lower()
    result_class = "healthy" if is_healthy else "diseased"

    st.markdown(f"""
    <div class="disease-result {result_class}">
        <h2>{'✅' if is_healthy else '⚠️'} {plant}</h2>
        <h3>{disease.title()}</h3>
        <p style="font-size: 1.2rem; margin-top: 1rem;">
            <strong>Confidence:</strong> {top_prob*100:.2f}%
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Confidence meter
    st.metric(label="Prediction Confidence", value=f"{top_prob*100:.1f}%")

    # Display top-k predictions
    st.markdown("### 📊 Top Predictions")

    for i, idx in enumerate(top_k_indices, 1):
        class_name = class_names[idx]
        prob = predictions[0][idx]
        plant_name, disease_name = parse_class_name(class_name)

        with st.expander(f"#{i}: {plant_name} - {disease_name} ({prob*100:.2f}%)", expanded=(i==1)):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.write(f"**Plant:** {plant_name}")
                st.write(f"**Condition:** {disease_name}")
                st.write(f"**Probability:** {prob*100:.2f}%")

            with col2:
                st.progress(float(prob))

    # Recommendations (simple example)
    if not is_healthy:
        st.markdown("### 💡 Recommendations")
        st.info("""
        **General recommendations for plant diseases:**
        - Remove and destroy infected plant parts
        - Ensure proper air circulation
        - Avoid overhead watering
        - Apply appropriate fungicides if needed
        - Consult with a local agricultural extension for specific treatment

        **Note:** This is an AI prediction. For critical decisions, please consult with a professional agronomist.
        """)


def main():
    """Main Streamlit application."""

    # Header
    st.markdown('<h1 class="main-header">🌿 Plant Disease Detection</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Upload an image of a plant leaf to detect diseases using AI</p>',
        unsafe_allow_html=True
    )

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This application uses a deep learning model (ResNet50) trained on the PlantVillage dataset to identify plant diseases from leaf images.

        **Model Details:**
        - Architecture: ResNet50 (Transfer Learning)
        - Training Data: 55,000+ images
        - Classes: 39 plant diseases + healthy
        - Accuracy: ~95% on test set
        """)

        st.markdown("---")

        st.header("🌱 Supported Plants")
        st.markdown("""
        - Apple
        - Blueberry
        - Cherry
        - Corn (Maize)
        - Grape
        - Orange
        - Peach
        - Pepper (Bell)
        - Potato
        - Raspberry
        - Soybean
        - Squash
        - Strawberry
        - Tomato
        """)

        st.markdown("---")

        st.header("👥 Team")
        st.markdown("""
        **Authors:**
        - Pratham Rajesh
        - Shreram Palanisamy

        **Date:** November 2025
        """)

        st.markdown("---")

        st.caption("Built with Streamlit & TensorFlow")

    # Main content
    try:
        # Load model and class names
        with st.spinner("Loading AI model..."):
            model = load_model()
            class_names = load_class_names()

        st.success(f"✅ Model loaded successfully! Ready to analyze {len(class_names)} plant conditions.")

        # File uploader
        st.markdown("### 📤 Upload Leaf Image")
        uploaded_file = st.file_uploader(
            "Choose a leaf image (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of a plant leaf for disease detection"
        )

        if uploaded_file is not None:
            # Display uploaded image
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Original Image")
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, use_column_width=True, caption="Uploaded leaf image")

                # Image info
                st.markdown("**Image Details:**")
                st.write(f"- Size: {image.size[0]} x {image.size[1]} pixels")
                st.write(f"- Format: {image.format}")

            with col2:
                st.markdown("#### Analysis")

                if st.button("🔍 Detect Disease", type="primary", use_container_width=True):
                    with st.spinner("Analyzing leaf image..."):
                        # Preprocess image
                        processed_img = preprocess_image(image)

                        # Make prediction
                        import time
                        start_time = time.time()
                        predictions = model.predict(processed_img, verbose=0)
                        inference_time = time.time() - start_time

                        # Display results
                        display_prediction_results(predictions, class_names, top_k=5)

                        # Show inference time
                        st.caption(f"⚡ Analysis completed in {inference_time:.3f} seconds")

        else:
            # Instructions when no file is uploaded
            st.info("👆 Please upload a leaf image to begin analysis")

            # Sample images section
            st.markdown("### 📸 Example Use Cases")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("""
                **Healthy Leaves**
                - Identify healthy plants
                - Verify plant condition
                - Track plant health over time
                """)

            with col2:
                st.markdown("""
                **Disease Detection**
                - Early disease identification
                - Fungal infections
                - Bacterial spots
                """)

            with col3:
                st.markdown("""
                **Viral Infections**
                - Leaf curl viruses
                - Mosaic patterns
                - Yellowing conditions
                """)

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.info("Please ensure the model is trained and all required files are in place.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p><strong>Disclaimer:</strong> This tool is for educational and research purposes.
        For critical agricultural decisions, please consult with professional agronomists or plant pathologists.</p>
        <p>© 2025 Plant Disease Detection Project | Built with ❤️ using Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
