"""
Natural Scene Classifier — Streamlit App
=========================================
A custom Convolutional Neural Network (built and trained from scratch —
no pretrained backbone / transfer learning) that classifies natural scene
photos into 6 categories: buildings, forest, glacier, mountain, sea, street.

Model:   4-block Conv2D + MaxPooling CNN, ~11M params
Dataset: Intel Image Classification (Kaggle), ~25,000 images
Result:  81.03% test accuracy / 0.5469 test loss on 3,000 held-out images

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
MODEL_PATH = "models/natural_scenes_BEST_model.keras"
IMG_SIZE = (150, 150)

# IMPORTANT: this order matches Keras' flow_from_directory alphabetical
# class indexing exactly as it was during training (verified against the
# training notebook's printed class_indices). Do not reorder this list —
# doing so silently scrambles every prediction.
CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

st.set_page_config(page_title="Natural Scene Classifier", page_icon="🏞️", layout="centered")


# ---------------------------------------------------------------------
# Model loading — cached so the ~42MB model loads once per session
# instead of on every single prediction (this was the biggest practical
# upgrade over a naive load-on-every-run implementation).
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    try:
        return keras.models.load_model(MODEL_PATH)
    except (OSError, IOError):
        st.error(
            f"Could not find the model file at `{MODEL_PATH}`. "
            "Make sure natural_scenes_BEST_model.keras is in the models/ folder "
            "(it should be committed to the repo — it's only ~42MB)."
        )
        st.stop()


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize, channel-normalize, and batch a PIL image to match model input.

    Mirrors the exact preprocessing used during training:
    resize -> 150x150, force 3 channels, scale to [0, 1], add batch dim.
    """
    image = image.resize(IMG_SIZE)
    img_array = np.array(image)

    # Handle grayscale (H, W) and RGBA (H, W, 4) uploads gracefully
    if img_array.ndim == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[..., :3]

    img_array = np.expand_dims(img_array, axis=0).astype("float32") / 255.0
    return img_array


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
st.title("🏞️ Natural Scene Classifier")
st.markdown("Upload a landscape photo to instantly classify it into one of six categories: **buildings, forest, glacier, mountain, sea, or street**.")

# 1. Hide the technical details in a clean, collapsible expander
with st.expander("ℹ️ About the Model"):
    st.markdown("""
    - **Architecture:** Custom 4-block CNN (trained from scratch)
    - **Accuracy:** 81.03% on unseen test data
   - **Learn More:** Check out the full training report and dataset details in the [GitHub Repository](https://github.com/alwin-1107/scenery-classification-app).
    """)

model = load_model()

uploaded_file = st.file_uploader("Drag and drop an image file here, or click to browse", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # 2. Create a professional side-by-side layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
    with col2:
        with st.spinner("Classifying..."):
            processed = preprocess_image(image)
            prediction = model.predict(processed, verbose=0)[0]

        predicted_idx = int(np.argmax(prediction))
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(prediction[predicted_idx])

        # 3. Big, bold metric for the top result
        st.metric(
            label="Top Prediction", 
            value=predicted_class.title(), 
            delta=f"{confidence:.2%} Confidence", 
            delta_color="normal"
        )
        
        # 4. Cleaned-up bar chart
        st.write("**All Class Probabilities**")
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(CLASS_NAMES, prediction, color="#4A90E2")
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1)
        
        # Remove top and right borders for a cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xticks(rotation=45)
        
        st.pyplot(fig)
else:
    st.info("Upload an image to get a prediction.")
