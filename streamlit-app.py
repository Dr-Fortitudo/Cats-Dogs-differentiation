import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2  # OpenCV for image processing

# Load the trained model
MODEL_PATH = "cats_dogs_model.h5"
model = load_model(MODEL_PATH)

# Define class labels
CLASS_LABELS = {0: "Cat", 1: "Dog", 2: "Not a Cat nor Dog"}

# Function to preprocess image
def preprocess_image(image):
    image = image.convert("RGB")  # Ensure 3-channel RGB
    image = image.resize((150, 150))  # Fixed size to match model input (150x150)
    img_array = np.array(image, dtype=np.float32) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Streamlit UI
st.title("Cat vs Dog Classifier")
st.write("Upload an image, and the model will classify it as a Cat, Dog, or Neither.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)  # Fixed deprecation warning
    
    try:
        # Preprocess the image
        img_array = preprocess_image(image)
        
        # Get model prediction
        prediction = model.predict(img_array)
        predicted_class = 0 if prediction[0][0] > 0.5 else 1
        
        # Optional: Threshold-based rejection (if confidence is too low, classify as "Neither")
        confidence = prediction[0][0]
        if 0.4 < confidence < 0.6:
            predicted_class = 2  # Not a cat nor dog
        
        st.write(f"Prediction: **{CLASS_LABELS[predicted_class]}**")
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
