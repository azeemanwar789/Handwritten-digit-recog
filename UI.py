import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
MODEL_PATH = r"C:\Azeem's Work\Project 2023-24\Handwritten_Digit_Recognition_(MNIST)\mnist_cnn.h5"
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Function to preprocess the image
def preprocess_image(image):
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28 pixels
    image = np.array(image) / 255.0  # Normalize pixel values
    image = image.reshape(1, 28, 28, 1)  # Reshape for model input
    return image

# Streamlit UI
st.title("🖊️ Handwritten Digit Recognition 🤖")
st.write("Upload an image of a handwritten digit, and the model will predict the number.")

# File uploader
uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_digit = np.argmax(prediction)

    # Show prediction
    st.write(f"### 🏆 Predicted Digit: **{predicted_digit}**")

    # Show confidence scores
    st.bar_chart(prediction[0])

# Footer
st.markdown("---")
st.write("Made with ❤️ using Streamlit & TensorFlow")
