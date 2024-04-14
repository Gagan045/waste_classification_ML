

import streamlit as st
import numpy as np
import cv2
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential, load_model
from PIL import Image

# Load your trained model
cnn_model = load_model("model_3.h5")  # Replace with your actual model path

# Define labels
labels = {0: "N", 1: "O", 2: "R"}  # Replace with your actual class labels

st.title("Waste Classification App")

# Upload an image for classification
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png"])

# Add a "Predict" button
if st.button("Predict"):
    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)

        # Check if the image is not None and then display it
        if image is not None:
            # Convert the PIL image to a NumPy array
            image_np = np.array(image)

            # Preprocess the image (resize, rescale, etc.)
            resized_image = cv2.resize(image_np, (150, 150))
            resized_image = resized_image / 255.0  # Rescale to [0, 1]

            # Make prediction
            prediction = cnn_model.predict(np.expand_dims(resized_image, axis=0))[0]
            predicted_class = labels[np.argmax(prediction)]
            confidence = prediction[np.argmax(prediction)]

            st.image(image_np, caption="Uploaded Image", use_column_width=True)
            st.write(f"Predicted Class: {predicted_class}")
            st.write(f"Confidence: {confidence:.2f}")
        else:
            st.write("No valid image uploaded.")