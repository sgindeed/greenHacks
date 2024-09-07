import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# Use relative paths for deployment
model_path = "plant_disease_prediction_model.h5"

# Check if model file exists
if not os.path.exists(model_path):
    st.error(f"Model file not found. Please ensure the file exists.")
    st.stop()

# Load the model
model = tf.keras.models.load_model(model_path)

# Load class indices
class_indices_path = "class_indices.json"
if not os.path.exists(class_indices_path):
    st.error(f"Class indices file not found. Please ensure the file exists.")
    st.stop()

with open(class_indices_path, 'r') as f:
    class_indices = json.load(f)

def load_and_preprocess_image(image, target_size=(224, 224)):
    img = Image.open(image)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

def predict_image_class(model, image, class_indices):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices.get(str(predicted_class_index), "Unknown class")
    return predicted_class_name

st.title('Plant Disease Detector for Multiple Crops')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            try:
                prediction = predict_image_class(model, uploaded_image, class_indices)
                st.success(f'Prediction: {prediction}')
            except Exception as e:
                st.error(f"An error occurred: {e}")
