import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("fine_tuned_mobilenetv2_brain_tumor.h5")
    return model

model = load_model()

# Class labels (same order as training)
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# App UI
st.title("🧠 Brain Tumor Classification")
st.write("Upload an MRI image and the model will predict the tumor type.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)
    predicted_class = class_names[np.argmax(preds)]

    st.markdown(f"### 🧾 Prediction: `{predicted_class}`")
    st.bar_chart(preds[0])
