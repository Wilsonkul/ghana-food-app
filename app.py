import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your trained model
model = tf.keras.models.load_model("ghana_food_model.h5")

# Your class names from training
class_names = ['Banku and Tilapia', 'Bofrot', 'Fufu with Lightsoup', 'Jollof Rice', 'Kenkey with Fried Fish', 'Okro soup']  # update if different

# App Title
st.title("Ghanaian Food Classifier")

# Image uploader
uploaded_file = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) / 255.0

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = 100 * np.max(prediction)

    st.markdown(f"**Prediction:** `{predicted_class}`")
    st.markdown(f"**Confidence:** `{confidence:.2f}%`")
