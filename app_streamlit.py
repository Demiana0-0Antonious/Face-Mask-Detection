import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("face_mask_model.h5")
class_labels = {0: "Mask", 1: "No Mask"}

def preprocess_image(image):
    # Convert numpy array to PIL if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((128, 128))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

st.title("Face Mask Detection")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    img_array = preprocess_image(image)
    preds = model.predict(img_array)[0]
    predicted_class = np.argmax(preds)
    confidence = preds[predicted_class]
    label = class_labels[predicted_class]
    st.write(f"Prediction: **{label}** ({confidence*100:.2f}%)")
