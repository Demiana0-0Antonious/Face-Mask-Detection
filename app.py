print("This code is RUNNING NOW NOW NOW!!!")

import os
import numpy as np
import tensorflow as tf
import gradio as gr
from PIL import Image
from tensorflow.keras.preprocessing import image as keras_image

# -----------------------------
# Load trained model
# -----------------------------
model = tf.keras.models.load_model("face_detect_model.h5")  # or face_mask_model.h5

# Class labels (must match training order!)
class_names = ["WithMask", "WithoutMask"]

# -----------------------------
# Common preprocessing function
# -----------------------------
def preprocess_image(img):
    if isinstance(img, np.ndarray):  # from Gradio
        img = Image.fromarray(img)
    if isinstance(img, str):  # from file path
        img = keras_image.load_img(img, target_size=(128, 128))
    else:  # PIL Image
        img = img.resize((128, 128))
    if img.mode != "RGB":
        img = img.convert("RGB")

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -----------------------------
# Unified prediction
# -----------------------------
def predict_mask(img):
    img_array = preprocess_image(img)
    prob = model.predict(img_array)[0][0]  # sigmoid → single probability

    # Map to dict for consistency
    scores = {
        "WithMask": float(1 - prob),
        "WithoutMask": float(prob)
    }
    print("Raw prob:", prob, "→ Scores:", scores)  # Debugging line
    return scores

# -----------------------------
# Offline test loop
# -----------------------------
def run_offline_test(test_folder):
    for fname in os.listdir(test_folder):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(test_folder, fname)
            scores = predict_mask(img_path)
            label = max(scores, key=scores.get)
            print(f"{fname} → Prediction: {label}, Scores: {scores}")

# -----------------------------
# Gradio app
# -----------------------------
iface = gr.Interface(
    fn=predict_mask,
    inputs=gr.Image(label="Upload Image", image_mode="RGB"),
    outputs=gr.Label(num_top_classes=2),
    title="Face Mask Detection",
    description="Upload an image to check whether a mask is present."
)

# Example offline test
# run_offline_test(r"D:\nti\face_mask_data\Face_Mask_training\test")

iface.launch(share=True)








