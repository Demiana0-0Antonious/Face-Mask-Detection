print('This code is RUNNING NOW NOW NOW!!! ')

import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your trained model
model = tf.keras.models.load_model("face_mask_model.h5")
class_labels = {0: "Mask", 1: "No Mask"}

def preprocess_image(image):
    # Convert numpy array (from Gradio) to PIL Image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((128, 128))  # Resize to your model's expected input size
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_mask(image):
    img_array = preprocess_image(image)
    preds = model.predict(img_array)[0]
    predicted_class = np.argmax(preds)
    confidence = preds[predicted_class]
    label = class_labels[predicted_class]
    return {label: float(confidence)}

iface = gr.Interface(
    fn=predict_mask,
    inputs=gr.Image(label="Upload Image", image_mode="RGB"),
    outputs=gr.Label(num_top_classes=2),
    title="Face Mask Detection",
    description="Upload an image for mask/no mask classification."
)

iface.launch(share=True)







