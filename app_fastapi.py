from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import tensorflow as tf

app = FastAPI()
model = tf.keras.models.load_model("face_mask_model.h5")
class_labels = {0: "Mask", 1: "No Mask"}

def preprocess_image(image):
    image = image.convert("RGB").resize((128, 128))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file)
    img_array = preprocess_image(image)
    preds = model.predict(img_array)[0]
    predicted_class = np.argmax(preds)
    confidence = float(preds[predicted_class])
    label = class_labels[predicted_class]
    return {"label": label, "confidence": confidence}

@app.get("/")
def root():
    return {"message": "FastAPI Mask Detection API is running!"}
