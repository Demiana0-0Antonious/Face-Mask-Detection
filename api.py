from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import tensorflow as tf
from fastapi.responses import JSONResponse

app = FastAPI()

# Load model once at startup
model = tf.keras.models.load_model("face_mask_model.h5")
class_labels = {0: "Mask", 1: "No Mask"}

def preprocess_image(image):
    """Convert image to 128x128 RGB array with normalization."""
    image = image.convert("RGB").resize((128, 128))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape (1, 128, 128, 3)
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file)
    except Exception:
        return JSONResponse(content={"error": f"Invalid image file: {file.filename}"}, status_code=400)
    
    img_array = preprocess_image(image)
    preds = model.predict(img_array, verbose=0)[0]
    predicted_class = np.argmax(preds)
    confidence = float(preds[predicted_class])
    label = class_labels[predicted_class]
    
    return {
        "filename": file.filename,
        "label": label,
        "confidence": confidence
    }

@app.get("/")
def root():
    return {"message": "FastAPI Mask Detection API is running!"}

