from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf

app = FastAPI()

# -----------------------------
# Load model once at startup
# -----------------------------
model = tf.keras.models.load_model("face_detect_model.h5")

# Match training order (must be consistent!)
class_labels = ["Mask", "No Mask"]

# -----------------------------
# Preprocessing function
# -----------------------------
def preprocess_image(image):
    """Convert image to 128x128 RGB array with normalization."""
    image = image.convert("RGB").resize((128, 128))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape (1, 128, 128, 3)
    return img_array

# -----------------------------
# Unified prediction function
# -----------------------------
def predict_mask(image):
    img_array = preprocess_image(image)
    preds = model.predict(img_array, verbose=0)[0][0]  # sigmoid → single prob

    # Map probability consistently
    scores = {
        "Mask": float(1 - preds),
        "No Mask": float(preds)
    }

    # Pick label with highest score
    label = max(scores, key=scores.get)
    confidence = scores[label]

    return label, confidence, scores

# -----------------------------
# API endpoint
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file)
    except Exception:
        return JSONResponse(
            content={"error": f"Invalid image file: {file.filename}"},
            status_code=400
        )

    label, confidence, scores = predict_mask(image)

    return {
        "filename": file.filename,
        "label": label,
        "confidence": confidence,
        "scores": scores  # useful for debugging / frontend
    }

@app.get("/")
def root():
    return {"message": "FastAPI Mask Detection API is running!"}
