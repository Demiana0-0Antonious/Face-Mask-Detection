import streamlit as st
import requests
from PIL import Image

st.title("😷 Face Mask Detection (via FastAPI)")

FASTAPI_URL = "http://127.0.0.1:8000/predict"  # Your API endpoint

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Send image to FastAPI
    files = {
        "file": (uploaded_file.name, uploaded_file.getvalue(), "image/jpeg")
    }
    response = requests.post(FASTAPI_URL, files=files)

    if response.status_code == 200:
        result = response.json()
        st.subheader(f"Prediction: **{result['label']}**")
        st.write(f"Confidence: {result['confidence']*100:.2f}%")
        
        # Show detailed class scores
        st.json(result["scores"])
    else:
        st.error(f"Error: {response.status_code} - {response.text}")

