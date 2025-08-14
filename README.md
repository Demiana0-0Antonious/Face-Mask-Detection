# Face-Mask-Detection
A simple deep learning project to detect whether a person is wearing a mask or not. Built using TensorFlow/Keras and deployed with Gradio / FastAPI / Streamlit.

## Features 
- Detects **Mask** vs **No Mask** in images.
- Web interface using Gradio / FastAPI / Streamlit.
- Easy to use: upload an image and get prediction with confidence scores.

## Model
- Model type: **Convolutional Neural Network (CNN)**  
- Total parameters: 31,938  
- Input shape: `(128, 128, 3)`  
- Output: 2 classes (`Mask` and `No Mask`)  
- Saved as: `face_mask_model.h5`

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/FaceMaskProject.git
cd FaceMaskProject

2. Create a conda environment:
conda create -n mask_env python=3.10 -y
conda activate mask_env

3. Install dependencies:
pip install -r requirements.txt
```markdown

## Usage

### Gradio
```bash
python app.py
Open the link shown in the terminal to test the app in your browser.

##FastAPI
```bash
uvicorn api:app --reload   
Visit http://127.0.0.1:8000 in your browser.

##Streamlit
```bash
streamlit run stream.py 
Open the link in the terminal to test the app.

