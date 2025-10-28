import streamlit as st
from ultralytics import YOLO
from PIL import Image
import torch

st.title("🛒 Shoplifting Detection MVP (YOLOv12)")
st.write("Upload an image and detect suspicious activity.")

# Load model once
@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    return model

model = load_model()

# Upload image
uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Inference
    st.write("Running detection...")
    results = model(img)

    # Show results
    res_plotted = results[0].plot()
    st.image(res_plotted, caption="Detection Results", use_column_width=True)
