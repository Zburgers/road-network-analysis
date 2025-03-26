# streamlit_dashboard/app.py

import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
from streamlit_dashboard.utils import predict_and_visualize

st.set_page_config(page_title="RoadNet: Live Road Segmentation", layout="centered")

st.title("🛣️ RoadNet - Satellite Road Segmentation")
st.markdown("Upload a satellite image and visualize road segmentation predictions using the trained model.")

# Upload image
uploaded_image = st.file_uploader("📷 Upload Satellite Image", type=["png", "jpg", "jpeg"])

# Load model
@st.cache_resource
def load_model():
    from src.model import get_deeplab_model
    model = get_deeplab_model()
    checkpoint = torch.load("checkpoint.pth", map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="📸 Uploaded Image", use_column_width=True)

    if st.button("🚀 Predict Roads"):
        with st.spinner("Processing..."):
            model = load_model()
            overlay = predict_and_visualize(model, image)
            st.image(overlay, caption="🧠 Predicted Road Segmentation", use_column_width=True)
