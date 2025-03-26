# streamlit_dashboard/utils.py

import torch
import numpy as np
from torchvision import transforms
from PIL import Image

def predict_and_visualize(model, pil_image):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(pil_image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
        pred_mask = (torch.sigmoid(output) > 0.5).squeeze().numpy()

    # Overlay prediction on original image
    image_np = np.array(pil_image.resize((512, 512))).copy()
    overlay = image_np.copy()
    overlay[pred_mask == 1] = [255, 0, 0]  # red roads
    blended = (0.6 * image_np + 0.4 * overlay).astype(np.uint8)
    return blended
