# src/utils.py

import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# ----------------------------
# ✅ Checkpointing Functions
# ----------------------------
def save_checkpoint(state, filename="checkpoint.pth"):
    torch.save(state, filename)
    print(f"[💾] Model checkpoint saved to {filename}")

def load_checkpoint(filename, model, optimizer=None):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])
    print(f"[📂] Checkpoint loaded from {filename}")
    return checkpoint.get("epoch", 0)


# ----------------------------
# 🔍 Image Utilities
# ----------------------------
def mask_to_rgb(mask):
    """Converts a 1-channel binary mask into 3-channel RGB for visualization"""
    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    rgb_mask[mask == 1] = [255, 255, 255]  # white for roads
    return rgb_mask


def visualize_sample(image, mask, pred=None):
    """Shows the input image, ground truth mask, and predicted mask if available"""
    plt.figure(figsize=(12, 4))
    
    # Image
    plt.subplot(1, 3 if pred is not None else 2, 1)
    plt.imshow(image.permute(1, 2, 0))
    plt.title("🛰️ Satellite Image")
    plt.axis("off")

    # Ground Truth
    plt.subplot(1, 3 if pred is not None else 2, 2)
    plt.imshow(mask_to_rgb(mask))
    plt.title("📌 Ground Truth")
    plt.axis("off")

    # Prediction
    if pred is not None:
        plt.subplot(1, 3, 3)
        plt.imshow(mask_to_rgb(pred))
        plt.title("🔮 Predicted")
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()


# ----------------------------
# 📦 Inference Helper
# ----------------------------
def predict_image(model, image_path, device, transform):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        pred_mask = (torch.sigmoid(output) > 0.5).float().squeeze().cpu().numpy()

    return np.array(image), pred_mask
