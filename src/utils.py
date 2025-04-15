# src/utils.py

import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import random
import shutil
from pathlib import Path

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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Added function: create_dirs
def create_dirs(paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)


def visualize_prediction(image, mask, prediction):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    plt.imshow(image, cmap='gray')
    plt.subplot(1, 3, 2)
    plt.title("Ground Truth Mask")
    plt.imshow(mask, cmap='gray')
    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask")
    plt.imshow(prediction, cmap='gray')
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


def save_checkpoint(model, optimizer, epoch, metrics, config, best=False):
    os.makedirs(config['train']['checkpoint_dir'], exist_ok=True)
    filename = 'best_model.pth' if best else 'last_epoch.pth'
    save_path = os.path.join(config['train']['checkpoint_dir'], filename)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }, save_path)


def load_checkpoint(filepath, model, optimizer):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    metrics = checkpoint.get('metrics', {'dice': 0.0})
    return model, optimizer, epoch + 1, metrics

def clear_corrupted_model_cache(model_name):
    """
    Clear potentially corrupted model cache files for a specific model.
    
    Args:
        model_name (str): Name of the model architecture (e.g., 'resnet50')
    
    Returns:
        bool: True if cache was cleared, False otherwise
    """
    print(f"Attempting to clear cache for {model_name}...")
    # Get the torch hub cache directory
    cache_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints")
    
    # Windows-specific PyTorch cache location
    windows_cache_dir = os.path.expanduser("~/AppData/Local/torch/hub/checkpoints")
    
    if os.path.exists(windows_cache_dir):
        cache_dir = windows_cache_dir
    
    # If the directory doesn't exist, there's nothing to clear
    if not os.path.exists(cache_dir):
        print(f"Cache directory {cache_dir} does not exist. Nothing to clear.")
        return False
        
    # Find all files related to the model
    model_files = [f for f in os.listdir(cache_dir) if model_name in f]
    
    if not model_files:
        print(f"No cached files found for {model_name}")
        return False
        
    # Remove each cached file
    for file_name in model_files:
        file_path = os.path.join(cache_dir, file_name)
        try:
            os.remove(file_path)
            print(f"Removed corrupted cache file: {file_path}")
        except Exception as e:
            print(f"Error removing {file_path}: {e}")
            
    return True
