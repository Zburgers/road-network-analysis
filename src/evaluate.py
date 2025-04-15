# src/evaluate.py

import torch
import numpy as np
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score

def dice_score(pred, target):
    smooth = 1e-6
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def evaluate_model(model, dataloader, device):
    model.eval()
    metrics = {
        'dice': [],
        'iou': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()

            # Calculate metrics
            dice = dice_score(preds, masks)
            metrics['dice'].append(dice.item())

            # Convert to numpy for sklearn metrics
            preds_np = preds.cpu().numpy().flatten()
            masks_np = masks.cpu().numpy().flatten()

            # Calculate other metrics
            metrics['iou'].append(jaccard_score(masks_np, preds_np))
            metrics['precision'].append(precision_score(masks_np, preds_np))
            metrics['recall'].append(recall_score(masks_np, preds_np))
            metrics['f1'].append(f1_score(masks_np, preds_np))

    # Calculate and print average metrics
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    print("\n📊 Model Performance Metrics:")
    print(f"🎯 Dice Score: {avg_metrics['dice']:.4f}")
    print(f"📐 IoU Score: {avg_metrics['iou']:.4f}")
    print(f"🎯 Precision: {avg_metrics['precision']:.4f}")
    print(f"🎯 Recall: {avg_metrics['recall']:.4f}")
    print(f"🎯 F1 Score: {avg_metrics['f1']:.4f}")

    return avg_metrics
