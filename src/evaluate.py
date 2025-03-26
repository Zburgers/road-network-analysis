# src/evaluate.py

import torch
import numpy as np
from sklearn.metrics import jaccard_score

def dice_score(pred, target):
    smooth = 1e-6
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def evaluate_model(model, dataloader, device):
    model.eval()
    dice_scores = []
    iou_scores = []

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()

            dice = dice_score(preds, masks)
            dice_scores.append(dice.item())

            iou = jaccard_score(masks.cpu().numpy().flatten(), preds.cpu().numpy().flatten())
            iou_scores.append(iou)

    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)
    print(f"📐 Dice Score: {avg_dice:.4f}, IoU Score: {avg_iou:.4f}")
    return avg_dice
