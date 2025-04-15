import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from datetime import datetime

from src.model import get_model, get_loss, get_optimizer, get_scheduler
from src.evaluate import evaluate_model
from src.utils import save_checkpoint, load_checkpoint
from src.data_loader import get_dataloaders


def train_model(config):
    device = torch.device(config['train'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")

    # Load datasets
    train_loader, val_loader = get_dataloaders(config)
    print(f"Training on {len(train_loader.dataset)} samples")
    print(f"Validating on {len(val_loader.dataset)} samples")

    # Setup model
    model = get_model(config).to(device)
    criterion = get_loss()
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    start_epoch = 0
    best_metrics = {'dice': 0.0}
    patience_counter = 0
    max_patience = config['train']['early_stopping_patience']

    # Resume training if checkpoint exists
    last_ckpt = os.path.join(config['train']['checkpoint_dir'], 'last_epoch.pth')
    if config['train'].get('resume_training', False) and os.path.exists(last_ckpt):
        model, optimizer, start_epoch, best_metrics = load_checkpoint(last_ckpt, model, optimizer)
        print(f"🔄 Resumed from epoch {start_epoch+1} with best Dice: {best_metrics['dice']:.4f}")

    # Training loop
    for epoch in range(start_epoch, config['train']['epochs']):
        model.train()
        epoch_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{config['train']['epochs']}]")

        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)
        print(f"\n📉 Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")

        # Validation
        print("\nValidating...")
        val_metrics = evaluate_model(model, val_loader, device)
        scheduler.step(val_metrics['dice'])

        # Save best model
        if val_metrics['dice'] > best_metrics['dice']:
            best_metrics = val_metrics
            save_checkpoint(model, optimizer, epoch, best_metrics, config, best=True)
            print(f"✅ New best model saved! Dice: {val_metrics['dice']:.4f}")
            patience_counter = 0
        else:
            print(f"⏭ No improvement. Best Dice: {best_metrics['dice']:.4f}")
            patience_counter += 1

        # Save last checkpoint always
        save_checkpoint(model, optimizer, epoch, best_metrics, config, best=False)

        # Early stopping
        if patience_counter >= max_patience:
            print(f"🛑 Early stopping triggered. Best Dice: {best_metrics['dice']:.4f}")
            break

        # Log epoch summary
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"Train Loss: {avg_loss:.4f}")
        print(f"Validation Metrics:")
        for metric, value in val_metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")
        print("-" * 50)

    return model, best_metrics
