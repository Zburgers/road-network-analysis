# src/train.py

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.model import get_model, get_loss, get_optimizer, get_scheduler
from src.evaluate import evaluate_model
from src.utils import save_checkpoint, load_data

def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load datasets
    train_dataset, val_dataset = load_data(config)
    train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    # Setup model
    model = get_model(config).to(device)
    criterion = get_loss()
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    
    best_dice = 0.0
    
    for epoch in range(config['train']['epochs']):
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
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
        # Validation and checkpoint
        dice_score = evaluate_model(model, val_loader, device)
        scheduler.step(dice_score)

        if dice_score > best_dice:
            best_dice = dice_score
            save_checkpoint(model, optimizer, epoch, best_dice, config)
            print(f"✅ New best model saved! Dice: {dice_score:.4f}")
        else:
            print(f"⏭ No improvement. Best Dice: {best_dice:.4f}")
