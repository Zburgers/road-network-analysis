import os
import yaml
import torch
import random
import numpy as np
import argparse  # <-- added

from src.train import train_model
from src.utils import set_seed, create_dirs

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Train or Evaluate RoadNet")
    parser.add_argument("--mode", choices=["train", "evaluate"], default="train", help="Choose to train or evaluate the model")
    args = parser.parse_args()
    
    config = load_config()
    
    # ✅ Patch missing values with safe defaults
    config.setdefault('train', {})
    config['train'].setdefault('lr', 1e-3)
    config['train'].setdefault('weight_decay', 0.0)
    config['train'].setdefault('epochs', 50)
    config['train'].setdefault('early_stopping_patience', 5)
    config['train'].setdefault('seed', 42)
    config['train'].setdefault('resume_training', False)
    config['train'].setdefault('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    config.setdefault('scheduler', {})
    config['scheduler'].setdefault('mode', 'min')
    config['scheduler'].setdefault('factor', 0.5)
    config['scheduler'].setdefault('patience', 3)
    config['scheduler'].setdefault('verbose', True)

    # Ensure checkpoint/output directories are derived or defaulted
    if 'checkpoint_dir' not in config['train']:
        checkpoint_path = config['train'].get('checkpoint_path', 'checkpoints/last_epoch.pth')
        config['train']['checkpoint_dir'] = os.path.dirname(checkpoint_path)

    if 'project' not in config:
        config['project'] = {}
    if 'output_dir' not in config['project']:
        config['project']['output_dir'] = 'outputs'

    # Set seed and create dirs
    set_seed(config['train']['seed'])
    create_dirs([
        config['project']['output_dir'],
        config['train']['checkpoint_dir']
    ])

    # ----------- Evaluate Mode -----------
    if args.mode == "evaluate":
        device = config['train']['device']
        from src.model import get_model
        model = get_model(config).to(device)
        ckpt_best = os.path.join(config['train']['checkpoint_dir'], 'best_checkpoint.pth')
        ckpt_last = os.path.join(config['train']['checkpoint_dir'], 'last_epoch.pth')
        ckpt = ckpt_best if os.path.exists(ckpt_best) else (ckpt_last if os.path.exists(ckpt_last) else None)
        if not ckpt:
            print("No checkpoint found for evaluation.")
            return
        checkpoint = torch.load(ckpt, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        from src.data_loader import get_dataloaders
        _, val_loader = get_dataloaders(config)
        from src.evaluate import evaluate_model
        print("Evaluating model accuracy...")
        evaluate_model(model, val_loader, device)
        return



if __name__ == "__main__":
    main()
