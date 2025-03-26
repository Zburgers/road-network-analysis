# main.py

import os
import yaml
import torch
import random
import numpy as np

from src.train import train_model
from src.utils import set_seed, create_dirs

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    set_seed(config['train']['seed'])

    # Create required directories
    create_dirs([config['project']['output_dir'], os.path.dirname(config['train']['checkpoint_path'])])

    # Start training
    train_model(config)

if __name__ == "__main__":
    main()
