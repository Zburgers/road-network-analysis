# src/data_loader.py

import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class RoadDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_size=256, augment=False):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_ids = sorted(os.listdir(images_dir))
        self.mask_ids = sorted(os.listdir(masks_dir))
        self.image_size = image_size
        self.augment = augment

        self.transform = self.get_transforms()

    def get_transforms(self):
        if self.augment:
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Rotate(limit=15, p=0.3),
                A.Normalize(),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_ids[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_ids[idx])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)  # Binary mask

        transformed = self.transform(image=image, mask=mask)
        return transformed['image'], transformed['mask'].unsqueeze(0)  # [1, H, W]

def get_dataloaders(config):
    train_dataset = RoadDataset(
        images_dir=config['data']['train_images_dir'],
        masks_dir=config['data']['train_masks_dir'],
        image_size=config['data']['image_size'],
        augment=config['data']['augmentations']
    )

    val_dataset = RoadDataset(
        images_dir=config['data']['val_images_dir'],
        masks_dir=config['data']['val_masks_dir'],
        image_size=config['data']['image_size'],
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )

    return train_loader, val_loader
