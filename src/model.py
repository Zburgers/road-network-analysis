# src/model.py

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class RoadSegmentationModel(nn.Module):
    def __init__(self, encoder_name='resnet34', pretrained=True, out_classes=1):
        super(RoadSegmentationModel, self).__init__()
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights='imagenet' if pretrained else None,
            in_channels=3,
            classes=out_classes,
            activation=None
        )

    def forward(self, x):
        return self.model(x)

def get_model(config):
    return RoadSegmentationModel(
        encoder_name=config['model']['encoder'],
        pretrained=config['model']['pretrained'],
        out_classes=1
    )

def get_loss():
    return nn.BCEWithLogitsLoss()

def get_optimizer(model, config):
    return torch.optim.Adam(
        model.parameters(),
        lr=config['train']['lr'],
        weight_decay=config['train']['weight_decay']
    )

def get_scheduler(optimizer, config):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )
