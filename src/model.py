import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes, device, pretrained=True):
    model = models.efficientnet_b4(pretrained=pretrained)
    
    # Modify classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    
    # Transfer to device
    model = model.to(device)
    
    # Freeze early layers
    for param in model.features.parameters():
        param.requires_grad = False
        
    # Unfreeze classifier
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    return model