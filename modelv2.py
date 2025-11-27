import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

class ImageEncoderImproved(nn.Module):
    def __init__(self, embedding_dim=512, dropout=0.2):
        super(ImageEncoderImproved, self).__init__()
        
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity() 
        
        # MODIFICATION: Regularized Projection Head
        # Added BatchNorm1d and Dropout to prevent overfitting
        self.projection_head = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),  # Normalize activations
            nn.GELU(),
            nn.Dropout(p=dropout), # Drop 20% of neurons during training
            nn.Linear(2048, embedding_dim)
        )

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.projection_head(features)
        return F.normalize(embeddings, p=2, dim=1)