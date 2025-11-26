import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

class ImageEncoder(nn.Module):
    def __init__(self, embedding_dim=512):
        super(ImageEncoder, self).__init__()
        
        # 1. Load ResNet50 with pretrained ImageNet weights
        # We use the modern 'weights' parameter instead of 'pretrained=True'
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        # 2. Replace the classification head (fc)
        # ResNet50's final feature map size is 2048
        # The prompt asks for two linear layers with GELU activation
        
        self.backbone.fc = nn.Identity() # Remove the original FC layer
        
        self.projection_head = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.GELU(),
            nn.Linear(2048, embedding_dim)
        )

    def forward(self, x):
        # Extract features from ResNet
        features = self.backbone(x)
        
        # Project to CLIP embedding space
        embeddings = self.projection_head(features)
        
        # Normalize embeddings to unit length (Crucial for Contrastive Learning)
        # This allows us to use dot product as cosine similarity
        return F.normalize(embeddings, p=2, dim=1)

if __name__ == "__main__":
    # Quick verification
    model = ImageEncoder()
    dummy_img = torch.randn(2, 3, 224, 224)
    out = model(dummy_img)
    print(f"Output shape: {out.shape}") # Should be [2, 512]