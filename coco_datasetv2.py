import torch
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CocoClipDatasetImproved(Dataset):
    def __init__(self, pt_file_path, image_root_dir, split="train"):
        """
        Args:
            pt_file_path (string): Path to the .pt file
            image_root_dir (string): Path to images
            split (string): 'train' or 'val'. Determines the transforms used.
        """
        self.data = torch.load(pt_file_path)
        self.image_root_dir = image_root_dir
        self.split = split
        
        # CLIP Normalization constants
        norm_mean = [0.48145466, 0.4578275, 0.40821073]
        norm_std = [0.26862954, 0.26130258, 0.27577711]

        if self.split == "train":
            # MODIFICATION: Strong Data Augmentation for Training
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)), # Randomly crop and resize
                transforms.RandomHorizontalFlip(),                   # Randomly flip
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Color noise
                transforms.ToTensor(),
                transforms.Normalize(mean=norm_mean, std=norm_std)
            ])
        else:
            # MODIFICATION: Correct Aspect Ratio Preserving Transform for Val
            self.transform = transforms.Compose([
                transforms.Resize(256),      # Resize smaller edge to 256
                transforms.CenterCrop(224),  # Crop center 224x224
                transforms.ToTensor(),
                transforms.Normalize(mean=norm_mean, std=norm_std)
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        img_name = item['file_name']
        img_path = os.path.join(self.image_root_dir, img_name)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback for corrupted images
            image = Image.new('RGB', (224, 224))
            
        image = self.transform(image)
        text_embedding = item['text_embedding']
        caption = item['caption']

        return {
            "image": image,
            "text_embedding": text_embedding,
            "caption": caption
        }