import torch
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CocoClipDataset(Dataset):
    def __init__(self, pt_file_path, image_root_dir):
        """
        Args:
            pt_file_path (string): Path to the .pt file created by preprocess_setup.py
            image_root_dir (string): Path to the folder containing images (train2014 or val2014)
        """
        print(f"Loading dataset from {pt_file_path}...")
        self.data = torch.load(pt_file_path)
        self.image_root_dir = image_root_dir
        
        # CLIP Standard normalization values
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load Image
        img_name = item['file_name']
        img_path = os.path.join(self.image_root_dir, img_name)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy tensor or handle error (usually dataset is clean)
            image = Image.new('RGB', (224, 224))
            
        image = self.transform(image)
        
        # Retrieve cached text embedding
        text_embedding = item['text_embedding']
        
        # Retrieve raw caption (useful for visualization/debug)
        caption = item['caption']

        return {
            "image": image,
            "text_embedding": text_embedding,
            "caption": caption
        }