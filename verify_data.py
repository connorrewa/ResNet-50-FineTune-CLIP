import matplotlib.pyplot as plt
import torch
import random
import torchvision.transforms as T
from coco_dataset import CocoClipDataset

# Helper to un-normalize for visualization
def unnormalize(tensor):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    return tensor * std + mean

def verify():
    # Paths
    train_pt = "./processed_data/train_data.pt"
    train_imgs = "./coco_data/train2014"
    
    # Initialize Dataset
    try:
        dataset = CocoClipDataset(train_pt, train_imgs)
    except FileNotFoundError:
        print("Dataset files not found. Please run 'preprocess_setup.py' first.")
        return

    print(f"Dataset successfully loaded with {len(dataset)} samples.")
    
    # Pick random samples
    indices = random.sample(range(len(dataset)), 3)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        image_tensor = sample['image']
        caption = sample['caption']
        
        # Prepare image for display
        image_disp = unnormalize(image_tensor)
        image_disp = image_disp.permute(1, 2, 0).numpy()
        # Clip values to [0,1] range to avoid matplotlib warnings
        image_disp = image_disp.clip(0, 1)
        
        axes[i].imshow(image_disp)
        axes[i].set_title(f"Caption: {caption[:30]}...", fontsize=9)
        axes[i].axis('off')
        
        print(f"Sample {i+1}:")
        print(f" - Caption: {caption}")
        print(f" - Image Shape: {sample['image'].shape}")
        print(f" - Text Emb Shape: {sample['text_embedding'].shape}")
        print("-" * 20)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    verify()