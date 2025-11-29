import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import CLIPTokenizer, CLIPTextModel
import matplotlib.pyplot as plt
import numpy as np
import os
import random

# --- IMPORTS FROM YOUR FILES ---
# Ensure modelv2.py and coco_dataset.py are in the same folder
from modelv2 import ImageEncoderImproved
from coco_dataset import CocoClipDataset

# --- CONFIGURATION ---
CONFIG = {
    "val_pt_path": "./processed_data/val_data.pt",
    "img_root_val": "./coco_data/val2014",
    # Make sure this points to your V2 weights
    "model_path": "./saved_models/clip_resnet_v2.pth", 
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "clip_model_name": "openai/clip-vit-base-patch32",
    "num_samples": 5  # How many images to test
}

def unnormalize_image(tensor):
    """Reverts normalization for visualization."""
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    return img.permute(1, 2, 0).clamp(0, 1).numpy()

def run_zero_shot(model, dataset, indices, classes, device):
    """
    Runs zero-shot classification on specific dataset indices using custom classes.
    """
    print(f"\n--- Running Zero-Shot on {len(indices)} images ---")
    print(f"Candidate Classes: {classes}")

    # 1. Prepare Text Embeddings for Classes
    tokenizer = CLIPTokenizer.from_pretrained(CONFIG['clip_model_name'])
    text_encoder = CLIPTextModel.from_pretrained(CONFIG['clip_model_name']).to(device)
    
    # Wrap classes in prompts
    prompts = [f"a photo of a {c}" for c in classes]
    
    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        text_out = text_encoder(**inputs)
        # Normalize text embeddings
        class_embeds = F.normalize(text_out.pooler_output, p=2, dim=1)

    # 2. Loop through images
    model.eval()
    
    # Setup plot
    fig, axes = plt.subplots(len(indices), 2, figsize=(10, 3 * len(indices)))
    if len(indices) == 1: axes = [axes] # Handle single case
    
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        image_tensor = sample['image'].unsqueeze(0).to(device)
        true_caption = sample['caption']
        
        # Get Image Embedding
        with torch.no_grad():
            image_embed = model(image_tensor) # Already normalized in model forward()
            
        # Calculate Similarity (Logits)
        # Scale by 1/0.07 (CLIP temperature) for sharper softmax
        logits = (image_embed @ class_embeds.T) / 0.07
        probs = F.softmax(logits, dim=1).cpu().numpy().flatten()
        
        # Get prediction
        pred_idx = np.argmax(probs)
        pred_label = classes[pred_idx]
        
        # --- Visualization ---
        ax_img = axes[i][0]
        ax_bar = axes[i][1]
        
        # Show Image
        ax_img.imshow(unnormalize_image(sample['image']))
        ax_img.axis('off')
        ax_img.set_title(f"True: {true_caption[:30]}...\nPred: {pred_label}", fontsize=10)
        
        # Show Probabilities
        y_pos = np.arange(len(classes))
        ax_bar.barh(y_pos, probs, align='center', color='skyblue')
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(classes)
        ax_bar.invert_yaxis()
        ax_bar.set_xlim(0, 1.0)
        ax_bar.set_xlabel('Probability')
    
    plt.tight_layout()
    plt.show()

def main():
    # 1. Load Dataset
    print("Loading Validation Dataset (Lightweight)...")
    val_dataset = CocoClipDataset(CONFIG['val_pt_path'], CONFIG['img_root_val'])
    
    # 2. Load Your V2 Model
    print(f"Loading Model from {CONFIG['model_path']}...")
    model = ImageEncoderImproved().to(CONFIG['device'])
    
    if os.path.exists(CONFIG['model_path']):
        # Strict=False helps skip potential auxiliary keys, but strict=True is better for exact matches
        model.load_state_dict(torch.load(CONFIG['model_path'], map_location=CONFIG['device']))
        print("Weights loaded successfully.")
    else:
        print("ERROR: Model weights not found!")
        return

    # 3. Pick 5 Random Indices
    # total_images = len(val_dataset)
    # indices = random.sample(range(total_images), CONFIG['num_samples'])
    
    # Or pick specific indices if you want to test specific images
    indices = [10, 50, 660, 250, 260]

    # 4. Define Candidate Classes
    # You can change these to whatever you think might be in the images
    candidate_classes = [
        "a person", "an animal", "a landscape"
    ]
    
    # 5. Run
    run_zero_shot(model, val_dataset, indices, candidate_classes, CONFIG['device'])

if __name__ == "__main__":
    main()