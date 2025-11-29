import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer, CLIPTextModel
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# --- IMPORTS ---
# NOTE: If you are using the v2 model, change this to:
# from modelv2 import ImageEncoderImproved as ImageEncoder
from modelv2 import ImageEncoderImproved
from coco_dataset import CocoClipDataset

# --- CONFIGURATION ---
CONFIG = {
    "val_pt_path": "./processed_data/val_data.pt",
    "img_root_val": "./coco_data/val2014",
    # Point this to the specific .pth file you want to visualize
    # Ensure this matches the architecture imported above (v1 vs v2)
    "model_path": "./saved_models/clip_resnet_v2.pth", 
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "clip_model_name": "openai/clip-vit-base-patch32"
}

def unnormalize_image(tensor):
    """Reverts normalization for visualization."""
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    # Move tensor to cpu before math if it's on gpu
    img = tensor.cpu() * std + mean
    return img.permute(1, 2, 0).clamp(0, 1).numpy()

def get_image_embeddings(model, dataloader, device):
    """Pre-computes image embeddings for the whole dataset."""
    model.eval()
    all_embeds = []
    print(f"Generating embeddings for {model.__class__.__name__}...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images = batch['image'].to(device)
            embeds = model(images)
            all_embeds.append(embeds.cpu())
    return torch.cat(all_embeds)

def visualize_retrieval(queries, dataset, img_embeds, device):
    """
    Retrieves and displays images for text queries, ensuring no duplicates.
    """
    tokenizer = CLIPTokenizer.from_pretrained(CONFIG['clip_model_name'])
    text_encoder = CLIPTextModel.from_pretrained(CONFIG['clip_model_name']).to(device)
    
    # Iterate over each query and produce a separate plot
    for query in queries:
        print(f"\nProcessing Query: '{query}'")
        
        # 1. Encode Text
        inputs = tokenizer([query], padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            text_embed = text_encoder(**inputs).pooler_output
            text_embed = F.normalize(text_embed, p=2, dim=1).cpu()

        # 2. Retrieve Top-50 (Fetch extra to allow for filtering duplicates)
        # We check top 50 to ensure we find at least 5 unique images
        sims = text_embed @ img_embeds.T
        scores, indices = torch.topk(sims, k=50)
        
        # 3. Filter for Unique Image IDs
        unique_matches = []
        seen_ids = set()
        
        for i, idx in enumerate(indices[0]):
            dataset_idx = idx.item()
            # Access the raw data to check the ID
            # dataset.data is the list of dicts loaded from .pt file
            img_id = dataset.data[dataset_idx]['image_id']
            
            if img_id not in seen_ids:
                seen_ids.add(img_id)
                score = scores[0][i].item()
                unique_matches.append((dataset_idx, score))
            
            # Stop once we have 5 unique images
            if len(unique_matches) == 5:
                break
        
        # 4. Plot
        # We will plot up to 5 images
        num_to_plot = 5
        fig, axes = plt.subplots(1, num_to_plot, figsize=(15, 4))
        fig.suptitle(f"Query: '{query}'", fontsize=16)
        
        # Handle case where only 1 image is plotted (axes is not a list)
        if num_to_plot == 1:
            axes = [axes]
            
        for i in range(num_to_plot):
            ax = axes[i]
            ax.axis('off') # Hide axis by default
            
            if i < len(unique_matches):
                img_idx, score = unique_matches[i]
                
                # Load and unnormalize image
                img_tensor = dataset[img_idx]['image']
                img = unnormalize_image(img_tensor)
                
                ax.imshow(img)
                ax.set_title(f"Rank {i+1}\nScore: {score:.3f}")

        plt.tight_layout()
        plt.show()

def main():
    # 1. Load Dataset
    print("Loading Validation Dataset...")
    val_dataset = CocoClipDataset(CONFIG['val_pt_path'], CONFIG['img_root_val'])
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # 2. Load Model
    print(f"Loading Model from {CONFIG['model_path']}...")
    model = ImageEncoderImproved().to(CONFIG['device'])
    
    if os.path.exists(CONFIG['model_path']):
        # strict=False allows loading slightly mismatched weights (e.g. if you added dropout/bn later)
        # but for v1 it should match exactly.
        try:
            model.load_state_dict(torch.load(CONFIG['model_path'], map_location=CONFIG['device']))
            print("Weights loaded successfully.")
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Ensure your model definition (model.py vs modelv2.py) matches the checkpoint.")
            return
    else:
        print(f"ERROR: Weights not found at {CONFIG['model_path']}")
        return

    # 3. Get Embeddings
    # Note: You can implement caching here as shown in previous answers to speed this up!
    img_embeds = get_image_embeddings(model, val_loader, CONFIG['device'])
    
    # 4. Define Queries
    queries = [
        "A baseball player swinging a bat",
        "A red double-decker bus",
        "A grazing zebra",
        "A messy kitchen",
        "Surfboards on the beach"
    ]
    
    # 5. Run Visualization
    visualize_retrieval(queries, val_dataset, img_embeds, CONFIG['device'])

if __name__ == "__main__":
    main()