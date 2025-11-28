import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer, CLIPTextModel
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
from PIL import Image

# Import custom modules
from model import ImageEncoder
from coco_dataset import CocoClipDataset

# --- Configuration ---
CONFIG = {
    "val_pt_path": "./processed_data/val_data.pt",      
    "img_root_val": "./coco_data/val2014",              
    "model_path": "./saved_models/clip_resnet_finetuned.pth", # Update this if using the improved model
    "batch_size": 64,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "clip_model_name": "openai/clip-vit-base-patch32"
}

def unnormalize_image(tensor):
    """Reverts normalization for visualization."""
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    return img.permute(1, 2, 0).clamp(0, 1).numpy()

def get_all_embeddings(model, dataloader, device):
    """
    Passes all validation images through the model and retrieves pre-computed text embeddings.
    """
    model.eval()
    all_img_embeds = []
    all_text_embeds = []
    
    print("Generating global embeddings for evaluation...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images = batch['image'].to(device)
            text_emb = batch['text_embedding'].to(device)
            
            # 1. Get Image Embeddings (Forward pass)
            img_emb = model(images)
            
            # 2. Get Text Embeddings & Normalize
            text_emb = F.normalize(text_emb, p=2, dim=1)
            
            all_img_embeds.append(img_emb.cpu())
            all_text_embeds.append(text_emb.cpu())
            
    return torch.cat(all_img_embeds), torch.cat(all_text_embeds)

def calculate_recall(image_embeds, text_embeds, image_ids, k_values=[1, 5, 10], batch_size=256):
    """
    Computes Recall@K using Image IDs to handle the 1-to-Many relationship 
    (1 Image <-> 5 Captions).
    """
    device = CONFIG['device']
    num_samples = image_embeds.shape[0]
    
    # Move embeddings to device for calculation
    image_embeds = image_embeds.to(device)
    text_embeds = text_embeds.to(device)
    
    # Convert image_ids to a tensor on device
    # This allows us to check match validity: if image_ids[i] == image_ids[j], it's a match.
    image_ids_tensor = torch.tensor(image_ids).to(device)
    
    print(f"\nComputing Recall metrics for {num_samples} samples...")
    
    max_k = max(k_values)
    results = {}
    
    # --- Image to Text Retrieval ---
    # Query: Image -> Target: Text
    print("--- Computing Image-to-Text Recall ---")
    correct_matches_i2t = {k: 0 for k in k_values}
    
    for start_idx in tqdm(range(0, num_samples, batch_size), desc="I2T Batches"):
        end_idx = min(start_idx + batch_size, num_samples)
        
        # Current Batch of Query Images
        img_batch = image_embeds[start_idx:end_idx]
        batch_ids = image_ids_tensor[start_idx:end_idx] # IDs of the queries
        
        # Similarity: (Batch, All_Texts)
        sim_batch = img_batch @ text_embeds.T 
        
        # Get Top-K Indices
        _, topk_indices = torch.topk(sim_batch, k=max_k, dim=1)
        
        # Ground Truth Check:
        # Retrieve the image_id for every item in the Top-K results
        # Shape: (Batch_Size, K)
        retrieved_ids = image_ids_tensor[topk_indices]
        
        # Check if the retrieved ID matches the Query ID
        # Shape: (Batch_Size, K)
        matches = (retrieved_ids == batch_ids.view(-1, 1))
        
        for k in k_values:
            # If ANY of the top K is a match, we count it
            hit = matches[:, :k].any(dim=1)
            correct_matches_i2t[k] += hit.sum().item()
            
    for k in k_values:
        recall = correct_matches_i2t[k] / num_samples
        results[f"I2T_R@{k}"] = recall
        print(f"I2T Recall@{k}: {recall:.4f}")

    # --- Text to Image Retrieval ---
    # Query: Text -> Target: Image
    print("\n--- Computing Text-to-Image Recall ---")
    correct_matches_t2i = {k: 0 for k in k_values}
    
    for start_idx in tqdm(range(0, num_samples, batch_size), desc="T2I Batches"):
        end_idx = min(start_idx + batch_size, num_samples)
        
        # Current Batch of Query Texts
        txt_batch = text_embeds[start_idx:end_idx]
        batch_ids = image_ids_tensor[start_idx:end_idx]
        
        # Similarity: (Batch, All_Images)
        sim_batch = txt_batch @ image_embeds.T 
        
        _, topk_indices = torch.topk(sim_batch, k=max_k, dim=1)
        
        retrieved_ids = image_ids_tensor[topk_indices]
        matches = (retrieved_ids == batch_ids.view(-1, 1))
        
        for k in k_values:
            hit = matches[:, :k].any(dim=1)
            correct_matches_t2i[k] += hit.sum().item()
            
    for k in k_values:
        recall = correct_matches_t2i[k] / num_samples
        results[f"T2I_R@{k}"] = recall
        print(f"T2I Recall@{k}: {recall:.4f}")
        
    return results

def visualize_text_to_image(model, dataset, all_img_embeds, device, query_text="a person riding a bike"):
    print(f"\n--- Visualizing Retrieval for query: '{query_text}' ---")
    
    tokenizer = CLIPTokenizer.from_pretrained(CONFIG['clip_model_name'])
    text_encoder = CLIPTextModel.from_pretrained(CONFIG['clip_model_name']).to(device)
    text_encoder.eval()
    
    inputs = tokenizer([query_text], padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        text_out = text_encoder(**inputs)
        text_embed = text_out.pooler_output
        text_embed = F.normalize(text_embed, p=2, dim=1).cpu()

    sims = text_embed @ all_img_embeds.T
    scores, indices = torch.topk(sims, k=5)
    indices = indices.squeeze().numpy()
    
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    fig.suptitle(f"Query: '{query_text}'", fontsize=14)
    
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        img = unnormalize_image(sample['image'])
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f"Rank {i+1}\nScore: {scores[0][i]:.3f}")
        
    plt.tight_layout()
    plt.show()

def visualize_zero_shot_classification(model, dataset, device, sample_idx=0, candidate_classes=["cat", "dog", "car"]):
    print(f"\n--- Visualizing Zero-Shot Classification ---")
    
    sample = dataset[sample_idx]
    image_tensor = sample['image'].unsqueeze(0).to(device)
    true_caption = sample['caption']
    
    model.eval()
    with torch.no_grad():
        image_embed = model(image_tensor)
        
    prompts = [f"a photo of a {c}" for c in candidate_classes]
    
    tokenizer = CLIPTokenizer.from_pretrained(CONFIG['clip_model_name'])
    text_encoder = CLIPTextModel.from_pretrained(CONFIG['clip_model_name']).to(device)
    
    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        text_out = text_encoder(**inputs)
        class_embeds = F.normalize(text_out.pooler_output, p=2, dim=1)
        
    logits = (image_embed @ class_embeds.T) / 0.07 
    probs = F.softmax(logits, dim=1).cpu().numpy().flatten()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(unnormalize_image(sample['image']))
    ax1.set_title(f"True Caption:\n{true_caption[:40]}...")
    ax1.axis('off')
    
    y_pos = np.arange(len(candidate_classes))
    ax2.barh(y_pos, probs, align='center', color='skyblue')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(candidate_classes)
    ax2.invert_yaxis()
    ax2.set_xlabel('Probability')
    ax2.set_title('Zero-Shot Classification')
    
    plt.tight_layout()
    plt.show()

def main():
    print("Loading Validation Dataset...")
    val_dataset = CocoClipDataset(CONFIG['val_pt_path'], CONFIG['img_root_val'])
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
    
    # [FIX] Extract Image IDs directly from the dataset list
    # The CocoClipDataset stores the loaded .pt data in self.data
    # Each item is a dict: {'image_id': ..., 'caption': ..., ...}
    print("Extracting Image IDs for evaluation...")
    val_image_ids = [item['image_id'] for item in val_dataset.data]
    
    print("Loading Model...")
    model = ImageEncoder().to(CONFIG['device'])
    if os.path.exists(CONFIG['model_path']):
        model.load_state_dict(torch.load(CONFIG['model_path'], map_location=CONFIG['device']))
        print("Pretrained weights loaded.")
    else:
        print(f"WARNING: Model not found at {CONFIG['model_path']}. Using random weights.")
    
    # 1. Compute Embeddings
    img_embeds, text_embeds = get_all_embeddings(model, val_loader, CONFIG['device'])
    
    # 2. Compute Quantitative Metrics (Passing Image IDs now)
    calculate_recall(img_embeds, text_embeds, val_image_ids)
    
    # 3. Qualitative
    visualize_text_to_image(model, val_dataset, img_embeds, CONFIG['device'], query_text="a group of people playing baseball")
    visualize_text_to_image(model, val_dataset, img_embeds, CONFIG['device'], query_text="a plate of food")
    
    visualize_zero_shot_classification(
        model, 
        val_dataset, 
        CONFIG['device'], 
        sample_idx=5, 
        candidate_classes=["person", "animal", "landscape"]
    )

if __name__ == "__main__":
    main()