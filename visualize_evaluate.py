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
    "val_pt_path": "./processed_data/val_data.pt",      # Path to processed val data
    "img_root_val": "./coco_data/val2014",              # Path to images
    "model_path": "./saved_models/clip_resnet_finetuned.pth", # Path to trained model
    "batch_size": 64,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "clip_model_name": "openai/clip-vit-base-patch32"  # Must match preprocess_setup.py
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
    Returns normalized tensors for cosine similarity.
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
            
            # 2. Get Text Embeddings
            # Normalize text embeddings here (Pre-computed ones might not be normalized)
            text_emb = F.normalize(text_emb, p=2, dim=1)
            
            all_img_embeds.append(img_emb.cpu())
            all_text_embeds.append(text_emb.cpu())
            
    # Concatenate all into one large tensor
    return torch.cat(all_img_embeds), torch.cat(all_text_embeds)

def calculate_recall(image_embeds, text_embeds, k_values=[1, 5, 10], batch_size=256):
    """
    Computes Recall@K using batched matrix multiplication to avoid OOM errors.
    """
    device = CONFIG['device']
    num_samples = image_embeds.shape[0]
    print(f"\nComputing Recall metrics for {num_samples} samples in batches...")
    
    # Ensure embeddings are on the correct device for fast computation
    # (Move to GPU if they fit; 100k x 512 floats is ~200MB, so these fit easily.
    # The result matrix N x N is the problem, not the input vectors).
    image_embeds = image_embeds.to(device)
    text_embeds = text_embeds.to(device)
    
    max_k = max(k_values)
    results = {}
    
    # --- Image to Text Retrieval ---
    print("--- Computing Image-to-Text Recall ---")
    correct_matches_i2t = {k: 0 for k in k_values}
    
    # Iterate over images in batches
    for start_idx in tqdm(range(0, num_samples, batch_size), desc="I2T Batches"):
        end_idx = min(start_idx + batch_size, num_samples)
        
        # Batch of images: (Batch_Size, 512)
        img_batch = image_embeds[start_idx:end_idx]
        
        # Calculate similarity against ALL texts: (Batch_Size, N)
        # This results in a smaller matrix that fits in memory (e.g., 256 x 100,000)
        sim_batch = img_batch @ text_embeds.T 
        
        # Get top-k for this batch
        _, topk_indices = torch.topk(sim_batch, k=max_k, dim=1)
        
        # Ground truth indices for this batch
        # The correct text for image i is at index i
        targets = torch.arange(start_idx, end_idx).to(device).view(-1, 1)
        
        for k in k_values:
            # Check if target is present in the top k predictions
            match = (topk_indices[:, :k] == targets).any(dim=1)
            correct_matches_i2t[k] += match.sum().item()
            
    for k in k_values:
        recall = correct_matches_i2t[k] / num_samples
        results[f"I2T_R@{k}"] = recall
        print(f"I2T Recall@{k}: {recall:.4f}")

    # --- Text to Image Retrieval ---
    print("\n--- Computing Text-to-Image Recall ---")
    correct_matches_t2i = {k: 0 for k in k_values}
    
    # Iterate over texts in batches
    for start_idx in tqdm(range(0, num_samples, batch_size), desc="T2I Batches"):
        end_idx = min(start_idx + batch_size, num_samples)
        
        # Batch of texts: (Batch_Size, 512)
        txt_batch = text_embeds[start_idx:end_idx]
        
        # Similarity against ALL images: (Batch_Size, N)
        sim_batch = txt_batch @ image_embeds.T 
        
        _, topk_indices = torch.topk(sim_batch, k=max_k, dim=1)
        targets = torch.arange(start_idx, end_idx).to(device).view(-1, 1)
        
        for k in k_values:
            match = (topk_indices[:, :k] == targets).any(dim=1)
            correct_matches_t2i[k] += match.sum().item()
            
    for k in k_values:
        recall = correct_matches_t2i[k] / num_samples
        results[f"T2I_R@{k}"] = recall
        print(f"T2I Recall@{k}: {recall:.4f}")
        
    return results

def visualize_text_to_image(model, dataset, all_img_embeds, device, query_text="a person riding a bike"):
    """
    Embeds a custom text query and retrieves the top 5 matching images from the dataset.
    """
    print(f"\n--- Visualizing Retrieval for query: '{query_text}' ---")
    
    # Load HuggingFace Text Encoder (to encode the new query)
    tokenizer = CLIPTokenizer.from_pretrained(CONFIG['clip_model_name'])
    text_encoder = CLIPTextModel.from_pretrained(CONFIG['clip_model_name']).to(device)
    text_encoder.eval()
    
    # Encode Query
    inputs = tokenizer([query_text], padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        text_out = text_encoder(**inputs)
        text_embed = text_out.pooler_output
        text_embed = F.normalize(text_embed, p=2, dim=1).cpu() # Normalize!

    # Compute Similarity against all validation images
    # Shape: (1, 512) @ (N, 512).T -> (1, N)
    # Note: This is small enough to do in one go since it's just 1 vector vs N
    sims = text_embed @ all_img_embeds.T
    scores, indices = torch.topk(sims, k=5)
    
    indices = indices.squeeze().numpy()
    
    # Plot
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
    """
    Takes one image and classifies it among a list of candidate classes.
    """
    print(f"\n--- Visualizing Zero-Shot Classification ---")
    
    # Get the image
    sample = dataset[sample_idx]
    image_tensor = sample['image'].unsqueeze(0).to(device) # Add batch dim
    true_caption = sample['caption']
    
    # Get Image Embedding
    model.eval()
    with torch.no_grad():
        image_embed = model(image_tensor) # Already normalized
        
    # Prepare Text Prompts ("a photo of a {class}")
    prompts = [f"a photo of a {c}" for c in candidate_classes]
    
    # Load HF Text Encoder
    tokenizer = CLIPTokenizer.from_pretrained(CONFIG['clip_model_name'])
    text_encoder = CLIPTextModel.from_pretrained(CONFIG['clip_model_name']).to(device)
    
    # Encode Classes
    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        text_out = text_encoder(**inputs)
        class_embeds = text_out.pooler_output
        class_embeds = F.normalize(class_embeds, p=2, dim=1)
        
    # Calculate Similarity & Softmax
    logits = (image_embed @ class_embeds.T)
    # Apply temperature scaling (standard CLIP uses learned temp, here we use raw or 1/0.07)
    logits = logits / 0.07 
    probs = F.softmax(logits, dim=1).cpu().numpy().flatten()
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Show Image
    ax1.imshow(unnormalize_image(sample['image']))
    ax1.set_title(f"True Caption:\n{true_caption[:40]}...")
    ax1.axis('off')
    
    # Show Probabilities
    y_pos = np.arange(len(candidate_classes))
    ax2.barh(y_pos, probs, align='center', color='skyblue')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(candidate_classes)
    ax2.invert_yaxis()  # labels read top-to-bottom
    ax2.set_xlabel('Probability')
    ax2.set_title('Zero-Shot Classification')
    
    plt.tight_layout()
    plt.show()

def main():
    print("Loading Validation Dataset...")
    # NOTE: Ensure you are pointing to the correct split
    val_dataset = CocoClipDataset(CONFIG['val_pt_path'], CONFIG['img_root_val'])
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
    
    print("Loading Model...")
    model = ImageEncoder().to(CONFIG['device'])
    if os.path.exists(CONFIG['model_path']):
        model.load_state_dict(torch.load(CONFIG['model_path'], map_location=CONFIG['device']))
        print("Pretrained weights loaded.")
    else:
        print("WARNING: trained model not found. Using random weights.")
    
    # 1. Compute Embeddings
    img_embeds, text_embeds = get_all_embeddings(model, val_loader, CONFIG['device'])
    
    # 2. Compute Quantitative Metrics (BATCHED to avoid OOM)
    calculate_recall(img_embeds, text_embeds)
    
    # 3. Qualitative: Text-to-Image Retrieval
    # You can change these queries to anything
    visualize_text_to_image(model, val_dataset, img_embeds, CONFIG['device'], query_text="a group of people playing baseball")
    visualize_text_to_image(model, val_dataset, img_embeds, CONFIG['device'], query_text="a plate of food")
    
    # 4. Qualitative: Zero-Shot Classification
    # Pick a random index or specific one to test
    visualize_zero_shot_classification(
        model, 
        val_dataset, 
        CONFIG['device'], 
        sample_idx=5, 
        candidate_classes=["person", "cat", "dog", "train", "food"]
    )

if __name__ == "__main__":
    main()