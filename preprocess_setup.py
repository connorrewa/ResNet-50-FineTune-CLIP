import os
import json
import torch
import random
import numpy as np
from PIL import Image
from transformers import CLIPTokenizer, CLIPTextModel
from tqdm import tqdm
from torch.utils.data import DataLoader

# Configuration
DATA_ROOT = "./coco_data"
OUTPUT_DIR = "./processed_data"
SUBSET_PERCENTAGE = 0.50
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP Model Name
MODEL_NAME = "openai/clip-vit-base-patch32"

def setup_directories():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_annotations(split_name):
    """Loads COCO caption annotations."""
    path = os.path.join(DATA_ROOT, "annotations", f"captions_{split_name}.json")
    print(f"Loading annotations from {path}...")
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def subset_data(data, percentage):
    """
    Subsets the data by unique image IDs to ensure no data leakage.
    Returns a list of (image_filename, caption_text, image_id) tuples.
    """
    # Group captions by image_id
    img_to_caps = {}
    img_id_to_filename = {img['id']: img['file_name'] for img in data['images']}
    
    for ann in data['annotations']:
        img_id = ann['image_id']
        caption = ann['caption']
        if img_id not in img_to_caps:
            img_to_caps[img_id] = []
        img_to_caps[img_id].append(caption)

    # Select random subset of Image IDs
    all_img_ids = list(img_to_caps.keys())
    
    # [CHANGE 1] Shuffle is now deterministic due to random.seed in main()
    random.shuffle(all_img_ids)
    
    num_to_keep = int(len(all_img_ids) * percentage)
    subset_img_ids = all_img_ids[:num_to_keep]
    
    print(f"Subsetting: Keeping {len(subset_img_ids)} images out of {len(all_img_ids)} ({percentage*100}%)")

    # Flatten back to a list of training pairs
    final_pairs = []
    for img_id in subset_img_ids:
        filename = img_id_to_filename[img_id]
        for cap in img_to_caps[img_id]:
            final_pairs.append({
                "image_id": img_id,
                "file_name": filename,
                "caption": cap
            })
            
    return final_pairs

def encode_and_save(pairs, split_name, tokenizer, text_encoder):
    """
    Encodes captions in batches and saves the pairs + embeddings to disk.
    """
    print(f"Encoding text for {split_name} ({len(pairs)} samples)...")
    
    # Prepare list for storage
    processed_dataset = []
    
    # Create batches
    batch_captions = []
    batch_meta = []
    
    text_encoder.eval()
    
    with torch.no_grad():
        for i, item in tqdm(enumerate(pairs), total=len(pairs)):
            batch_captions.append(item['caption'])
            batch_meta.append(item)
            
            if len(batch_captions) == BATCH_SIZE or i == len(pairs) - 1:
                # Tokenize
                inputs = tokenizer(
                    batch_captions, 
                    padding=True, 
                    truncation=True, 
                    max_length=77, 
                    return_tensors="pt"
                ).to(DEVICE)
                
                # [CHANGE 2] Encode using pooler_output (EOS token) instead of index 0 (CLS)
                # CLIP uses the EOS token to represent the sentence.
                # pooler_output in HuggingFace CLIPTextModel extracts the EOS feature automatically.
                outputs = text_encoder(**inputs)
                text_embeddings = outputs.pooler_output
                
                text_embeddings = text_embeddings.cpu()
                
                # Save to list
                for j, meta in enumerate(batch_meta):
                    processed_dataset.append({
                        "image_id": meta['image_id'],
                        "file_name": meta['file_name'],
                        "caption": meta['caption'],
                        "text_embedding": text_embeddings[j] # Save as tensor
                    })
                
                # Reset batch
                batch_captions = []
                batch_meta = []

    # Save to disk
    save_path = os.path.join(OUTPUT_DIR, f"{split_name}_data.pt")
    print(f"Saving {split_name} dataset to {save_path}...")
    torch.save(processed_dataset, save_path)

def main():
    # [CHANGE 3] Set Seed for Reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    setup_directories()
    
    # Load CLIP components
    print("Loading CLIP Text Encoder...")
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME)
    text_encoder = CLIPTextModel.from_pretrained(MODEL_NAME).to(DEVICE)

    # Process Train
    train_data = load_annotations("train2014")
    train_pairs = subset_data(train_data, SUBSET_PERCENTAGE)
    encode_and_save(train_pairs, "train", tokenizer, text_encoder)

    # Process Val
    val_data = load_annotations("val2014")
    val_pairs = subset_data(val_data, SUBSET_PERCENTAGE)
    encode_and_save(val_pairs, "val", tokenizer, text_encoder)
    
    print("Preprocessing complete!")

if __name__ == "__main__":
    main()