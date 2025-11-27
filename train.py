import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm

# Import your custom modules
from model import ImageEncoder
from coco_dataset import CocoClipDataset

# --- Configuration ---
CONFIG = {
    "train_pt_path": "./processed_data/train_data.pt",  # From preprocess_setup.py
    "val_pt_path": "./processed_data/val_data.pt",      # From preprocess_setup.py
    "img_root_train": "./coco_data/train2014",          # Adjust if needed
    "img_root_val": "./coco_data/val2014",              # Adjust if needed
    "batch_size": 32,
    "learning_rate": 1e-4,  # Lower LR for fine-tuning
    "epochs": 5,
    "temperature": 0.07,    # Standard CLIP temperature
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_path": "./saved_models/clip_resnet_finetuned.pth",
    "plot_path": "./training_loss_curve.png"
}

def info_nce_loss(image_embeddings, text_embeddings, temperature, device):
    """
    Calculates the InfoNCE (Contrastive) Loss.
    
    Args:
        image_embeddings: Tensor of shape (batch_size, embed_dim)
        text_embeddings: Tensor of shape (batch_size, embed_dim)
        temperature: Scalar to scale logits
    """
    # 1. Calculate Cosine Similarity Matrix
    # Both embeddings are already normalized in the model/dataset, 
    # so dot product equals cosine similarity.
    # Shape: (batch_size, batch_size)
    logits = (image_embeddings @ text_embeddings.T) / temperature

    # 2. Create Labels
    # The image at index i corresponds to the text at index i.
    # So the targets are the diagonal elements (0, 1, 2, ... batch_size-1)
    batch_size = image_embeddings.shape[0]
    targets = torch.arange(batch_size).to(device)

    # 3. Calculate Symmetric Loss
    # Loss for Image-to-Text (rows)
    loss_i2t = nn.functional.cross_entropy(logits, targets)
    # Loss for Text-to-Image (columns)
    loss_t2i = nn.functional.cross_entropy(logits.T, targets)

    # Average the two losses
    return (loss_i2t + loss_t2i) / 2

def train_one_epoch(model, dataloader, optimizer, device, epoch_idx):
    model.train()
    running_loss = 0.0
    
    # Progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch_idx+1} [Train]")
    
    for batch in pbar:
        # Move data to device
        images = batch['image'].to(device)
        text_embeddings = batch['text_embedding'].to(device) # Frozen embeddings [cite: 35]
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass (Image Encoder only) [cite: 36]
        img_embeddings = model(images)
        
        # Calculate Loss
        loss = info_nce_loss(img_embeddings, text_embeddings, CONFIG['temperature'], device)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Stats
        running_loss += loss.item()
        pbar.set_postfix({"Loss": loss.item()})
        
    return running_loss / len(dataloader)

def validate(model, dataloader, device, epoch_idx):
    model.eval()
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch_idx+1} [Val]")
    
    with torch.no_grad():
        for batch in pbar:
            images = batch['image'].to(device)
            text_embeddings = batch['text_embedding'].to(device)
            
            img_embeddings = model(images)
            loss = info_nce_loss(img_embeddings, text_embeddings, CONFIG['temperature'], device)
            
            running_loss += loss.item()
            pbar.set_postfix({"Val Loss": loss.item()})
            
    return running_loss / len(dataloader)

def plot_training_curves(train_losses, val_losses, save_path):
    """Generates and saves the loss plot as required[cite: 78, 99]."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("CLIP Fine-tuning Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    print(f"Loss plot saved to {save_path}")

def main():
    # Setup directories
    os.makedirs(os.path.dirname(CONFIG['save_path']), exist_ok=True)
    
    print(f"Using device: {CONFIG['device']}") # [cite: 79]
    print("Initializing Datasets...")
    
    # Load Datasets
    train_dataset = CocoClipDataset(CONFIG['train_pt_path'], CONFIG['img_root_train'])
    val_dataset = CocoClipDataset(CONFIG['val_pt_path'], CONFIG['img_root_val'])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
    
    # Initialize Model
    print("Initializing Model...")
    model = ImageEncoder().to(CONFIG['device']) # ResNet50 backbone [cite: 67]
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    # Track metrics
    train_losses = []
    val_losses = []
    
    start_time = time.time()
    
    # Training Loop
    for epoch in range(CONFIG['epochs']):
        train_loss = train_one_epoch(model, train_loader, optimizer, CONFIG['device'], epoch)
        val_loss = validate(model, val_loader, CONFIG['device'], epoch)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save checkpoint every epoch (optional, but good practice)
        torch.save(model.state_dict(), CONFIG['save_path'])

    total_time = time.time() - start_time
    print(f"\nTraining Complete. Total time: {total_time/60:.2f} minutes.") # [cite: 79, 98]
    
    # Generate Report Artifacts
    plot_training_curves(train_losses, val_losses, CONFIG['plot_path'])
    print(f"Model saved to {CONFIG['save_path']}")

if __name__ == "__main__":
    main()