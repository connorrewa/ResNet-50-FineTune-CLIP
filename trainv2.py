import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm

# Import IMPROVED modules
from modelv2 import ImageEncoderImproved
from coco_dataset import CocoClipDataset as CocoClipDatasetImproved
# --- Configuration ---
CONFIG = {
    "train_pt_path": "./processed_data/train_data.pt",
    "val_pt_path": "./processed_data/val_data.pt",
    "img_root_train": "./coco_data/train2014",
    "img_root_val": "./coco_data/val2014",
    "batch_size": 64,
    "learning_rate": 1e-4,
    "epochs": 10,           # Increased epochs to let scheduler work
    "temperature": 0.07,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_path": "./saved_models/clip_resnet_v2.pth", # New filename
    "plot_path": "./training_loss_curve_v2.png"
}

def info_nce_loss(image_embeddings, text_embeddings, temperature, device):
    logits = (image_embeddings @ text_embeddings.T) / temperature
    targets = torch.arange(image_embeddings.shape[0]).to(device)
    loss_i2t = nn.functional.cross_entropy(logits, targets)
    loss_t2i = nn.functional.cross_entropy(logits.T, targets)
    return (loss_i2t + loss_t2i) / 2

def train_one_epoch(model, dataloader, optimizer, device, epoch_idx):
    model.train()
    running_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch_idx+1} [Train]")
    
    for batch in pbar:
        images = batch['image'].to(device)
        text_embeddings = batch['text_embedding'].to(device)
        
        optimizer.zero_grad()
        img_embeddings = model(images)
        loss = info_nce_loss(img_embeddings, text_embeddings, CONFIG['temperature'], device)
        loss.backward()
        optimizer.step()
        
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

def main():
    os.makedirs(os.path.dirname(CONFIG['save_path']), exist_ok=True)
    
    print("Initializing IMPROVED Datasets...")
    # Note 'split' parameter
    train_dataset = CocoClipDatasetImproved(CONFIG['train_pt_path'], CONFIG['img_root_train'])
    val_dataset = CocoClipDatasetImproved(CONFIG['val_pt_path'], CONFIG['img_root_val'])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
    
    print("Initializing IMPROVED Model...")
    model = ImageEncoderImproved().to(CONFIG['device'])
    
    # MODIFICATION: AdamW Optimizer + Weight Decay
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-4)
    
    # MODIFICATION: Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
    
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    
    start_time = time.time()
    
    for epoch in range(CONFIG['epochs']):
        train_loss = train_one_epoch(model, train_loader, optimizer, CONFIG['device'], epoch)
        val_loss = validate(model, val_loader, CONFIG['device'], epoch)
        
        # Step the scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1} - Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {current_lr:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CONFIG['save_path'])
            print(f"  -> Saved Best Model!")

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Improved CLIP Training (Augmentation + Regularization)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(CONFIG['plot_path'])
    
    print(f"Training finished in {(time.time()-start_time)/60:.2f} mins.")

if __name__ == "__main__":
    main()