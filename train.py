import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import os

# Import your custom modules
from coco_dataset import CocoClipDataset
from model import ImageEncoder

# --- Configuration ---
BATCH_SIZE = 64        # Decrease to 32 if you run out of GPU memory
LEARNING_RATE = 1e-4   # Standard starting point for fine-tuning
EPOCHS = 10            # Number of passes through the data
TEMPERATURE = 0.07     # Softmax temperature (standard in CLIP)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
# --- InfoNCE Loss Definition ---
def info_nce_loss(image_embeddings, text_embeddings, temperature=TEMPERATURE):
    """
    Calculates the InfoNCE Loss (Symmetric Cross Entropy).
    
    Args:
        image_embeddings: Tensor of shape (Batch_Size, Embed_Dim)
        text_embeddings: Tensor of shape (Batch_Size, Embed_Dim)
    """
    # 1. Calculate Cosine Similarity (logits)
    # Since vectors are normalized in the model, dot product == cosine similarity
    # logits shape: (Batch_Size, Batch_Size)
    logits = (image_embeddings @ text_embeddings.T) / temperature
    
    # 2. Create Labels
    # The image at index i should match the text at index i.
    # So the 'correct' class for row i is index i.
    batch_size = image_embeddings.shape[0]
    labels = torch.arange(batch_size, dtype=torch.long).to(DEVICE)
    
    # 3. Calculate Loss (Symmetric)
    # Loss for Image-to-Text (rows)
    loss_i2t = nn.CrossEntropyLoss()(logits, labels)
    # Loss for Text-to-Image (columns)
    loss_t2i = nn.CrossEntropyLoss()(logits.T, labels)
    
    return (loss_i2t + loss_t2i) / 2

# --- Training Loop ---
def train_model():
    print(f"Hardware used: {DEVICE}")
    
    # 1. Load Data
    train_dataset = CocoClipDataset("./processed_data/train_data.pt", "./coco_data/train2014")
    val_dataset = CocoClipDataset("./processed_data/val_data.pt", "./coco_data/val2014")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # 2. Initialize Model
    model = ImageEncoder().to(DEVICE)
    
    # Optimizer (AdamW is generally better for Transformers/ResNets than SGD)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Tracking metrics
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            images = batch['image'].to(DEVICE)
            # Text embeddings are already cached, so we just load them
            text_embeddings = batch['text_embedding'].to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass (Image Encoder)
            img_embeddings = model(images)
            
            # Calculate Loss
            loss = info_nce_loss(img_embeddings, text_embeddings)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            
        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        
        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch['image'].to(DEVICE)
                text_embeddings = batch['text_embedding'].to(DEVICE)
                
                img_embeddings = model(images)
                loss = info_nce_loss(img_embeddings, text_embeddings)
                running_val_loss += loss.item()
                
        epoch_val_loss = running_val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)
        
        print(f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
        
        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved Best Model!")

    total_time = time.time() - start_time
    print(f"\nTraining Complete in {total_time/60:.2f} minutes.")
    
    # --- 3. Plotting Results ---
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('InfoNCE Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')
    print("Loss plot saved to loss_curve.png")

    # Save training metadata to text file (for report)
    with open("training_log.txt", "w") as f:
        f.write(f"Total Training Time: {total_time/60:.2f} minutes\n")
        f.write(f"Hardware: {DEVICE}\n")
        f.write(f"Best Validation Loss: {best_val_loss:.4f}\n")
        f.write(f"Parameters: LR={LEARNING_RATE}, BS={BATCH_SIZE}, Temp={TEMPERATURE}\n")

if __name__ == "__main__":
    train_model()