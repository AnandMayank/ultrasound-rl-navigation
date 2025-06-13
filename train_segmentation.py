import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from PIL import Image
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

from core.segmentation_model import ResNetUNet, combined_loss
from core.utils import visualize_segmentation


class UltrasoundDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('L')
        mask = Image.open(self.mask_paths[idx]).convert('L')
        
        image = image.resize((256, 256))
        mask = mask.resize((256, 256))
        
        image = np.array(image, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32) / 255.0
        
        image = torch.FloatTensor(image).unsqueeze(0)
        mask = torch.FloatTensor(mask).unsqueeze(0)
        
        return image, mask


def train_segmentation_model(data_dir, save_dir, num_epochs=20, batch_size=8, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    image_paths = sorted(glob.glob(os.path.join(data_dir, "images", "*.png")))
    mask_paths = sorted(glob.glob(os.path.join(data_dir, "masks", "*.png")))
    
    if len(image_paths) != len(mask_paths):
        raise ValueError("Number of images and masks must match")
    
    split_idx = int(0.8 * len(image_paths))
    train_images = image_paths[:split_idx]
    train_masks = mask_paths[:split_idx]
    val_images = image_paths[split_idx:]
    val_masks = mask_paths[split_idx:]
    
    train_dataset = UltrasoundDataset(train_images, train_masks)
    val_dataset = UltrasoundDataset(val_images, val_masks)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = ResNetUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        
        for batch_idx, (images, masks) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = combined_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        model.eval()
        epoch_val_loss = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = combined_loss(outputs, masks)
                epoch_val_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
        
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch+1}.pth"))
    
    torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pth"))
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Progress')
    plt.savefig(os.path.join(save_dir, "training_curves.png"))
    plt.close()
    
    return model, train_losses, val_losses


if __name__ == "__main__":
    data_dir = "./abdominal_US"
    save_dir = "./trained_models/segmentation"
    
    model, train_losses, val_losses = train_segmentation_model(
        data_dir=data_dir,
        save_dir=save_dir,
        num_epochs=20,
        batch_size=8,
        learning_rate=0.001
    )
    
    print("Training completed!")
