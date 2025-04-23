import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.models.unet import UNet
from src.data.fastmri_data import create_data_loaders
from src.utils.metrics import mse, ssim

def train_epoch(model, data_loader, optimizer, device):
    model.train()
    running_loss = 0.0
    running_mse = 0.0
    running_ssim = 0.0
    
    for batch in tqdm(data_loader, desc='Training'):
        image = batch['image'].to(device)
        target = batch['target'].to(device)
        
        optimizer.zero_grad()
        output = model(image)
        
        # Compute losses
        loss = mse(output, target)  # Using MSE as primary loss
        ssim_value = ssim(output, target)
        
        loss.backward()
        optimizer.step()
        
        # Update metrics
        running_loss += loss.item()
        running_mse += loss.item()  # MSE is same as loss in this case
        running_ssim += ssim_value.item()
    
    # Calculate averages
    avg_loss = running_loss / len(data_loader)
    avg_mse = running_mse / len(data_loader)
    avg_ssim = running_ssim / len(data_loader)
    
    return avg_loss, avg_mse, avg_ssim

def validate(model, data_loader, device):
    model.eval()
    running_mse = 0.0
    running_ssim = 0.0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Validation'):
            image = batch['image'].to(device)
            target = batch['target'].to(device)
            
            output = model(image)
            
            # Compute metrics
            mse_value = mse(output, target)
            ssim_value = ssim(output, target)
            
            running_mse += mse_value.item()
            running_ssim += ssim_value.item()
    
    # Calculate averages
    avg_mse = running_mse / len(data_loader)
    avg_ssim = running_ssim / len(data_loader)
    
    return avg_mse, avg_ssim

def train(data_path, num_epochs=50, batch_size=4, lr=1e-4):
    # Set up device (CPU, CUDA, or MPS)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    # Create model and move to device
    model = UNet().to(device)
    print(f"Model moved to {device}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        data_path,
        batch_size=batch_size
    )
    print(f"Created data loaders with batch size {batch_size}")
    
    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Create directory for saving models if it doesn't exist
    os.makedirs('checkpoints', exist_ok=True)
    
    # Training loop
    best_val_ssim = 0.0
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        # Train
        train_loss, train_mse, train_ssim = train_epoch(
            model, train_loader, optimizer, device
        )
        
        # Validate
        val_mse, val_ssim = validate(model, val_loader, device)
        
        # Print metrics
        print(f'Training - Loss: {train_loss:.4f}, MSE: {train_mse:.4f}, SSIM: {train_ssim:.4f}')
        print(f'Validation - MSE: {val_mse:.4f}, SSIM: {val_ssim:.4f}')
        
        # Save best model
        if val_ssim > best_val_ssim:
            best_val_ssim = val_ssim
            model_path = os.path.join('checkpoints', 'best_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f'Saved new best model to {model_path}!')

if __name__ == '__main__':
    # Set paths
    data_path = 'data/FastMRI_knee_singlecoil/extracted/train/singlecoil_train'
    
    # Start training with smaller batch size for MPS
    train(data_path, batch_size=4, num_epochs=50) 