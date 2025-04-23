"""
Training script for FastMRI model with ROI-based loss weighting.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from pathlib import Path
import logging

from src.models.unet import UNet
from src.data.fastmri_data import create_data_loaders
from src.utils.metrics import mse, ssim
from src.utils.roi_detection import ROILossWeighting
from src.utils.metrics import calculate_metrics
from src.utils.visualize import visualize_training

class ROIWeightedLoss(nn.Module):
    def __init__(self, base_loss_fn, roi_weighting):
        """
        Initialize ROI-weighted loss function.
        
        Args:
            base_loss_fn: Base loss function (e.g., L1Loss, MSELoss)
            roi_weighting: ROILossWeighting instance
        """
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.roi_weighting = roi_weighting
        self.previous_weights = None

    def forward(self, pred, target, input_image):
        """
        Calculate weighted loss based on ROI detection.
        
        Args:
            pred: Model predictions
            target: Ground truth
            input_image: Input image for ROI detection
            
        Returns:
            Weighted loss value
        """
        # Convert tensors to numpy for ROI detection
        input_np = input_image.detach().cpu().numpy()
        
        # Get loss weights for each image in the batch
        batch_weights = []
        for img in input_np:
            weights = self.roi_weighting.get_loss_weights(
                img[0],  # Take first channel
                self.previous_weights
            )
            batch_weights.append(weights)
        
        # Convert weights back to tensor
        weights = torch.from_numpy(np.stack(batch_weights)).to(pred.device)
        self.previous_weights = weights[0].cpu().numpy()  # Store for next iteration
        
        # Calculate weighted loss
        loss = self.base_loss_fn(pred * weights, target * weights)
        return loss

def train_epoch(model, dataloader, optimizer, loss_fn, device, epoch):
    """
    Train for one epoch with ROI-weighted loss.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader instance
        optimizer: Optimizer instance
        loss_fn: ROI-weighted loss function
        device: Device to train on
        epoch: Current epoch number
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    
    with tqdm(dataloader, desc=f'Epoch {epoch}') as pbar:
        for batch_idx, (input_data, target) in enumerate(pbar):
            input_data = input_data.to(device)
            target = target.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(input_data)
            
            # Calculate loss with ROI weighting
            loss = loss_fn(output, target, input_data)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update progress
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)

def validate(model, dataloader, loss_fn, device):
    """
    Validate the model with ROI-weighted loss.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader instance
        loss_fn: ROI-weighted loss function
        device: Device to validate on
        
    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    total_loss = 0
    all_metrics = []
    
    with torch.no_grad():
        for input_data, target in dataloader:
            input_data = input_data.to(device)
            target = target.to(device)
            
            # Forward pass
            output = model(input_data)
            
            # Calculate loss with ROI weighting
            loss = loss_fn(output, target, input_data)
            total_loss += loss.item()
            
            # Calculate metrics
            metrics = calculate_metrics(output, target)
            all_metrics.append(metrics)
    
    # Average metrics
    avg_metrics = {
        k: np.mean([m[k] for m in all_metrics])
        for k in all_metrics[0].keys()
    }
    avg_metrics['loss'] = total_loss / len(dataloader)
    
    return avg_metrics

def train_model(model, train_loader, val_loader, config):
    """
    Train the model with ROI-based loss weighting.
    
    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        config: Training configuration
    """
    device = torch.device(config.device)
    model = model.to(device)
    
    # Initialize loss function with ROI weighting
    base_loss_fn = nn.L1Loss()
    roi_weighting = ROILossWeighting(
        base_weight=1.0,
        roi_weight=2.0,
        smooth_factor=0.5
    )
    loss_fn = ROIWeightedLoss(base_loss_fn, roi_weighting)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(config.num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, epoch)
        
        # Validate
        val_metrics = validate(model, val_loader, loss_fn, device)
        
        # Log metrics
        logging.info(f'Epoch {epoch}:')
        logging.info(f'  Train Loss: {train_loss:.4f}')
        logging.info(f'  Val Loss: {val_metrics["loss"]:.4f}')
        logging.info(f'  Val PSNR: {val_metrics["psnr"]:.2f}')
        logging.info(f'  Val SSIM: {val_metrics["ssim"]:.4f}')
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_metrics': val_metrics
            }, Path(config.checkpoint_dir) / 'best_model.pth')
        
        # Visualize results
        if epoch % config.visualize_frequency == 0:
            visualize_training(model, val_loader, device, epoch)

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