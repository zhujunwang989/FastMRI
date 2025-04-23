"""
Training script for FastMRI model with enhanced ROI-based loss weighting.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
import logging
import json
from datetime import datetime

from ..models.unet import UNet
from ..data.fastmri_data import create_data_loaders
from .loss_weighting import EnhancedROIWeightedLoss, WeightingConfig
from ..utils.metrics import calculate_metrics

def setup_logging(log_dir: str):
    """Set up logging configuration."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'training_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return log_file

def train_epoch(model, dataloader, optimizer, loss_fn, device, epoch):
    """
    Train for one epoch with enhanced ROI-weighted loss.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader instance
        optimizer: Optimizer instance
        loss_fn: Enhanced ROI-weighted loss function
        device: Device to train on
        epoch: Current epoch number
        
    Returns:
        Dictionary of training metrics
    """
    model.train()
    total_loss = 0
    all_metrics = []
    
    with tqdm(dataloader, desc=f'Epoch {epoch}') as pbar:
        for batch_idx, (input_data, target) in enumerate(pbar):
            input_data = input_data.to(device)
            target = target.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(input_data)
            
            # Calculate loss with ROI weighting
            loss, loss_metadata = loss_fn(output, target, input_data)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            metrics = calculate_metrics(output, target)
            metrics.update(loss_metadata)
            all_metrics.append(metrics)
            
            # Update progress
            total_loss += loss.item()
            pbar.set_postfix({
                'loss': loss.item(),
                'mse': metrics['mse'],
                'ssim': metrics['ssim'],
                'mean_weight': loss_metadata['mean_weight']
            })
    
    # Average metrics
    avg_metrics = {
        k: np.mean([m[k] for m in all_metrics])
        for k in all_metrics[0].keys()
    }
    avg_metrics['loss'] = total_loss / len(dataloader)
    
    return avg_metrics

def validate(model, dataloader, loss_fn, device):
    """
    Validate the model with enhanced ROI-weighted loss.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader instance
        loss_fn: Enhanced ROI-weighted loss function
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
            loss, loss_metadata = loss_fn(output, target, input_data)
            total_loss += loss.item()
            
            # Calculate metrics
            metrics = calculate_metrics(output, target)
            metrics.update(loss_metadata)
            all_metrics.append(metrics)
    
    # Average metrics
    avg_metrics = {
        k: np.mean([m[k] for m in all_metrics])
        for k in all_metrics[0].keys()
    }
    avg_metrics['loss'] = total_loss / len(dataloader)
    
    return avg_metrics

def train_with_roi(model, train_loader, val_loader, config):
    """
    Train the model with enhanced ROI-based loss weighting.
    
    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        config: Training configuration
    """
    # Set up logging
    log_file = setup_logging(config.log_dir)
    logging.info(f"Starting training with ROI enhancement")
    logging.info(f"Configuration: {config.__dict__}")
    
    device = torch.device(config.device)
    model = model.to(device)
    
    # Initialize loss function with ROI weighting
    weighting_config = WeightingConfig(
        base_weight=1.0,
        roi_weight=2.0,
        smooth_factor=0.5,
        confidence_threshold=0.5,
        min_area=1000,
        max_area=50000
    )
    base_loss_fn = nn.L1Loss()
    loss_fn = EnhancedROIWeightedLoss(base_loss_fn, weighting_config)
    
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
        train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, device, epoch)
        
        # Validate
        val_metrics = validate(model, val_loader, loss_fn, device)
        
        # Log metrics
        logging.info(f'Epoch {epoch}:')
        logging.info(f'  Train Loss: {train_metrics["loss"]:.4f}')
        logging.info(f'  Train MSE: {train_metrics["mse"]:.4f}')
        logging.info(f'  Train SSIM: {train_metrics["ssim"]:.4f}')
        logging.info(f'  Val Loss: {val_metrics["loss"]:.4f}')
        logging.info(f'  Val MSE: {val_metrics["mse"]:.4f}')
        logging.info(f'  Val SSIM: {val_metrics["ssim"]:.4f}')
        logging.info(f'  Val PSNR: {val_metrics["psnr"]:.2f}')
        logging.info(f'  Mean Weight: {val_metrics["mean_weight"]:.2f}')
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_metrics': val_metrics,
                'weighting_config': weighting_config.__dict__,
                'weight_statistics': loss_fn.get_weight_statistics()
            }
            torch.save(checkpoint, Path(config.checkpoint_dir) / 'best_model_roi.pth')
            logging.info(f'Saved new best model with loss {best_val_loss:.4f}')
    
    # Save final statistics
    final_stats = {
        'best_val_loss': best_val_loss,
        'weight_statistics': loss_fn.get_weight_statistics(),
        'training_log': str(log_file)
    }
    
    with open(Path(config.log_dir) / 'training_stats.json', 'w') as f:
        json.dump(final_stats, f, indent=2)
    
    logging.info("Training completed")
    logging.info(f"Best validation loss: {best_val_loss:.4f}")
    logging.info(f"Final weight statistics: {loss_fn.get_weight_statistics()}")

if __name__ == '__main__':
    # Example usage
    from dataclasses import dataclass
    
    @dataclass
    class TrainingConfig:
        device: str = 'cuda'
        learning_rate: float = 1e-4
        weight_decay: float = 1e-5
        num_epochs: int = 50
        batch_size: int = 16
        log_dir: str = 'logs'
        checkpoint_dir: str = 'checkpoints'
    
    # Create model and data loaders
    model = UNet()
    train_loader, val_loader = create_data_loaders(
        'data/FastMRI_knee_singlecoil/extracted/train/singlecoil_train',
        batch_size=TrainingConfig.batch_size
    )
    
    # Train with ROI enhancement
    train_with_roi(model, train_loader, val_loader, TrainingConfig()) 