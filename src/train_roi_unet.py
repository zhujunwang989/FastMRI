import os
import sys

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.models.roi_unet import ROIUNet, ROIUNetTrainer
from src.data.fastmri_data import create_data_loaders
import logging
import os
from datetime import datetime
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_training(config):
    """
    Setup training components
    """
    # Create model
    model = ROIUNet(in_channels=1, out_channels=1)
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Setup trainer
    trainer = ROIUNetTrainer(
        model=model,
        optimizer=optimizer,
        device=config['device']
    )
    
    return trainer

def train(config, train_loader, val_loader):
    """
    Main training loop
    """
    trainer = setup_training(config)
    
    # Create checkpoint directory
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    best_val_loss = float('inf')
    start_time = datetime.now()
    
    # Calculate total steps for progress tracking
    total_batches = len(train_loader)
    total_val_batches = len(val_loader)
    
    for epoch in range(config['num_epochs']):
        print(f"\rEpoch {epoch+1}/{config['num_epochs']}")
        
        # Training
        train_loss, train_metrics = trainer.train_epoch(train_loader, epoch)
        
        # Validation
        val_loss, val_metrics = trainer.validate_epoch(val_loader, epoch)
        
        # Print metrics
        print(f"Training - Loss: {train_loss:.4f}, MSE: {train_metrics['mse']:.4f}, SSIM: {train_metrics['ssim']:.4f}")
        print(f"Validation - MSE: {val_metrics['mse']:.4f}, SSIM: {val_metrics['ssim']:.4f}")
        
        # Save checkpoint if best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(
                config['checkpoint_dir'],
                f'roi_unet_best.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'train_metrics': train_metrics,
            }, checkpoint_path)
        
        # Save periodic checkpoint
        if (epoch + 1) % config['checkpoint_frequency'] == 0:
            checkpoint_path = os.path.join(
                config['checkpoint_dir'],
                f'roi_unet_epoch_{epoch+1}.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'train_metrics': train_metrics,
            }, checkpoint_path)

if __name__ == '__main__':
    # Training configuration
    # Check for MPS (Metal Performance Shaders) availability
    if torch.backends.mps.is_available():
        device = 'mps'
        logger.info("Using MPS (Metal Performance Shaders) device")
    elif torch.cuda.is_available():
        device = 'cuda'
        logger.info("Using CUDA device")
    else:
        device = 'cpu'
        logger.info("Using CPU device")

    config = {
        'learning_rate': 1e-4,
        'num_epochs': 50,
        'batch_size': 32,
        'device': device,
        'checkpoint_dir': 'checkpoints/roi_unet',
        'checkpoint_frequency': 10,
        'data_path': 'Data/FastMRI_knee_singlecoil/extracted/train/singlecoil_train',
        'num_workers': 4,
        'pin_memory': True,
    }
    
    # Setup logging with more detailed format
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    
    # Log training configuration
    logger.info("Training configuration:")
    for key, value in config.items():
        logger.info(f"{key}: {value}")
    
    # Create data loaders
    logger.info('Creating data loaders...')
    train_loader, val_loader = create_data_loaders(
        data_path=config['data_path'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )
    
    # Enable cuDNN benchmarking for faster training
    if config['device'] in ['cuda', 'mps']:
        torch.backends.cudnn.benchmark = True
        logger.info('GPU acceleration is available, enabled cuDNN benchmarking')
    
    # Start training
    logger.info('Starting ROI-UNet training...')
    train(config, train_loader, val_loader) 