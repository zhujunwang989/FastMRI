"""
Utility functions for saving and loading model parameters for deployment.
"""

import os
import torch
from pathlib import Path
from model_config import ModelConfig

def save_model_parameters(model, epoch, optimizer=None, scheduler=None, 
                        metrics=None, save_dir='saved_models'):
    """
    Save model parameters and related training state.
    
    Args:
        model: The PyTorch model
        epoch: Current epoch number
        optimizer: Optional optimizer state
        scheduler: Optional scheduler state
        metrics: Optional dictionary of metrics (e.g., {'val_loss': 0.123})
        save_dir: Directory to save the parameters
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Prepare the state dictionary
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'config': ModelConfig.get_config_dict()  # Save configuration with the model
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    # Save the full checkpoint for training continuation
    full_checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, full_checkpoint_path)
    
    # Save a lightweight version for deployment (only model parameters)
    deploy_checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': ModelConfig.get_config_dict()
    }
    deploy_path = os.path.join(save_dir, 'model_deploy.pt')
    torch.save(deploy_checkpoint, deploy_path)
    
    print(f"Saved full checkpoint to: {full_checkpoint_path}")
    print(f"Saved deployment model to: {deploy_path}")

def load_model_parameters(model, path, device='cpu', training=False):
    """
    Load model parameters for either training continuation or deployment.
    
    Args:
        model: The PyTorch model to load parameters into
        path: Path to the checkpoint file
        device: Device to load the parameters to ('cpu' or 'cuda')
        training: If True, load full checkpoint; if False, load only deployment-necessary parts
    
    Returns:
        If training=True: (model, optimizer, scheduler, epoch, metrics)
        If training=False: model
    """
    checkpoint = torch.load(path, map_location=device)
    
    # Verify configuration compatibility
    saved_config = checkpoint.get('config', {})
    current_config = ModelConfig.get_config_dict()
    
    # Check for critical mismatches
    critical_params = ['in_channels', 'out_channels', 'hidden_channels', 'num_pools']
    for param in critical_params:
        if saved_config.get(param) != current_config.get(param):
            raise ValueError(f"Configuration mismatch for {param}. "
                           f"Saved: {saved_config.get(param)}, "
                           f"Current: {current_config.get(param)}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if training:
        return {
            'model': model,
            'optimizer_state_dict': checkpoint.get('optimizer_state_dict'),
            'scheduler_state_dict': checkpoint.get('scheduler_state_dict'),
            'epoch': checkpoint.get('epoch', 0),
            'metrics': checkpoint.get('metrics', {})
        }
    
    return model

# Example usage
if __name__ == '__main__':
    # Example of saving parameters
    # model = YourModel()
    # save_model_parameters(model, epoch=10)
    
    # Example of loading for deployment
    # model = YourModel()
    # model = load_model_parameters(model, 'saved_models/model_deploy.pt')
    pass 