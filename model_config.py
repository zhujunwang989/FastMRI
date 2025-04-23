"""
Configuration file for FastMRI model parameters and training settings.
"""

class ModelConfig:
    # Model Architecture Parameters
    in_channels = 1  # Input channels for MRI images
    out_channels = 1  # Output channels for reconstructed images
    hidden_channels = 32  # Number of hidden channels in the U-Net
    num_pools = 4  # Number of pooling layers in U-Net
    
    # Training Parameters
    batch_size = 16
    learning_rate = 1e-4
    num_epochs = 50
    weight_decay = 1e-5
    
    # Data Parameters
    train_val_split = 0.8  # 80% training, 20% validation
    acceleration_factor = 4  # Acceleration factor for k-space undersampling
    center_fractions = [0.08]  # Center fraction of k-space to preserve
    
    # Optimizer and Loss Parameters
    optimizer = 'Adam'
    loss_function = 'L1Loss'  # Options: L1Loss, MSELoss, SSIMLoss
    
    # Data Augmentation
    use_data_augmentation = True
    random_flip = True
    random_rotate = True
    
    # Checkpointing
    save_frequency = 5  # Save model every N epochs
    checkpoint_dir = 'checkpoints'
    
    # Logging
    log_dir = 'runs'
    log_frequency = 100  # Log metrics every N batches
    
    # Hardware
    device = 'cuda'  # 'cuda' or 'cpu'
    num_workers = 4  # Number of data loading workers
    
    # Model Specific Parameters
    normalization = 'instance'  # Options: batch, instance, group
    activation = 'ReLU'  # Options: ReLU, LeakyReLU
    use_residual = True
    use_attention = True
    
    # Early Stopping
    patience = 10  # Number of epochs to wait before early stopping
    min_delta = 1e-4  # Minimum change in validation loss to qualify as an improvement

    @classmethod
    def get_config_dict(cls):
        """Returns the configuration as a dictionary."""
        return {k: v for k, v in cls.__dict__.items() 
                if not k.startswith('__') and not callable(getattr(cls, k))}

# Example usage:
if __name__ == '__main__':
    config = ModelConfig()
    print("Model Configuration:")
    for key, value in ModelConfig.get_config_dict().items():
        print(f"{key}: {value}") 