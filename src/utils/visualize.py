import torch
import matplotlib.pyplot as plt
import numpy as np
from src.models.unet import UNet
from src.data.fastmri_data import FastMRIDataset
import h5py

def load_model(checkpoint_path, device):
    """Load the trained model."""
    model = UNet().to(device)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model

def visualize_results(data_path, checkpoint_path, num_examples=6):
    """Visualize original vs reconstructed images."""
    # Set up device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = load_model(checkpoint_path, device)
    
    # Create dataset
    dataset = FastMRIDataset(data_path)
    
    # Set up the plot with 3 rows and 4 columns
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    plt.suptitle('Original vs Reconstructed Images', fontsize=16)
    
    with torch.no_grad():
        for i in range(num_examples):
            # Calculate row and column indices
            row = i // 2
            col = (i % 2) * 2  # Multiply by 2 to leave space for reconstruction
            
            # Get a random sample
            idx = np.random.randint(len(dataset))
            sample = dataset[idx]
            
            # Get original and target images
            image = sample['image'].unsqueeze(0).to(device)  # Add batch dimension
            target = sample['target']
            
            # Get reconstruction
            reconstruction = model(image)
            
            # Convert tensors to numpy arrays
            target = target.squeeze().cpu().numpy()
            reconstruction = reconstruction.squeeze().cpu().numpy()
            
            # Plot original
            axes[row, col].imshow(target, cmap='gray')
            axes[row, col].set_title(f'Original {i+1}')
            axes[row, col].axis('off')
            
            # Plot reconstruction
            axes[row, col + 1].imshow(reconstruction, cmap='gray')
            axes[row, col + 1].set_title(f'Reconstructed {i+1}')
            axes[row, col + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('comparison_results.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    data_path = 'data/FastMRI_knee_singlecoil/extracted/train/singlecoil_train'
    checkpoint_path = 'checkpoints/best_model.pth'
    visualize_results(data_path, checkpoint_path) 