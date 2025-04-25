import torch
import torch.nn.functional as F

def gaussian_kernel(size=11, sigma=1.5):
    """
    Create a Gaussian kernel for SSIM calculation
    """
    coords = torch.arange(size).float() - size // 2
    coords = coords.view(1, -1).expand(size, -1)
    coords = coords.pow(2) + coords.t().pow(2)
    kernel = torch.exp(-coords / (2 * sigma ** 2))
    return kernel / kernel.sum()

def ssim(x, y, kernel_size=11, sigma=1.5):
    """
    Calculate SSIM between two images
    Args:
        x, y: Input images (B, C, H, W)
        kernel_size: Size of Gaussian kernel
        sigma: Standard deviation of Gaussian kernel
    """
    if x.size() != y.size():
        raise ValueError('Input images must have the same dimensions.')
        
    # Constants for stability
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    # Create Gaussian kernel
    kernel = gaussian_kernel(kernel_size, sigma).to(x.device)
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.expand(x.size(1), -1, -1, -1)
    
    # Calculate means
    mu_x = F.conv2d(x, kernel, groups=x.size(1), padding=kernel_size//2)
    mu_y = F.conv2d(y, kernel, groups=y.size(1), padding=kernel_size//2)
    
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y
    
    # Calculate variances and covariance
    sigma_x_sq = F.conv2d(x * x, kernel, groups=x.size(1), padding=kernel_size//2) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, kernel, groups=y.size(1), padding=kernel_size//2) - mu_y_sq
    sigma_xy = F.conv2d(x * y, kernel, groups=x.size(1), padding=kernel_size//2) - mu_xy
    
    # Calculate SSIM
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
    
    return ssim_map.mean()

def calculate_metrics(pred, target):
    """
    Calculate multiple metrics between prediction and target
    Returns:
        dict: Dictionary containing MSE and SSIM values
    """
    mse = F.mse_loss(pred, target)
    ssim_val = ssim(pred, target)
    
    return {
        'mse': mse.item(),
        'ssim': ssim_val.item()
    } 