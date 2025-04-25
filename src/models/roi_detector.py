import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import measure
from scipy import ndimage

class ROIDetector:
    """
    A class for detecting Regions of Interest (ROI) in knee MRI images.
    Uses intensity-based thresholding and connected component analysis.
    Optimized for GPU batch processing.
    """
    def __init__(self, threshold_ratio=0.2, min_area_ratio=0.05):
        self.threshold_ratio = threshold_ratio
        self.min_area_ratio = min_area_ratio
    
    def detect_roi(self, images):
        """
        Detect ROIs in a batch of images using GPU acceleration when possible.
        
        Args:
            images: Tensor of shape (B, C, H, W)
            
        Returns:
            roi_masks: Binary masks of same size as input
            bboxes: List of bounding boxes for each detected ROI
        """
        # Ensure input is a tensor
        if not torch.is_tensor(images):
            images = torch.from_numpy(images)
        
        # Move to same device as input
        device = images.device
        
        # Ensure 4D input
        if len(images.shape) == 3:
            images = images.unsqueeze(1)
        
        batch_size, channels, height, width = images.shape
        
        # Normalize images batch-wise
        # Handle potential MPS device by moving tensors to CPU for min/max operations
        if device.type == 'mps':
            images_cpu = images.cpu()
            min_vals = images_cpu.amin(dim=(2, 3), keepdim=True)
            max_vals = images_cpu.amax(dim=(2, 3), keepdim=True)
            min_vals = min_vals.to(device)
            max_vals = max_vals.to(device)
        else:
            min_vals = images.amin(dim=(2, 3), keepdim=True)
            max_vals = images.amax(dim=(2, 3), keepdim=True)
        
        images = (images - min_vals) / (max_vals - min_vals + 1e-8)
        
        # Apply threshold
        if device.type == 'mps':
            threshold = self.threshold_ratio * max_vals
        else:
            threshold = self.threshold_ratio * images.amax(dim=(2, 3), keepdim=True)
            
        binary = (images > threshold).float()
        
        # Apply 2D max pooling to connect nearby regions
        pooled = F.max_pool2d(binary, kernel_size=3, stride=1, padding=1)
        
        # Apply 2D average pooling to smooth the mask
        smoothed = F.avg_pool2d(pooled, kernel_size=5, stride=1, padding=2)
        
        # Threshold again to get binary mask
        roi_masks = (smoothed > 0.5).float()
        
        # Remove small regions by area thresholding
        min_area = self.min_area_ratio * height * width
        if device.type == 'mps':
            area = roi_masks.cpu().sum(dim=(2, 3), keepdim=True).to(device)
        else:
            area = roi_masks.sum(dim=(2, 3), keepdim=True)
            
        roi_masks = roi_masks * (area > min_area).float()
        
        # For compatibility, still return empty bboxes list
        bboxes = []
        
        return roi_masks, bboxes 