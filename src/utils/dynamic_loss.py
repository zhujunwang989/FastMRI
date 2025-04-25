import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import SSIM

class DynamicROILoss(nn.Module):
    """
    Dynamic loss function that combines MSE and SSIM losses with ROI weighting.
    The loss is weighted more heavily in the ROI regions.
    """
    def __init__(self, alpha=0.5, roi_weight=2.0):
        """
        Args:
            alpha: Weight between MSE (alpha) and SSIM (1-alpha) losses
            roi_weight: Weight multiplier for loss in ROI regions
        """
        super().__init__()
        self.alpha = alpha
        self.roi_weight = roi_weight
        self.ssim_module = SSIM(data_range=1.0, size_average=True, channel=1)
        
    def forward(self, pred, target, roi_mask):
        """
        Compute the weighted loss.
        
        Args:
            pred: Predicted images (B, C, H, W)
            target: Target images (B, C, H, W)
            roi_mask: Binary ROI masks (B, C, H, W)
        """
        # Convert roi_mask to same device as pred
        if torch.is_tensor(roi_mask):
            roi_mask = roi_mask.to(pred.device)
        else:
            roi_mask = torch.from_numpy(roi_mask).to(pred.device)
        
        # Ensure all inputs are float tensors
        pred = pred.float()
        target = target.float()
        roi_mask = roi_mask.float()
        
        # Compute MSE loss with ROI weighting
        mse_loss = F.mse_loss(pred, target, reduction='none')
        weighted_mse = mse_loss * (1 + (self.roi_weight - 1) * roi_mask)
        mse_loss = weighted_mse.mean()
        
        # Compute SSIM loss (1 - SSIM to make it a loss)
        ssim_loss = 1 - self.ssim_module(pred, target)
        
        # Combine losses
        total_loss = self.alpha * mse_loss + (1 - self.alpha) * ssim_loss
        
        return total_loss 