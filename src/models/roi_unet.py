import torch
import torch.nn as nn
from .unet import UNet, ConvBlock
from .roi_detector import ROIDetector
from src.utils.dynamic_loss import DynamicROILoss
from tqdm import tqdm
from .metrics import calculate_metrics

class ROIUNet(nn.Module):
    """
    Enhanced UNet model that incorporates ROI detection for improved knee MRI reconstruction.
    Combines the standard UNet architecture with an ROI detection system for focused learning.
    """
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        # Base UNet model
        self.unet = UNet(in_channels, out_channels)
        
        # ROI detector
        self.roi_detector = ROIDetector()
        
        # Initialize loss function
        self.criterion = DynamicROILoss()
        
    def forward(self, x):
        # If input is a dictionary (from FastMRI dataset), get the image
        if isinstance(x, dict):
            x = x['image']
        
        # Ensure input is in the right format (B, C, H, W)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        # Get ROI mask (already a tensor from optimized detector)
        roi_mask, _ = self.roi_detector.detect_roi(x)
        
        # UNet reconstruction
        recon = self.unet(x)
        
        # Store ROI mask for loss calculation
        self.last_roi_mask = roi_mask
        
        return recon
    
    def compute_loss(self, pred, target):
        """
        Compute the ROI-weighted loss
        """
        # If target is a dictionary (from FastMRI dataset), get the target image
        if isinstance(target, dict):
            target = target['target']
            
        # Ensure target is in the right format (B, C, H, W)
        if len(target.shape) == 3:
            target = target.unsqueeze(1)
            
        return self.criterion(pred, target, self.last_roi_mask)

class ROIUNetTrainer:
    """
    Trainer class for ROIUNet model
    """
    def __init__(self, model, optimizer, device='cuda'):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        # Ensure model parameters are float32
        self.model = self.model.to(dtype=torch.float32)
        self.model.to(device)
        
    def train_step(self, input_data, target):
        """
        Single training step
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        try:
            # Move data to device and ensure float32
            if isinstance(input_data, dict):
                input_data = {k: v.to(self.device, dtype=torch.float32) if torch.is_tensor(v) else v 
                             for k, v in input_data.items()}
            else:
                input_data = input_data.to(self.device, dtype=torch.float32)
                
            if isinstance(target, dict):
                target = {k: v.to(self.device, dtype=torch.float32) if torch.is_tensor(v) else v 
                         for k, v in target.items()}
            else:
                target = target.to(self.device, dtype=torch.float32)
            
            # Forward pass
            output = self.model(input_data)
            
            # Compute loss with ROI weighting
            loss = self.model.compute_loss(output, target)
            
            # Calculate additional metrics
            if isinstance(target, dict):
                target_img = target['target']
            else:
                target_img = target
                
            metrics = calculate_metrics(output, target_img)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            return loss.item(), metrics
            
        except Exception as e:
            print(f"\nError in train_step: {str(e)}")
            print(f"Input type: {type(input_data)}")
            if isinstance(input_data, dict):
                print(f"Input keys: {input_data.keys()}")
                print(f"Input image shape: {input_data['image'].shape}")
            raise e
    
    def validate_step(self, input_data, target):
        """
        Single validation step
        """
        self.model.eval()
        with torch.no_grad():
            if isinstance(input_data, dict):
                input_data = {k: v.to(self.device, dtype=torch.float32) if torch.is_tensor(v) else v 
                             for k, v in input_data.items()}
            else:
                input_data = input_data.to(self.device, dtype=torch.float32)
                
            if isinstance(target, dict):
                target = {k: v.to(self.device, dtype=torch.float32) if torch.is_tensor(v) else v 
                         for k, v in target.items()}
            else:
                target = target.to(self.device, dtype=torch.float32)
            
            output = self.model(input_data)
            loss = self.model.compute_loss(output, target)
            
            # Calculate additional metrics
            if isinstance(target, dict):
                target_img = target['target']
            else:
                target_img = target
                
            metrics = calculate_metrics(output, target_img)
            
        return loss.item(), output, metrics

    def train_epoch(self, train_loader, epoch):
        """
        Train for one epoch
        """
        total_loss = 0
        total_metrics = {'mse': 0, 'ssim': 0}
        num_batches = len(train_loader)
        
        for batch_idx, data in enumerate(train_loader):
            # For FastMRI dataset, input and target are the same dictionary
            loss, metrics = self.train_step(data, data)
            total_loss += loss
            
            # Update metrics
            for key in total_metrics:
                total_metrics[key] += metrics[key]
            
            # Update progress bar
            progress = (batch_idx + 1) / num_batches * 100
            filled_length = int(50 * (batch_idx + 1) // num_batches)
            bar = '█' * filled_length + '-' * (50 - filled_length)
            print(f'\rTraining: {progress:3.0f}%|{bar}|', end='')
        print()  # New line after progress bar
        
        # Calculate final averages
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        return total_loss / num_batches, avg_metrics
    
    def validate_epoch(self, val_loader, epoch):
        """
        Validate for one epoch
        """
        total_loss = 0
        total_metrics = {'mse': 0, 'ssim': 0}
        num_batches = len(val_loader)
        
        for batch_idx, data in enumerate(val_loader):
            # For FastMRI dataset, input and target are the same dictionary
            loss, _, metrics = self.validate_step(data, data)
            total_loss += loss
            
            # Update metrics
            for key in total_metrics:
                total_metrics[key] += metrics[key]
            
            # Update progress bar
            progress = (batch_idx + 1) / num_batches * 100
            filled_length = int(50 * (batch_idx + 1) // num_batches)
            bar = '█' * filled_length + '-' * (50 - filled_length)
            print(f'\rValidation: {progress:3.0f}%|{bar}|', end='')
        print()  # New line after progress bar
        
        # Calculate final averages
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        return total_loss / num_batches, avg_metrics 