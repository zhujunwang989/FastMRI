"""
Advanced loss weighting module for FastMRI training.
Implements dynamic loss weighting based on ROI detection.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from .roi_detector import AdvancedROIDetector

@dataclass
class WeightingConfig:
    """Configuration for loss weighting."""
    base_weight: float = 1.0
    roi_weight: float = 2.0
    smooth_factor: float = 0.5
    confidence_threshold: float = 0.5
    min_area: int = 1000
    max_area: int = 50000

class EnhancedROIWeighting:
    def __init__(self, config: WeightingConfig):
        """
        Initialize enhanced ROI-based loss weighting.
        
        Args:
            config: Weighting configuration
        """
        self.config = config
        self.roi_detector = AdvancedROIDetector(
            min_area=config.min_area,
            max_area=config.max_area,
            confidence_threshold=config.confidence_threshold
        )
        self.previous_weights = None
        self.weight_history = []

    def get_loss_weights(self, 
                        image: np.ndarray,
                        previous_weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """
        Get loss weights based on ROI detection with enhanced features.
        
        Args:
            image: Input MRI image
            previous_weights: Optional previous weight map for smoothing
            
        Returns:
            Tuple of (weight map, metadata)
        """
        # Detect ROI
        roi_mask, metadata = self.roi_detector.detect_roi(image)
        
        # Calculate base weights
        weight_map = np.ones_like(image, dtype=np.float32) * self.config.base_weight
        
        # Apply ROI weights with confidence scaling
        confidence = metadata['confidence']
        roi_weight = self.config.roi_weight * confidence
        weight_map[roi_mask > 0] = roi_weight
        
        # Apply Gaussian smoothing
        weight_map = cv2.GaussianBlur(weight_map, (5, 5), 0)
        
        # Smooth with previous weights if available
        if previous_weights is not None:
            weight_map = (self.config.smooth_factor * weight_map + 
                         (1 - self.config.smooth_factor) * previous_weights)
        
        # Store weight history
        self.weight_history.append({
            'mean_weight': np.mean(weight_map),
            'max_weight': np.max(weight_map),
            'roi_area': metadata['total_area'],
            'confidence': confidence
        })
        
        return weight_map, metadata

class EnhancedROIWeightedLoss(nn.Module):
    def __init__(self, 
                 base_loss_fn: nn.Module,
                 config: WeightingConfig):
        """
        Initialize enhanced ROI-weighted loss function.
        
        Args:
            base_loss_fn: Base loss function (e.g., L1Loss, MSELoss)
            config: Weighting configuration
        """
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.weighting = EnhancedROIWeighting(config)
        self.previous_weights = None
        self.metadata_history = []

    def forward(self, 
                pred: torch.Tensor,
                target: torch.Tensor,
                input_image: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Calculate weighted loss with enhanced features.
        
        Args:
            pred: Model predictions
            target: Ground truth
            input_image: Input image for ROI detection
            
        Returns:
            Tuple of (loss value, metadata)
        """
        # Convert tensors to numpy for ROI detection
        input_np = input_image.detach().cpu().numpy()
        
        # Get loss weights for each image in the batch
        batch_weights = []
        batch_metadata = []
        
        for img in input_np:
            weights, metadata = self.weighting.get_loss_weights(
                img[0],  # Take first channel
                self.previous_weights
            )
            batch_weights.append(weights)
            batch_metadata.append(metadata)
        
        # Convert weights back to tensor
        weights = torch.from_numpy(np.stack(batch_weights)).to(pred.device)
        self.previous_weights = weights[0].cpu().numpy()
        
        # Calculate weighted loss
        loss = self.base_loss_fn(pred * weights, target * weights)
        
        # Store metadata
        self.metadata_history.append({
            'batch_metadata': batch_metadata,
            'mean_weight': weights.mean().item(),
            'max_weight': weights.max().item()
        })
        
        return loss, {
            'mean_weight': weights.mean().item(),
            'max_weight': weights.max().item(),
            'batch_metadata': batch_metadata
        }

    def get_weight_statistics(self) -> Dict:
        """Get statistics about weight history."""
        if not self.weighting.weight_history:
            return {}
        
        weights = self.weighting.weight_history
        return {
            'mean_weight': np.mean([w['mean_weight'] for w in weights]),
            'max_weight': np.max([w['max_weight'] for w in weights]),
            'mean_confidence': np.mean([w['confidence'] for w in weights]),
            'mean_roi_area': np.mean([w['roi_area'] for w in weights])
        } 