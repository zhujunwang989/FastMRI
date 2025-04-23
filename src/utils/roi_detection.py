"""
ROI Detection module for FastMRI knee images.
Uses traditional computer vision techniques to detect knee joint regions.
"""

import numpy as np
import cv2
from scipy import ndimage
from typing import Tuple, Dict, Optional

class ROIDetector:
    def __init__(self, 
                 min_area: int = 1000,
                 max_area: int = 50000,
                 threshold_value: int = 127,
                 blur_kernel_size: int = 5):
        """
        Initialize ROI detector for knee MRI images.
        
        Args:
            min_area: Minimum area of detected regions
            max_area: Maximum area of detected regions
            threshold_value: Threshold value for image binarization
            blur_kernel_size: Size of Gaussian blur kernel
        """
        self.min_area = min_area
        self.max_area = max_area
        self.threshold_value = threshold_value
        self.blur_kernel_size = blur_kernel_size

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image for ROI detection.
        
        Args:
            image: Input MRI image (2D numpy array)
            
        Returns:
            Preprocessed image
        """
        # Normalize image to 0-255 range
        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image, (self.blur_kernel_size, self.blur_kernel_size), 0)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )
        
        return binary

    def detect_roi(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Detect ROI in the knee MRI image.
        
        Args:
            image: Input MRI image (2D numpy array)
            
        Returns:
            Tuple of (ROI mask, ROI metadata)
        """
        # Preprocess image
        binary = self.preprocess_image(image)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        valid_contours = [
            cnt for cnt in contours 
            if self.min_area < cv2.contourArea(cnt) < self.max_area
        ]
        
        # Create ROI mask
        roi_mask = np.zeros_like(image, dtype=np.uint8)
        cv2.drawContours(roi_mask, valid_contours, -1, 1, -1)
        
        # Calculate ROI metadata
        roi_metadata = {
            'num_regions': len(valid_contours),
            'total_area': np.sum(roi_mask),
            'center_of_mass': ndimage.center_of_mass(roi_mask),
            'bounding_boxes': [cv2.boundingRect(cnt) for cnt in valid_contours]
        }
        
        return roi_mask, roi_metadata

    def calculate_roi_weights(self, 
                            roi_mask: np.ndarray, 
                            base_weight: float = 1.0,
                            roi_weight: float = 2.0) -> np.ndarray:
        """
        Calculate loss weights based on ROI detection.
        
        Args:
            roi_mask: Binary mask of detected ROI
            base_weight: Base weight for non-ROI regions
            roi_weight: Weight multiplier for ROI regions
            
        Returns:
            Weight map for loss calculation
        """
        weight_map = np.ones_like(roi_mask, dtype=np.float32) * base_weight
        weight_map[roi_mask > 0] = roi_weight
        
        # Apply Gaussian smoothing to weight map
        weight_map = cv2.GaussianBlur(weight_map, (5, 5), 0)
        
        return weight_map

class ROILossWeighting:
    def __init__(self, 
                 base_weight: float = 1.0,
                 roi_weight: float = 2.0,
                 smooth_factor: float = 0.5):
        """
        Initialize ROI-based loss weighting.
        
        Args:
            base_weight: Base weight for non-ROI regions
            roi_weight: Weight multiplier for ROI regions
            smooth_factor: Smoothing factor for weight transitions
        """
        self.roi_detector = ROIDetector()
        self.base_weight = base_weight
        self.roi_weight = roi_weight
        self.smooth_factor = smooth_factor

    def get_loss_weights(self, 
                        image: np.ndarray,
                        previous_weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get loss weights based on ROI detection.
        
        Args:
            image: Input MRI image
            previous_weights: Optional previous weight map for smoothing
            
        Returns:
            Weight map for loss calculation
        """
        # Detect ROI
        roi_mask, _ = self.roi_detector.detect_roi(image)
        
        # Calculate new weights
        new_weights = self.roi_detector.calculate_roi_weights(
            roi_mask,
            self.base_weight,
            self.roi_weight
        )
        
        # Smooth weights if previous weights are provided
        if previous_weights is not None:
            weights = (self.smooth_factor * new_weights + 
                      (1 - self.smooth_factor) * previous_weights)
        else:
            weights = new_weights
            
        return weights 