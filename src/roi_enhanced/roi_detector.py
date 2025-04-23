"""
Advanced ROI Detection module for FastMRI knee images.
Uses multiple detection methods and combines their results.
"""

import numpy as np
import cv2
from scipy import ndimage
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass

@dataclass
class ROIDetectionResult:
    """Container for ROI detection results."""
    mask: np.ndarray
    confidence: float
    method: str
    metadata: Dict

class AdvancedROIDetector:
    def __init__(self,
                 min_area: int = 1000,
                 max_area: int = 50000,
                 confidence_threshold: float = 0.5):
        """
        Initialize advanced ROI detector with multiple detection methods.
        
        Args:
            min_area: Minimum area of detected regions
            max_area: Maximum area of detected regions
            confidence_threshold: Minimum confidence for detection
        """
        self.min_area = min_area
        self.max_area = max_area
        self.confidence_threshold = confidence_threshold
        
        # Initialize different detection methods
        self.methods = {
            'threshold': self._detect_by_threshold,
            'watershed': self._detect_by_watershed,
            'edge': self._detect_by_edge
        }

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for detection."""
        # Normalize to 0-255
        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        image = clahe.apply(image)
        
        return image

    def _detect_by_threshold(self, image: np.ndarray) -> ROIDetectionResult:
        """Detect ROI using adaptive thresholding."""
        # Preprocess
        processed = self._preprocess_image(image)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            processed,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by area
        valid_contours = [
            cnt for cnt in contours 
            if self.min_area < cv2.contourArea(cnt) < self.max_area
        ]
        
        # Create mask
        mask = np.zeros_like(image, dtype=np.uint8)
        cv2.drawContours(mask, valid_contours, -1, 1, -1)
        
        # Calculate confidence based on contour properties
        confidence = min(1.0, len(valid_contours) / 5.0)  # Simple heuristic
        
        return ROIDetectionResult(
            mask=mask,
            confidence=confidence,
            method='threshold',
            metadata={'num_contours': len(valid_contours)}
        )

    def _detect_by_watershed(self, image: np.ndarray) -> ROIDetectionResult:
        """Detect ROI using watershed segmentation."""
        # Preprocess
        processed = self._preprocess_image(image)
        
        # Apply Otsu's thresholding
        _, binary = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Noise removal
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        # Find unknown region
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown==255] = 0
        
        # Apply watershed
        markers = cv2.watershed(cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR), markers)
        
        # Create mask
        mask = np.zeros_like(image, dtype=np.uint8)
        mask[markers > 1] = 1
        
        # Calculate confidence
        confidence = min(1.0, np.sum(mask) / (image.shape[0] * image.shape[1] * 0.3))
        
        return ROIDetectionResult(
            mask=mask,
            confidence=confidence,
            method='watershed',
            metadata={'num_regions': len(np.unique(markers)) - 1}
        )

    def _detect_by_edge(self, image: np.ndarray) -> ROIDetectionResult:
        """Detect ROI using edge detection and contour analysis."""
        # Preprocess
        processed = self._preprocess_image(image)
        
        # Edge detection
        edges = cv2.Canny(processed, 50, 150)
        
        # Dilate edges
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by area and shape
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_area < area < self.max_area:
                # Check shape regularity
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.3:  # Filter out very irregular shapes
                        valid_contours.append(cnt)
        
        # Create mask
        mask = np.zeros_like(image, dtype=np.uint8)
        cv2.drawContours(mask, valid_contours, -1, 1, -1)
        
        # Calculate confidence
        confidence = min(1.0, len(valid_contours) / 3.0)
        
        return ROIDetectionResult(
            mask=mask,
            confidence=confidence,
            method='edge',
            metadata={'num_contours': len(valid_contours)}
        )

    def detect_roi(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Detect ROI using multiple methods and combine results.
        
        Args:
            image: Input MRI image (2D numpy array)
            
        Returns:
            Tuple of (combined ROI mask, metadata)
        """
        # Run all detection methods
        results = []
        for method_name, method_func in self.methods.items():
            try:
                result = method_func(image)
                if result.confidence >= self.confidence_threshold:
                    results.append(result)
            except Exception as e:
                print(f"Error in {method_name} detection: {str(e)}")
        
        if not results:
            # Fallback to threshold method if no confident detections
            results = [self._detect_by_threshold(image)]
        
        # Combine results
        combined_mask = np.zeros_like(image, dtype=np.uint8)
        total_confidence = 0
        
        for result in results:
            combined_mask = np.logical_or(combined_mask, result.mask)
            total_confidence += result.confidence
        
        # Normalize confidence
        avg_confidence = total_confidence / len(results)
        
        # Create metadata
        metadata = {
            'num_methods': len(results),
            'confidence': avg_confidence,
            'methods_used': [r.method for r in results],
            'center_of_mass': ndimage.center_of_mass(combined_mask),
            'total_area': np.sum(combined_mask)
        }
        
        return combined_mask.astype(np.uint8), metadata 