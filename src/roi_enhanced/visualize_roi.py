"""
Visualization tools for ROI detection results.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import torch
from .roi_detector import AdvancedROIDetector, ROIDetectionResult

class ROIVisualizer:
    def __init__(self, save_dir: str = 'visualizations/roi'):
        """
        Initialize ROI visualizer.
        
        Args:
            save_dir: Directory to save visualizations
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up color maps
        self.cmap = plt.cm.viridis
        self.mask_cmap = plt.cm.Reds
        self.edge_cmap = plt.cm.Greens

    def visualize_detection(self, 
                          image: np.ndarray,
                          results: List[ROIDetectionResult],
                          save_path: Optional[str] = None,
                          show: bool = True) -> None:
        """
        Visualize ROI detection results from multiple methods.
        
        Args:
            image: Original MRI image
            results: List of detection results from different methods
            save_path: Optional path to save the visualization
            show: Whether to display the plot
        """
        n_methods = len(results)
        fig, axes = plt.subplots(2, n_methods + 1, figsize=(5 * (n_methods + 1), 10))
        
        # Plot original image
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Plot combined mask
        combined_mask = np.zeros_like(image, dtype=np.uint8)
        for result in results:
            combined_mask = np.logical_or(combined_mask, result.mask)
        
        axes[1, 0].imshow(combined_mask, cmap=self.mask_cmap)
        axes[1, 0].set_title('Combined Mask')
        axes[1, 0].axis('off')
        
        # Plot individual method results
        for i, result in enumerate(results, 1):
            # Plot mask
            axes[0, i].imshow(result.mask, cmap=self.mask_cmap)
            axes[0, i].set_title(f'{result.method}\nConfidence: {result.confidence:.2f}')
            axes[0, i].axis('off')
            
            # Plot overlay
            overlay = image.copy()
            overlay[result.mask > 0] = overlay[result.mask > 0] * 0.7 + 0.3 * 255
            axes[1, i].imshow(overlay, cmap='gray')
            axes[1, i].set_title(f'{result.method} Overlay')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()

    def visualize_weight_map(self,
                           image: np.ndarray,
                           weight_map: np.ndarray,
                           metadata: Dict,
                           save_path: Optional[str] = None,
                           show: bool = True) -> None:
        """
        Visualize loss weight map.
        
        Args:
            image: Original MRI image
            weight_map: Weight map for loss calculation
            metadata: ROI detection metadata
            save_path: Optional path to save the visualization
            show: Whether to display the plot
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot original image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Plot weight map
        im = axes[1].imshow(weight_map, cmap=self.cmap)
        axes[1].set_title('Weight Map')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
        
        # Plot overlay
        overlay = image.copy()
        overlay = overlay * 0.7 + 0.3 * weight_map * 255
        axes[2].imshow(overlay, cmap='gray')
        axes[2].set_title('Weight Overlay')
        axes[2].axis('off')
        
        # Add metadata text
        metadata_text = '\n'.join([
            f'Confidence: {metadata["confidence"]:.2f}',
            f'Methods: {", ".join(metadata["methods_used"])}',
            f'ROI Area: {metadata["total_area"]}'
        ])
        fig.text(0.5, 0.01, metadata_text, ha='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()

    def visualize_batch_results(self,
                              images: torch.Tensor,
                              weight_maps: torch.Tensor,
                              metadata_list: List[Dict],
                              batch_idx: int,
                              save_dir: Optional[str] = None) -> None:
        """
        Visualize ROI detection results for a batch of images.
        
        Args:
            images: Batch of input images
            weight_maps: Batch of weight maps
            metadata_list: List of metadata for each image
            batch_idx: Batch index for saving
            save_dir: Optional directory to save visualizations
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(len(images)):
            image = images[i].cpu().numpy()
            weight_map = weight_maps[i].cpu().numpy()
            metadata = metadata_list[i]
            
            if save_dir:
                save_path = save_dir / f'batch_{batch_idx}_image_{i}.png'
            else:
                save_path = None
            
            self.visualize_weight_map(
                image[0],  # Take first channel
                weight_map[0],
                metadata,
                save_path=save_path,
                show=False
            )

    def create_detection_video(self,
                             images: List[np.ndarray],
                             detector: AdvancedROIDetector,
                             output_path: str,
                             fps: int = 10) -> None:
        """
        Create a video showing ROI detection results over a sequence of images.
        
        Args:
            images: List of input images
            detector: ROI detector instance
            output_path: Path to save the video
            fps: Frames per second
        """
        # Get first image to determine size
        height, width = images[0].shape
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for image in images:
            # Detect ROI
            roi_mask, metadata = detector.detect_roi(image)
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(image, cmap='gray')
            ax.imshow(roi_mask, cmap=self.mask_cmap, alpha=0.3)
            ax.set_title(f'Confidence: {metadata["confidence"]:.2f}')
            ax.axis('off')
            
            # Convert plot to image
            fig.canvas.draw()
            plot_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            # Convert to BGR for OpenCV
            plot_image = cv2.cvtColor(plot_image, cv2.COLOR_RGB2BGR)
            
            # Write frame
            out.write(plot_image)
            
            plt.close()
        
        out.release()

def visualize_training_progress(model: torch.nn.Module,
                              dataloader: torch.utils.data.DataLoader,
                              device: torch.device,
                              epoch: int,
                              save_dir: str = 'visualizations/training') -> None:
    """
    Visualize model predictions and ROI detection during training.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader instance
        device: Device to run inference on
        epoch: Current epoch number
        save_dir: Directory to save visualizations
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    visualizer = ROIVisualizer(save_dir)
    detector = AdvancedROIDetector()
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (input_data, target) in enumerate(dataloader):
            if batch_idx >= 5:  # Visualize only first 5 batches
                break
                
            input_data = input_data.to(device)
            target = target.to(device)
            
            # Get model predictions
            output = model(input_data)
            
            # Detect ROI and calculate weights
            weight_maps = []
            metadata_list = []
            
            for img in input_data.cpu().numpy():
                roi_mask, metadata = detector.detect_roi(img[0])
                weight_maps.append(roi_mask)
                metadata_list.append(metadata)
            
            # Visualize results
            visualizer.visualize_batch_results(
                input_data,
                torch.from_numpy(np.stack(weight_maps)),
                metadata_list,
                batch_idx,
                save_dir / f'epoch_{epoch}'
            ) 