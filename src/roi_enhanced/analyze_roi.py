"""
Analysis and visualization script for ROI detection results.
This script can be run after training to generate visualizations for reporting.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import List, Dict, Tuple
import seaborn as sns
from tqdm import tqdm

from ..models.unet import UNet
from ..data.fastmri_data import create_data_loaders
from .roi_detector import AdvancedROIDetector
from .loss_weighting import WeightingConfig
from .visualize_roi import ROIVisualizer

def load_trained_model(model_path: str, device: str = 'cuda') -> Tuple[UNet, Dict]:
    """
    Load a trained model and its configuration.
    
    Args:
        model_path: Path to the trained model checkpoint
        device: Device to load the model on
        
    Returns:
        Tuple of (model, config)
    """
    checkpoint = torch.load(model_path, map_location=device)
    model = UNet().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint

def analyze_roi_detection(detector: AdvancedROIDetector,
                         dataloader: torch.utils.data.DataLoader,
                         device: str,
                         save_dir: str = 'analysis/roi_detection') -> Dict:
    """
    Analyze ROI detection performance on a dataset.
    
    Args:
        detector: ROI detector instance
        dataloader: DataLoader for the dataset
        device: Device to run analysis on
        save_dir: Directory to save results
        
    Returns:
        Dictionary of analysis results
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'confidence_scores': [],
        'roi_areas': [],
        'methods_used': [],
        'detection_times': []
    }
    
    visualizer = ROIVisualizer(save_dir)
    
    for batch_idx, (input_data, _) in enumerate(tqdm(dataloader, desc='Analyzing ROI detection')):
        if batch_idx >= 100:  # Analyze first 100 batches
            break
            
        input_data = input_data.to(device)
        
        for img in input_data:
            # Detect ROI
            roi_mask, metadata = detector.detect_roi(img[0].cpu().numpy())
            
            # Store results
            results['confidence_scores'].append(metadata['confidence'])
            results['roi_areas'].append(metadata['total_area'])
            results['methods_used'].extend(metadata['methods_used'])
            
            # Visualize detection
            visualizer.visualize_detection(
                img[0].cpu().numpy(),
                [ROIDetectionResult(
                    mask=roi_mask,
                    confidence=metadata['confidence'],
                    method='combined',
                    metadata=metadata
                )],
                save_path=save_dir / f'detection_{batch_idx}.png',
                show=False
            )
    
    # Calculate statistics
    stats = {
        'mean_confidence': np.mean(results['confidence_scores']),
        'std_confidence': np.std(results['confidence_scores']),
        'mean_area': np.mean(results['roi_areas']),
        'std_area': np.std(results['roi_areas']),
        'method_distribution': {
            method: results['methods_used'].count(method)
            for method in set(results['methods_used'])
        }
    }
    
    # Save results
    with open(save_dir / 'analysis_results.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats

def compare_with_without_roi(model: UNet,
                           dataloader: torch.utils.data.DataLoader,
                           device: str,
                           save_dir: str = 'analysis/comparison') -> Dict:
    """
    Compare model performance with and without ROI weighting.
    
    Args:
        model: Trained model
        dataloader: DataLoader for the dataset
        device: Device to run comparison on
        save_dir: Directory to save results
        
    Returns:
        Dictionary of comparison results
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'with_roi': {'psnr': [], 'ssim': [], 'loss': []},
        'without_roi': {'psnr': [], 'ssim': [], 'loss': []}
    }
    
    visualizer = ROIVisualizer(save_dir)
    detector = AdvancedROIDetector()
    
    for batch_idx, (input_data, target) in enumerate(tqdm(dataloader, desc='Comparing performance')):
        if batch_idx >= 50:  # Compare first 50 batches
            break
            
        input_data = input_data.to(device)
        target = target.to(device)
        
        # Get model predictions
        with torch.no_grad():
            output = model(input_data)
        
        # Calculate metrics without ROI weighting
        loss = torch.nn.L1Loss()(output, target)
        psnr = 10 * torch.log10(1.0 / loss)
        ssim = calculate_ssim(output, target)
        
        results['without_roi']['loss'].append(loss.item())
        results['without_roi']['psnr'].append(psnr.item())
        results['without_roi']['ssim'].append(ssim.item())
        
        # Calculate metrics with ROI weighting
        for i, img in enumerate(input_data):
            roi_mask, metadata = detector.detect_roi(img[0].cpu().numpy())
            weight_map = np.ones_like(roi_mask, dtype=np.float32)
            weight_map[roi_mask > 0] = 2.0  # Apply ROI weighting
            
            # Calculate weighted metrics
            weighted_loss = torch.nn.L1Loss()(
                output[i] * torch.from_numpy(weight_map).to(device),
                target[i] * torch.from_numpy(weight_map).to(device)
            )
            weighted_psnr = 10 * torch.log10(1.0 / weighted_loss)
            weighted_ssim = calculate_ssim(
                output[i] * torch.from_numpy(weight_map).to(device),
                target[i] * torch.from_numpy(weight_map).to(device)
            )
            
            results['with_roi']['loss'].append(weighted_loss.item())
            results['with_roi']['psnr'].append(weighted_psnr.item())
            results['with_roi']['ssim'].append(weighted_ssim.item())
            
            # Visualize comparison
            visualizer.visualize_weight_map(
                img[0].cpu().numpy(),
                weight_map,
                metadata,
                save_path=save_dir / f'comparison_{batch_idx}_{i}.png',
                show=False
            )
    
    # Calculate statistics
    stats = {
        'with_roi': {
            'mean_loss': np.mean(results['with_roi']['loss']),
            'mean_psnr': np.mean(results['with_roi']['psnr']),
            'mean_ssim': np.mean(results['with_roi']['ssim'])
        },
        'without_roi': {
            'mean_loss': np.mean(results['without_roi']['loss']),
            'mean_psnr': np.mean(results['without_roi']['psnr']),
            'mean_ssim': np.mean(results['without_roi']['ssim'])
        }
    }
    
    # Create comparison plots
    create_comparison_plots(results, save_dir)
    
    # Save results
    with open(save_dir / 'comparison_results.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats

def create_comparison_plots(results: Dict, save_dir: Path) -> None:
    """
    Create comparison plots for the analysis.
    
    Args:
        results: Dictionary of comparison results
        save_dir: Directory to save plots
    """
    # Set style
    plt.style.use('seaborn')
    
    # Create metrics comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['loss', 'psnr', 'ssim']
    titles = ['Loss', 'PSNR', 'SSIM']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        data = [
            results['with_roi'][metric],
            results['without_roi'][metric]
        ]
        
        sns.boxplot(data=data, ax=axes[i])
        axes[i].set_title(title)
        axes[i].set_xticklabels(['With ROI', 'Without ROI'])
    
    plt.tight_layout()
    plt.savefig(save_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create scatter plot of PSNR vs SSIM
    plt.figure(figsize=(10, 10))
    plt.scatter(
        results['with_roi']['psnr'],
        results['with_roi']['ssim'],
        label='With ROI',
        alpha=0.5
    )
    plt.scatter(
        results['without_roi']['psnr'],
        results['without_roi']['ssim'],
        label='Without ROI',
        alpha=0.5
    )
    plt.xlabel('PSNR')
    plt.ylabel('SSIM')
    plt.title('PSNR vs SSIM Comparison')
    plt.legend()
    plt.savefig(save_dir / 'psnr_ssim_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to run the analysis."""
    # Set up paths
    model_path = 'checkpoints/best_model_roi.pth'
    data_path = 'data/FastMRI_knee_singlecoil/extracted/train/singlecoil_train'
    
    # Create directories
    analysis_dir = Path('analysis')
    analysis_dir.mkdir(exist_ok=True)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, config = load_trained_model(model_path, device)
    
    # Create data loader
    _, val_loader = create_data_loaders(data_path, batch_size=16)
    
    # Run analysis
    print("Analyzing ROI detection...")
    roi_stats = analyze_roi_detection(
        AdvancedROIDetector(),
        val_loader,
        device,
        save_dir='analysis/roi_detection'
    )
    
    print("\nComparing performance with and without ROI...")
    comparison_stats = compare_with_without_roi(
        model,
        val_loader,
        device,
        save_dir='analysis/comparison'
    )
    
    print("\nAnalysis complete! Results saved in 'analysis' directory.")
    print("\nROI Detection Statistics:")
    print(json.dumps(roi_stats, indent=2))
    print("\nPerformance Comparison:")
    print(json.dumps(comparison_stats, indent=2))

if __name__ == '__main__':
    main() 