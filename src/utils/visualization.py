"""Visualization utilities for medical imaging."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
import torch


def plot_3d_volume_slices(volume: np.ndarray,
                          num_slices: int = 12,
                          axis: int = 0,
                          title: str = "Volume Slices",
                          cmap: str = 'gray',
                          figsize: Tuple[int, int] = (15, 10)):
    """
    Plot evenly spaced slices from a 3D volume.
    
    Args:
        volume: 3D numpy array [D, H, W]
        num_slices: Number of slices to display
        axis: Axis along which to slice (0=depth, 1=height, 2=width)
        title: Plot title
        cmap: Colormap
        figsize: Figure size
    """
    if isinstance(volume, torch.Tensor):
        volume = volume.cpu().numpy()
    
    # Move slicing axis to first position
    volume = np.moveaxis(volume, axis, 0)
    depth = volume.shape[0]
    
    indices = np.linspace(0, depth - 1, num_slices, dtype=int)
    
    rows = int(np.ceil(np.sqrt(num_slices)))
    cols = int(np.ceil(num_slices / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if num_slices > 1 else [axes]
    
    for idx, slice_idx in enumerate(indices):
        axes[idx].imshow(volume[slice_idx], cmap=cmap)
        axes[idx].set_title(f'Slice {slice_idx}/{depth}')
        axes[idx].axis('off')
    
    for idx in range(len(indices), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_attention_maps(volume: np.ndarray,
                        attention: np.ndarray,
                        num_samples: int = 6):
    """
    Visualize attention maps overlaid on CT slices.
    
    Args:
        volume: Input volume [D, H, W]
        attention: Attention weights [D, H, W]
        num_samples: Number of slices to show
    """
    depth = volume.shape[0]
    indices = np.linspace(0, depth - 1, num_samples, dtype=int)
    
    fig, axes = plt.subplots(2, num_samples, figsize=(18, 6))
    
    for idx, slice_idx in enumerate(indices):
        # Original slice
        axes[0, idx].imshow(volume[slice_idx], cmap='gray')
        axes[0, idx].set_title(f'Slice {slice_idx}')
        axes[0, idx].axis('off')
        
        # Attention overlay
        axes[1, idx].imshow(volume[slice_idx], cmap='gray')
        axes[1, idx].imshow(attention[slice_idx], cmap='hot', alpha=0.5)
        axes[1, idx].set_title('Attention')
        axes[1, idx].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_training_curves(train_losses: List[float],
                         val_losses: Optional[List[float]] = None,
                         title: str = "Training Curves",
                         figsize: Tuple[int, int] = (12, 5)):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        title: Plot title
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    
    if val_losses:
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def visualize_mask(volume: np.ndarray,
                   mask: np.ndarray,
                   slice_idx: Optional[int] = None):
    """
    Visualize masked regions on a CT slice.
    
    Args:
        volume: Input volume [D, H, W]
        mask: Binary mask [D, H, W]
        slice_idx: Slice to visualize (middle if None)
    """
    if slice_idx is None:
        slice_idx = volume.shape[0] // 2
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(volume[slice_idx], cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(mask[slice_idx], cmap='Reds', alpha=0.7)
    axes[1].set_title('Mask')
    axes[1].axis('off')
    
    masked = volume[slice_idx].copy()
    masked[mask[slice_idx] > 0] = masked.min()
    axes[2].imshow(masked, cmap='gray')
    axes[2].set_title('Masked Volume')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
