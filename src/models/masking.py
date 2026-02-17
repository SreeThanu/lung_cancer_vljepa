"""
3D Block Masking Strategy for CT-JEPA.

Masking Strategy:
- Random contiguous 3D cuboid/block masking
- Mask ratio: 30-50% of volume
- Context encoder sees only visible patches
- Target encoder sees full volume (but we predict only masked regions)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Optional


class BlockMask3D:
    """Generate 3D block masks for JEPA."""
    
    def __init__(self,
                 input_size: Tuple[int, int, int],
                 patch_size: Tuple[int, int, int],
                 num_blocks: int = 4,
                 min_block_scale: float = 0.15,
                 max_block_scale: float = 0.35,
                 aspect_ratio_min: float = 0.3,
                 aspect_ratio_max: float = 3.0):
        """
        Initialize 3D block masking.
        
        Args:
            input_size: Input volume size (D, H, W)
            patch_size: Patch size for tokenization
            num_blocks: Number of masked blocks
            min_block_scale: Minimum block size (relative to volume)
            max_block_scale: Maximum block size (relative to volume)
            aspect_ratio_min: Minimum aspect ratio
            aspect_ratio_max: Maximum aspect ratio
        """
        self.input_size = input_size
        self.patch_size = patch_size
        self.num_blocks = num_blocks
        self.min_block_scale = min_block_scale
        self.max_block_scale = max_block_scale
        self.aspect_ratio_min = aspect_ratio_min
        self.aspect_ratio_max = aspect_ratio_max
        
        # Calculate number of patches per dimension
        self.num_patches = tuple(i // p for i, p in zip(input_size, patch_size))
        self.total_patches = np.prod(self.num_patches)
        
    def __call__(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate block masks for a batch.
        
        Returns:
            visible_masks: Boolean mask for visible patches [B, num_patches]
            target_masks: Boolean mask for prediction targets [B, num_patches]
        """
        visible_masks = []
        target_masks = []
        
        for _ in range(batch_size):
            mask = self._generate_single_mask()
            visible_masks.append(~mask)  # Visible = NOT masked
            target_masks.append(mask)     # Predict masked regions
        
        visible_masks = torch.stack(visible_masks)
        target_masks = torch.stack(target_masks)
        
        return visible_masks, target_masks
    
    def _generate_single_mask(self) -> torch.Tensor:
        """
        Generate a single mask with multiple random blocks.
        
        Returns:
            Boolean mask [num_patches_d * num_patches_h * num_patches_w]
        """
        mask_3d = torch.zeros(self.num_patches, dtype=torch.bool)
        
        for _ in range(self.num_blocks):
            # Sample block size
            scale = np.random.uniform(self.min_block_scale, self.max_block_scale)
            
            # Sample aspect ratios for 3D block
            aspect_d = np.random.uniform(self.aspect_ratio_min, self.aspect_ratio_max)
            aspect_h = np.random.uniform(self.aspect_ratio_min, self.aspect_ratio_max)
            aspect_w = 1.0 / (aspect_d * aspect_h)
            
            # Calculate block dimensions in patches
            block_d = int(np.round(np.sqrt(scale * self.num_patches[0] * aspect_d)))
            block_h = int(np.round(np.sqrt(scale * self.num_patches[1] * aspect_h)))
            block_w = int(np.round(np.sqrt(scale * self.num_patches[2] * aspect_w)))
            
            # Clamp to valid range
            block_d = min(max(block_d, 1), self.num_patches[0])
            block_h = min(max(block_h, 1), self.num_patches[1])
            block_w = min(max(block_w, 1), self.num_patches[2])
            
            # Random position
            start_d = np.random.randint(0, self.num_patches[0] - block_d + 1)
            start_h = np.random.randint(0, self.num_patches[1] - block_h + 1)
            start_w = np.random.randint(0, self.num_patches[2] - block_w + 1)
            
            # Apply mask
            mask_3d[start_d:start_d+block_d, 
                   start_h:start_h+block_h, 
                   start_w:start_w+block_w] = True
        
        # Flatten to 1D
        return mask_3d.flatten()


class MaskCollator:
    """
    Collator for creating batches with masks for JEPA training.
    
    Usage in DataLoader:
        loader = DataLoader(dataset, collate_fn=MaskCollator(...))
    """
    
    def __init__(self,
                 input_size: Tuple[int, int, int] = (96, 96, 96),
                 patch_size: Tuple[int, int, int] = (16, 16, 16),
                 num_blocks: int = 4,
                 min_block_scale: float = 0.15,
                 max_block_scale: float = 0.35):
        """
        Initialize mask collator.
        
        Args:
            input_size: Input volume size
            patch_size: Patch size
            num_blocks: Number of masked blocks
            min_block_scale: Min block scale
            max_block_scale: Max block scale
        """
        self.masker = BlockMask3D(
            input_size=input_size,
            patch_size=patch_size,
            num_blocks=num_blocks,
            min_block_scale=min_block_scale,
            max_block_scale=max_block_scale
        )
    
    def __call__(self, batch: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create batch with masks.
        
        Args:
            batch: List of volumes from dataset
            
        Returns:
            volumes: Stacked volumes [B, C, D, H, W]
            visible_masks: Visible patch masks [B, num_patches]
            target_masks: Target patch masks [B, num_patches]
        """
        volumes = torch.stack(batch)
        batch_size = volumes.size(0)
        
        visible_masks, target_masks = self.masker(batch_size)
        
        return volumes, visible_masks, target_masks
