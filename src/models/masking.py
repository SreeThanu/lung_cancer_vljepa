"""
3D masking strategies for CT-JEPA.

- BlockMask3D: target masking regions (predicted tokens)
- ContextMask3D: contiguous visible-context region for context encoder
"""

import torch
import numpy as np
from typing import Tuple, List, Optional


class BlockMask3D:
    """Generate 3D target masks for JEPA."""

    def __init__(
        self,
        input_size: Tuple[int, int, int],
        patch_size: Tuple[int, int, int],
        num_blocks: int = 4,
        min_block_scale: float = 0.30,
        max_block_scale: float = 0.50,
        aspect_ratio_min: float = 0.75,
        aspect_ratio_max: float = 1.50,
    ):
        self.input_size = input_size
        self.patch_size = patch_size
        self.num_blocks = num_blocks
        self.min_block_scale = min_block_scale
        self.max_block_scale = max_block_scale
        self.aspect_ratio_min = aspect_ratio_min
        self.aspect_ratio_max = aspect_ratio_max

        self.num_patches = tuple(i // p for i, p in zip(input_size, patch_size))
        self.total_patches = int(np.prod(self.num_patches))

    def __call__(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        visible_masks = []
        target_masks = []

        for _ in range(batch_size):
            target = self._generate_single_mask()
            visible_masks.append(~target)
            target_masks.append(target)

        visible_masks = torch.stack(visible_masks)
        target_masks = torch.stack(target_masks)
        return visible_masks, target_masks

    def _generate_single_mask(self) -> torch.Tensor:
        """Generate one target mask with effective 30-50% masking."""
        target_ratio = float(np.random.uniform(self.min_block_scale, self.max_block_scale))
        target_count = max(1, int(round(target_ratio * self.total_patches)))

        mask_3d = torch.zeros(self.num_patches, dtype=torch.bool)

        attempts = 0
        max_attempts = 200

        while mask_3d.sum().item() < target_count and attempts < max_attempts:
            attempts += 1

            # sample cuboid volume around remaining need
            remaining = target_count - mask_3d.sum().item()
            approx_volume = max(1, remaining // max(self.num_blocks, 1))

            aspect_d = float(np.random.uniform(self.aspect_ratio_min, self.aspect_ratio_max))
            aspect_h = float(np.random.uniform(self.aspect_ratio_min, self.aspect_ratio_max))
            aspect_w = 1.0 / max(aspect_d * aspect_h, 1e-6)

            block_d = int(round((approx_volume * aspect_d) ** (1.0 / 3.0)))
            block_h = int(round((approx_volume * aspect_h) ** (1.0 / 3.0)))
            block_w = int(round((approx_volume * aspect_w) ** (1.0 / 3.0)))

            block_d = min(max(block_d, 1), self.num_patches[0])
            block_h = min(max(block_h, 1), self.num_patches[1])
            block_w = min(max(block_w, 1), self.num_patches[2])

            start_d = int(np.random.randint(0, self.num_patches[0] - block_d + 1))
            start_h = int(np.random.randint(0, self.num_patches[1] - block_h + 1))
            start_w = int(np.random.randint(0, self.num_patches[2] - block_w + 1))

            mask_3d[
                start_d : start_d + block_d,
                start_h : start_h + block_h,
                start_w : start_w + block_w,
            ] = True

        flat = mask_3d.flatten()

        # Keep exact target_count for consistent masked-token counts per sample.
        current_count = int(flat.sum().item())
        if current_count > target_count:
            idx = torch.where(flat)[0]
            perm = torch.randperm(idx.numel())
            drop = idx[perm[: current_count - target_count]]
            flat[drop] = False
        elif current_count < target_count:
            idx = torch.where(~flat)[0]
            perm = torch.randperm(idx.numel())
            add = idx[perm[: target_count - current_count]]
            flat[add] = True

        return flat


class ContextMask3D:
    """Single contiguous context region with 40-60% visible patches."""

    def __init__(
        self,
        input_size: Tuple[int, int, int],
        patch_size: Tuple[int, int, int],
        keep_ratio_min: float = 0.40,
        keep_ratio_max: float = 0.60,
    ):
        self.input_size = input_size
        self.patch_size = patch_size
        self.keep_ratio_min = keep_ratio_min
        self.keep_ratio_max = keep_ratio_max

        self.num_patches = tuple(i // p for i, p in zip(input_size, patch_size))
        self.total_patches = int(np.prod(self.num_patches))

    def __call__(
        self,
        batch_size: int,
        target_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        visible_masks = []
        for b in range(batch_size):
            visible = self._single_context_mask()
            if target_masks is not None:
                visible = visible & (~target_masks[b].bool())
                if visible.sum() == 0:
                    visible = ~target_masks[b].bool()
            visible_masks.append(visible)
        return torch.stack(visible_masks)

    def _single_context_mask(self) -> torch.Tensor:
        
        keep_ratio = float(np.random.uniform(self.keep_ratio_min, self.keep_ratio_max))
        keep_count = max(1, int(round(keep_ratio * self.total_patches)))
    
        side = keep_count ** (1.0 / 3.0)
        dims = np.array(self.num_patches, dtype=np.int32)
        context_dims = np.maximum(1, np.minimum(dims, np.round(side).astype(np.int32)))
    
        while int(np.prod(context_dims)) < keep_count:
            axis = int(np.argmin(context_dims / np.maximum(dims, 1)))
            if context_dims[axis] < dims[axis]:
                context_dims[axis] += 1
            else:
                break
    
        d, h, w = [int(v) for v in context_dims]
    
        start_d = int(np.random.randint(0, self.num_patches[0] - d + 1))
        start_h = int(np.random.randint(0, self.num_patches[1] - h + 1))
        start_w = int(np.random.randint(0, self.num_patches[2] - w + 1))
    
        vis_3d = torch.zeros(self.num_patches, dtype=torch.bool)
        vis_3d[start_d:start_d + d, start_h:start_h + h, start_w:start_w + w] = True
    
        # Shrink the last face to trim excess — preserves cuboid shape
        while int(vis_3d.sum().item()) > keep_count and context_dims[2] > 1:
            context_dims[2] -= 1
            vis_3d[start_d:start_d + context_dims[0],
                   start_h:start_h + context_dims[1],
                   start_w + context_dims[2]:start_w + context_dims[2] + 1] = False
    
        flat = vis_3d.flatten()
    
        # Only expand if significantly undershooting — no random scatter
        if int(flat.sum().item()) < keep_count:
            pass  # Accept the slight undershoot rather than break continuity
    
        return flat


class MaskCollator:
    """Collator for creating batches with masks for JEPA training."""

    def __init__(
        self,
        input_size: Tuple[int, int, int] = (96, 96, 96),
        patch_size: Tuple[int, int, int] = (16, 16, 16),
        num_blocks: int = 4,
        min_block_scale: float = 0.30,
        max_block_scale: float = 0.50,
    ):
        self.masker = BlockMask3D(
            input_size=input_size,
            patch_size=patch_size,
            num_blocks=num_blocks,
            min_block_scale=min_block_scale,
            max_block_scale=max_block_scale,
        )

    def __call__(self, batch: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        volumes = torch.stack(batch)
        batch_size = volumes.size(0)

        visible_masks, target_masks = self.masker(batch_size)
        return volumes, visible_masks, target_masks
