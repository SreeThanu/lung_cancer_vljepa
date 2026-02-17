"""
3D Grad-CAM for ViT3D Lung Cancer Model

Supports:
- Patch-level attention maps
- Tumor localization heatmaps
- Medical explainability

Designed for:
- LungCancerClassifier
- ViT3DEncoder
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


class GradCAM3D:
    """
    3D Grad-CAM implementation for ViT3D encoder.
    """

    def __init__(self, model, target_layer):
        """
        Args:
            model: Full classification model
            target_layer: Layer to hook (e.g., encoder.transformer.layers[-1])
        """
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self._register_hooks()

    # --------------------------------------------------------

    def _register_hooks(self):

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    # --------------------------------------------------------

    def generate(self, volume: torch.Tensor, class_idx: int = None):
        """
        Generate Grad-CAM heatmap.

        Args:
            volume: [1, 1, 96, 96, 96]
            class_idx: Target class index

        Returns:
            3D heatmap numpy array
        """

        self.model.eval()

        volume = volume.requires_grad_(True)

        output = self.model(volume)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        loss = output[:, class_idx]
        loss.backward()

        # Get gradients & activations
        grads = self.gradients         # [B, N, D]
        acts = self.activations        # [B, N, D]

        # Global average pooling over tokens
        weights = grads.mean(dim=1, keepdim=True)   # [B,1,D]

        cam = (weights * acts).sum(dim=-1)          # [B,N]
        cam = F.relu(cam)

        cam = cam.detach().cpu().numpy()[0]

        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

    # --------------------------------------------------------

    def project_to_volume(self,
                          cam_tokens: np.ndarray,
                          input_shape: Tuple[int, int, int],
                          patch_size: Tuple[int, int, int]):
        """
        Convert token-level CAM to full 3D volume heatmap.
        """

        D, H, W = input_shape
        pd, ph, pw = patch_size

        d_patches = D // pd
        h_patches = H // ph
        w_patches = W // pw

        cam_grid = cam_tokens.reshape(
            d_patches,
            h_patches,
            w_patches
        )

        heatmap = np.repeat(cam_grid, pd, axis=0)
        heatmap = np.repeat(heatmap, ph, axis=1)
        heatmap = np.repeat(heatmap, pw, axis=2)

        return heatmap

    # --------------------------------------------------------

    def overlay_slice(self,
                      volume: np.ndarray,
                      heatmap: np.ndarray,
                      slice_idx: int = 48):
        """
        Visualize 2D slice overlay.
        """

        plt.figure(figsize=(8, 6))

        plt.imshow(volume[slice_idx], cmap="gray")
        plt.imshow(heatmap[slice_idx], cmap="jet", alpha=0.4)

        plt.title("Grad-CAM Tumor Localization")
        plt.axis("off")
        plt.show()
