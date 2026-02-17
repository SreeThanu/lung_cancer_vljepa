"""
3D Vision Transformer Encoder for CT-JEPA

Stable masking version:
- Always returns [B, N, D]
- Mask only zeroes tokens (no slicing)
- No variable-length tensor outputs
- JEPA safe
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


# ============================================================
# PATCH EMBEDDING
# ============================================================

class PatchEmbed3D(nn.Module):
    """
    3D Patch Embedding using Conv3D projection
    """

    def __init__(
        self,
        input_size: Tuple[int, int, int] = (96, 96, 96),
        patch_size: Tuple[int, int, int] = (16, 16, 16),
        in_channels: int = 1,
        embed_dim: int = 768,
    ):
        super().__init__()

        self.input_size = input_size
        self.patch_size = patch_size

        self.num_patches = tuple(i // p for i, p in zip(input_size, patch_size))
        self.total_patches = (
            self.num_patches[0]
            * self.num_patches[1]
            * self.num_patches[2]
        )

        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, D, H, W]
        returns: [B, N, embed_dim]
        """
        x = self.proj(x)          # [B, D, D', H', W']
        x = x.flatten(2)          # [B, D, N]
        x = x.transpose(1, 2)     # [B, N, D]
        return x


# ============================================================
# VISION TRANSFORMER ENCODER
# ============================================================

class ViT3DEncoder(nn.Module):
    """
    3D ViT Encoder for JEPA and classification

    Input:
        [B, 1, 96, 96, 96]

    Output:
        [B, N, embed_dim]
    """

    def __init__(
        self,
        input_size: Tuple[int, int, int] = (96, 96, 96),
        patch_size: Tuple[int, int, int] = (16, 16, 16),
        in_channels: int = 1,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbed3D(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        num_patches = self.patch_embed.total_patches

        # Learnable positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim)
        )

        # Transformer encoder block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
        )

        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    # --------------------------------------------------------
    # Weight Initialization
    # --------------------------------------------------------

    def _init_weights(self):

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        for m in self.modules():

            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)

            elif isinstance(m, nn.Conv3d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, C, D, H, W]
            mask: [B, N] boolean (True = visible)

        Returns:
            [B, N, embed_dim]
        """

        # 1️⃣ Patch embedding
        x = self.patch_embed(x)  # [B, N, D]

        # 2️⃣ Add position embeddings
        x = x + self.pos_embed

        # 3️⃣ Apply masking (JEPA context encoder only)
        # IMPORTANT:
        # We DO NOT slice tokens anymore.
        # We only zero-out masked tokens.
        if mask is not None:
            mask = mask.to(x.dtype).unsqueeze(-1)  # [B, N, 1]
            x = x * mask

        # 4️⃣ Transformer
        x = self.transformer(x)

        # 5️⃣ Final normalization
        x = self.norm(x)

        return x

    # --------------------------------------------------------

    def get_num_patches(self):
        return self.patch_embed.total_patches
