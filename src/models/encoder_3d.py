"""
3D Vision Transformer Encoder for CT-JEPA.

JEPA-critical behavior:
- Context encoder receives only visible tokens (masked tokens are removed)
- Variable visible-token counts are padded for batched transformer execution
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


# ============================================================
# PATCH EMBEDDING
# ============================================================

class PatchEmbed3D(nn.Module):
    """3D Patch Embedding using Conv3D projection."""

    def __init__(
        self,
        input_size: Tuple[int, int, int] = (96, 96, 96),
        patch_size: Tuple[int, int, int] = (16, 16, 16),
        in_channels: int = 1,
        embed_dim: int = 384,
    ):
        super().__init__()

        self.input_size = input_size
        self.patch_size = patch_size

        self.num_patches = tuple(i // p for i, p in zip(input_size, patch_size))
        self.total_patches = (
            self.num_patches[0] * self.num_patches[1] * self.num_patches[2]
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
        x = self.proj(x)      # [B, D, D', H', W']
        x = x.flatten(2)      # [B, D, N]
        x = x.transpose(1, 2) # [B, N, D]
        return x


# ============================================================
# VISION TRANSFORMER ENCODER
# ============================================================

class ViT3DEncoder(nn.Module):
    """3D ViT Encoder for JEPA and downstream classification."""

    def __init__(
        self,
        input_size: Tuple[int, int, int] = (96, 96, 96),
        patch_size: Tuple[int, int, int] = (16, 16, 16),
        in_channels: int = 1,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed3D(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        num_patches = self.patch_embed.total_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

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

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, C, D, H, W]
            mask: [B, N] boolean, True = visible

        Returns:
            If mask is None: [B, N, embed_dim]
            If mask is provided: [B, max_visible, embed_dim] (padded)
        """
        x = self.patch_embed(x)  # [B, N, D]
        x = x + self.pos_embed[:, : x.size(1), :]

        key_padding_mask = None

        if mask is not None:
            mask = mask.bool()
            if mask.shape != x.shape[:2]:
                raise ValueError(
                    f"Mask shape {mask.shape} must match [B, N] = {x.shape[:2]}"
                )

            visible_tokens = []
            visible_counts = []

            for b in range(x.size(0)):
                tokens_b = x[b][mask[b]]  # [num_visible, D]

                # Safety guard for extreme masks
                if tokens_b.size(0) == 0:
                    tokens_b = x[b, :1, :]

                visible_tokens.append(tokens_b)
                visible_counts.append(tokens_b.size(0))

            max_visible = max(visible_counts)

            padded = x.new_zeros((x.size(0), max_visible, x.size(-1)))
            key_padding_mask = torch.ones(
                (x.size(0), max_visible), dtype=torch.bool, device=x.device
            )

            for b, tokens_b in enumerate(visible_tokens):
                n = tokens_b.size(0)
                padded[b, :n] = tokens_b
                key_padding_mask[b, :n] = False

            x = padded

        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x

    def get_num_patches(self):
        return self.patch_embed.total_patches
