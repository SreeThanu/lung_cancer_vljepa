"""
JEPA Predictor Network.

Predictor receives context latents (visible tokens) and target mask, then predicts
latents only for masked tokens in the original token order.
"""

import torch
import torch.nn as nn


class JEPAPredictor(nn.Module):
    """Transformer predictor for JEPA latent prediction."""

    def __init__(
        self,
        embed_dim: int = 384,
        predictor_embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_num_patches: int = 512,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.predictor_embed_dim = predictor_embed_dim
        self.max_num_patches = max_num_patches

        self.proj_in = nn.Linear(embed_dim, predictor_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_num_patches, predictor_embed_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=predictor_embed_dim,
            nhead=num_heads,
            dim_feedforward=int(predictor_embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.proj_out = nn.Linear(predictor_embed_dim, embed_dim)
        self.norm = nn.LayerNorm(predictor_embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        context_latents: torch.Tensor,
        target_mask: torch.Tensor,
        num_total_patches: int,
        visible_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            context_latents: [B, max_visible, embed_dim] (padded visible tokens)
            target_mask: [B, N] bool, True for masked/target tokens
            num_total_patches: N
            visible_mask: [B, N] bool, True for context-visible tokens

        Returns:
            [B, max_masked, embed_dim] (padded masked-token predictions)
        """
        B = context_latents.size(0)
        N = int(num_total_patches)
        dtype = context_latents.dtype  # FIX: pin dtype from input once, use everywhere

        if N > self.max_num_patches:
            raise ValueError(
                f"num_total_patches={N} exceeds max_num_patches={self.max_num_patches}."
            )

        target_mask = target_mask.bool()
        if target_mask.shape != (B, N):
            raise ValueError(
                f"target_mask shape {target_mask.shape} must be (B, N)=({B}, {N})."
            )

        if visible_mask is None:
            visible_mask = ~target_mask

        visible_mask = visible_mask.bool()
        if visible_mask.shape != (B, N):
            raise ValueError(
                f"visible_mask shape {visible_mask.shape} must be (B, N)=({B}, {N})."
            )

        context_proj = self.proj_in(context_latents).to(dtype)  # FIX: ensure proj output matches dtype

        full_sequence = self.mask_token.expand(B, N, -1).clone().to(dtype)  # FIX: cast mask token to input dtype

        visible_counts = visible_mask.sum(dim=1)

        if int(visible_counts.max().item()) > context_proj.size(1):
            raise ValueError(
                "context_latents has fewer tokens than visible_mask requires."
            )

        for b in range(B):
            n_vis = int(visible_counts[b].item())
            if n_vis > 0:
                full_sequence[b, visible_mask[b]] = context_proj[b, :n_vis]  # both are dtype now

        full_sequence = full_sequence + self.pos_embed[:, :N, :].to(dtype)  # FIX: cast pos_embed to match

        predicted = self.transformer(full_sequence)
        predicted = self.norm(predicted)
        predicted = self.proj_out(predicted)  # [B, N, embed_dim]

        masked_counts = target_mask.sum(dim=1)
        max_masked = int(masked_counts.max().item())

        masked_predictions = predicted.new_zeros(B, max_masked, self.embed_dim)
        for b in range(B):
            n_masked = int(masked_counts[b].item())
            if n_masked > 0:
                masked_predictions[b, :n_masked] = predicted[b, target_mask[b]]

        return masked_predictions


class MLPPredictor(nn.Module):
    """Simple MLP predictor (optional alternative)."""

    def __init__(self, embed_dim: int = 384, hidden_dim: int = 1024, num_layers: int = 4):
        super().__init__()

        layers = []
        for i in range(num_layers):
            in_dim = embed_dim if i == 0 else hidden_dim
            out_dim = embed_dim if i == num_layers - 1 else hidden_dim

            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.GELU())
                layers.append(nn.LayerNorm(out_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)