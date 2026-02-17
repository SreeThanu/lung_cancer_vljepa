"""
JEPA Predictor Network.

Role:
- Maps context latent representations to target latent space
- Predicts target encoder outputs from context encoder outputs
- Lightweight compared to encoders
"""

import torch
import torch.nn as nn
from typing import Optional


class JEPAPredictor(nn.Module):
    """
    Predictor network for JEPA.
    
    Architecture:
    - Input: Context latents from visible patches
    - Output: Predicted target latents for masked patches
    - Can be: MLP, Lightweight Transformer, or Cross-Attention
    """
    
    def __init__(self,
                 embed_dim: int = 768,
                 predictor_embed_dim: int = 384,
                 depth: int = 6,
                 num_heads: int = 6,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        """
        Initialize JEPA predictor.
        
        Args:
            embed_dim: Encoder embedding dimension
            predictor_embed_dim: Predictor internal dimension (typically smaller)
            depth: Number of transformer layers
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.predictor_embed_dim = predictor_embed_dim
        
        # Project encoder dim to predictor dim
        self.proj_in = nn.Linear(embed_dim, predictor_embed_dim)
        
        # Mask tokens (learnable embeddings for masked positions)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        
        # Transformer predictor
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=predictor_embed_dim,
            nhead=num_heads,
            dim_feedforward=int(predictor_embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth
        )
        
        # Project back to encoder dimension
        self.proj_out = nn.Linear(predictor_embed_dim, embed_dim)
        
        # Layer norm
        self.norm = nn.LayerNorm(predictor_embed_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self,
                context_latents: torch.Tensor,
                target_mask: torch.Tensor,
                num_total_patches: int) -> torch.Tensor:
        """
        Predict target latents from context latents.
        
        Forward Flow:
        1. Project context latents to predictor dimension
        2. Create mask tokens for masked positions
        3. Combine visible + mask tokens
        4. Apply transformer
        5. Extract predictions for masked positions only
        6. Project back to encoder dimension
        
        Args:
            context_latents: Context encoder outputs [B, num_visible, embed_dim]
            target_mask: Boolean mask of target patches [B, num_patches]
            num_total_patches: Total number of patches in full volume
            
        Returns:
            Predicted target latents [B, num_masked, embed_dim]
        """
        B = context_latents.size(0)
        
        # Project context to predictor dimension
        context_proj = self.proj_in(context_latents)  # [B, num_visible, predictor_dim]
        
        # Create full sequence with mask tokens
        num_masked = target_mask.sum(dim=1)[0].item()  # Number of masked patches
        
        # Expand mask tokens
        mask_tokens = self.mask_token.expand(B, num_masked, -1)
        
        # Concatenate: [visible patches, mask tokens]
        full_sequence = torch.cat([context_proj, mask_tokens], dim=1)
        
        # Apply transformer
        predicted = self.transformer(full_sequence)
        predicted = self.norm(predicted)
        
        # Extract only predictions for masked positions
        # (last num_masked tokens correspond to predictions)
        masked_predictions = predicted[:, -num_masked:, :]
        
        # Project back to encoder dimension
        masked_predictions = self.proj_out(masked_predictions)  # [B, num_masked, embed_dim]
        
        return masked_predictions


class MLPPredictor(nn.Module):
    """
    Simple MLP-based predictor (alternative to transformer).
    
    Lighter weight, good for quick experiments.
    """
    
    def __init__(self,
                 embed_dim: int = 768,
                 hidden_dim: int = 2048,
                 num_layers: int = 4):
        """
        Initialize MLP predictor.
        
        Args:
            embed_dim: Input/output dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of MLP layers
        """
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
        """Forward through MLP."""
        return self.mlp(x)
