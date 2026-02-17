"""
Momentum Encoder (Target Encoder) for JEPA.

Key Features:
- EMA (Exponential Moving Average) update
- Stop-gradient (no backprop through target)
- Initialized as copy of context encoder
"""

import torch
import torch.nn as nn
from typing import Optional
import copy


class MomentumEncoder(nn.Module):
    """
    Momentum-updated target encoder for JEPA.
    
    EMA Update Rule:
        θ_target = momentum * θ_target + (1 - momentum) * θ_context
    
    Where:
    - θ_target: Target encoder parameters
    - θ_context: Context encoder parameters  
    - momentum: EMA decay coefficient (typically 0.996-1.0)
    """
    
    def __init__(self,
                 base_encoder: nn.Module,
                 momentum: float = 0.996):
        """
        Initialize momentum encoder.
        
        Args:
            base_encoder: The context encoder to copy
            momentum: EMA momentum coefficient
        """
        super().__init__()
        
        # Create a copy of the base encoder
        self.encoder = copy.deepcopy(base_encoder)
        
        # Freeze all parameters (no gradient computation)
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.momentum = momentum
    
    @torch.no_grad()
    def update(self, base_encoder: nn.Module, momentum: Optional[float] = None):
        """
        Update target encoder parameters using EMA.
        
        Args:
            base_encoder: Context encoder with updated parameters
            momentum: Optional momentum override
        """
        if momentum is None:
            momentum = self.momentum
        
        # EMA update for all parameters
        for param_target, param_base in zip(
            self.encoder.parameters(), 
            base_encoder.parameters()
        ):
            param_target.data.mul_(momentum).add_(
                param_base.data, alpha=1.0 - momentum
            )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through target encoder.
        
        CRITICAL: Uses torch.no_grad() - no gradients flow through target!
        
        Args:
            x: Input volume [B, C, D, H, W]
            mask: Optional mask (typically None for target encoder)
            
        Returns:
            Target latent representations
        """
        with torch.no_grad():
            return self.encoder(x, mask)


def get_momentum_schedule(base_momentum: float = 0.996,
                          final_momentum: float = 1.0,
                          epochs: int = 100,
                          warmup_epochs: int = 10) -> list:
    """
    Create momentum schedule with warmup.
    
    Typical Schedule:
    - Warmup: linearly increase from 0.996 to 0.998
    - Later: cosine increase to 1.0
    
    Args:
        base_momentum: Starting momentum
        final_momentum: Final momentum (typically 1.0)
        epochs: Total training epochs
        warmup_epochs: Number of warmup epochs
        
    Returns:
        List of momentum values for each epoch
    """
    import numpy as np
    
    schedule = []
    
    for epoch in range(epochs):
        if epoch < warmup_epochs:
            # Linear warmup
            momentum = base_momentum + (1.0 - base_momentum) * 0.2 * (epoch / warmup_epochs)
        else:
            # Cosine schedule
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            momentum = final_momentum - (final_momentum - base_momentum) * 0.5 * (1 + np.cos(np.pi * progress))
        
        schedule.append(momentum)
    
    return schedule
