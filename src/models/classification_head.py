"""Classification head for downstream task."""

import torch
import torch.nn as nn
from typing import List


class ClassificationHead(nn.Module):
    """
    Classification head for fine-tuning on lung cancer detection.
    
    Architecture:
    - Global pooling of encoder outputs
    - MLP with dropout
    - Binary classification output
    """
    
    def __init__(self,
                 embed_dim: int = 768,
                 hidden_dims: List[int] = [512, 256],
                 num_classes: int = 2,
                 dropout: float = 0.3):
        """
        Initialize classification head.
        
        Args:
            embed_dim: Input dimension from encoder
            hidden_dims: List of hidden layer dimensions
            num_classes: Number of output classes (2 for binary)
            dropout: Dropout rate
        """
        super().__init__()
        
        layers = []
        in_dim = embed_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        # Final classification layer
        layers.append(nn.Linear(in_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Encoder outputs [B, num_patches, embed_dim]
            
        Returns:
            Class logits [B, num_classes]
        """
        # Global average pooling
        x = x.mean(dim=1)  # [B, embed_dim]
        
        # Classification
        x = self.classifier(x)  # [B, num_classes]
        
        return x
