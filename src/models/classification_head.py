"""Classification head for downstream lung cancer prediction."""

import torch
import torch.nn as nn
from typing import List, Optional


class ClassificationHead(nn.Module):
    """
    Classification head for fine-tuning/linear-probe style training.

    Architecture:
    - Global average pooling over token dimension for encoder outputs [B, N, D]
    - MLP with dropout
    - Binary classification logits
    """

    def __init__(
        self,
        embed_dim: int = 384,
        hidden_dims: Optional[List[int]] = None,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256]

        layers = []
        in_dim = embed_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: encoder features [B, N, D] or pooled features [B, D]

        Returns:
            logits [B, num_classes]
        """
        if x.ndim == 3:
            x = x.mean(dim=1)
        elif x.ndim != 2:
            raise ValueError(f"Expected [B, N, D] or [B, D], got shape {tuple(x.shape)}")

        return self.classifier(x)
