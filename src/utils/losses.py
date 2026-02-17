"""Loss functions for JEPA training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class JEPALoss(nn.Module):
    """
    JEPA Latent Prediction Loss.
    
    Options:
    - MSE (L2): Mean Squared Error in latent space
    - Smooth L1: Huber loss variant
    - Cosine Similarity: Normalized cosine distance
    
    CRITICAL: This is NOT pixel reconstruction loss!
    We predict latent representations, not pixels.
    """
    
    def __init__(self, loss_type: str = 'smooth_l1'):
        """
        Initialize JEPA loss.
        
        Args:
            loss_type: Type of loss ('mse', 'smooth_l1', 'cosine')
        """
        super().__init__()
        self.loss_type = loss_type
    
    def forward(self,
                predictions: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Compute latent prediction loss.
        
        Args:
            predictions: Predicted latents from predictor [B, num_masked, embed_dim]
            targets: Target latents from target encoder [B, num_masked, embed_dim]
            
        Returns:
            Scalar loss
        """
        if self.loss_type == 'mse':
            # Mean Squared Error
            loss = F.mse_loss(predictions, targets)
            
        elif self.loss_type == 'smooth_l1':
            # Smooth L1 (Huber) loss
            loss = F.smooth_l1_loss(predictions, targets)
            
        elif self.loss_type == 'cosine':
            # Cosine similarity loss
            # Normalize vectors
            predictions_norm = F.normalize(predictions, p=2, dim=-1)
            targets_norm = F.normalize(targets, p=2, dim=-1)
            
            # Cosine similarity (1 = identical, -1 = opposite)
            cosine_sim = (predictions_norm * targets_norm).sum(dim=-1)
            
            # Convert to distance loss (0 = identical, 2 = opposite)
            loss = (1 - cosine_sim).mean()
            
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in classification.
    
    Useful for downstream classification task if dataset is imbalanced.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor (0-1)
            gamma: Focusing parameter (higher = more focus on hard examples)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Model logits [B, num_classes]
            targets: Ground truth labels [B]
            
        Returns:
            Scalar loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()
