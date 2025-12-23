"""
Focal Loss for Imbalanced Classification
Addresses the class imbalance problem by down-weighting easy examples
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    FL(pt) = -alpha * (1-pt)^gamma * log(pt)
    
    Args:
        alpha: Class weights (can be tensor or float)
        gamma: Focusing parameter (typically 2.0)
        reduction: 'mean' or 'sum'
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [N, C] logits
            targets: [N] class indices
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # probability of correct class
        
        # Apply focal term
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Apply class weights if provided
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                # Move alpha to same device as targets
                alpha_device = self.alpha.to(targets.device)
                # Get alpha for each sample's target class
                alpha_t = alpha_device.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class BalancedFocalLoss(nn.Module):
    """
    Balanced variant of Focal Loss that automatically computes class weights
    """
    
    def __init__(self, gamma=2.0, beta=0.999, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.beta = beta
        self.reduction = reduction
        self.class_counts = None
        self.effective_num = None
        
    def update_class_counts(self, targets, num_classes=3):
        """Update running class counts for balanced weights"""
        if self.class_counts is None:
            self.class_counts = torch.zeros(num_classes, device=targets.device)
        
        for i in range(num_classes):
            self.class_counts[i] = torch.sum(targets == i).float()
        
        # Compute effective number of samples
        self.effective_num = 1.0 - torch.pow(self.beta, self.class_counts)
        self.effective_num = torch.where(
            self.class_counts > 0,
            self.effective_num,
            torch.ones_like(self.effective_num)
        )
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [N, C] logits
            targets: [N] class indices
        """
        # Update class statistics
        self.update_class_counts(targets, num_classes=inputs.size(1))
        
        # Compute balanced weights
        weights = (1.0 - self.beta) / self.effective_num
        weights = weights / weights.sum() * len(weights)
        
        # Apply focal loss with balanced weights
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Get weight for each sample
        sample_weights = weights.gather(0, targets)
        
        # Apply focal term and weights
        focal_loss = sample_weights * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss