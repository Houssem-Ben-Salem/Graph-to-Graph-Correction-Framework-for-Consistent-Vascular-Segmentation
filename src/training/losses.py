"""Loss functions for medical image segmentation"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation"""
    
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        # Apply sigmoid to predictions if not already applied
        if predictions.min() < 0 or predictions.max() > 1:
            predictions = torch.sigmoid(predictions)
        
        # Flatten tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Calculate Dice coefficient
        intersection = (predictions * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        
        # Return Dice loss (1 - Dice coefficient)
        return 1 - dice


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance"""
    
    def __init__(self, alpha=1, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce
    
    def forward(self, predictions, targets):
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(predictions)
        
        # Calculate focal loss
        ce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        
        if self.reduce:
            return torch.mean(loss)
        else:
            return loss


class CombinedLoss(nn.Module):
    """Combined Dice and Focal loss"""
    
    def __init__(self, dice_weight=0.5, focal_weight=0.5, focal_gamma=2.0, smooth=1e-6):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        self.dice_loss = DiceLoss(smooth=smooth)
        self.focal_loss = FocalLoss(gamma=focal_gamma)
    
    def forward(self, predictions, targets):
        dice = self.dice_loss(predictions, targets)
        focal = self.focal_loss(predictions, targets)
        
        return self.dice_weight * dice + self.focal_weight * focal


class BoundaryLoss(nn.Module):
    """Boundary-aware loss that emphasizes vessel boundaries"""
    
    def __init__(self, boundary_weight=2.0):
        super(BoundaryLoss, self).__init__()
        self.boundary_weight = boundary_weight
        self.dice_loss = DiceLoss()
        
    def forward(self, predictions, targets):
        # Standard dice loss
        dice = self.dice_loss(predictions, targets)
        
        # Calculate boundary loss
        boundary_targets = self._extract_boundaries(targets)
        boundary_loss = F.binary_cross_entropy_with_logits(
            predictions, boundary_targets, reduction='mean'
        )
        
        return dice + self.boundary_weight * boundary_loss
    
    def _extract_boundaries(self, masks):
        """Extract boundary pixels using morphological operations"""
        # Simple boundary extraction using erosion
        kernel = torch.ones(1, 1, 3, 3, 3, device=masks.device)
        eroded = F.conv3d(masks, kernel, padding=1) / kernel.sum()
        boundaries = masks - (eroded > 0.99).float()
        return boundaries


class TverskyLoss(nn.Module):
    """Tversky loss - generalization of Dice loss"""
    
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha  # Weight for false positives
        self.beta = beta    # Weight for false negatives
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions)
        
        # Flatten tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # True positives, false positives, false negatives
        TP = (predictions * targets).sum()
        FP = (predictions * (1 - targets)).sum()
        FN = ((1 - predictions) * targets).sum()
        
        # Tversky index
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        return 1 - tversky


class WeightedCombinedLoss(nn.Module):
    """Weighted combination of multiple losses with adaptive weighting"""
    
    def __init__(self, losses_config):
        super(WeightedCombinedLoss, self).__init__()
        
        self.losses = nn.ModuleDict()
        self.weights = {}
        
        for loss_name, config in losses_config.items():
            if loss_name == 'dice':
                self.losses[loss_name] = DiceLoss(smooth=config.get('smooth', 1e-6))
            elif loss_name == 'focal':
                self.losses[loss_name] = FocalLoss(
                    alpha=config.get('alpha', 1),
                    gamma=config.get('gamma', 2)
                )
            elif loss_name == 'tversky':
                self.losses[loss_name] = TverskyLoss(
                    alpha=config.get('alpha', 0.7),
                    beta=config.get('beta', 0.3)
                )
            elif loss_name == 'boundary':
                self.losses[loss_name] = BoundaryLoss(
                    boundary_weight=config.get('boundary_weight', 2.0)
                )
            
            self.weights[loss_name] = config.get('weight', 1.0)
    
    def forward(self, predictions, targets):
        total_loss = 0
        loss_dict = {}
        
        for loss_name, loss_fn in self.losses.items():
            loss_value = loss_fn(predictions, targets)
            weighted_loss = self.weights[loss_name] * loss_value
            total_loss += weighted_loss
            loss_dict[loss_name] = loss_value.item()
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict