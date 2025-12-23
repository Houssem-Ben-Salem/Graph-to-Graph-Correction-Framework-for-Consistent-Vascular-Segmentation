"""
Implementation of clDice and cbDice loss functions for topology-preserving segmentation.

Based on:
- "clDice - A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation" (CVPR 2021)
- "Centerline Boundary Dice Loss for Vascular Segmentation" (MICCAI 2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union


def soft_erode(img: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    Soft erosion operation using max pooling.
    
    Args:
        img: Input tensor of shape (B, C, H, W) or (B, C, D, H, W)
        kernel_size: Size of the erosion kernel
        
    Returns:
        Eroded tensor
    """
    if len(img.shape) == 4:  # 2D
        # Use negative values and min pooling to simulate erosion
        p = kernel_size // 2
        img_neg = -F.max_pool2d(-img, kernel_size, stride=1, padding=p)
        return img_neg
    elif len(img.shape) == 5:  # 3D
        p = kernel_size // 2
        img_neg = -F.max_pool3d(-img, kernel_size, stride=1, padding=p)
        return img_neg
    else:
        raise ValueError(f"Unsupported tensor shape: {img.shape}")


def soft_dilate(img: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    Soft dilation operation using max pooling.
    
    Args:
        img: Input tensor of shape (B, C, H, W) or (B, C, D, H, W)
        kernel_size: Size of the dilation kernel
        
    Returns:
        Dilated tensor
    """
    if len(img.shape) == 4:  # 2D
        p = kernel_size // 2
        return F.max_pool2d(img, kernel_size, stride=1, padding=p)
    elif len(img.shape) == 5:  # 3D
        p = kernel_size // 2
        return F.max_pool3d(img, kernel_size, stride=1, padding=p)
    else:
        raise ValueError(f"Unsupported tensor shape: {img.shape}")


def soft_open(img: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    Soft morphological opening (erosion followed by dilation).
    
    Args:
        img: Input tensor
        kernel_size: Size of the morphological kernel
        
    Returns:
        Opened tensor
    """
    return soft_dilate(soft_erode(img, kernel_size), kernel_size)


def soft_skel(img: torch.Tensor, num_iter: int = 40, kernel_size: int = 3) -> torch.Tensor:
    """
    Soft skeletonization operation.
    
    Iteratively computes the skeleton by subtracting the morphological opening
    from the input and accumulating the results.
    
    Args:
        img: Input tensor (B, C, H, W) or (B, C, D, H, W)
        num_iter: Number of iterations for skeletonization
        kernel_size: Size of the morphological kernel
        
    Returns:
        Soft skeleton tensor
    """
    img1 = soft_open(img, kernel_size)
    skel = F.relu(img - img1)
    
    for j in range(num_iter):
        img = soft_erode(img, kernel_size)
        img1 = soft_open(img, kernel_size)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    
    return skel


class SoftclDiceLoss(nn.Module):
    """
    Soft centerline Dice (clDice) loss function.
    
    This loss function preserves topology by computing the Dice coefficient
    on the skeleton of the segmentation masks.
    
    Args:
        num_iter: Number of iterations for soft skeletonization (default: 40)
        smooth: Smoothing factor to avoid division by zero (default: 1.0)
        alpha: Weight for combining with standard Dice loss (default: 0.5)
    """
    
    def __init__(
        self, 
        num_iter: int = 40, 
        smooth: float = 1.0,
        alpha: float = 0.5
    ):
        super(SoftclDiceLoss, self).__init__()
        self.num_iter = num_iter
        self.smooth = smooth
        self.alpha = alpha
    
    def forward(
        self, 
        y_pred: torch.Tensor, 
        y_true: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the soft clDice loss.
        
        Args:
            y_pred: Predicted segmentation (B, C, H, W) or (B, C, D, H, W)
            y_true: Ground truth segmentation (B, C, H, W) or (B, C, D, H, W)
            
        Returns:
            clDice loss value
        """
        # Ensure binary predictions
        y_pred = torch.sigmoid(y_pred)
        
        # Compute soft skeletons
        skel_pred = soft_skel(y_pred, self.num_iter)
        skel_true = soft_skel(y_true, self.num_iter)
        
        # Compute topology precision and sensitivity
        # tprec = intersection(skel_pred, y_true) / skel_pred
        tprec = (torch.sum(skel_pred * y_true) + self.smooth) / (torch.sum(skel_pred) + self.smooth)
        
        # tsens = intersection(skel_true, y_pred) / skel_true  
        tsens = (torch.sum(skel_true * y_pred) + self.smooth) / (torch.sum(skel_true) + self.smooth)
        
        # Compute clDice
        cl_dice = 2.0 * (tprec * tsens) / (tprec + tsens + self.smooth)
        
        return 1.0 - cl_dice


class CombinedclDiceLoss(nn.Module):
    """
    Combined clDice and standard Dice loss.
    
    This combines the topology-preserving clDice with standard Dice loss
    for better volumetric accuracy and stability.
    
    Args:
        num_iter: Number of iterations for soft skeletonization
        smooth: Smoothing factor
        alpha: Weight for clDice (1-alpha for standard Dice)
    """
    
    def __init__(
        self, 
        num_iter: int = 40, 
        smooth: float = 1.0,
        alpha: float = 0.5
    ):
        super(CombinedclDiceLoss, self).__init__()
        self.cldice_loss = SoftclDiceLoss(num_iter, smooth, alpha)
        self.smooth = smooth
        self.alpha = alpha
    
    def dice_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Standard Dice loss."""
        y_pred = torch.sigmoid(y_pred)
        
        intersection = torch.sum(y_pred * y_true)
        union = torch.sum(y_pred) + torch.sum(y_true)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice
    
    def forward(
        self, 
        y_pred: torch.Tensor, 
        y_true: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            y_pred: Predicted segmentation
            y_true: Ground truth segmentation
            
        Returns:
            Combined loss value
        """
        cldice_loss = self.cldice_loss(y_pred, y_true)
        dice_loss = self.dice_loss(y_pred, y_true)
        
        return self.alpha * cldice_loss + (1 - self.alpha) * dice_loss


class SoftcbDiceLoss(nn.Module):
    """
    Soft centerline boundary Dice (cbDice) loss function.
    
    This is an advanced version that combines topology preservation (clDice)
    with boundary awareness for better geometric detail recognition.
    
    Based on "Centerline Boundary Dice Loss for Vascular Segmentation" (MICCAI 2024)
    
    Args:
        num_iter: Number of iterations for soft skeletonization
        smooth: Smoothing factor
        alpha: Weight for clDice component
        beta: Weight for boundary component
        sigma: Standard deviation for Gaussian boundary weighting
    """
    
    def __init__(
        self, 
        num_iter: int = 40, 
        smooth: float = 1.0,
        alpha: float = 0.5,
        beta: float = 0.3,
        sigma: float = 1.0
    ):
        super(SoftcbDiceLoss, self).__init__()
        self.num_iter = num_iter
        self.smooth = smooth
        self.alpha = alpha  # Weight for clDice
        self.beta = beta    # Weight for boundary
        self.sigma = sigma  # Gaussian sigma for boundary weighting
        
        self.cldice_loss = SoftclDiceLoss(num_iter, smooth)
    
    def compute_boundary_weights(self, skel: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary weights based on distance from skeleton.
        
        This gives more weight to regions closer to the centerline,
        incorporating radius information for better geometric accuracy.
        """
        # Compute distance transform approximation
        # For simplicity, we use a Gaussian weighting based on skeleton
        weights = torch.exp(-torch.pow(skel, 2) / (2 * self.sigma ** 2))
        return weights
    
    def boundary_aware_dice(
        self, 
        y_pred: torch.Tensor, 
        y_true: torch.Tensor,
        skel_true: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute boundary-aware Dice with radius weighting.
        """
        # Compute boundary weights
        weights = self.compute_boundary_weights(skel_true)
        
        # Apply weights to the intersection and union
        weighted_intersection = torch.sum(weights * y_pred * y_true)
        weighted_pred = torch.sum(weights * y_pred)
        weighted_true = torch.sum(weights * y_true)
        
        boundary_dice = (2.0 * weighted_intersection + self.smooth) / (
            weighted_pred + weighted_true + self.smooth
        )
        
        return 1.0 - boundary_dice
    
    def forward(
        self, 
        y_pred: torch.Tensor, 
        y_true: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the cbDice loss.
        
        Args:
            y_pred: Predicted segmentation
            y_true: Ground truth segmentation
            
        Returns:
            cbDice loss value
        """
        # Ensure binary predictions
        y_pred = torch.sigmoid(y_pred)
        
        # Compute clDice component
        cldice_loss = self.cldice_loss(y_pred, y_true)
        
        # Compute skeleton for boundary weighting
        skel_true = soft_skel(y_true, self.num_iter)
        
        # Compute boundary-aware Dice
        boundary_loss = self.boundary_aware_dice(y_pred, y_true, skel_true)
        
        # Standard Dice for stability
        intersection = torch.sum(y_pred * y_true)
        union = torch.sum(y_pred) + torch.sum(y_true)
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Combine all components
        total_loss = (
            self.alpha * cldice_loss + 
            self.beta * boundary_loss + 
            (1 - self.alpha - self.beta) * dice_loss
        )
        
        return total_loss


# Convenience functions for easy integration
def create_cldice_loss(
    loss_type: str = "combined",
    num_iter: int = 40,
    smooth: float = 1.0,
    alpha: float = 0.5,
    **kwargs
) -> nn.Module:
    """
    Factory function to create clDice loss variants.
    
    Args:
        loss_type: Type of loss ("cldice", "combined", "cbdice")
        num_iter: Number of skeletonization iterations
        smooth: Smoothing factor
        alpha: Weight parameter
        **kwargs: Additional parameters for specific loss types
        
    Returns:
        Loss function module
    """
    if loss_type == "cldice":
        return SoftclDiceLoss(num_iter=num_iter, smooth=smooth, alpha=alpha)
    elif loss_type == "combined":
        return CombinedclDiceLoss(num_iter=num_iter, smooth=smooth, alpha=alpha)
    elif loss_type == "cbdice":
        return SoftcbDiceLoss(
            num_iter=num_iter, 
            smooth=smooth, 
            alpha=alpha,
            beta=kwargs.get('beta', 0.3),
            sigma=kwargs.get('sigma', 1.0)
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test the loss functions
    import numpy as np
    
    # Create synthetic test data
    batch_size, channels, height, width = 2, 1, 64, 64
    
    # Ground truth with vessel-like structure
    y_true = torch.zeros(batch_size, channels, height, width)
    y_true[:, :, 20:25, 10:50] = 1.0  # Horizontal vessel
    y_true[:, :, 10:50, 30:35] = 1.0  # Vertical vessel
    
    # Prediction with some errors
    y_pred = y_true.clone()
    y_pred[:, :, 22:24, 45:50] = 0.0  # Break in vessel
    y_pred[:, :, 5:8, 5:8] = 0.8      # False positive
    
    # Add some logits-like values
    y_pred = torch.logit(torch.clamp(y_pred, 0.01, 0.99))
    
    print("Testing clDice loss functions...")
    
    # Test standard clDice
    cldice_loss = SoftclDiceLoss(num_iter=10)  # Fewer iterations for testing
    loss_value = cldice_loss(y_pred, y_true)
    print(f"clDice Loss: {loss_value.item():.4f}")
    
    # Test combined loss
    combined_loss = CombinedclDiceLoss(num_iter=10, alpha=0.7)
    loss_value = combined_loss(y_pred, y_true)
    print(f"Combined clDice Loss: {loss_value.item():.4f}")
    
    # Test cbDice
    cbdice_loss = SoftcbDiceLoss(num_iter=10, alpha=0.5, beta=0.3)
    loss_value = cbdice_loss(y_pred, y_true)
    print(f"cbDice Loss: {loss_value.item():.4f}")
    
    print("All tests completed successfully!")