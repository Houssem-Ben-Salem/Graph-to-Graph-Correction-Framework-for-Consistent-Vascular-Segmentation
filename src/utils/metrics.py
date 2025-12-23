"""Evaluation metrics for medical image segmentation"""

import torch
import numpy as np
from scipy import ndimage
from sklearn.metrics import precision_score, recall_score
import SimpleITK as sitk


def compute_dice_score(pred, target, smooth=1e-6):
    """Compute Dice coefficient"""
    pred = pred.flatten()
    target = target.flatten()
    
    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice


def compute_jaccard_score(pred, target, smooth=1e-6):
    """Compute Jaccard index (IoU)"""
    pred = pred.flatten()
    target = target.flatten()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    jaccard = (intersection + smooth) / (union + smooth)
    return jaccard


def compute_hausdorff_distance(pred, target, spacing=None):
    """Compute Hausdorff distance between segmentations"""
    try:
        if spacing is None:
            spacing = [1.0, 1.0, 1.0]
        
        # Convert to SimpleITK images
        pred_img = sitk.GetImageFromArray(pred.astype(np.uint8))
        target_img = sitk.GetImageFromArray(target.astype(np.uint8))
        
        pred_img.SetSpacing(spacing)
        target_img.SetSpacing(spacing)
        
        # Compute Hausdorff distance
        hausdorff_filter = sitk.HausdorffDistanceImageFilter()
        hausdorff_filter.Execute(pred_img, target_img)
        
        return hausdorff_filter.GetHausdorffDistance()
    
    except Exception:
        # Fallback to simple edge-based calculation if SimpleITK fails
        return _compute_hausdorff_fallback(pred, target, spacing)


def _compute_hausdorff_fallback(pred, target, spacing):
    """Fallback Hausdorff distance calculation"""
    # Extract surface points
    pred_surface = _extract_surface_points(pred)
    target_surface = _extract_surface_points(target)
    
    if len(pred_surface) == 0 or len(target_surface) == 0:
        return float('inf')
    
    # Apply spacing
    if spacing is not None:
        pred_surface = pred_surface * np.array(spacing)
        target_surface = target_surface * np.array(spacing)
    
    # Compute distances
    from scipy.spatial.distance import cdist
    
    # Distance from pred to target
    dist_pred_to_target = cdist(pred_surface, target_surface)
    max_dist_pred = np.min(dist_pred_to_target, axis=1).max()
    
    # Distance from target to pred
    dist_target_to_pred = cdist(target_surface, pred_surface)
    max_dist_target = np.min(dist_target_to_pred, axis=1).max()
    
    return max(max_dist_pred, max_dist_target)


def _extract_surface_points(mask):
    """Extract surface points from binary mask"""
    # Use morphological operations to find boundary
    struct = ndimage.generate_binary_structure(mask.ndim, 1)
    eroded = ndimage.binary_erosion(mask, struct)
    boundary = mask & ~eroded
    
    return np.column_stack(np.where(boundary))


def compute_sensitivity(pred, target):
    """Compute sensitivity (recall/true positive rate)"""
    pred = pred.flatten()
    target = target.flatten()
    
    if target.sum() == 0:
        return 1.0 if pred.sum() == 0 else 0.0
    
    return recall_score(target, pred, zero_division=0)


def compute_specificity(pred, target):
    """Compute specificity (true negative rate)"""
    pred = pred.flatten()
    target = target.flatten()
    
    # Compute for negative class
    return recall_score(1 - target, 1 - pred, zero_division=0)


def compute_precision(pred, target):
    """Compute precision (positive predictive value)"""
    pred = pred.flatten()
    target = target.flatten()
    
    if pred.sum() == 0:
        return 1.0 if target.sum() == 0 else 0.0
    
    return precision_score(target, pred, zero_division=0)


def compute_volume_similarity(pred, target, spacing=None):
    """Compute volume similarity coefficient"""
    if spacing is not None:
        voxel_volume = np.prod(spacing)
        pred_volume = float(pred.sum()) * voxel_volume
        target_volume = float(target.sum()) * voxel_volume
    else:
        pred_volume = float(pred.sum())
        target_volume = float(target.sum())
    
    if pred_volume + target_volume == 0:
        return 1.0
    
    # Use numpy to handle potential overflow more gracefully
    volume_diff = np.abs(pred_volume - target_volume)
    volume_sum = pred_volume + target_volume
    
    # Check for potential overflow/invalid values
    if not np.isfinite(volume_diff) or not np.isfinite(volume_sum) or volume_sum == 0:
        return 0.0
    
    return 1.0 - volume_diff / volume_sum


def compute_average_symmetric_surface_distance(pred, target, spacing=None):
    """Compute Average Symmetric Surface Distance (ASSD)"""
    try:
        if spacing is None:
            spacing = [1.0, 1.0, 1.0]
        
        # Convert to SimpleITK images
        pred_img = sitk.GetImageFromArray(pred.astype(np.uint8))
        target_img = sitk.GetImageFromArray(target.astype(np.uint8))
        
        pred_img.SetSpacing(spacing)
        target_img.SetSpacing(spacing)
        
        # Compute ASSD
        hausdorff_filter = sitk.HausdorffDistanceImageFilter()
        hausdorff_filter.Execute(pred_img, target_img)
        
        return hausdorff_filter.GetAverageHausdorffDistance()
    
    except Exception:
        return _compute_assd_fallback(pred, target, spacing)


def _compute_assd_fallback(pred, target, spacing):
    """Fallback ASSD calculation"""
    pred_surface = _extract_surface_points(pred)
    target_surface = _extract_surface_points(target)
    
    if len(pred_surface) == 0 or len(target_surface) == 0:
        return float('inf')
    
    if spacing is not None:
        pred_surface = pred_surface * np.array(spacing)
        target_surface = target_surface * np.array(spacing)
    
    from scipy.spatial.distance import cdist
    
    # Distance from pred to target
    dist_pred_to_target = cdist(pred_surface, target_surface)
    min_dist_pred = np.min(dist_pred_to_target, axis=1)
    
    # Distance from target to pred
    dist_target_to_pred = cdist(target_surface, pred_surface)
    min_dist_target = np.min(dist_target_to_pred, axis=1)
    
    # Average symmetric surface distance
    assd = (min_dist_pred.mean() + min_dist_target.mean()) / 2.0
    return assd


def compute_metrics(predictions, targets, spacing=None, include_surface_metrics=True):
    """
    Compute comprehensive evaluation metrics
    
    Args:
        predictions: Binary predictions (torch.Tensor or numpy.ndarray)
        targets: Ground truth masks (torch.Tensor or numpy.ndarray)
        spacing: Voxel spacing for surface metrics
        include_surface_metrics: Whether to compute surface-based metrics
    
    Returns:
        Dictionary of computed metrics
    """
    # Convert to numpy if torch tensors
    if torch.is_tensor(predictions):
        predictions = predictions.detach().cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()
    
    # Ensure binary
    predictions = (predictions > 0.5).astype(np.uint8)
    targets = (targets > 0.5).astype(np.uint8)
    
    metrics = {}
    
    # Volume-based metrics
    metrics['dice'] = float(compute_dice_score(predictions, targets))
    metrics['jaccard'] = float(compute_jaccard_score(predictions, targets))
    metrics['sensitivity'] = float(compute_sensitivity(predictions, targets))
    metrics['specificity'] = float(compute_specificity(predictions, targets))
    metrics['precision'] = float(compute_precision(predictions, targets))
    metrics['volume_similarity'] = float(compute_volume_similarity(predictions, targets, spacing))
    
    # F1 score (harmonic mean of precision and recall)
    if metrics['precision'] + metrics['sensitivity'] > 0:
        metrics['f1_score'] = 2 * (metrics['precision'] * metrics['sensitivity']) / \
                             (metrics['precision'] + metrics['sensitivity'])
    else:
        metrics['f1_score'] = 0.0
    
    # Surface-based metrics (more computationally expensive)
    if include_surface_metrics:
        try:
            metrics['hausdorff_distance'] = float(compute_hausdorff_distance(predictions, targets, spacing))
            metrics['avg_surface_distance'] = float(compute_average_symmetric_surface_distance(predictions, targets, spacing))
        except Exception as e:
            metrics['hausdorff_distance'] = float('inf')
            metrics['avg_surface_distance'] = float('inf')
    
    return metrics


def compute_batch_metrics(predictions, targets, spacing=None):
    """Compute metrics for a batch of predictions"""
    batch_metrics = []
    
    for i in range(len(predictions)):
        pred = predictions[i]
        target = targets[i]
        vol_spacing = spacing[i] if spacing is not None else None
        
        metrics = compute_metrics(pred, target, vol_spacing, include_surface_metrics=False)
        batch_metrics.append(metrics)
    
    # Compute average metrics
    avg_metrics = {}
    for key in batch_metrics[0].keys():
        values = [m[key] for m in batch_metrics if not np.isnan(m[key]) and not np.isinf(m[key])]
        avg_metrics[key] = np.mean(values) if values else 0.0
        avg_metrics[f'{key}_std'] = np.std(values) if len(values) > 1 else 0.0
    
    return avg_metrics, batch_metrics