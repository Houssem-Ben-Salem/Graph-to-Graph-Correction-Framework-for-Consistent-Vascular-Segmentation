"""
Mask preprocessing for graph extraction
Implements morphological cleaning and quality assessment for binary segmentation masks
"""

import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage import gaussian_filter, binary_fill_holes, label, binary_erosion, binary_dilation
from skimage.morphology import remove_small_objects, ball
from typing import Tuple, Dict, Optional
import logging


class MaskPreprocessor:
    """
    Preprocessor for binary segmentation masks before graph extraction
    """
    
    def __init__(self, 
                 min_component_size: int = 27,  # 3x3x3 voxels
                 gaussian_sigma: float = 0.5,
                 fill_holes: bool = True):
        """
        Initialize mask preprocessor
        
        Args:
            min_component_size: Minimum volume for connected components (voxels)
            gaussian_sigma: Standard deviation for Gaussian smoothing (for predicted masks)
            fill_holes: Whether to fill small holes in vessels
        """
        self.min_component_size = min_component_size
        self.gaussian_sigma = gaussian_sigma
        self.fill_holes = fill_holes
        self.logger = logging.getLogger(__name__)
    
    def preprocess_mask(self, 
                       mask: np.ndarray, 
                       is_prediction: bool = True,
                       voxel_spacing: Optional[Tuple[float, float, float]] = None) -> Tuple[np.ndarray, Dict]:
        """
        Preprocess binary segmentation mask for graph extraction
        
        Args:
            mask: Binary segmentation mask (0s and 1s)
            is_prediction: Whether this is a predicted mask (vs ground truth)
            voxel_spacing: Voxel spacing in mm (z, y, x)
            
        Returns:
            Tuple of (cleaned_mask, preprocessing_info)
        """
        if mask.dtype != bool and mask.dtype != np.uint8:
            raise ValueError(f"Mask must be binary (bool or uint8), got {mask.dtype}")
        
        # Convert to boolean for processing
        mask_bool = mask.astype(bool)
        original_volume = np.sum(mask_bool)
        
        preprocessing_info = {
            'original_volume': original_volume,
            'is_prediction': is_prediction,
            'voxel_spacing': voxel_spacing,
            'steps_applied': []
        }
        
        self.logger.info(f"Preprocessing {'predicted' if is_prediction else 'ground truth'} mask "
                        f"with volume {original_volume} voxels")
        
        # Step 1: Remove small disconnected components
        mask_cleaned = self._remove_small_components(mask_bool, preprocessing_info)
        
        # Step 2: Fill small holes
        if self.fill_holes:
            mask_cleaned = self._fill_small_holes(mask_cleaned, preprocessing_info)
        
        # Step 3: Apply Gaussian smoothing (only for predicted masks)
        if is_prediction and self.gaussian_sigma > 0:
            mask_cleaned = self._apply_gaussian_smoothing(mask_cleaned, preprocessing_info)
        
        # Step 4: Final validation and cleanup
        mask_cleaned = self._final_cleanup(mask_cleaned, preprocessing_info)
        
        final_volume = np.sum(mask_cleaned)
        preprocessing_info['final_volume'] = final_volume
        preprocessing_info['volume_change'] = final_volume - original_volume
        preprocessing_info['volume_retention'] = final_volume / original_volume if original_volume > 0 else 0
        
        self.logger.info(f"Preprocessing complete. Volume retention: "
                        f"{preprocessing_info['volume_retention']:.3f}")
        
        return mask_cleaned.astype(np.uint8), preprocessing_info
    
    def _remove_small_components(self, mask: np.ndarray, info: Dict) -> np.ndarray:
        """Remove disconnected components smaller than threshold"""
        # Label connected components
        labeled_mask, num_features = label(mask)
        
        if num_features == 0:
            info['steps_applied'].append('remove_small_components: no components found')
            return mask
        
        # Calculate component sizes
        component_sizes = np.bincount(labeled_mask.flat)[1:]  # Skip background (0)
        
        # Keep only components larger than threshold
        large_components = np.where(component_sizes >= self.min_component_size)[0] + 1
        
        if len(large_components) == 0:
            self.logger.warning("No components larger than threshold found")
            info['steps_applied'].append('remove_small_components: all components removed')
            return np.zeros_like(mask, dtype=bool)
        
        # Create mask with only large components
        mask_filtered = np.isin(labeled_mask, large_components)
        
        removed_volume = np.sum(mask) - np.sum(mask_filtered)
        info['steps_applied'].append(f'remove_small_components: removed {removed_volume} voxels '
                                   f'from {num_features - len(large_components)} components')
        
        return mask_filtered
    
    def _fill_small_holes(self, mask: np.ndarray, info: Dict) -> np.ndarray:
        """Fill small holes within vessels"""
        original_volume = np.sum(mask)
        
        # Fill holes using binary_fill_holes
        mask_filled = binary_fill_holes(mask)
        
        filled_volume = np.sum(mask_filled) - original_volume
        info['steps_applied'].append(f'fill_holes: filled {filled_volume} voxels')
        
        return mask_filled
    
    def _apply_gaussian_smoothing(self, mask: np.ndarray, info: Dict) -> np.ndarray:
        """Apply light Gaussian smoothing to reduce noise in predicted masks"""
        # Convert to float for smoothing
        mask_float = mask.astype(np.float32)
        
        # Apply Gaussian filter
        mask_smoothed = gaussian_filter(mask_float, sigma=self.gaussian_sigma)
        
        # Threshold back to binary (use 0.5 threshold)
        mask_binary = mask_smoothed > 0.5
        
        volume_change = np.sum(mask_binary) - np.sum(mask)
        info['steps_applied'].append(f'gaussian_smoothing: sigma={self.gaussian_sigma}, '
                                   f'volume_change={volume_change}')
        
        return mask_binary
    
    def _final_cleanup(self, mask: np.ndarray, info: Dict) -> np.ndarray:
        """Final cleanup and validation"""
        # Ensure mask is still binary
        mask_clean = mask.astype(bool)
        
        # Remove any remaining isolated voxels (morphological opening)
        structure = ball(1)  # 3x3x3 structuring element
        mask_clean = binary_erosion(mask_clean, structure)
        mask_clean = binary_dilation(mask_clean, structure)
        
        cleanup_volume_change = np.sum(mask_clean) - np.sum(mask)
        info['steps_applied'].append(f'final_cleanup: volume_change={cleanup_volume_change}')
        
        return mask_clean
    
    def generate_confidence_map(self, 
                               predicted_mask: np.ndarray,
                               prediction_probabilities: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate confidence map for predicted masks based on local consistency
        
        Args:
            predicted_mask: Binary predicted mask
            prediction_probabilities: Raw prediction probabilities (optional)
            
        Returns:
            Confidence map with values between 0 and 1
        """
        if prediction_probabilities is not None:
            # Use prediction probabilities as base confidence
            confidence = prediction_probabilities.copy()
        else:
            # Generate confidence based on local consistency
            confidence = np.ones_like(predicted_mask, dtype=np.float32)
        
        # Reduce confidence near boundaries and isolated regions
        mask_bool = predicted_mask.astype(bool)
        
        # Distance from edges (internal distance)
        distance_internal = ndi.distance_transform_edt(mask_bool)
        
        # Distance from outside (external distance)  
        distance_external = ndi.distance_transform_edt(~mask_bool)
        
        # Combine distances for boundary-aware confidence
        boundary_confidence = np.minimum(distance_internal, distance_external)
        boundary_confidence = np.minimum(boundary_confidence / 3.0, 1.0)  # Normalize
        
        # Combine with original confidence
        if prediction_probabilities is not None:
            final_confidence = confidence * boundary_confidence
        else:
            final_confidence = boundary_confidence
        
        return final_confidence