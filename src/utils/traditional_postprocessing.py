"""
Traditional post-processing methods for vascular segmentation.
These serve as baseline comparison methods for the graph-to-graph correction framework.
"""

import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import binary_opening, binary_closing, binary_dilation, binary_erosion
from skimage import measure
from skimage.morphology import disk, ball, remove_small_objects, remove_small_holes
from typing import Union, Tuple, Optional
import warnings


class MorphologicalPostProcessor:
    """
    Traditional morphological post-processing for vascular segmentation cleanup.
    
    This class implements standard binary morphological operations typically used
    to clean up segmentation masks by removing noise, filling gaps, and smoothing boundaries.
    """
    
    def __init__(
        self,
        opening_radius: int = 2,
        closing_radius: int = 3,
        min_object_size: int = 100,
        min_hole_size: int = 50,
        use_3d_structuring_elements: bool = True
    ):
        """
        Initialize morphological post-processor.
        
        Args:
            opening_radius: Radius for morphological opening (removes small objects/spurs)
            closing_radius: Radius for morphological closing (fills gaps and holes)
            min_object_size: Minimum size of objects to keep (in voxels)
            min_hole_size: Minimum size of holes to fill (in voxels)
            use_3d_structuring_elements: Whether to use 3D structuring elements for 3D data
        """
        self.opening_radius = opening_radius
        self.closing_radius = closing_radius
        self.min_object_size = min_object_size
        self.min_hole_size = min_hole_size
        self.use_3d_structuring_elements = use_3d_structuring_elements
    
    def process(
        self, 
        mask: Union[np.ndarray, torch.Tensor],
        return_torch: bool = False
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply morphological post-processing to a binary mask.
        
        Args:
            mask: Binary mask to process (numpy array or torch tensor)
            return_torch: Whether to return result as torch tensor
            
        Returns:
            Processed binary mask
        """
        # Convert to numpy if needed
        if isinstance(mask, torch.Tensor):
            input_was_torch = True
            original_device = mask.device
            mask_np = mask.detach().cpu().numpy()
        else:
            input_was_torch = False
            mask_np = mask.copy()
        
        # Ensure binary
        mask_np = (mask_np > 0.5).astype(np.uint8)
        
        # Apply morphological operations
        processed_mask = self._apply_morphological_sequence(mask_np)
        
        # Convert back to original format if needed
        if return_torch or input_was_torch:
            processed_mask = torch.from_numpy(processed_mask.astype(np.float32))
            if input_was_torch:
                processed_mask = processed_mask.to(original_device)
        
        return processed_mask
    
    def _apply_morphological_sequence(self, mask: np.ndarray) -> np.ndarray:
        """Apply the standard morphological processing sequence."""
        
        # Step 1: Morphological opening (removes small objects and spurs)
        if self.opening_radius > 0:
            structuring_element = self._get_structuring_element(
                self.opening_radius, mask.ndim
            )
            mask = binary_opening(mask, structure=structuring_element)
        
        # Step 2: Remove small objects
        if self.min_object_size > 0:
            mask = remove_small_objects(
                mask.astype(bool), min_size=self.min_object_size
            ).astype(np.uint8)
        
        # Step 3: Morphological closing (fills gaps and holes)
        if self.closing_radius > 0:
            structuring_element = self._get_structuring_element(
                self.closing_radius, mask.ndim
            )
            mask = binary_closing(mask, structure=structuring_element)
        
        # Step 4: Fill small holes
        if self.min_hole_size > 0:
            mask = remove_small_holes(
                mask.astype(bool), area_threshold=self.min_hole_size
            ).astype(np.uint8)
        
        return mask
    
    def _get_structuring_element(self, radius: int, ndim: int):
        """Get appropriate structuring element for morphological operations."""
        if ndim == 2:
            return disk(radius)
        elif ndim == 3 and self.use_3d_structuring_elements:
            return ball(radius)
        else:
            # Fallback to cross-shaped structuring element for 3D
            se = np.zeros((2*radius+1,) * ndim, dtype=bool)
            center = radius
            for i in range(ndim):
                idx = [center] * ndim
                for j in range(-radius, radius+1):
                    idx[i] = center + j
                    if all(0 <= idx[k] < se.shape[k] for k in range(ndim)):
                        se[tuple(idx)] = True
            return se


class LargestConnectedComponentFilter:
    """
    Largest Connected Component (LCC) filtering for vascular segmentation.
    
    This is a simple but common post-processing method that keeps only the largest
    connected component and discards everything else as "noise". While naive,
    it's widely used in practice and serves as an important baseline.
    """
    
    def __init__(self, connectivity: Optional[int] = None, min_size_ratio: float = 0.01):
        """
        Initialize LCC filter.
        
        Args:
            connectivity: Connectivity for connected components (None for max connectivity)
            min_size_ratio: Minimum size ratio relative to largest component to keep
                          (allows keeping multiple large components if desired)
        """
        self.connectivity = connectivity
        self.min_size_ratio = min_size_ratio
    
    def process(
        self, 
        mask: Union[np.ndarray, torch.Tensor],
        keep_n_largest: int = 1,
        return_torch: bool = False
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Filter mask to keep only largest connected component(s).
        
        Args:
            mask: Binary mask to process
            keep_n_largest: Number of largest components to keep (default: 1)
            return_torch: Whether to return result as torch tensor
            
        Returns:
            Filtered binary mask
        """
        # Convert to numpy if needed
        if isinstance(mask, torch.Tensor):
            input_was_torch = True
            original_device = mask.device
            mask_np = mask.detach().cpu().numpy()
        else:
            input_was_torch = False
            mask_np = mask.copy()
        
        # Ensure binary
        mask_np = (mask_np > 0.5).astype(np.uint8)
        
        # Find connected components
        if mask_np.ndim == 2:
            connectivity = self.connectivity or 2  # 8-connectivity for 2D
        else:
            connectivity = self.connectivity or 3  # 26-connectivity for 3D
        
        labeled_mask = measure.label(mask_np, connectivity=connectivity)
        
        if labeled_mask.max() == 0:  # No components found
            filtered_mask = mask_np
        else:
            # Get component sizes
            component_sizes = np.bincount(labeled_mask.ravel())
            component_sizes[0] = 0  # Ignore background
            
            # Find largest component(s)
            largest_indices = np.argsort(component_sizes)[::-1][:keep_n_largest]
            
            # Apply size threshold if specified
            if self.min_size_ratio > 0:
                min_size = component_sizes[largest_indices[0]] * self.min_size_ratio
                largest_indices = [
                    idx for idx in largest_indices 
                    if component_sizes[idx] >= min_size
                ]
            
            # Create filtered mask
            filtered_mask = np.zeros_like(mask_np)
            for idx in largest_indices:
                if idx > 0:  # Skip background
                    filtered_mask[labeled_mask == idx] = 1
        
        # Convert back to original format if needed
        if return_torch or input_was_torch:
            filtered_mask = torch.from_numpy(filtered_mask.astype(np.float32))
            if input_was_torch:
                filtered_mask = filtered_mask.to(original_device)
        
        return filtered_mask
    
    def get_component_stats(self, mask: Union[np.ndarray, torch.Tensor]) -> dict:
        """Get statistics about connected components in the mask."""
        # Convert to numpy if needed
        if isinstance(mask, torch.Tensor):
            mask_np = mask.detach().cpu().numpy()
        else:
            mask_np = mask.copy()
        
        # Ensure binary
        mask_np = (mask_np > 0.5).astype(np.uint8)
        
        # Find connected components
        if mask_np.ndim == 2:
            connectivity = self.connectivity or 2
        else:
            connectivity = self.connectivity or 3
        
        labeled_mask = measure.label(mask_np, connectivity=connectivity)
        
        if labeled_mask.max() == 0:
            return {
                'num_components': 0,
                'largest_size': 0,
                'total_size': int(mask_np.sum()),
                'size_distribution': []
            }
        
        # Get component sizes
        component_sizes = np.bincount(labeled_mask.ravel())
        component_sizes = component_sizes[1:]  # Remove background
        
        return {
            'num_components': len(component_sizes),
            'largest_size': int(component_sizes.max()),
            'total_size': int(mask_np.sum()),
            'size_distribution': sorted(component_sizes.tolist(), reverse=True),
            'size_ratios': (component_sizes / component_sizes.max()).tolist()
        }


class CombinedTraditionalPostProcessor:
    """
    Combined traditional post-processing pipeline.
    
    Applies both morphological operations and connected component filtering
    in a sensible order to maximize cleanup effectiveness.
    """
    
    def __init__(
        self,
        morphological_params: Optional[dict] = None,
        lcc_params: Optional[dict] = None,
        apply_order: str = 'morph_then_lcc'  # or 'lcc_then_morph' or 'morph_only' or 'lcc_only'
    ):
        """
        Initialize combined post-processor.
        
        Args:
            morphological_params: Parameters for morphological processor
            lcc_params: Parameters for LCC filter
            apply_order: Order of operations ('morph_then_lcc', 'lcc_then_morph', etc.)
        """
        self.apply_order = apply_order
        
        if morphological_params is None:
            morphological_params = {}
        if lcc_params is None:
            lcc_params = {}
        
        self.morph_processor = MorphologicalPostProcessor(**morphological_params)
        self.lcc_filter = LargestConnectedComponentFilter(**lcc_params)
    
    def process(
        self, 
        mask: Union[np.ndarray, torch.Tensor],
        return_torch: bool = False
    ) -> Union[np.ndarray, torch.Tensor]:
        """Apply combined post-processing pipeline."""
        
        if self.apply_order == 'morph_only':
            return self.morph_processor.process(mask, return_torch=return_torch)
        
        elif self.apply_order == 'lcc_only':
            return self.lcc_filter.process(mask, return_torch=return_torch)
        
        elif self.apply_order == 'morph_then_lcc':
            # First morphological cleanup, then LCC filtering
            intermediate = self.morph_processor.process(mask, return_torch=False)
            return self.lcc_filter.process(intermediate, return_torch=return_torch)
        
        elif self.apply_order == 'lcc_then_morph':
            # First LCC filtering, then morphological cleanup
            intermediate = self.lcc_filter.process(mask, return_torch=False)
            return self.morph_processor.process(intermediate, return_torch=return_torch)
        
        else:
            raise ValueError(f"Unknown apply_order: {self.apply_order}")


# Convenience functions for easy usage
def apply_morphological_postprocessing(
    mask: Union[np.ndarray, torch.Tensor],
    **kwargs
) -> Union[np.ndarray, torch.Tensor]:
    """Convenience function for morphological post-processing."""
    processor = MorphologicalPostProcessor(**kwargs)
    return processor.process(mask)


def apply_lcc_filtering(
    mask: Union[np.ndarray, torch.Tensor],
    **kwargs
) -> Union[np.ndarray, torch.Tensor]:
    """Convenience function for LCC filtering."""
    filter_obj = LargestConnectedComponentFilter(**kwargs)
    return filter_obj.process(mask)


def apply_combined_traditional_postprocessing(
    mask: Union[np.ndarray, torch.Tensor],
    **kwargs
) -> Union[np.ndarray, torch.Tensor]:
    """Convenience function for combined traditional post-processing."""
    processor = CombinedTraditionalPostProcessor(**kwargs)
    return processor.process(mask)