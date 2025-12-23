"""
Template-Based Volume Reconstructor
Main reconstruction pipeline that converts corrected graphs to volumetric masks
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
import time
from pathlib import Path

from src.models.graph_extraction.vascular_graph import VascularGraph
from src.models.reconstruction.vessel_templates import VesselTemplate, TemplateFactory
from src.models.reconstruction.template_placement import TemplatePlacer, PlacementConfig
from src.models.reconstruction.sdf_renderer import SDFRenderer, RenderingConfig, VoxelGrid


@dataclass
class ReconstructionConfig:
    """Configuration for volume reconstruction"""
    # Volume properties
    target_resolution: Tuple[int, int, int] = (256, 256, 256)  # Target volume size
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # Voxel dimensions
    auto_compute_bounds: bool = True  # Automatically compute volume bounds
    volume_padding: float = 10.0  # Padding around vessel structures
    
    # Template placement
    placement_config: PlacementConfig = None
    
    # Rendering
    rendering_config: RenderingConfig = None
    
    # Integration with original predictions
    blend_with_original: bool = True  # Blend with original U-Net prediction
    original_blend_weight: float = 0.3  # Weight for original prediction
    confidence_weighting: bool = True  # Use confidence for blending
    
    # Post-processing
    apply_morphological_closing: bool = True  # Fill small gaps
    closing_kernel_size: int = 3  # Morphological closing kernel size
    apply_smoothing: bool = True  # Apply final smoothing
    smoothing_sigma: float = 0.5  # Gaussian smoothing sigma
    
    # Quality control
    min_component_size: int = 100  # Minimum connected component size
    connectivity_validation: bool = True  # Validate vessel connectivity
    
    def __post_init__(self):
        if self.placement_config is None:
            self.placement_config = PlacementConfig()
        
        if self.rendering_config is None:
            self.rendering_config = RenderingConfig(
                resolution=self.target_resolution,
                voxel_size=self.voxel_spacing,
                parallel_processing=True,
                anti_aliasing=True,
                template_blending=True
            )


@dataclass
class ReconstructionResult:
    """Result of volume reconstruction"""
    reconstructed_mask: Optional[np.ndarray] = None  # Final reconstructed volume
    sdf_volume: Optional[np.ndarray] = None  # SDF representation (if requested)
    probability_mask: Optional[np.ndarray] = None  # Probability representation
    
    # Quality metrics
    reconstruction_quality: float = 0.0  # Overall quality score
    volume_ratio: float = 0.0  # Volume compared to original
    connectivity_score: float = 0.0  # Connectivity preservation
    
    # Process information
    templates_used: int = 0  # Number of templates used
    rendering_time: float = 0.0  # Time spent rendering
    total_time: float = 0.0  # Total reconstruction time
    
    # Intermediate results (for debugging)
    placement_result: Optional[object] = None
    template_volumes: Optional[Dict] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class VolumeReconstructor:
    """Template-based volume reconstruction system"""
    
    def __init__(self, config: ReconstructionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.template_placer = TemplatePlacer(config.placement_config)
        self.sdf_renderer = None  # Initialized based on graph bounds
        
        # State
        self.current_graph = None
        self.current_bounds = None
    
    def reconstruct_from_graph(self, 
                             corrected_graph: VascularGraph,
                             original_prediction: Optional[np.ndarray] = None,
                             reference_spacing: Optional[Tuple[float, float, float]] = None,
                             return_intermediates: bool = False) -> ReconstructionResult:
        """
        Reconstruct volume from corrected graph
        
        Args:
            corrected_graph: Graph with corrected topology and geometry
            original_prediction: Original U-Net prediction for blending
            reference_spacing: Reference voxel spacing from original volume
            return_intermediates: Whether to return intermediate results
        """
        start_time = time.time()
        self.logger.info(f"Starting volume reconstruction for graph with {len(corrected_graph.nodes)} nodes")
        
        # Initialize reconstruction
        self.current_graph = corrected_graph
        result = ReconstructionResult()
        
        try:
            # Step 1: Compute volume bounds and initialize renderer
            self._initialize_renderer_for_graph(corrected_graph, reference_spacing)
            
            # Step 2: Place templates
            self.logger.info("Placing vessel templates...")
            placement_start = time.time()
            
            placement_result = self.template_placer.place_templates(corrected_graph, reference_spacing)
            result.templates_used = placement_result.total_templates
            
            if return_intermediates:
                result.placement_result = placement_result
            
            placement_time = time.time() - placement_start
            self.logger.info(f"Template placement completed in {placement_time:.2f}s")
            
            # Step 3: Render templates to volume
            self.logger.info("Rendering templates to volume...")
            rendering_start = time.time()
            
            all_templates = []
            for template_list in placement_result.templates.values():
                all_templates.extend(template_list)
            
            if not all_templates:
                self.logger.warning("No templates to render!")
                result.reconstructed_mask = np.zeros(self.config.target_resolution, dtype=np.uint8)
                result.warnings.append("No templates generated")
                return result
            
            # Render SDF volume
            sdf_volume = self.sdf_renderer.render_templates(all_templates)
            
            # Convert to binary mask
            binary_mask = self.sdf_renderer.sdf_to_binary_mask(sdf_volume)
            
            # Store intermediate results if requested
            if return_intermediates:
                result.sdf_volume = sdf_volume.data
                result.probability_mask = self.sdf_renderer.sdf_to_probability_mask(sdf_volume)
                result.template_volumes = self._compute_template_volumes(all_templates)
            
            result.rendering_time = time.time() - rendering_start
            self.logger.info(f"Rendering completed in {result.rendering_time:.2f}s")
            
            # Step 4: Blend with original prediction if available
            if self.config.blend_with_original and original_prediction is not None:
                self.logger.info("Blending with original prediction...")
                binary_mask = self._blend_with_original(binary_mask, original_prediction, placement_result)
            
            # Step 5: Post-processing
            self.logger.info("Applying post-processing...")
            binary_mask = self._apply_post_processing(binary_mask)
            
            # Step 6: Quality assessment
            result.reconstruction_quality = self._assess_reconstruction_quality(
                binary_mask, corrected_graph, original_prediction
            )
            
            result.volume_ratio = self._compute_volume_ratio(binary_mask, original_prediction)
            result.connectivity_score = self._assess_connectivity(binary_mask, corrected_graph)
            
            # Final result
            result.reconstructed_mask = binary_mask
            result.total_time = time.time() - start_time
            result.warnings.extend(placement_result.warnings)
            
            self.logger.info(f"Reconstruction completed in {result.total_time:.2f}s")
            self.logger.info(f"Quality: {result.reconstruction_quality:.3f}, "
                           f"Volume ratio: {result.volume_ratio:.3f}, "
                           f"Connectivity: {result.connectivity_score:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Reconstruction failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Return empty result with error
            result.reconstructed_mask = np.zeros(self.config.target_resolution, dtype=np.uint8)
            result.warnings.append(f"Reconstruction failed: {str(e)}")
            result.total_time = time.time() - start_time
            
            return result
    
    def _initialize_renderer_for_graph(self, graph: VascularGraph, 
                                     reference_spacing: Optional[Tuple[float, float, float]]):
        """Initialize SDF renderer based on graph bounds"""
        # Compute graph bounds
        node_positions = np.array([node['position'][:3] for node in graph.nodes])
        
        if len(node_positions) == 0:
            raise ValueError("Graph has no nodes!")
        
        min_coords = np.min(node_positions, axis=0)
        max_coords = np.max(node_positions, axis=0)
        
        # Add padding - convert to float to avoid dtype casting issues
        padding = self.config.volume_padding
        min_coords = min_coords.astype(float)
        max_coords = max_coords.astype(float)
        min_coords -= padding
        max_coords += padding
        
        # Compute volume properties
        volume_size = max_coords - min_coords
        
        if self.config.auto_compute_bounds:
            # Adjust resolution to maintain aspect ratio
            aspect_ratios = volume_size / np.min(volume_size)
            base_resolution = min(self.config.target_resolution)
            
            adjusted_resolution = tuple(int(base_resolution * ratio) for ratio in aspect_ratios)
            
            # Clamp to reasonable limits
            adjusted_resolution = tuple(
                max(32, min(512, res)) for res in adjusted_resolution
            )
            
            voxel_size = volume_size / np.array(adjusted_resolution)
        else:
            adjusted_resolution = self.config.target_resolution
            voxel_size = np.array(self.config.voxel_spacing)
            
            if reference_spacing is not None:
                voxel_size = np.array(reference_spacing)
        
        # Update rendering config
        rendering_config = RenderingConfig(
            resolution=adjusted_resolution,
            voxel_size=tuple(voxel_size),
            origin=tuple(min_coords),
            parallel_processing=self.config.rendering_config.parallel_processing,
            anti_aliasing=self.config.rendering_config.anti_aliasing,
            template_blending=self.config.rendering_config.template_blending
        )
        
        # Initialize renderer
        self.sdf_renderer = SDFRenderer(rendering_config)
        
        self.logger.info(f"Renderer initialized:")
        self.logger.info(f"  Resolution: {adjusted_resolution}")
        self.logger.info(f"  Voxel size: {voxel_size}")
        self.logger.info(f"  Origin: {min_coords}")
        self.logger.info(f"  Volume size: {volume_size}")
    
    def _blend_with_original(self, reconstructed_mask: np.ndarray, 
                           original_prediction: np.ndarray,
                           placement_result) -> np.ndarray:
        """Blend reconstructed mask with original prediction"""
        
        # Resize original prediction to match reconstructed mask if necessary
        if original_prediction.shape != reconstructed_mask.shape:
            from scipy import ndimage
            zoom_factors = np.array(reconstructed_mask.shape) / np.array(original_prediction.shape)
            original_prediction = ndimage.zoom(original_prediction, zoom_factors, order=1)
            original_prediction = (original_prediction > 0.5).astype(np.uint8)
        
        # Simple weighted blending
        if self.config.confidence_weighting:
            # Use placement quality as confidence
            confidence = max(placement_result.placement_quality, 0.1)
            blend_weight = confidence * (1 - self.config.original_blend_weight)
        else:
            blend_weight = 1 - self.config.original_blend_weight
        
        # Blend: high confidence regions use reconstruction, low confidence use original
        blended = (reconstructed_mask.astype(float) * blend_weight + 
                  original_prediction.astype(float) * self.config.original_blend_weight)
        
        return (blended > 0.5).astype(np.uint8)
    
    def _apply_post_processing(self, mask: np.ndarray) -> np.ndarray:
        """Apply post-processing to reconstructed mask"""
        processed_mask = mask.copy()
        
        # Morphological closing to fill gaps
        if self.config.apply_morphological_closing:
            from scipy import ndimage
            
            kernel_size = self.config.closing_kernel_size
            kernel = np.ones((kernel_size, kernel_size, kernel_size))
            
            processed_mask = ndimage.binary_closing(processed_mask, structure=kernel)
            processed_mask = processed_mask.astype(np.uint8)
        
        # Remove small components
        if self.config.min_component_size > 0:
            processed_mask = self._remove_small_components(processed_mask)
        
        # Gaussian smoothing
        if self.config.apply_smoothing:
            from scipy import ndimage
            
            # Convert to float for smoothing
            float_mask = processed_mask.astype(float)
            smoothed = ndimage.gaussian_filter(float_mask, sigma=self.config.smoothing_sigma)
            
            # Convert back to binary
            processed_mask = (smoothed > 0.5).astype(np.uint8)
        
        return processed_mask
    
    def _remove_small_components(self, mask: np.ndarray) -> np.ndarray:
        """Remove small connected components"""
        from scipy import ndimage
        
        # Label connected components
        labeled_mask, num_components = ndimage.label(mask)
        
        if num_components == 0:
            return mask
        
        # Compute component sizes
        component_sizes = ndimage.sum(mask, labeled_mask, range(1, num_components + 1))
        
        # Keep only large components
        large_components = np.where(component_sizes >= self.config.min_component_size)[0] + 1
        
        # Create mask with only large components
        filtered_mask = np.isin(labeled_mask, large_components)
        
        removed_components = num_components - len(large_components)
        if removed_components > 0:
            self.logger.info(f"Removed {removed_components} small components")
        
        return filtered_mask.astype(np.uint8)
    
    def _compute_template_volumes(self, templates: List[VesselTemplate]) -> Dict:
        """Compute individual template volumes for analysis"""
        template_volumes = {
            'cylinders': [],
            'bifurcations': [],
            'total_estimated': 0.0
        }
        
        for template in templates:
            volume = template.get_volume_estimate()
            template_volumes['total_estimated'] += volume
            
            if hasattr(template, 'params') and hasattr(template.params, 'start_point'):
                # Cylindrical template
                template_volumes['cylinders'].append(volume)
            else:
                # Bifurcation template
                template_volumes['bifurcations'].append(volume)
        
        return template_volumes
    
    def _assess_reconstruction_quality(self, reconstructed_mask: np.ndarray,
                                     graph: VascularGraph,
                                     original_prediction: Optional[np.ndarray]) -> float:
        """Assess overall reconstruction quality"""
        quality_scores = []
        
        # Template coverage quality
        if hasattr(self, 'template_placer') and self.template_placer.placement_warnings:
            warning_penalty = max(0, 1.0 - len(self.template_placer.placement_warnings) * 0.05)
            quality_scores.append(warning_penalty)
        else:
            quality_scores.append(1.0)
        
        # Volume consistency
        reconstructed_volume = np.sum(reconstructed_mask)
        if reconstructed_volume > 0:
            # Reasonable volume (not too sparse or too dense)
            total_voxels = np.prod(reconstructed_mask.shape)
            fill_ratio = reconstructed_volume / total_voxels
            
            # Optimal fill ratio for vascular structures: 0.01 - 0.1
            if 0.01 <= fill_ratio <= 0.1:
                volume_quality = 1.0
            elif fill_ratio < 0.01:
                volume_quality = fill_ratio / 0.01  # Penalize too sparse
            else:
                volume_quality = max(0, 1.0 - (fill_ratio - 0.1) / 0.1)  # Penalize too dense
            
            quality_scores.append(volume_quality)
        else:
            quality_scores.append(0.0)
        
        # Connectivity quality (placeholder)
        connectivity_quality = 0.8  # TODO: Implement proper connectivity assessment
        quality_scores.append(connectivity_quality)
        
        return np.mean(quality_scores)
    
    def _compute_volume_ratio(self, reconstructed_mask: np.ndarray,
                            original_prediction: Optional[np.ndarray]) -> float:
        """Compute volume ratio compared to original"""
        reconstructed_volume = np.sum(reconstructed_mask)
        
        if original_prediction is not None:
            # Resize if necessary
            if original_prediction.shape != reconstructed_mask.shape:
                from scipy import ndimage
                zoom_factors = np.array(reconstructed_mask.shape) / np.array(original_prediction.shape)
                original_prediction = ndimage.zoom(original_prediction, zoom_factors, order=1)
                original_prediction = (original_prediction > 0.5).astype(np.uint8)
            
            original_volume = np.sum(original_prediction)
            
            if original_volume > 0:
                return reconstructed_volume / original_volume
        
        # Fallback: compute ratio against total volume
        total_voxels = np.prod(reconstructed_mask.shape)
        return reconstructed_volume / total_voxels
    
    def _assess_connectivity(self, reconstructed_mask: np.ndarray, 
                           graph: VascularGraph) -> float:
        """Assess connectivity preservation"""
        if not self.config.connectivity_validation:
            return 1.0
        
        try:
            from scipy import ndimage
            
            # Label connected components
            labeled_mask, num_components = ndimage.label(reconstructed_mask)
            
            # Ideally, we should have one main connected component
            if num_components == 0:
                return 0.0
            elif num_components == 1:
                return 1.0
            else:
                # Penalize multiple components
                component_sizes = ndimage.sum(reconstructed_mask, labeled_mask, 
                                            range(1, num_components + 1))
                largest_component_size = np.max(component_sizes)
                total_size = np.sum(component_sizes)
                
                # Score based on fraction in largest component
                return largest_component_size / total_size
        
        except Exception as e:
            self.logger.warning(f"Connectivity assessment failed: {e}")
            return 0.5  # Neutral score


class BatchVolumeReconstructor:
    """Batch processor for multiple graphs"""
    
    def __init__(self, config: ReconstructionConfig):
        self.reconstructor = VolumeReconstructor(config)
        self.logger = logging.getLogger(__name__)
    
    def reconstruct_batch(self, 
                         corrected_graphs: List[VascularGraph],
                         original_predictions: Optional[List[np.ndarray]] = None,
                         reference_spacings: Optional[List[Tuple[float, float, float]]] = None) -> List[ReconstructionResult]:
        """Reconstruct volumes for batch of graphs"""
        
        self.logger.info(f"Starting batch reconstruction for {len(corrected_graphs)} graphs")
        
        results = []
        
        for i, graph in enumerate(corrected_graphs):
            self.logger.info(f"Reconstructing graph {i+1}/{len(corrected_graphs)}")
            
            # Get corresponding data
            original_pred = original_predictions[i] if original_predictions else None
            ref_spacing = reference_spacings[i] if reference_spacings else None
            
            # Reconstruct
            result = self.reconstructor.reconstruct_from_graph(
                graph, original_pred, ref_spacing
            )
            
            results.append(result)
        
        # Compute batch statistics
        self._log_batch_statistics(results)
        
        return results
    
    def _log_batch_statistics(self, results: List[ReconstructionResult]):
        """Log batch reconstruction statistics"""
        if not results:
            return
        
        qualities = [r.reconstruction_quality for r in results]
        times = [r.total_time for r in results]
        templates = [r.templates_used for r in results]
        
        self.logger.info("Batch reconstruction summary:")
        self.logger.info(f"  Average quality: {np.mean(qualities):.3f} ± {np.std(qualities):.3f}")
        self.logger.info(f"  Average time: {np.mean(times):.2f}s ± {np.std(times):.2f}s")
        self.logger.info(f"  Average templates: {np.mean(templates):.1f} ± {np.std(templates):.1f}")
        
        success_rate = sum(1 for r in results if r.reconstruction_quality > 0.5) / len(results)
        self.logger.info(f"  Success rate: {success_rate:.1%}")


def test_volume_reconstruction():
    """Test volume reconstruction functionality"""
    from src.models.graph_extraction.vascular_graph import VascularGraph
    
    # Create test graph
    nodes = [
        {'position': [0, 0, 0], 'radius_voxels': 2.0, 'type': 'normal'},
        {'position': [10, 0, 0], 'radius_voxels': 1.8, 'type': 'bifurcation'},
        {'position': [15, 5, 0], 'radius_voxels': 1.2, 'type': 'normal'},
        {'position': [15, -5, 0], 'radius_voxels': 1.2, 'type': 'normal'},
        {'position': [25, 5, 0], 'radius_voxels': 1.0, 'type': 'normal'},
        {'position': [25, -5, 0], 'radius_voxels': 1.0, 'type': 'normal'},
    ]
    
    edges = [
        {'source': 0, 'target': 1, 'confidence': 0.9},
        {'source': 1, 'target': 2, 'confidence': 0.8},
        {'source': 1, 'target': 3, 'confidence': 0.8},
        {'source': 2, 'target': 4, 'confidence': 0.7},
        {'source': 3, 'target': 5, 'confidence': 0.7},
    ]
    
    test_graph = VascularGraph(nodes=nodes, edges=edges)
    
    # Create reconstruction config
    config = ReconstructionConfig(
        target_resolution=(64, 64, 64),
        auto_compute_bounds=True,
        blend_with_original=False,
        apply_post_processing=True
    )
    
    # Create reconstructor
    reconstructor = VolumeReconstructor(config)
    
    # Reconstruct
    print("Starting volume reconstruction...")
    result = reconstructor.reconstruct_from_graph(
        test_graph, 
        return_intermediates=True
    )
    
    print(f"Reconstruction completed:")
    print(f"  Quality: {result.reconstruction_quality:.3f}")
    print(f"  Templates used: {result.templates_used}")
    print(f"  Rendering time: {result.rendering_time:.2f}s")
    print(f"  Total time: {result.total_time:.2f}s")
    print(f"  Output shape: {result.reconstructed_mask.shape}")
    print(f"  Fill ratio: {np.mean(result.reconstructed_mask):.4f}")
    
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
    
    return result


if __name__ == "__main__":
    # Run tests
    result = test_volume_reconstruction()
    print("Volume reconstruction tests completed successfully!")