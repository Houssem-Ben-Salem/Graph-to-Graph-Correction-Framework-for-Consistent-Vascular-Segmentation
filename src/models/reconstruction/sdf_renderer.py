"""
Signed Distance Field (SDF) Renderer
High-performance volumetric rendering for vessel templates
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from src.models.reconstruction.vessel_templates import VesselTemplate, CylindricalTemplate, BifurcationTemplate


@dataclass
class RenderingConfig:
    """Configuration for SDF rendering"""
    resolution: Tuple[int, int, int] = (128, 128, 128)  # Volume dimensions
    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # Voxel spacing
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Volume origin
    
    # Rendering parameters
    max_distance: float = 10.0  # Maximum SDF distance to compute
    smoothing_kernel_size: float = 1.0  # Smoothing radius
    anti_aliasing: bool = True  # Enable anti-aliasing 
    parallel_processing: bool = True  # Use parallel processing
    chunk_size: int = 32  # Chunk size for parallel processing
    
    # Quality settings
    supersampling: int = 1  # Supersampling factor (1 = no supersampling)
    boundary_smoothing: bool = True  # Smooth template boundaries
    template_blending: bool = True  # Blend overlapping templates


@dataclass  
class VoxelGrid:
    """3D voxel grid representation"""
    data: np.ndarray  # 3D array of voxel values
    origin: np.ndarray  # Grid origin in world coordinates
    voxel_size: np.ndarray  # Voxel dimensions
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.data.shape
    
    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get world coordinate bounds"""
        max_coords = self.origin + np.array(self.shape) * self.voxel_size
        return self.origin, max_coords
    
    def world_to_voxel(self, world_coords: np.ndarray) -> np.ndarray:
        """Convert world coordinates to voxel indices"""
        return (world_coords - self.origin) / self.voxel_size
    
    def voxel_to_world(self, voxel_coords: np.ndarray) -> np.ndarray:
        """Convert voxel indices to world coordinates"""
        return voxel_coords * self.voxel_size + self.origin
    
    def get_voxel_centers(self) -> np.ndarray:
        """Get world coordinates of all voxel centers"""
        i, j, k = np.mgrid[0:self.shape[0], 0:self.shape[1], 0:self.shape[2]]
        voxel_indices = np.stack([i.ravel(), j.ravel(), k.ravel()], axis=1)
        world_coords = self.voxel_to_world(voxel_indices + 0.5)  # +0.5 for centers
        return world_coords


class SDFRenderer:
    """High-performance SDF renderer for vessel templates"""
    
    def __init__(self, config: RenderingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Pre-compute coordinate grids for efficiency
        self._initialize_coordinate_grids()
    
    def _initialize_coordinate_grids(self):
        """Pre-compute coordinate grids"""
        self.logger.info("Initializing coordinate grids...")
        
        # Create voxel grid
        origin = np.array(self.config.origin)
        voxel_size = np.array(self.config.voxel_size)
        
        self.voxel_grid = VoxelGrid(
            data=np.zeros(self.config.resolution),
            origin=origin,
            voxel_size=voxel_size
        )
        
        # Pre-compute world coordinates for all voxels
        self.world_coordinates = self.voxel_grid.get_voxel_centers()
        
        self.logger.info(f"Grid shape: {self.config.resolution}")
        self.logger.info(f"World bounds: {self.voxel_grid.bounds}")
        self.logger.info(f"Total voxels: {np.prod(self.config.resolution):,}")
    
    def render_template(self, template: VesselTemplate) -> VoxelGrid:
        """Render a single template to volume"""
        start_time = time.time()
        
        # Check if template intersects with rendering volume
        template_min, template_max = template.get_bounding_box()
        volume_min, volume_max = self.voxel_grid.bounds
        
        if not self._bounding_boxes_intersect(template_min, template_max, volume_min, volume_max):
            self.logger.debug("Template outside rendering volume, skipping")
            return VoxelGrid(
                data=np.full(self.config.resolution, self.config.max_distance),
                origin=self.voxel_grid.origin,
                voxel_size=self.voxel_grid.voxel_size
            )
        
        # Compute SDF for all voxels
        if self.config.parallel_processing:
            sdf_values = self._compute_sdf_parallel(template)
        else:
            sdf_values = self._compute_sdf_sequential(template)
        
        # Reshape to 3D grid
        sdf_volume = sdf_values.reshape(self.config.resolution)
        
        # Apply post-processing
        if self.config.anti_aliasing:
            sdf_volume = self._apply_anti_aliasing(sdf_volume)
        
        if self.config.boundary_smoothing:
            sdf_volume = self._apply_boundary_smoothing(sdf_volume)
        
        render_time = time.time() - start_time
        self.logger.debug(f"Template rendered in {render_time:.3f}s")
        
        return VoxelGrid(
            data=sdf_volume,
            origin=self.voxel_grid.origin,
            voxel_size=self.voxel_grid.voxel_size
        )
    
    def render_templates(self, templates: List[VesselTemplate]) -> VoxelGrid:
        """Render multiple templates with blending"""
        if not templates:
            return VoxelGrid(
                data=np.full(self.config.resolution, self.config.max_distance),
                origin=self.voxel_grid.origin,
                voxel_size=self.voxel_grid.voxel_size
            )
        
        start_time = time.time()
        self.logger.info(f"Rendering {len(templates)} templates...")
        
        # Render each template individually
        template_volumes = []
        
        if self.config.parallel_processing and len(templates) > 1:
            # Parallel template rendering
            with ThreadPoolExecutor(max_workers=min(4, len(templates))) as executor:
                futures = [executor.submit(self.render_template, template) 
                          for template in templates]
                
                for future in as_completed(futures):
                    template_volumes.append(future.result())
        else:
            # Sequential rendering
            for template in templates:
                template_volumes.append(self.render_template(template))
        
        # Blend templates
        combined_volume = self._blend_template_volumes(template_volumes)
        
        render_time = time.time() - start_time
        self.logger.info(f"All templates rendered in {render_time:.3f}s")
        
        return combined_volume
    
    def _compute_sdf_parallel(self, template: VesselTemplate) -> np.ndarray:
        """Compute SDF using parallel processing"""
        total_voxels = len(self.world_coordinates)
        chunk_size = self.config.chunk_size**3
        
        sdf_values = np.zeros(total_voxels)
        
        # Process in chunks
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for start_idx in range(0, total_voxels, chunk_size):
                end_idx = min(start_idx + chunk_size, total_voxels)
                chunk_coords = self.world_coordinates[start_idx:end_idx]
                
                future = executor.submit(template.compute_sdf, chunk_coords)
                futures.append((future, start_idx, end_idx))
            
            # Collect results
            for future, start_idx, end_idx in futures:
                sdf_values[start_idx:end_idx] = future.result()
        
        return sdf_values
    
    def _compute_sdf_sequential(self, template: VesselTemplate) -> np.ndarray:
        """Compute SDF sequentially (for debugging or small volumes)"""
        return template.compute_sdf(self.world_coordinates)
    
    def _blend_template_volumes(self, volumes: List[VoxelGrid]) -> VoxelGrid:
        """Blend multiple template volumes"""
        if len(volumes) == 1:
            return volumes[0]
        
        # Extract SDF data
        sdf_arrays = [vol.data for vol in volumes]
        
        if self.config.template_blending:
            # Smooth minimum blending
            blended_sdf = self._smooth_minimum_blend(sdf_arrays)
        else:
            # Simple minimum (union)
            blended_sdf = np.minimum.reduce(sdf_arrays)
        
        return VoxelGrid(
            data=blended_sdf,
            origin=self.voxel_grid.origin,
            voxel_size=self.voxel_grid.voxel_size
        )
    
    def _smooth_minimum_blend(self, sdf_arrays: List[np.ndarray], 
                            smoothing_factor: float = 0.5) -> np.ndarray:
        """Smooth minimum blending for better template fusion"""
        if len(sdf_arrays) == 1:
            return sdf_arrays[0]
        
        # Start with first array
        result = sdf_arrays[0].copy()
        
        # Progressively blend with remaining arrays
        for sdf_array in sdf_arrays[1:]:
            # Smooth minimum operation
            # smin(a, b, k) = -log(exp(-k*a) + exp(-k*b)) / k
            k = 1.0 / smoothing_factor
            
            # Numerically stable computation
            diff = result - sdf_array
            stable_exp = np.exp(-k * np.abs(diff))
            
            where_result_smaller = diff < 0
            smin = np.where(
                where_result_smaller,
                result - np.log(1 + stable_exp) / k,
                sdf_array - np.log(1 + stable_exp) / k
            )
            
            result = smin
        
        return result
    
    def _apply_anti_aliasing(self, sdf_volume: np.ndarray) -> np.ndarray:
        """Apply anti-aliasing to reduce staircase artifacts"""
        # Simple Gaussian smoothing
        from scipy import ndimage
        
        sigma = 0.5  # Smoothing strength
        return ndimage.gaussian_filter(sdf_volume, sigma=sigma)
    
    def _apply_boundary_smoothing(self, sdf_volume: np.ndarray) -> np.ndarray:
        """Apply boundary smoothing near zero-level set"""
        # Smooth only near the surface (|sdf| < threshold)
        from scipy import ndimage
        
        threshold = 2.0  # Distance threshold for smoothing
        mask = np.abs(sdf_volume) < threshold
        
        if np.any(mask):
            # Apply light smoothing only near surface
            smoothed = ndimage.gaussian_filter(sdf_volume, sigma=0.3)
            sdf_volume = np.where(mask, smoothed, sdf_volume)
        
        return sdf_volume
    
    def _bounding_boxes_intersect(self, min1: np.ndarray, max1: np.ndarray,
                                min2: np.ndarray, max2: np.ndarray) -> bool:
        """Check if two 3D bounding boxes intersect"""
        return np.all(min1 < max2) and np.all(min2 < max1)
    
    def sdf_to_binary_mask(self, sdf_volume: VoxelGrid, 
                          threshold: float = 0.0) -> np.ndarray:
        """Convert SDF volume to binary mask"""
        return (sdf_volume.data <= threshold).astype(np.uint8)
    
    def sdf_to_probability_mask(self, sdf_volume: VoxelGrid,
                              transition_width: float = 1.0) -> np.ndarray:
        """Convert SDF to smooth probability mask"""
        # Sigmoid function for smooth transition
        sigmoid = 1.0 / (1.0 + np.exp(sdf_volume.data / transition_width))
        return sigmoid
    
    def extract_surface_mesh(self, sdf_volume: VoxelGrid, 
                           iso_value: float = 0.0) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Extract surface mesh using marching cubes"""
        try:
            from skimage import measure
            
            # Run marching cubes
            vertices, faces, normals, _ = measure.marching_cubes(
                sdf_volume.data, 
                level=iso_value,
                spacing=sdf_volume.voxel_size
            )
            
            # Transform vertices to world coordinates
            vertices += sdf_volume.origin
            
            return vertices, faces
            
        except ImportError:
            self.logger.error("scikit-image not available for mesh extraction")
            return None
        except Exception as e:
            self.logger.error(f"Mesh extraction failed: {e}")
            return None
    
    def compute_volume_statistics(self, sdf_volume: VoxelGrid) -> Dict:
        """Compute volume statistics from SDF"""
        inside_mask = sdf_volume.data <= 0
        
        # Volume estimation
        voxel_volume = np.prod(sdf_volume.voxel_size)
        total_volume = np.sum(inside_mask) * voxel_volume
        
        # Surface area estimation (approximate)
        boundary_mask = np.abs(sdf_volume.data) <= 1.0
        surface_area = np.sum(boundary_mask) * (voxel_volume ** (2/3))
        
        return {
            'total_volume': total_volume,
            'surface_area': surface_area,
            'inside_voxels': np.sum(inside_mask),
            'boundary_voxels': np.sum(boundary_mask),
            'fill_ratio': np.mean(inside_mask),
            'min_sdf': np.min(sdf_volume.data),
            'max_sdf': np.max(sdf_volume.data)
        }


class BatchSDFRenderer:
    """Batch renderer for processing multiple graphs"""
    
    def __init__(self, config: RenderingConfig):
        self.renderer = SDFRenderer(config)
        self.logger = logging.getLogger(__name__)
    
    def render_graph_batch(self, template_batches: List[Dict[str, List[VesselTemplate]]],
                          output_format: str = 'binary') -> List[np.ndarray]:
        """Render a batch of template sets"""
        results = []
        
        for i, templates_dict in enumerate(template_batches):
            self.logger.info(f"Rendering graph {i+1}/{len(template_batches)}")
            
            # Combine all templates
            all_templates = []
            for template_list in templates_dict.values():
                all_templates.extend(template_list)
            
            # Render templates
            sdf_volume = self.renderer.render_templates(all_templates)
            
            # Convert to requested format
            if output_format == 'binary':
                result = self.renderer.sdf_to_binary_mask(sdf_volume)
            elif output_format == 'probability':
                result = self.renderer.sdf_to_probability_mask(sdf_volume)
            elif output_format == 'sdf':
                result = sdf_volume.data
            else:
                raise ValueError(f"Unknown output format: {output_format}")
            
            results.append(result)
        
        return results


def test_sdf_renderer():
    """Test SDF renderer functionality"""
    from src.models.reconstruction.vessel_templates import CylindricalTemplate, CylinderParameters
    
    # Create test configuration
    config = RenderingConfig(
        resolution=(64, 64, 64),
        voxel_size=(0.5, 0.5, 0.5),
        origin=(-16.0, -16.0, -16.0),
        parallel_processing=True
    )
    
    # Create renderer
    renderer = SDFRenderer(config)
    
    # Create test templates
    cylinder_params = CylinderParameters(
        start_point=np.array([-10, 0, 0]),
        end_point=np.array([10, 0, 0]),
        start_radius=3.0,
        end_radius=2.0
    )
    
    cylinder = CylindricalTemplate(cylinder_params)
    
    # Render template
    print("Rendering cylinder template...")
    start_time = time.time()
    
    sdf_volume = renderer.render_template(cylinder)
    
    render_time = time.time() - start_time
    print(f"Rendering completed in {render_time:.3f}s")
    
    # Convert to binary mask
    binary_mask = renderer.sdf_to_binary_mask(sdf_volume)
    
    # Compute statistics
    stats = renderer.compute_volume_statistics(sdf_volume)
    
    print("Volume statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.3f}")
    
    print(f"Binary mask shape: {binary_mask.shape}")
    print(f"Binary mask fill ratio: {np.mean(binary_mask):.3f}")
    
    return renderer, sdf_volume, binary_mask


if __name__ == "__main__":
    # Run tests
    renderer, sdf_volume, binary_mask = test_sdf_renderer()
    print("SDF renderer tests completed successfully!")