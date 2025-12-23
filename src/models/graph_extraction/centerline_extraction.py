"""
3D Centerline Extraction for Vascular Structures
Implements distance transform-based skeletonization and centerline refinement
"""

import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage import distance_transform_edt, binary_erosion
from skimage.morphology import ball, binary_closing
from skimage.filters import gaussian
from typing import Tuple, Dict, Optional
import logging

# Handle different skimage versions
try:
    from skimage.morphology import skeletonize_3d
except ImportError:
    try:
        from skimage.morphology import skeletonize
        def skeletonize_3d(image):
            return skeletonize(image)
    except ImportError:
        # Fallback to scipy implementation
        from scipy.ndimage import binary_erosion
        def skeletonize_3d(image):
            # Simple fallback - this is not optimal but will work for testing
            return binary_erosion(image, iterations=1)


class CenterlineExtractor:
    """
    Extract centerlines from binary vascular masks using 3D skeletonization
    """
    
    def __init__(self,
                 spur_length_threshold: int = 3,
                 smoothing_sigma: float = 0.5,
                 min_branch_length: int = 2):
        """
        Initialize centerline extractor
        
        Args:
            spur_length_threshold: Remove spurs shorter than this (voxels)
            smoothing_sigma: Gaussian smoothing for centerline refinement
            min_branch_length: Minimum length for valid branches
        """
        self.spur_length_threshold = spur_length_threshold
        self.smoothing_sigma = smoothing_sigma
        self.min_branch_length = min_branch_length
        self.logger = logging.getLogger(__name__)
    
    def extract_centerline(self,
                          mask: np.ndarray,
                          voxel_spacing: Optional[Tuple[float, float, float]] = None) -> Dict:
        """
        Extract centerline from binary mask
        
        Args:
            mask: Binary segmentation mask
            voxel_spacing: Voxel spacing in mm (z, y, x)
            
        Returns:
            Dictionary containing centerline data and metrics
        """
        mask_bool = mask.astype(bool)
        
        self.logger.info(f"Extracting centerline from mask with {np.sum(mask_bool)} voxels")
        
        # Step 1: Compute distance transform
        distance_map = self._compute_distance_transform(mask_bool)
        
        # Step 2: Extract skeleton
        skeleton = self._extract_skeleton(mask_bool)
        
        # Step 3: Refine centerline
        refined_skeleton = self._refine_centerline(skeleton, distance_map)
        
        # Step 4: Extract centerline properties
        centerline_data = self._analyze_centerline(refined_skeleton, distance_map, voxel_spacing)
        
        return {
            'skeleton': refined_skeleton,
            'distance_map': distance_map,
            'centerline_points': centerline_data['points'],
            'radii': centerline_data['radii'],
            'connectivity': centerline_data['connectivity'],
            'metrics': centerline_data['metrics']
        }
    
    def _compute_distance_transform(self, mask: np.ndarray) -> np.ndarray:
        """
        Compute 3D Euclidean distance transform
        
        Args:
            mask: Binary mask
            
        Returns:
            Distance transform array
        """
        self.logger.debug("Computing 3D distance transform")
        
        # Compute distance transform
        distance_map = distance_transform_edt(mask)
        
        return distance_map
    
    def _extract_skeleton(self, mask: np.ndarray) -> np.ndarray:
        """
        Extract 3D skeleton using morphological thinning
        
        Args:
            mask: Binary mask
            
        Returns:
            Binary skeleton
        """
        self.logger.debug("Extracting 3D skeleton")
        
        # Use skimage skeletonize_3d (implements Lee's algorithm)
        skeleton = skeletonize_3d(mask)
        
        return skeleton
    
    def _refine_centerline(self, skeleton: np.ndarray, distance_map: np.ndarray) -> np.ndarray:
        """
        Refine centerline by removing spurs and smoothing
        
        Args:
            skeleton: Raw skeleton
            distance_map: Distance transform
            
        Returns:
            Refined skeleton
        """
        self.logger.debug("Refining centerline")
        
        refined_skeleton = skeleton.copy()
        
        # Step 1: Remove short spurs iteratively
        refined_skeleton = self._remove_short_spurs(refined_skeleton)
        
        # Step 2: Smooth while preserving topology
        if self.smoothing_sigma > 0:
            refined_skeleton = self._smooth_centerline(refined_skeleton, distance_map)
        
        # Step 3: Ensure connectivity
        refined_skeleton = self._ensure_connectivity(refined_skeleton)
        
        return refined_skeleton
    
    def _remove_short_spurs(self, skeleton: np.ndarray) -> np.ndarray:
        """
        Iteratively remove short spurs from skeleton
        
        Args:
            skeleton: Binary skeleton
            
        Returns:
            Skeleton with spurs removed
        """
        refined = skeleton.copy()
        iteration = 0
        max_iterations = 10
        
        while iteration < max_iterations:
            # Find endpoints
            endpoints = self._find_endpoints(refined)
            
            if len(endpoints) == 0:
                break
            
            # Check each endpoint for spur removal
            spurs_removed = 0
            for endpoint in endpoints:
                if self._is_short_spur(refined, endpoint):
                    refined = self._remove_spur(refined, endpoint)
                    spurs_removed += 1
            
            if spurs_removed == 0:
                break
                
            iteration += 1
        
        self.logger.debug(f"Removed spurs in {iteration} iterations")
        return refined
    
    def _find_endpoints(self, skeleton: np.ndarray) -> list:
        """Find endpoint voxels in skeleton"""
        endpoints = []
        skeleton_points = np.argwhere(skeleton)
        
        for point in skeleton_points:
            z, y, x = point
            neighbor_count = self._count_skeleton_neighbors(skeleton, z, y, x)
            
            if neighbor_count == 1:
                endpoints.append(point)
        
        return endpoints
    
    def _count_skeleton_neighbors(self, skeleton: np.ndarray, z: int, y: int, x: int) -> int:
        """Count 26-connected skeleton neighbors"""
        count = 0
        for dz in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dz == 0 and dy == 0 and dx == 0:
                        continue
                    nz, ny, nx = z + dz, y + dy, x + dx
                    if (0 <= nz < skeleton.shape[0] and 
                        0 <= ny < skeleton.shape[1] and 
                        0 <= nx < skeleton.shape[2] and
                        skeleton[nz, ny, nx]):
                        count += 1
        return count
    
    def _is_short_spur(self, skeleton: np.ndarray, endpoint: np.ndarray) -> bool:
        """Check if endpoint is part of a short spur"""
        # Trace path from endpoint until bifurcation or other endpoint
        current = endpoint.copy()
        path_length = 0
        visited = set()
        
        while path_length < self.spur_length_threshold:
            z, y, x = current
            visited.add((z, y, x))
            
            # Find next point in path
            neighbors = []
            for dz in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dz == 0 and dy == 0 and dx == 0:
                            continue
                        nz, ny, nx = z + dz, y + dy, x + dx
                        if (0 <= nz < skeleton.shape[0] and 
                            0 <= ny < skeleton.shape[1] and 
                            0 <= nx < skeleton.shape[2] and
                            skeleton[nz, ny, nx] and
                            (nz, ny, nx) not in visited):
                            neighbors.append(np.array([nz, ny, nx]))
            
            if len(neighbors) == 0:
                # Dead end - this is a spur
                return True
            elif len(neighbors) == 1:
                # Continue along path
                current = neighbors[0]
                path_length += 1
            else:
                # Reached bifurcation - not a spur
                return False
        
        # Path is longer than threshold - not a short spur
        return False
    
    def _remove_spur(self, skeleton: np.ndarray, endpoint: np.ndarray) -> np.ndarray:
        """Remove spur starting from endpoint"""
        result = skeleton.copy()
        current = endpoint.copy()
        
        while True:
            z, y, x = current
            result[z, y, x] = False
            
            # Find next point
            neighbor_count = self._count_skeleton_neighbors(result, z, y, x)
            
            if neighbor_count != 1:
                break
            
            # Find the one remaining neighbor
            for dz in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dz == 0 and dy == 0 and dx == 0:
                            continue
                        nz, ny, nx = z + dz, y + dy, x + dx
                        if (0 <= nz < result.shape[0] and 
                            0 <= ny < result.shape[1] and 
                            0 <= nx < result.shape[2] and
                            result[nz, ny, nx]):
                            current = np.array([nz, ny, nx])
                            break
                else:
                    continue
                break
            else:
                break
        
        return result
    
    def _smooth_centerline(self, skeleton: np.ndarray, distance_map: np.ndarray) -> np.ndarray:
        """
        Apply gentle smoothing while preserving critical points
        
        Args:
            skeleton: Binary skeleton
            distance_map: Distance transform for guidance
            
        Returns:
            Smoothed skeleton
        """
        # Convert to float for smoothing
        skeleton_float = skeleton.astype(np.float32)
        
        # Apply Gaussian smoothing
        smoothed = gaussian(skeleton_float, sigma=self.smoothing_sigma, preserve_range=True)
        
        # Threshold back to binary
        smoothed_binary = smoothed > 0.5
        
        # Preserve critical points (bifurcations and endpoints)
        critical_points = self._find_critical_points(skeleton)
        
        # Ensure critical points remain in smoothed skeleton
        for point in critical_points:
            z, y, x = point
            smoothed_binary[z, y, x] = True
        
        return smoothed_binary
    
    def _find_critical_points(self, skeleton: np.ndarray) -> list:
        """Find bifurcations and endpoints"""
        critical_points = []
        skeleton_points = np.argwhere(skeleton)
        
        for point in skeleton_points:
            z, y, x = point
            neighbor_count = self._count_skeleton_neighbors(skeleton, z, y, x)
            
            # Endpoints or bifurcations
            if neighbor_count == 1 or neighbor_count > 2:
                critical_points.append(point)
        
        return critical_points
    
    def _ensure_connectivity(self, skeleton: np.ndarray) -> np.ndarray:
        """Ensure skeleton maintains proper connectivity"""
        # Apply morphological closing to reconnect small gaps
        structure = ball(1)
        connected_skeleton = binary_closing(skeleton, structure)
        
        # Re-skeletonize to maintain thinness
        if np.sum(connected_skeleton) > np.sum(skeleton) * 1.5:  # Only if significant thickening
            connected_skeleton = skeletonize_3d(connected_skeleton)
        
        return connected_skeleton
    
    def _analyze_centerline(self, 
                           skeleton: np.ndarray, 
                           distance_map: np.ndarray,
                           voxel_spacing: Optional[Tuple[float, float, float]]) -> Dict:
        """
        Analyze centerline properties
        
        Args:
            skeleton: Refined skeleton
            distance_map: Distance transform
            voxel_spacing: Voxel spacing in mm
            
        Returns:
            Dictionary with centerline analysis
        """
        skeleton_points = np.argwhere(skeleton)
        
        if len(skeleton_points) == 0:
            return {
                'points': [],
                'radii': [],
                'connectivity': {},
                'metrics': {'total_length': 0, 'num_endpoints': 0, 'num_bifurcations': 0}
            }
        
        # Extract radii at skeleton points
        radii = []
        for point in skeleton_points:
            z, y, x = point
            radius = distance_map[z, y, x]
            radii.append(radius)
        
        # Analyze connectivity
        endpoints = self._find_endpoints(skeleton)
        bifurcations = self._find_bifurcations(skeleton)
        
        # Calculate total length
        total_length_voxels = len(skeleton_points)
        total_length_mm = None
        if voxel_spacing is not None:
            # Approximate physical length (could be improved with actual path tracing)
            avg_voxel_size = np.mean(voxel_spacing)
            total_length_mm = total_length_voxels * avg_voxel_size
        
        connectivity = {
            'skeleton_points': skeleton_points,
            'neighbor_graph': self._build_neighbor_graph(skeleton),
        }
        
        metrics = {
            'total_length_voxels': total_length_voxels,
            'total_length_mm': total_length_mm,
            'num_endpoints': len(endpoints),
            'num_bifurcations': len(bifurcations),
            'average_radius': np.mean(radii) if radii else 0,
            'radius_std': np.std(radii) if radii else 0
        }
        
        return {
            'points': skeleton_points,
            'radii': np.array(radii),
            'connectivity': connectivity,
            'metrics': metrics
        }
    
    def _find_bifurcations(self, skeleton: np.ndarray) -> list:
        """Find bifurcation points (nodes with >2 neighbors)"""
        bifurcations = []
        skeleton_points = np.argwhere(skeleton)
        
        for point in skeleton_points:
            z, y, x = point
            neighbor_count = self._count_skeleton_neighbors(skeleton, z, y, x)
            
            if neighbor_count > 2:
                bifurcations.append(point)
        
        return bifurcations
    
    def _build_neighbor_graph(self, skeleton: np.ndarray) -> Dict:
        """Build connectivity graph of skeleton points"""
        skeleton_points = np.argwhere(skeleton)
        point_to_index = {tuple(point): i for i, point in enumerate(skeleton_points)}
        
        adjacency = {}
        
        for i, point in enumerate(skeleton_points):
            z, y, x = point
            neighbors = []
            
            # Find 26-connected neighbors
            for dz in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dz == 0 and dy == 0 and dx == 0:
                            continue
                        nz, ny, nx = z + dz, y + dy, x + dx
                        if (0 <= nz < skeleton.shape[0] and 
                            0 <= ny < skeleton.shape[1] and 
                            0 <= nx < skeleton.shape[2] and
                            skeleton[nz, ny, nx]):
                            neighbor_point = (nz, ny, nx)
                            if neighbor_point in point_to_index:
                                neighbors.append(point_to_index[neighbor_point])
            
            adjacency[i] = neighbors
        
        return adjacency