"""
Quality assessment for segmentation masks
Implements connectivity analysis and artifact detection
"""

import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage import label
from typing import Dict, Tuple, List, Optional
import logging

# Handle different skimage versions for skeletonize_3d
try:
    from skimage.morphology import skeletonize_3d
except ImportError:
    try:
        from skimage.morphology import skeletonize
        def skeletonize_3d(image):
            return skeletonize(image)
    except ImportError:
        from scipy.ndimage import binary_erosion
        def skeletonize_3d(image):
            return binary_erosion(image, iterations=1)


class QualityAssessment:
    """
    Quality assessment for binary segmentation masks
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def assess_mask_quality(self, 
                           mask: np.ndarray,
                           voxel_spacing: Optional[Tuple[float, float, float]] = None) -> Dict:
        """
        Comprehensive quality assessment of segmentation mask
        
        Args:
            mask: Binary segmentation mask
            voxel_spacing: Voxel spacing in mm (z, y, x)
            
        Returns:
            Dictionary with quality metrics
        """
        mask_bool = mask.astype(bool)
        
        quality_metrics = {
            'voxel_spacing': voxel_spacing,
            'total_volume_voxels': np.sum(mask_bool),
            'total_volume_mm3': None,
            'connectivity': {},
            'topology': {},
            'artifacts': {},
            'overall_quality_score': 0.0
        }
        
        # Calculate physical volume if spacing provided
        if voxel_spacing is not None:
            voxel_volume_mm3 = np.prod(voxel_spacing)
            quality_metrics['total_volume_mm3'] = quality_metrics['total_volume_voxels'] * voxel_volume_mm3
        
        # Connectivity analysis
        quality_metrics['connectivity'] = self._analyze_connectivity(mask_bool)
        
        # Topology analysis
        quality_metrics['topology'] = self._analyze_topology(mask_bool)
        
        # Artifact detection
        quality_metrics['artifacts'] = self._detect_artifacts(mask_bool, voxel_spacing)
        
        # Overall quality score
        quality_metrics['overall_quality_score'] = self._compute_overall_quality(quality_metrics)
        
        return quality_metrics
    
    def _analyze_connectivity(self, mask: np.ndarray) -> Dict:
        """Analyze connectivity of the mask"""
        # Label connected components
        labeled_mask, num_components = label(mask)
        
        if num_components == 0:
            return {
                'num_components': 0,
                'largest_component_ratio': 0.0,
                'component_sizes': [],
                'connectivity_score': 0.0
            }
        
        # Calculate component sizes
        component_sizes = np.bincount(labeled_mask.flat)[1:]  # Skip background
        component_sizes_sorted = np.sort(component_sizes)[::-1]  # Descending order
        
        # Calculate connectivity metrics
        largest_component_ratio = component_sizes_sorted[0] / np.sum(component_sizes)
        
        # Connectivity score (higher is better)
        # Penalize many small components, reward single large component
        if num_components == 1:
            connectivity_score = 1.0
        else:
            # Score based on size distribution
            size_entropy = -np.sum((component_sizes / np.sum(component_sizes)) * 
                                  np.log(component_sizes / np.sum(component_sizes) + 1e-10))
            max_entropy = np.log(num_components)
            connectivity_score = 1.0 - (size_entropy / max_entropy if max_entropy > 0 else 0)
        
        return {
            'num_components': num_components,
            'largest_component_ratio': largest_component_ratio,
            'component_sizes': component_sizes_sorted.tolist(),
            'connectivity_score': connectivity_score
        }
    
    def _analyze_topology(self, mask: np.ndarray) -> Dict:
        """Analyze topological properties"""
        if np.sum(mask) == 0:
            return {
                'skeleton_length': 0,
                'num_endpoints': 0,
                'num_bifurcations': 0,
                'topology_score': 0.0
            }
        
        try:
            # Extract skeleton for topology analysis
            skeleton = skeletonize_3d(mask)
            skeleton_volume = np.sum(skeleton)
            
            if skeleton_volume == 0:
                return {
                    'skeleton_length': 0,
                    'num_endpoints': 0,
                    'num_bifurcations': 0,
                    'topology_score': 0.0
                }
            
            # Analyze skeleton connectivity
            endpoints, bifurcations = self._analyze_skeleton_topology(skeleton)
            
            # Topology score based on reasonable vessel structure
            # Good topology: reasonable ratio of endpoints to bifurcations
            if len(bifurcations) == 0:
                if len(endpoints) == 2:
                    topology_score = 1.0  # Simple tube
                else:
                    topology_score = 0.5  # Multiple endpoints without bifurcations
            else:
                # Expected: roughly 2 endpoints per bifurcation for tree structure
                expected_endpoints = len(bifurcations) + 1
                endpoint_ratio = min(len(endpoints) / expected_endpoints, 
                                   expected_endpoints / len(endpoints)) if expected_endpoints > 0 else 0
                topology_score = endpoint_ratio
            
            return {
                'skeleton_length': skeleton_volume,
                'num_endpoints': len(endpoints),
                'num_bifurcations': len(bifurcations),
                'topology_score': topology_score
            }
            
        except Exception as e:
            self.logger.warning(f"Topology analysis failed: {e}")
            return {
                'skeleton_length': 0,
                'num_endpoints': 0,
                'num_bifurcations': 0,
                'topology_score': 0.0
            }
    
    def _analyze_skeleton_topology(self, skeleton: np.ndarray) -> Tuple[List, List]:
        """Analyze skeleton to find endpoints and bifurcations"""
        endpoints = []
        bifurcations = []
        
        # Find skeleton points
        skeleton_points = np.argwhere(skeleton)
        
        for point in skeleton_points:
            z, y, x = point
            
            # Count 26-connected neighbors
            neighbor_count = 0
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
                            neighbor_count += 1
            
            if neighbor_count == 1:
                endpoints.append(point)
            elif neighbor_count > 2:
                bifurcations.append(point)
        
        return endpoints, bifurcations
    
    def _detect_artifacts(self, mask: np.ndarray, voxel_spacing: Optional[Tuple[float, float, float]]) -> Dict:
        """Detect various artifacts in the mask"""
        artifacts = {
            'isolated_pixels': 0,
            'elongated_structures': 0,
            'unrealistic_shapes': 0,
            'artifact_score': 1.0  # Higher is better (fewer artifacts)
        }
        
        # Detect isolated pixels (single voxel components)
        labeled_mask, _ = label(mask)
        component_sizes = np.bincount(labeled_mask.flat)[1:]
        artifacts['isolated_pixels'] = np.sum(component_sizes == 1)
        
        # Detect unrealistic elongated structures
        if voxel_spacing is not None:
            artifacts['elongated_structures'] = self._detect_elongated_structures(mask, voxel_spacing)
        
        # Calculate overall artifact score
        total_voxels = np.sum(mask)
        if total_voxels > 0:
            artifact_ratio = (artifacts['isolated_pixels'] + artifacts['elongated_structures']) / total_voxels
            artifacts['artifact_score'] = max(0.0, 1.0 - artifact_ratio)
        
        return artifacts
    
    def _detect_elongated_structures(self, mask: np.ndarray, voxel_spacing: Tuple[float, float, float]) -> int:
        """Detect unrealistically elongated structures"""
        # This is a simplified implementation
        # Could be enhanced with more sophisticated shape analysis
        
        labeled_mask, num_components = label(mask)
        elongated_count = 0
        
        for component_id in range(1, num_components + 1):
            component_mask = (labeled_mask == component_id)
            
            # Calculate component bounding box
            coords = np.argwhere(component_mask)
            if len(coords) == 0:
                continue
                
            min_coords = np.min(coords, axis=0)
            max_coords = np.max(coords, axis=0)
            
            # Physical dimensions
            dimensions = (max_coords - min_coords) * np.array(voxel_spacing)
            
            # Check for extreme aspect ratios
            max_dim = np.max(dimensions)
            min_dim = np.min(dimensions[dimensions > 0]) if np.any(dimensions > 0) else 1
            
            if max_dim / min_dim > 20:  # Very elongated
                elongated_count += 1
        
        return elongated_count
    
    def _compute_overall_quality(self, metrics: Dict) -> float:
        """Compute overall quality score from individual metrics"""
        weights = {
            'connectivity': 0.4,
            'topology': 0.3,
            'artifacts': 0.3
        }
        
        connectivity_score = metrics['connectivity'].get('connectivity_score', 0.0)
        topology_score = metrics['topology'].get('topology_score', 0.0)
        artifact_score = metrics['artifacts'].get('artifact_score', 0.0)
        
        overall_score = (weights['connectivity'] * connectivity_score +
                        weights['topology'] * topology_score +
                        weights['artifacts'] * artifact_score)
        
        return overall_score
    
    def generate_quality_report(self, metrics: Dict) -> str:
        """Generate human-readable quality report"""
        report = ["=== Mask Quality Assessment ==="]
        
        # Basic info
        report.append(f"Total volume: {metrics['total_volume_voxels']} voxels")
        if metrics['total_volume_mm3'] is not None:
            report.append(f"Physical volume: {metrics['total_volume_mm3']:.2f} mm³")
        
        # Connectivity
        conn = metrics['connectivity']
        report.append(f"\nConnectivity:")
        report.append(f"  Components: {conn['num_components']}")
        report.append(f"  Largest component ratio: {conn['largest_component_ratio']:.3f}")
        report.append(f"  Connectivity score: {conn['connectivity_score']:.3f}")
        
        # Topology
        topo = metrics['topology']
        report.append(f"\nTopology:")
        report.append(f"  Skeleton length: {topo['skeleton_length']} voxels")
        report.append(f"  Endpoints: {topo['num_endpoints']}")
        report.append(f"  Bifurcations: {topo['num_bifurcations']}")
        report.append(f"  Topology score: {topo['topology_score']:.3f}")
        
        # Artifacts
        art = metrics['artifacts']
        report.append(f"\nArtifacts:")
        report.append(f"  Isolated pixels: {art['isolated_pixels']}")
        report.append(f"  Elongated structures: {art['elongated_structures']}")
        report.append(f"  Artifact score: {art['artifact_score']:.3f}")
        
        # Overall
        report.append(f"\nOverall Quality Score: {metrics['overall_quality_score']:.3f}")
        
        return "\n".join(report)