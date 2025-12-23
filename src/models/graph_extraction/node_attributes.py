"""
Node Attribute Extraction for Vascular Graphs
Implements comprehensive geometric, structural, and context attribute extraction
"""

import numpy as np
import scipy.ndimage as ndi
from scipy.spatial import KDTree
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm


class NodeAttributeExtractor:
    """
    Extract comprehensive attributes for graph nodes
    """
    
    def __init__(self, neighborhood_radius: float = 3.0):
        """
        Initialize node attribute extractor
        
        Args:
            neighborhood_radius: Radius for local neighborhood analysis
        """
        self.neighborhood_radius = neighborhood_radius
        self.logger = logging.getLogger(__name__)
    
    def extract_all_attributes(self,
                             nodes: Dict,
                             distance_map: np.ndarray,
                             skeleton: np.ndarray,
                             original_mask: Optional[np.ndarray] = None,
                             prediction_probabilities: Optional[np.ndarray] = None,
                             confidence_map: Optional[np.ndarray] = None,
                             voxel_spacing: Optional[Tuple[float, float, float]] = None) -> List[Dict]:
        """
        Extract all node attributes
        
        Args:
            nodes: Node placement results
            distance_map: Distance transform
            skeleton: Binary skeleton
            original_mask: Original segmentation mask
            prediction_probabilities: Raw prediction probabilities (for predicted masks)
            confidence_map: Confidence map
            voxel_spacing: Voxel spacing in mm
            
        Returns:
            List of attribute dictionaries for each node
        """
        positions = nodes['positions']
        types = nodes['types']
        
        self.logger.info(f"Extracting attributes for {len(positions)} nodes")
        
        attributes = []
        
        # Use tqdm for progress tracking
        pbar = tqdm(enumerate(zip(positions, types)), total=len(positions), 
                   desc="Extracting node attributes", leave=False)
        
        for i, (pos, node_type) in pbar:
            node_attrs = {}
            
            # Basic identification
            node_attrs['node_id'] = i
            node_attrs['node_type'] = node_type
            
            # Geometric attributes
            geometric_attrs = self._extract_geometric_attributes(
                pos, distance_map, skeleton, voxel_spacing
            )
            node_attrs.update(geometric_attrs)
            
            # Structural attributes
            structural_attrs = self._extract_structural_attributes(
                pos, skeleton, positions, i
            )
            node_attrs.update(structural_attrs)
            
            # Context attributes (for predicted masks)
            if prediction_probabilities is not None or confidence_map is not None:
                context_attrs = self._extract_context_attributes(
                    pos, prediction_probabilities, confidence_map, original_mask
                )
                node_attrs.update(context_attrs)
            
            attributes.append(node_attrs)
            
            # Update progress bar every 10 nodes to reduce overhead
            if i % 10 == 0:
                pbar.set_postfix({'processed': i+1})
        
        pbar.close()
        return attributes
    
    def _extract_geometric_attributes(self,
                                    position: np.ndarray,
                                    distance_map: np.ndarray,
                                    skeleton: np.ndarray,
                                    voxel_spacing: Optional[Tuple[float, float, float]]) -> Dict:
        """Extract geometric attributes for a node"""
        z, y, x = position.astype(int)
        
        # Ensure coordinates are within bounds
        z = np.clip(z, 0, distance_map.shape[0] - 1)
        y = np.clip(y, 0, distance_map.shape[1] - 1)
        x = np.clip(x, 0, distance_map.shape[2] - 1)
        
        # Basic position
        coordinates_voxel = position
        
        # Physical coordinates
        if voxel_spacing is not None:
            coordinates_physical = position * np.array(voxel_spacing)
        else:
            coordinates_physical = position
        
        # Local vessel radius from distance transform
        radius_voxels = distance_map[z, y, x]
        
        # Refined radius using local sphere fitting
        refined_radius = self._refine_radius_estimate(position, distance_map)
        
        # Physical radius
        if voxel_spacing is not None:
            radius_mm = refined_radius * np.mean(voxel_spacing)
        else:
            radius_mm = refined_radius
        
        # Local curvature
        curvature = self._calculate_local_curvature(position, skeleton)
        
        # Vessel direction (tangent vector)
        direction = self._calculate_vessel_direction(position, skeleton)
        
        return {
            'coordinates_voxel': coordinates_voxel,
            'coordinates_physical': coordinates_physical,
            'radius_voxels': refined_radius,
            'radius_mm': radius_mm,
            'local_curvature': curvature,
            'vessel_direction': direction,
        }
    
    def _refine_radius_estimate(self, position: np.ndarray, distance_map: np.ndarray) -> float:
        """Refine radius estimate using local sphere fitting"""
        # Get local neighborhood
        neighborhood = self._get_local_neighborhood(position, distance_map)
        
        if len(neighborhood['positions']) < 5:
            # Fall back to simple distance transform value
            z, y, x = position.astype(int)
            z = np.clip(z, 0, distance_map.shape[0] - 1)
            y = np.clip(y, 0, distance_map.shape[1] - 1)
            x = np.clip(x, 0, distance_map.shape[2] - 1)
            return distance_map[z, y, x]
        
        # Use maximum distance in neighborhood as refined estimate
        # This helps with noisy distance transforms
        max_radius = np.max(neighborhood['distances'])
        median_radius = np.median(neighborhood['distances'])
        
        # Take weighted average favoring median to reduce noise
        refined_radius = 0.7 * median_radius + 0.3 * max_radius
        
        return refined_radius
    
    def _get_local_neighborhood(self, position: np.ndarray, distance_map: np.ndarray) -> Dict:
        """Get local neighborhood around position"""
        z, y, x = position.astype(int)
        radius = int(self.neighborhood_radius)
        
        # Define neighborhood bounds
        z_min = max(0, z - radius)
        z_max = min(distance_map.shape[0], z + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(distance_map.shape[1], y + radius + 1)
        x_min = max(0, x - radius)
        x_max = min(distance_map.shape[2], x + radius + 1)
        
        # Extract neighborhood
        local_region = distance_map[z_min:z_max, y_min:y_max, x_min:x_max]
        
        # Get coordinates of all points in neighborhood
        coords = np.mgrid[z_min:z_max, y_min:y_max, x_min:x_max]
        coords = coords.reshape(3, -1).T
        
        # Filter to points within radius
        distances_to_center = np.linalg.norm(coords - position, axis=1)
        within_radius = distances_to_center <= self.neighborhood_radius
        
        neighborhood_coords = coords[within_radius]
        neighborhood_distances = local_region.flat[within_radius]
        
        # Only include points with positive distance (inside vessel)
        positive_mask = neighborhood_distances > 0
        
        return {
            'positions': neighborhood_coords[positive_mask],
            'distances': neighborhood_distances[positive_mask]
        }
    
    def _calculate_local_curvature(self, position: np.ndarray, skeleton: np.ndarray) -> float:
        """Calculate local curvature using skeleton geometry"""
        # Find nearby skeleton points
        skeleton_points = np.argwhere(skeleton)
        
        if len(skeleton_points) < 3:
            return 0.0
        
        # Find closest skeleton points
        distances = np.linalg.norm(skeleton_points - position, axis=1)
        closest_indices = np.argsort(distances)[:min(5, len(distances))]
        closest_points = skeleton_points[closest_indices]
        
        if len(closest_points) < 3:
            return 0.0
        
        # Fit local curve and calculate curvature
        try:
            # Center points around position
            centered_points = closest_points - position
            
            # Use PCA to find principal direction
            pca = PCA(n_components=1)
            pca.fit(centered_points)
            principal_direction = pca.components_[0]
            
            # Project points onto principal direction
            projections = np.dot(centered_points, principal_direction)
            
            # Sort by projection
            sorted_indices = np.argsort(projections)
            sorted_points = closest_points[sorted_indices]
            
            # Calculate discrete curvature
            if len(sorted_points) >= 3:
                # Use three consecutive points
                p1, p2, p3 = sorted_points[0], sorted_points[1], sorted_points[2]
                
                v1 = p2 - p1
                v2 = p3 - p2
                
                # Calculate curvature
                cross_product = np.cross(v1, v2)
                cross_magnitude = np.linalg.norm(cross_product)
                v1_magnitude = np.linalg.norm(v1)
                
                if v1_magnitude > 0:
                    curvature = cross_magnitude / (v1_magnitude ** 3)
                else:
                    curvature = 0.0
            else:
                curvature = 0.0
                
        except Exception:
            curvature = 0.0
        
        return curvature
    
    def _calculate_vessel_direction(self, position: np.ndarray, skeleton: np.ndarray) -> np.ndarray:
        """Calculate local vessel direction (tangent vector)"""
        # Find nearby skeleton points
        skeleton_points = np.argwhere(skeleton)
        
        if len(skeleton_points) < 2:
            return np.array([0.0, 0.0, 1.0])  # Default direction
        
        # Find closest skeleton points
        distances = np.linalg.norm(skeleton_points - position, axis=1)
        closest_indices = np.argsort(distances)[:min(3, len(distances))]
        closest_points = skeleton_points[closest_indices]
        
        if len(closest_points) < 2:
            return np.array([0.0, 0.0, 1.0])
        
        # Calculate direction using PCA
        try:
            centered_points = closest_points - np.mean(closest_points, axis=0)
            pca = PCA(n_components=1)
            pca.fit(centered_points)
            direction = pca.components_[0]
            
            # Normalize
            direction = direction / np.linalg.norm(direction)
            
        except Exception:
            direction = np.array([0.0, 0.0, 1.0])
        
        return direction
    
    def _extract_structural_attributes(self,
                                     position: np.ndarray,
                                     skeleton: np.ndarray,
                                     all_positions: np.ndarray,
                                     node_index: int) -> Dict:
        """Extract structural attributes"""
        # Degree will be calculated during edge creation
        degree = 0  # Placeholder
        
        # Distance to other nodes
        if len(all_positions) > 1:
            other_positions = np.delete(all_positions, node_index, axis=0)
            distances_to_others = np.linalg.norm(other_positions - position, axis=1)
            nearest_neighbor_distance = np.min(distances_to_others)
        else:
            nearest_neighbor_distance = float('inf')
        
        # Local density (number of nearby nodes)
        if len(all_positions) > 1:
            distances_to_all = np.linalg.norm(all_positions - position, axis=1)
            local_density = np.sum(distances_to_all <= 5.0) - 1  # Exclude self
        else:
            local_density = 0
        
        return {
            'degree': degree,  # Will be updated during edge creation
            'nearest_neighbor_distance': nearest_neighbor_distance,
            'local_node_density': local_density,
        }
    
    def _extract_context_attributes(self,
                                  position: np.ndarray,
                                  prediction_probabilities: Optional[np.ndarray],
                                  confidence_map: Optional[np.ndarray],
                                  original_mask: Optional[np.ndarray]) -> Dict:
        """Extract context attributes for predicted masks"""
        z, y, x = position.astype(int)
        
        # Ensure coordinates are within bounds
        if prediction_probabilities is not None:
            z = np.clip(z, 0, prediction_probabilities.shape[0] - 1)
            y = np.clip(y, 0, prediction_probabilities.shape[1] - 1)
            x = np.clip(x, 0, prediction_probabilities.shape[2] - 1)
        
        context_attrs = {}
        
        # U-Net confidence at node location
        if prediction_probabilities is not None:
            unet_confidence = prediction_probabilities[z, y, x]
            context_attrs['unet_confidence'] = float(unet_confidence)
            
            # Local confidence statistics
            local_stats = self._calculate_local_confidence_stats(
                position, prediction_probabilities
            )
            context_attrs.update(local_stats)
        
        # Confidence map value
        if confidence_map is not None:
            if (0 <= z < confidence_map.shape[0] and 
                0 <= y < confidence_map.shape[1] and 
                0 <= x < confidence_map.shape[2]):
                confidence_value = confidence_map[z, y, x]
                context_attrs['confidence_map_value'] = float(confidence_value)
        
        # Uncertainty measure
        if prediction_probabilities is not None:
            uncertainty = self._calculate_local_uncertainty(position, prediction_probabilities)
            context_attrs['uncertainty_measure'] = uncertainty
        
        return context_attrs
    
    def _calculate_local_confidence_stats(self,
                                        position: np.ndarray,
                                        prediction_probabilities: np.ndarray) -> Dict:
        """Calculate local confidence statistics in 3x3x3 neighborhood"""
        z, y, x = position.astype(int)
        
        # Define neighborhood bounds
        z_min = max(0, z - 1)
        z_max = min(prediction_probabilities.shape[0], z + 2)
        y_min = max(0, y - 1)
        y_max = min(prediction_probabilities.shape[1], y + 2)
        x_min = max(0, x - 1)
        x_max = min(prediction_probabilities.shape[2], x + 2)
        
        # Extract neighborhood
        neighborhood = prediction_probabilities[z_min:z_max, y_min:y_max, x_min:x_max]
        neighborhood_flat = neighborhood.flatten()
        
        return {
            'local_confidence_mean': float(np.mean(neighborhood_flat)),
            'local_confidence_std': float(np.std(neighborhood_flat)),
            'local_confidence_min': float(np.min(neighborhood_flat)),
            'local_confidence_max': float(np.max(neighborhood_flat)),
        }
    
    def _calculate_local_uncertainty(self,
                                   position: np.ndarray,
                                   prediction_probabilities: np.ndarray) -> float:
        """Calculate local uncertainty using entropy"""
        # Get local neighborhood
        z, y, x = position.astype(int)
        radius = 2
        
        z_min = max(0, z - radius)
        z_max = min(prediction_probabilities.shape[0], z + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(prediction_probabilities.shape[1], y + radius + 1)
        x_min = max(0, x - radius)
        x_max = min(prediction_probabilities.shape[2], x + radius + 1)
        
        neighborhood = prediction_probabilities[z_min:z_max, y_min:y_max, x_min:x_max]
        
        # Calculate entropy as uncertainty measure
        # Convert probabilities to binary entropy
        probs = neighborhood.flatten()
        probs = np.clip(probs, 1e-10, 1 - 1e-10)  # Avoid log(0)
        
        entropy = -np.mean(probs * np.log2(probs) + (1 - probs) * np.log2(1 - probs))
        
        return float(entropy)