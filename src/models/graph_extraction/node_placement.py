"""
Strategic Node Placement for Vascular Graph Construction
Implements adaptive node placement along vessel centerlines with critical point detection
"""

import numpy as np
from scipy.spatial import distance_matrix, KDTree
from scipy.interpolate import splprep, splev
from typing import List, Tuple, Dict, Optional
import logging


class NodePlacer:
    """
    Strategic node placement for vascular graph construction
    """
    
    def __init__(self,
                 base_density: float = 2.0,  # Base spacing in voxels
                 curvature_sensitivity: float = 7.0,  # Alpha parameter for curvature adaptation
                 min_distance: float = 0.5,  # Minimum distance between nodes
                 max_distance: float = 3.0,  # Maximum distance between nodes
                 bifurcation_buffer: int = 2,  # Buffer around bifurcations
                 max_nodes: int = 1000):  # Maximum number of nodes to prevent memory issues
        """
        Initialize node placer
        
        Args:
            base_density: Base node spacing in voxels
            curvature_sensitivity: Sensitivity to curvature for adaptive sampling
            min_distance: Minimum allowed distance between nodes
            max_distance: Maximum allowed distance between nodes
            bifurcation_buffer: Number of buffer nodes around bifurcations
        """
        self.base_density = base_density
        self.curvature_sensitivity = curvature_sensitivity
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.bifurcation_buffer = bifurcation_buffer
        self.max_nodes = max_nodes
        self.logger = logging.getLogger(__name__)
    
    def place_nodes(self,
                    centerline_data: Dict,
                    voxel_spacing: Optional[Tuple[float, float, float]] = None) -> Dict:
        """
        Place nodes strategically along vessel centerlines
        
        Args:
            centerline_data: Output from CenterlineExtractor
            voxel_spacing: Voxel spacing in mm (z, y, x)
            
        Returns:
            Dictionary containing node placement results
        """
        skeleton = centerline_data['skeleton']
        skeleton_points = centerline_data['centerline_points']
        distance_map = centerline_data['distance_map']
        connectivity = centerline_data['connectivity']
        
        if len(skeleton_points) == 0:
            return {
                'nodes': [],
                'node_types': [],
                'node_attributes': [],
                'critical_points': {'bifurcations': [], 'endpoints': []}
            }
        
        self.logger.info(f"Placing nodes on centerline with {len(skeleton_points)} skeleton points")
        
        # Adaptive density adjustment for large volumes
        adaptive_base_density = self.base_density
        adaptive_min_distance = self.min_distance
        
        if len(skeleton_points) > 2000:  # Large volume
            # Reduce density to keep node count manageable
            scale_factor = min(3.0, len(skeleton_points) / 2000)
            adaptive_base_density = self.base_density * scale_factor
            adaptive_min_distance = self.min_distance * scale_factor
            self.logger.info(f"Large volume detected, adjusting density: base={adaptive_base_density:.2f}, min_dist={adaptive_min_distance:.2f}")
        
        # Step 1: Detect critical points
        critical_points = self._detect_critical_points(skeleton_points, connectivity['neighbor_graph'])
        
        # Step 2: Calculate curvature along centerline
        curvature_map = self._calculate_curvature(skeleton_points, connectivity['neighbor_graph'])
        
        # Step 3: Adaptive node placement
        nodes = self._adaptive_node_placement(
            skeleton_points, 
            curvature_map, 
            critical_points,
            distance_map,
            voxel_spacing,
            adaptive_base_density,
            adaptive_min_distance
        )
        
        # Step 4: Node validation and refinement
        validated_nodes = self._validate_nodes(nodes, distance_map)
        
        # Step 5: Extract node attributes
        node_attributes = self._extract_node_attributes(
            validated_nodes, 
            distance_map, 
            critical_points,
            voxel_spacing
        )
        
        return {
            'positions': validated_nodes['positions'],
            'types': validated_nodes['types'],
            'node_attributes': node_attributes,
            'critical_points': critical_points
        }
    
    def _detect_critical_points(self, skeleton_points: np.ndarray, neighbor_graph: Dict) -> Dict:
        """
        Detect bifurcations and endpoints in skeleton
        
        Args:
            skeleton_points: Array of skeleton point coordinates
            neighbor_graph: Adjacency information
            
        Returns:
            Dictionary with bifurcations and endpoints
        """
        bifurcations = []
        endpoints = []
        
        for i, point in enumerate(skeleton_points):
            neighbor_count = len(neighbor_graph.get(i, []))
            
            if neighbor_count == 1:
                endpoints.append({'index': i, 'position': point})
            elif neighbor_count > 2:
                bifurcations.append({'index': i, 'position': point})
        
        self.logger.debug(f"Detected {len(bifurcations)} bifurcations and {len(endpoints)} endpoints")
        
        return {
            'bifurcations': bifurcations,
            'endpoints': endpoints
        }
    
    def _calculate_curvature(self, skeleton_points: np.ndarray, neighbor_graph: Dict) -> np.ndarray:
        """
        Calculate discrete curvature at each skeleton point
        
        Args:
            skeleton_points: Array of skeleton point coordinates
            neighbor_graph: Adjacency information
            
        Returns:
            Array of curvature values
        """
        curvature = np.zeros(len(skeleton_points))
        
        for i, point in enumerate(skeleton_points):
            neighbors = neighbor_graph.get(i, [])
            
            if len(neighbors) >= 2:
                # Get neighboring points for curvature calculation
                neighbor_positions = skeleton_points[neighbors]
                
                # Calculate curvature using discrete approximation
                if len(neighbor_positions) >= 2:
                    # Use two neighbors for angle calculation
                    v1 = neighbor_positions[0] - point
                    v2 = neighbor_positions[1] - point
                    
                    # Normalize vectors
                    v1_norm = np.linalg.norm(v1)
                    v2_norm = np.linalg.norm(v2)
                    
                    if v1_norm > 0 and v2_norm > 0:
                        v1_unit = v1 / v1_norm
                        v2_unit = v2 / v2_norm
                        
                        # Calculate angle between vectors
                        cos_angle = np.clip(np.dot(v1_unit, v2_unit), -1.0, 1.0)
                        angle = np.arccos(cos_angle)
                        
                        # Curvature approximation
                        curvature[i] = angle
            
            elif len(neighbors) == 1:
                # For points with one neighbor, check if we can use second-order neighbors
                curvature[i] = self._calculate_endpoint_curvature(i, skeleton_points, neighbor_graph)
        
        return curvature
    
    def _calculate_endpoint_curvature(self, point_idx: int, skeleton_points: np.ndarray, neighbor_graph: Dict) -> float:
        """Calculate curvature for endpoint using path tracing"""
        # Trace path from endpoint to get direction vectors
        path = self._trace_path_from_point(point_idx, skeleton_points, neighbor_graph, max_length=5)
        
        if len(path) < 3:
            return 0.0
        
        # Calculate curvature along the path
        total_curvature = 0.0
        count = 0
        
        for i in range(1, len(path) - 1):
            v1 = skeleton_points[path[i]] - skeleton_points[path[i-1]]
            v2 = skeleton_points[path[i+1]] - skeleton_points[path[i]]
            
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            
            if v1_norm > 0 and v2_norm > 0:
                v1_unit = v1 / v1_norm
                v2_unit = v2 / v2_norm
                
                cos_angle = np.clip(np.dot(v1_unit, v2_unit), -1.0, 1.0)
                angle = np.arccos(cos_angle)
                total_curvature += angle
                count += 1
        
        return total_curvature / count if count > 0 else 0.0
    
    def _trace_path_from_point(self, start_idx: int, skeleton_points: np.ndarray, neighbor_graph: Dict, max_length: int) -> List[int]:
        """Trace path from a point along the skeleton"""
        path = [start_idx]
        current = start_idx
        visited = {start_idx}
        
        while len(path) < max_length:
            neighbors = [n for n in neighbor_graph.get(current, []) if n not in visited]
            
            if not neighbors:
                break
            
            # Choose the neighbor (for linear structures, should be only one)
            next_point = neighbors[0]
            path.append(next_point)
            visited.add(next_point)
            current = next_point
        
        return path
    
    def _adaptive_node_placement(self,
                                skeleton_points: np.ndarray,
                                curvature_map: np.ndarray,
                                critical_points: Dict,
                                distance_map: np.ndarray,
                                voxel_spacing: Optional[Tuple[float, float, float]],
                                base_density: float,
                                min_distance: float) -> Dict:
        """
        Place nodes adaptively based on curvature and critical points
        
        Args:
            skeleton_points: Skeleton point coordinates
            curvature_map: Curvature values at each point
            critical_points: Bifurcations and endpoints
            distance_map: Distance transform
            voxel_spacing: Voxel spacing
            
        Returns:
            Dictionary with node positions and types
        """
        node_positions = []
        node_types = []
        node_indices = []  # Track original skeleton indices
        
        # First, add all critical points
        for bifurcation in critical_points['bifurcations']:
            node_positions.append(bifurcation['position'])
            node_types.append('bifurcation')
            node_indices.append(bifurcation['index'])
            
            # Add buffer nodes around bifurcations
            buffer_nodes = self._place_bifurcation_buffer_nodes(
                bifurcation, skeleton_points, curvature_map
            )
            for buffer_node in buffer_nodes:
                node_positions.append(buffer_node['position'])
                node_types.append('buffer')
                node_indices.append(buffer_node['index'])
        
        for endpoint in critical_points['endpoints']:
            node_positions.append(endpoint['position'])
            node_types.append('endpoint')
            node_indices.append(endpoint['index'])
        
        # Convert to numpy arrays for distance calculations
        if node_positions:
            existing_nodes = np.array(node_positions)
        else:
            existing_nodes = np.empty((0, 3))
        
        # Add regular nodes along vessel segments
        regular_nodes = self._place_regular_nodes(
            skeleton_points, curvature_map, existing_nodes, voxel_spacing, 
            base_density, min_distance
        )
        
        for regular_node in regular_nodes:
            node_positions.append(regular_node['position'])
            node_types.append('regular')
            node_indices.append(regular_node['index'])
        
        return {
            'positions': np.array(node_positions),
            'types': node_types,
            'skeleton_indices': node_indices
        }
    
    def _place_bifurcation_buffer_nodes(self,
                                       bifurcation: Dict,
                                       skeleton_points: np.ndarray,
                                       curvature_map: np.ndarray) -> List[Dict]:
        """Place buffer nodes around bifurcations"""
        buffer_nodes = []
        bifurcation_pos = bifurcation['position']
        
        # Find nearby skeleton points within buffer distance
        distances = np.linalg.norm(skeleton_points - bifurcation_pos, axis=1)
        buffer_candidates = np.where((distances > 0) & (distances <= self.bifurcation_buffer))[0]
        
        for candidate_idx in buffer_candidates:
            buffer_nodes.append({
                'position': skeleton_points[candidate_idx],
                'index': candidate_idx
            })
        
        return buffer_nodes
    
    def _place_regular_nodes(self,
                           skeleton_points: np.ndarray,
                           curvature_map: np.ndarray,
                           existing_nodes: np.ndarray,
                           voxel_spacing: Optional[Tuple[float, float, float]],
                           base_density: float,
                           min_distance: float) -> List[Dict]:
        """Place regular nodes with adaptive density"""
        regular_nodes = []
        
        # Build KDTree for efficient distance queries
        if len(existing_nodes) > 0:
            existing_tree = KDTree(existing_nodes)
        else:
            existing_tree = None
        
        for i, point in enumerate(skeleton_points):
            # Hard limit on number of nodes
            total_nodes = len(existing_nodes) + len(regular_nodes)
            if total_nodes >= self.max_nodes:
                self.logger.warning(f"Reached maximum node limit ({self.max_nodes}), stopping placement")
                break
                
            # Check if point is already covered by existing nodes
            if existing_tree is not None:
                distances_to_existing, _ = existing_tree.query(point.reshape(1, -1), k=1)
                if distances_to_existing[0] < min_distance:
                    continue
            
            # Calculate adaptive spacing based on curvature
            curvature = curvature_map[i]
            adaptive_density = base_density * (1 + self.curvature_sensitivity * curvature)
            
            # Ensure spacing is within bounds
            target_spacing = np.clip(1.0 / adaptive_density, min_distance, self.max_distance)
            
            # Check if we should place a node here
            should_place = True
            
            if existing_tree is not None:
                distances_to_existing, _ = existing_tree.query(point.reshape(1, -1), k=1)
                if distances_to_existing[0] < target_spacing:
                    should_place = False
            
            if should_place:
                regular_nodes.append({
                    'position': point,
                    'index': i
                })
                
                # Update existing nodes for next iterations
                if existing_tree is not None:
                    # Rebuild tree with new node (inefficient but simple)
                    new_nodes = np.vstack([existing_nodes, point.reshape(1, -1)])
                    existing_tree = KDTree(new_nodes)
                    existing_nodes = new_nodes
                else:
                    existing_nodes = point.reshape(1, -1)
                    existing_tree = KDTree(existing_nodes)
        
        return regular_nodes
    
    def _validate_nodes(self, nodes: Dict, distance_map: np.ndarray) -> Dict:
        """Validate and refine node placement"""
        positions = nodes['positions']
        types = nodes['types']
        skeleton_indices = nodes['skeleton_indices']
        
        if len(positions) == 0:
            return nodes
        
        # Remove nodes that are too close to each other
        valid_indices = []
        used_positions = []
        
        # Prioritize critical points
        type_priority = {'bifurcation': 0, 'endpoint': 1, 'buffer': 2, 'regular': 3}
        
        # Sort by priority
        sorted_indices = sorted(range(len(positions)), 
                              key=lambda i: type_priority.get(types[i], 4))
        
        for idx in sorted_indices:
            pos = positions[idx]
            
            # Check distance to already accepted nodes
            too_close = False
            for used_pos in used_positions:
                if np.linalg.norm(pos - used_pos) < self.min_distance:
                    too_close = True
                    break
            
            if not too_close:
                valid_indices.append(idx)
                used_positions.append(pos)
        
        # Filter to valid nodes
        validated_positions = positions[valid_indices]
        validated_types = [types[i] for i in valid_indices]
        validated_indices = [skeleton_indices[i] for i in valid_indices]
        
        self.logger.info(f"Validated {len(validated_positions)} nodes from {len(positions)} candidates")
        
        return {
            'positions': validated_positions,
            'types': validated_types,
            'skeleton_indices': validated_indices
        }
    
    def _extract_node_attributes(self,
                               nodes: Dict,
                               distance_map: np.ndarray,
                               critical_points: Dict,
                               voxel_spacing: Optional[Tuple[float, float, float]]) -> List[Dict]:
        """Extract attributes for each node"""
        attributes = []
        positions = nodes['positions']
        types = nodes['types']
        
        for i, (pos, node_type) in enumerate(zip(positions, types)):
            z, y, x = pos.astype(int)
            
            # Ensure coordinates are within bounds
            z = np.clip(z, 0, distance_map.shape[0] - 1)
            y = np.clip(y, 0, distance_map.shape[1] - 1)
            x = np.clip(x, 0, distance_map.shape[2] - 1)
            
            # Extract basic attributes
            radius_voxels = distance_map[z, y, x]
            
            # Convert to physical coordinates if spacing provided
            if voxel_spacing is not None:
                physical_coords = pos * np.array(voxel_spacing)
                radius_mm = radius_voxels * np.mean(voxel_spacing)
            else:
                physical_coords = pos
                radius_mm = radius_voxels
            
            # Calculate degree (will be updated during edge creation)
            degree = 0  # Placeholder
            
            # Distance to nearest bifurcation
            bifurcation_distance = self._calculate_bifurcation_distance(
                pos, critical_points['bifurcations']
            )
            
            node_attributes = {
                'coordinates_voxel': pos,
                'coordinates_physical': physical_coords,
                'radius_voxels': radius_voxels,
                'radius_mm': radius_mm,
                'node_type': node_type,
                'degree': degree,  # Will be updated
                'bifurcation_distance': bifurcation_distance,
            }
            
            attributes.append(node_attributes)
        
        return attributes
    
    def _calculate_bifurcation_distance(self, position: np.ndarray, bifurcations: List[Dict]) -> float:
        """Calculate distance to nearest bifurcation"""
        if not bifurcations:
            return float('inf')
        
        bifurcation_positions = np.array([b['position'] for b in bifurcations])
        distances = np.linalg.norm(bifurcation_positions - position, axis=1)
        
        return np.min(distances)