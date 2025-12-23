"""
Edge Creation and Attribute Computation for Vascular Graphs
Implements edge generation following vessel connectivity and attribute extraction
"""

import numpy as np
from scipy.spatial import KDTree, distance_matrix
from scipy.ndimage import binary_dilation
from typing import List, Dict, Tuple, Optional
import networkx as nx
from collections import deque
import logging
from tqdm import tqdm


class EdgeCreator:
    """
    Create edges between nodes following vessel connectivity
    """
    
    def __init__(self,
                 max_edge_length: float = 10.0,
                 connectivity_tolerance: float = 2.0):
        """
        Initialize edge creator
        
        Args:
            max_edge_length: Maximum allowed edge length (voxels)
            connectivity_tolerance: Tolerance for connectivity detection
        """
        self.max_edge_length = max_edge_length
        self.connectivity_tolerance = connectivity_tolerance
        self.logger = logging.getLogger(__name__)
    
    def create_edges(self,
                    nodes: Dict,
                    node_attributes: List[Dict],
                    centerline_data: Dict,
                    distance_map: np.ndarray,
                    voxel_spacing: Optional[Tuple[float, float, float]] = None) -> Dict:
        """
        Create edges between nodes following vessel connectivity
        
        Args:
            nodes: Node placement results
            node_attributes: Node attributes
            centerline_data: Centerline extraction results
            distance_map: Distance transform
            voxel_spacing: Voxel spacing in mm
            
        Returns:
            Dictionary containing edges and their attributes
        """
        positions = nodes['positions']
        skeleton = centerline_data['skeleton']
        
        if len(positions) < 2:
            return {
                'edges': [],
                'edge_attributes': [],
                'adjacency_matrix': np.zeros((len(positions), len(positions))),
                'connectivity_graph': {}
            }
        
        self.logger.info(f"Creating edges for {len(positions)} nodes")
        
        # Step 1: Build skeleton connectivity graph
        skeleton_graph = self._build_skeleton_graph(skeleton)
        
        # Step 2: Find valid connections between nodes
        edges = self._find_node_connections(positions, skeleton, skeleton_graph)
        
        # Step 3: Validate and filter edges
        validated_edges = self._validate_edges(edges, positions, distance_map)
        
        # Step 4: Extract edge attributes
        edge_attributes = self._extract_edge_attributes(
            validated_edges, positions, skeleton, distance_map, voxel_spacing
        )
        
        # Step 5: Update node degrees
        updated_node_attributes = self._update_node_degrees(
            node_attributes, validated_edges
        )
        
        # Step 6: Build adjacency structures
        adjacency_matrix = self._build_adjacency_matrix(validated_edges, len(positions))
        connectivity_graph = self._build_connectivity_graph(validated_edges, len(positions))
        
        return {
            'edges': validated_edges,
            'edge_attributes': edge_attributes,
            'adjacency_matrix': adjacency_matrix,
            'connectivity_graph': connectivity_graph,
            'updated_node_attributes': updated_node_attributes
        }
    
    def _build_skeleton_graph(self, skeleton: np.ndarray) -> nx.Graph:
        """Build NetworkX graph from skeleton connectivity"""
        skeleton_points = np.argwhere(skeleton)
        
        if len(skeleton_points) == 0:
            return nx.Graph()
        
        # Create mapping from coordinates to indices
        coord_to_idx = {tuple(point): i for i, point in enumerate(skeleton_points)}
        
        # Build graph
        G = nx.Graph()
        G.add_nodes_from(range(len(skeleton_points)))
        
        # Add edges based on 26-connectivity
        for i, point in enumerate(skeleton_points):
            z, y, x = point
            
            # Check 26-connected neighbors
            for dz in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dz == 0 and dy == 0 and dx == 0:
                            continue
                        
                        neighbor = (z + dz, y + dy, x + dx)
                        
                        if neighbor in coord_to_idx:
                            j = coord_to_idx[neighbor]
                            if i != j:
                                G.add_edge(i, j, weight=np.linalg.norm([dz, dy, dx]))
        
        return G
    
    def _find_node_connections(self,
                             positions: np.ndarray,
                             skeleton: np.ndarray,
                             skeleton_graph: nx.Graph) -> List[Tuple[int, int]]:
        """Find valid connections between nodes following skeleton paths"""
        skeleton_points = np.argwhere(skeleton)
        
        if len(skeleton_points) == 0:
            return []
        
        # Build KDTree for skeleton points
        skeleton_tree = KDTree(skeleton_points)
        
        edges = []
        
        # More efficient approach: only check nearby nodes
        node_tree = KDTree(positions)
        
        # Use tqdm for progress tracking
        pbar = tqdm(enumerate(positions), total=len(positions), desc="Finding edges", leave=False)
        
        for i, pos1 in pbar:
            # Find nearby nodes within max_edge_length
            nearby_indices = node_tree.query_ball_point(pos1, r=self.max_edge_length)
            
            # Update progress bar with edge count
            pbar.set_postfix({'edges': len(edges), 'checking': len(nearby_indices)})
            
            for j in nearby_indices:
                if j <= i:  # Skip self and already processed pairs
                    continue
                
                pos2 = positions[j]
                
                # Quick distance check
                if np.linalg.norm(pos2 - pos1) > self.max_edge_length:
                    continue
                
                # Check if nodes can be connected
                if self._are_nodes_connected(pos1, pos2, skeleton, skeleton_tree, skeleton_graph):
                    edges.append((i, j))
        
        pbar.close()
        return edges
    
    def _are_nodes_connected(self,
                           pos1: np.ndarray,
                           pos2: np.ndarray,
                           skeleton: np.ndarray,
                           skeleton_tree: KDTree,
                           skeleton_graph: nx.Graph) -> bool:
        """Check if two nodes are connected via skeleton path"""
        # Quick distance check first
        direct_distance = np.linalg.norm(pos2 - pos1)
        if direct_distance > self.max_edge_length:
            return False
        
        # For very close nodes, assume connected
        if direct_distance < self.connectivity_tolerance:
            return True
        
        # Find closest skeleton points to each node
        _, idx1 = skeleton_tree.query(pos1, k=1)
        _, idx2 = skeleton_tree.query(pos2, k=1)
        
        # If same skeleton point, they're connected
        if idx1 == idx2:
            return True
        
        # For larger graphs, use a simple heuristic instead of path finding
        if len(skeleton_graph) > 1000:
            # Just check if nodes are reasonably close and aligned
            return direct_distance <= self.max_edge_length * 0.8
        
        # Check if there's a path in skeleton graph
        try:
            if nx.has_path(skeleton_graph, idx1, idx2):
                # Calculate path length
                path_length = nx.shortest_path_length(skeleton_graph, idx1, idx2)
                
                # Only allow connection if path is reasonable
                return path_length <= self.max_edge_length * 1.5
            else:
                return False
        except (nx.NetworkXError, nx.NodeNotFound):
            return False
    
    def _validate_edges(self,
                       edges: List[Tuple[int, int]],
                       positions: np.ndarray,
                       distance_map: np.ndarray) -> List[Tuple[int, int]]:
        """Validate edges based on geometric and anatomical constraints"""
        validated_edges = []
        
        for edge in edges:
            i, j = edge
            pos1, pos2 = positions[i], positions[j]
            
            # Check edge length constraint
            edge_length = np.linalg.norm(pos2 - pos1)
            if edge_length > self.max_edge_length:
                continue
            
            # Check if edge passes through valid vessel region
            if self._edge_passes_through_vessel(pos1, pos2, distance_map):
                validated_edges.append(edge)
        
        self.logger.info(f"Validated {len(validated_edges)} edges from {len(edges)} candidates")
        return validated_edges
    
    def _edge_passes_through_vessel(self,
                                  pos1: np.ndarray,
                                  pos2: np.ndarray,
                                  distance_map: np.ndarray) -> bool:
        """Check if edge path passes through vessel region"""
        # Sample points along edge
        num_samples = max(3, int(np.linalg.norm(pos2 - pos1)))
        t_values = np.linspace(0, 1, num_samples)
        
        for t in t_values:
            sample_pos = pos1 + t * (pos2 - pos1)
            z, y, x = sample_pos.astype(int)
            
            # Check bounds
            if (0 <= z < distance_map.shape[0] and
                0 <= y < distance_map.shape[1] and
                0 <= x < distance_map.shape[2]):
                
                # Check if point is inside vessel (positive distance)
                if distance_map[z, y, x] <= 0:
                    return False
            else:
                return False
        
        return True
    
    def _extract_edge_attributes(self,
                               edges: List[Tuple[int, int]],
                               positions: np.ndarray,
                               skeleton: np.ndarray,
                               distance_map: np.ndarray,
                               voxel_spacing: Optional[Tuple[float, float, float]]) -> List[Dict]:
        """Extract attributes for each edge"""
        edge_attributes = []
        
        # Use tqdm for progress tracking
        pbar = tqdm(edges, desc="Extracting edge attributes", leave=False)
        
        for edge in pbar:
            i, j = edge
            pos1, pos2 = positions[i], positions[j]
            
            # Basic edge properties
            direction_vector = pos2 - pos1
            euclidean_length = np.linalg.norm(direction_vector)
            direction_unit = direction_vector / euclidean_length if euclidean_length > 0 else np.zeros(3)
            
            # Physical length
            if voxel_spacing is not None:
                physical_length = euclidean_length * np.mean(voxel_spacing)
            else:
                physical_length = euclidean_length
            
            # Calculate path length along skeleton
            path_length = self._calculate_skeleton_path_length(pos1, pos2, skeleton)
            
            # Average radius along edge
            average_radius = self._calculate_average_radius_along_edge(pos1, pos2, distance_map)
            
            # Path curvature
            path_curvature = self._calculate_path_curvature(pos1, pos2, skeleton)
            
            # Confidence score (for predicted masks)
            confidence_score = 1.0  # Default, will be updated if confidence map available
            
            edge_attrs = {
                'source_node': i,
                'target_node': j,
                'euclidean_length': euclidean_length,
                'physical_length': physical_length,
                'path_length': path_length,
                'direction_vector': direction_unit,
                'average_radius': average_radius,
                'path_curvature': path_curvature,
                'confidence_score': confidence_score,
            }
            
            edge_attributes.append(edge_attrs)
            
            # Update progress bar every 50 edges
            if len(edge_attributes) % 50 == 0:
                pbar.set_postfix({'processed': len(edge_attributes)})
        
        pbar.close()
        return edge_attributes
    
    def _calculate_skeleton_path_length(self,
                                      pos1: np.ndarray,
                                      pos2: np.ndarray,
                                      skeleton: np.ndarray) -> float:
        """Calculate actual path length along skeleton"""
        # For now, use Euclidean distance as approximation
        # Could be improved with actual path tracing
        return np.linalg.norm(pos2 - pos1)
    
    def _calculate_average_radius_along_edge(self,
                                           pos1: np.ndarray,
                                           pos2: np.ndarray,
                                           distance_map: np.ndarray) -> float:
        """Calculate average vessel radius along edge path"""
        # Sample points along edge
        num_samples = max(3, int(np.linalg.norm(pos2 - pos1)))
        t_values = np.linspace(0, 1, num_samples)
        
        radii = []
        for t in t_values:
            sample_pos = pos1 + t * (pos2 - pos1)
            z, y, x = sample_pos.astype(int)
            
            # Check bounds
            if (0 <= z < distance_map.shape[0] and
                0 <= y < distance_map.shape[1] and
                0 <= x < distance_map.shape[2]):
                radii.append(distance_map[z, y, x])
        
        return np.mean(radii) if radii else 0.0
    
    def _calculate_path_curvature(self,
                                pos1: np.ndarray,
                                pos2: np.ndarray,
                                skeleton: np.ndarray) -> float:
        """Calculate total angular change along edge path"""
        # Simple approximation using straight line
        # Could be improved with actual skeleton path analysis
        return 0.0  # Placeholder
    
    def _update_node_degrees(self,
                           node_attributes: List[Dict],
                           edges: List[Tuple[int, int]]) -> List[Dict]:
        """Update node degree information based on edges"""
        updated_attributes = []
        
        # Count degree for each node
        degree_count = [0] * len(node_attributes)
        for edge in edges:
            i, j = edge
            degree_count[i] += 1
            degree_count[j] += 1
        
        # Update attributes
        for i, attrs in enumerate(node_attributes):
            updated_attrs = attrs.copy()
            updated_attrs['degree'] = degree_count[i]
            updated_attributes.append(updated_attrs)
        
        return updated_attributes
    
    def _build_adjacency_matrix(self,
                              edges: List[Tuple[int, int]],
                              num_nodes: int) -> np.ndarray:
        """Build adjacency matrix from edge list"""
        adjacency = np.zeros((num_nodes, num_nodes), dtype=bool)
        
        for edge in edges:
            i, j = edge
            adjacency[i, j] = True
            adjacency[j, i] = True  # Undirected graph
        
        return adjacency
    
    def _build_connectivity_graph(self,
                                edges: List[Tuple[int, int]],
                                num_nodes: int) -> Dict:
        """Build connectivity graph as adjacency list"""
        graph = {i: [] for i in range(num_nodes)}
        
        for edge in edges:
            i, j = edge
            graph[i].append(j)
            graph[j].append(i)
        
        return graph