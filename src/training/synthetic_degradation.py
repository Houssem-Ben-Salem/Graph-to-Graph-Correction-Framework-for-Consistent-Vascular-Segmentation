"""
Synthetic Graph Degradation Pipeline
Implements realistic graph degradation to simulate U-Net prediction errors for training
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
import copy
import logging
from enum import Enum
from dataclasses import dataclass
import random

from src.models.graph_extraction.vascular_graph import VascularGraph


class DegradationType(Enum):
    """Types of degradation that can be applied to graphs"""
    SPATIAL_NOISE = "spatial_noise"
    RADIUS_DISTORTION = "radius_distortion"
    TOPOLOGY_BREAKS = "topology_breaks"
    MISSING_CONNECTIONS = "missing_connections"
    FALSE_CONNECTIONS = "false_connections"
    NODE_TYPE_ERRORS = "node_type_errors"
    FEATURE_CORRUPTION = "feature_corruption"
    VESSEL_FRAGMENTATION = "vessel_fragmentation"


@dataclass
class DegradationConfig:
    """Configuration for graph degradation"""
    
    # Spatial degradation
    spatial_noise_std: float = 0.5
    position_jitter_max: float = 2.0
    
    # Radius degradation
    radius_noise_factor: float = 0.3
    radius_outlier_prob: float = 0.05
    
    # Topology degradation
    edge_removal_prob: float = 0.1
    false_edge_prob: float = 0.05
    connection_break_prob: float = 0.15
    
    # Node degradation
    node_type_error_prob: float = 0.1
    node_removal_prob: float = 0.02
    
    # Feature corruption
    feature_noise_prob: float = 0.2
    feature_corruption_strength: float = 0.4
    
    # Vessel fragmentation
    fragmentation_prob: float = 0.08
    fragment_min_length: int = 5
    
    # Global parameters
    degradation_strength: float = 0.3  # Overall degradation intensity [0, 1]
    preserve_connectivity: bool = True
    maintain_graph_validity: bool = True


class SyntheticDegradationPipeline:
    """
    Comprehensive pipeline for generating synthetic graph degradations
    that simulate realistic U-Net prediction errors
    """
    
    def __init__(self, config: Optional[DegradationConfig] = None):
        """
        Initialize degradation pipeline
        
        Args:
            config: Degradation configuration parameters
        """
        self.config = config or DegradationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize degradation methods
        self.degradation_methods = {
            DegradationType.SPATIAL_NOISE: self._apply_spatial_noise,
            DegradationType.RADIUS_DISTORTION: self._apply_radius_distortion,
            DegradationType.TOPOLOGY_BREAKS: self._apply_topology_breaks,
            DegradationType.MISSING_CONNECTIONS: self._apply_missing_connections,
            DegradationType.FALSE_CONNECTIONS: self._apply_false_connections,
            DegradationType.NODE_TYPE_ERRORS: self._apply_node_type_errors,
            DegradationType.FEATURE_CORRUPTION: self._apply_feature_corruption,
            DegradationType.VESSEL_FRAGMENTATION: self._apply_vessel_fragmentation
        }
        
    def degrade_graph(self, 
                     gt_graph: VascularGraph, 
                     degradation_level: str = "medium",
                     specific_degradations: Optional[List[DegradationType]] = None,
                     seed: Optional[int] = None) -> Tuple[VascularGraph, Dict]:
        """
        Apply synthetic degradation to create realistic prediction graph
        
        Args:
            gt_graph: Ground truth vascular graph
            degradation_level: "easy", "medium", "hard", or "expert"
            specific_degradations: List of specific degradations to apply
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (degraded_graph, degradation_metadata)
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Adjust config based on degradation level
        config = self._get_level_config(degradation_level)
        
        # Deep copy the graph to avoid modifying original
        degraded_graph = self._deep_copy_graph(gt_graph)
        
        # Track applied degradations
        degradation_metadata = {
            'degradation_level': degradation_level,
            'applied_degradations': [],
            'degradation_stats': {},
            'original_stats': self._get_graph_stats(gt_graph),
            'seed': seed
        }
        
        # Determine which degradations to apply
        if specific_degradations is None:
            degradations_to_apply = self._select_degradations_for_level(degradation_level)
        else:
            degradations_to_apply = specific_degradations
        
        # Apply degradations in order
        for degradation_type in degradations_to_apply:
            if degradation_type in self.degradation_methods:
                try:
                    degraded_graph, stats = self.degradation_methods[degradation_type](
                        degraded_graph, config
                    )
                    degradation_metadata['applied_degradations'].append(degradation_type.value)
                    degradation_metadata['degradation_stats'][degradation_type.value] = stats
                    
                except Exception as e:
                    self.logger.warning(f"Failed to apply {degradation_type.value}: {e}")
        
        # Validate degraded graph
        if config.maintain_graph_validity:
            degraded_graph = self._validate_and_fix_graph(degraded_graph)
        
        # Update metadata with final stats
        degradation_metadata['final_stats'] = self._get_graph_stats(degraded_graph)
        degradation_metadata['degradation_summary'] = self._compute_degradation_summary(
            degradation_metadata['original_stats'], 
            degradation_metadata['final_stats']
        )
        
        # Update graph metadata
        degraded_graph.metadata.update({
            'synthetic_degradation': True,
            'degradation_level': degradation_level,
            'degradation_metadata': degradation_metadata
        })
        
        return degraded_graph, degradation_metadata
    
    def _get_level_config(self, level: str) -> DegradationConfig:
        """Get configuration for specific degradation level"""
        config = copy.deepcopy(self.config)
        
        if level == "easy":
            config.degradation_strength = 0.15
            config.spatial_noise_std = 0.3
            config.edge_removal_prob = 0.05
            config.node_type_error_prob = 0.05
            
        elif level == "medium":
            config.degradation_strength = 0.3
            config.spatial_noise_std = 0.5
            config.edge_removal_prob = 0.1
            config.node_type_error_prob = 0.1
            
        elif level == "hard":
            config.degradation_strength = 0.5
            config.spatial_noise_std = 0.8
            config.edge_removal_prob = 0.15
            config.node_type_error_prob = 0.15
            
        elif level == "expert":
            config.degradation_strength = 0.7
            config.spatial_noise_std = 1.0
            config.edge_removal_prob = 0.2
            config.node_type_error_prob = 0.2
            config.preserve_connectivity = False
        
        return config
    
    def _select_degradations_for_level(self, level: str) -> List[DegradationType]:
        """Select appropriate degradations for each difficulty level"""
        
        base_degradations = [
            DegradationType.SPATIAL_NOISE,
            DegradationType.RADIUS_DISTORTION,
            DegradationType.FEATURE_CORRUPTION
        ]
        
        if level in ["medium", "hard", "expert"]:
            base_degradations.extend([
                DegradationType.MISSING_CONNECTIONS,
                DegradationType.NODE_TYPE_ERRORS
            ])
        
        if level in ["hard", "expert"]:
            base_degradations.extend([
                DegradationType.TOPOLOGY_BREAKS,
                DegradationType.FALSE_CONNECTIONS,
                DegradationType.VESSEL_FRAGMENTATION
            ])
        
        return base_degradations
    
    def _apply_spatial_noise(self, graph: VascularGraph, config: DegradationConfig) -> Tuple[VascularGraph, Dict]:
        """Apply spatial noise to node positions"""
        stats = {'nodes_affected': 0, 'avg_displacement': 0.0}
        
        for node in graph.nodes:
            if 'position' in node and np.random.random() < config.degradation_strength:
                # Convert position to numpy array
                if hasattr(node['position'], 'tolist'):
                    position = np.array(node['position'].tolist())
                else:
                    position = np.array(node['position'])
                
                # Add gaussian noise
                noise = np.random.normal(0, config.spatial_noise_std, 3)
                
                # Add occasional larger displacements
                if np.random.random() < 0.1:
                    noise += np.random.uniform(-config.position_jitter_max, 
                                             config.position_jitter_max, 3)
                
                new_position = position + noise
                node['position'] = new_position.tolist()
                
                stats['nodes_affected'] += 1
                stats['avg_displacement'] += np.linalg.norm(noise)
        
        if stats['nodes_affected'] > 0:
            stats['avg_displacement'] /= stats['nodes_affected']
        
        return graph, stats
    
    def _apply_radius_distortion(self, graph: VascularGraph, config: DegradationConfig) -> Tuple[VascularGraph, Dict]:
        """Apply distortion to vessel radii"""
        stats = {'nodes_affected': 0, 'outliers_created': 0}
        
        for node in graph.nodes:
            if 'radius_voxels' in node and np.random.random() < config.degradation_strength:
                original_radius = float(node['radius_voxels'])
                
                # Apply noise
                noise_factor = 1 + np.random.normal(0, config.radius_noise_factor)
                new_radius = original_radius * noise_factor
                
                # Occasionally create outliers
                if np.random.random() < config.radius_outlier_prob:
                    outlier_factor = np.random.choice([0.1, 5.0])  # Very small or very large
                    new_radius = original_radius * outlier_factor
                    stats['outliers_created'] += 1
                
                # Ensure positive radius
                node['radius_voxels'] = max(0.1, new_radius)
                
                # Update physical radius if present
                if 'radius_mm' in node:
                    node['radius_mm'] = node['radius_voxels'] * 0.764  # Approximate conversion
                
                stats['nodes_affected'] += 1
        
        return graph, stats
    
    def _apply_topology_breaks(self, graph: VascularGraph, config: DegradationConfig) -> Tuple[VascularGraph, Dict]:
        """Apply topology breaks by removing critical connections"""
        stats = {'connections_broken': 0, 'components_created': 0}
        
        # Find critical edges (those whose removal would disconnect the graph)
        critical_edges = self._find_critical_edges(graph)
        
        # Break some connections
        edges_to_remove = []
        for i, edge in enumerate(graph.edges):
            if np.random.random() < config.connection_break_prob:
                # Be more conservative with critical edges if preserve_connectivity is True
                if config.preserve_connectivity and i in critical_edges:
                    if np.random.random() < 0.3:  # Only 30% chance to break critical edges
                        edges_to_remove.append(i)
                else:
                    edges_to_remove.append(i)
        
        # Remove edges (in reverse order to maintain indices)
        for edge_idx in reversed(edges_to_remove):
            graph.edges.pop(edge_idx)
            stats['connections_broken'] += 1
        
        return graph, stats
    
    def _apply_missing_connections(self, graph: VascularGraph, config: DegradationConfig) -> Tuple[VascularGraph, Dict]:
        """Remove edges to simulate missing connections"""
        stats = {'edges_removed': 0}
        
        edges_to_remove = []
        for i, edge in enumerate(graph.edges):
            if np.random.random() < config.edge_removal_prob:
                edges_to_remove.append(i)
        
        # Remove edges (in reverse order)
        for edge_idx in reversed(edges_to_remove):
            graph.edges.pop(edge_idx)
            stats['edges_removed'] += 1
        
        return graph, stats
    
    def _apply_false_connections(self, graph: VascularGraph, config: DegradationConfig) -> Tuple[VascularGraph, Dict]:
        """Add false edges to simulate spurious connections"""
        stats = {'false_edges_added': 0}
        
        num_nodes = len(graph.nodes)
        if num_nodes < 2:
            return graph, stats
        
        # Determine number of false edges to add
        num_false_edges = int(len(graph.edges) * config.false_edge_prob)
        
        existing_connections = set()
        for edge in graph.edges:
            existing_connections.add((edge['source'], edge['target']))
            existing_connections.add((edge['target'], edge['source']))
        
        for _ in range(num_false_edges):
            # Select random nodes
            source = np.random.randint(0, num_nodes)
            target = np.random.randint(0, num_nodes)
            
            if source != target and (source, target) not in existing_connections:
                # Create false edge
                false_edge = {
                    'source': source,
                    'target': target,
                    'euclidean_length': np.random.uniform(1.0, 10.0),
                    'average_radius': np.random.uniform(0.5, 2.0),
                    'path_curvature': np.random.uniform(0.0, 0.5),
                    'confidence_score': np.random.uniform(0.3, 0.7),  # Low confidence for false edges
                    'synthetic_false_edge': True
                }
                
                graph.edges.append(false_edge)
                existing_connections.add((source, target))
                existing_connections.add((target, source))
                stats['false_edges_added'] += 1
        
        return graph, stats
    
    def _apply_node_type_errors(self, graph: VascularGraph, config: DegradationConfig) -> Tuple[VascularGraph, Dict]:
        """Apply node type classification errors"""
        stats = {'type_errors': 0}
        
        for node in graph.nodes:
            if np.random.random() < config.node_type_error_prob:
                original_type = node.get('type', 'regular')
                
                # Common type confusion patterns
                if original_type == 'bifurcation':
                    # Bifurcation misclassified as regular
                    node['type'] = 'regular'
                elif original_type == 'endpoint':
                    # Endpoint misclassified as regular
                    node['type'] = 'regular'
                elif original_type == 'regular':
                    # Regular misclassified as bifurcation or endpoint
                    node['type'] = np.random.choice(['bifurcation', 'endpoint'])
                
                stats['type_errors'] += 1
        
        return graph, stats
    
    def _apply_feature_corruption(self, graph: VascularGraph, config: DegradationConfig) -> Tuple[VascularGraph, Dict]:
        """Apply corruption to node and edge features"""
        stats = {'corrupted_nodes': 0, 'corrupted_edges': 0}
        
        # Corrupt node features
        for node in graph.nodes:
            if np.random.random() < config.feature_noise_prob:
                # Corrupt curvature
                if 'local_curvature' in node:
                    noise = np.random.normal(0, config.feature_corruption_strength)
                    node['local_curvature'] = max(0, float(node['local_curvature']) + noise)
                
                # Corrupt vessel direction
                if 'vessel_direction' in node:
                    direction = np.array(node['vessel_direction'])
                    noise = np.random.normal(0, config.feature_corruption_strength * 0.3, 3)
                    new_direction = direction + noise
                    # Renormalize
                    norm = np.linalg.norm(new_direction)
                    if norm > 0:
                        new_direction = new_direction / norm
                    node['vessel_direction'] = new_direction.tolist()
                
                stats['corrupted_nodes'] += 1
        
        # Corrupt edge features
        for edge in graph.edges:
            if np.random.random() < config.feature_noise_prob:
                # Corrupt path curvature
                if 'path_curvature' in edge:
                    noise = np.random.normal(0, config.feature_corruption_strength * 0.5)
                    edge['path_curvature'] = max(0, float(edge['path_curvature']) + noise)
                
                stats['corrupted_edges'] += 1
        
        return graph, stats
    
    def _apply_vessel_fragmentation(self, graph: VascularGraph, config: DegradationConfig) -> Tuple[VascularGraph, Dict]:
        """Fragment vessels by breaking long paths"""
        stats = {'vessels_fragmented': 0}
        
        # Find long vessel paths
        vessel_paths = self._find_vessel_paths(graph)
        
        for path in vessel_paths:
            if len(path) > config.fragment_min_length and np.random.random() < config.fragmentation_prob:
                # Choose random break points
                num_breaks = np.random.randint(1, min(3, len(path) // config.fragment_min_length + 1))
                break_points = np.random.choice(
                    range(1, len(path) - 1), 
                    size=min(num_breaks, len(path) - 2), 
                    replace=False
                )
                
                # Remove edges at break points
                edges_to_remove = []
                for break_point in break_points:
                    # Find edge connecting path[break_point-1] to path[break_point]
                    for i, edge in enumerate(graph.edges):
                        if ((edge['source'] == path[break_point-1] and edge['target'] == path[break_point]) or
                            (edge['source'] == path[break_point] and edge['target'] == path[break_point-1])):
                            edges_to_remove.append(i)
                            break
                
                # Remove edges (in reverse order)
                for edge_idx in reversed(sorted(edges_to_remove)):
                    graph.edges.pop(edge_idx)
                
                stats['vessels_fragmented'] += 1
        
        return graph, stats
    
    def _deep_copy_graph(self, graph: VascularGraph) -> VascularGraph:
        """Create deep copy of graph"""
        return VascularGraph(
            nodes=copy.deepcopy(graph.nodes),
            edges=copy.deepcopy(graph.edges),
            global_properties=copy.deepcopy(graph.global_properties),
            metadata=copy.deepcopy(graph.metadata)
        )
    
    def _get_graph_stats(self, graph: VascularGraph) -> Dict:
        """Get basic statistics about the graph"""
        node_types = {}
        for node in graph.nodes:
            node_type = node.get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        return {
            'num_nodes': len(graph.nodes),
            'num_edges': len(graph.edges),
            'node_types': node_types,
            'connectivity': len(graph.edges) / max(len(graph.nodes), 1)
        }
    
    def _compute_degradation_summary(self, original_stats: Dict, final_stats: Dict) -> Dict:
        """Compute summary of degradation effects"""
        return {
            'node_loss_rate': (original_stats['num_nodes'] - final_stats['num_nodes']) / max(original_stats['num_nodes'], 1),
            'edge_loss_rate': (original_stats['num_edges'] - final_stats['num_edges']) / max(original_stats['num_edges'], 1),
            'connectivity_change': final_stats['connectivity'] - original_stats['connectivity']
        }
    
    def _find_critical_edges(self, graph: VascularGraph) -> List[int]:
        """Find edges whose removal would disconnect the graph"""
        # Simplified implementation - in practice would use more sophisticated graph analysis
        critical_edges = []
        
        # Build adjacency list
        adjacency = {}
        for i, node in enumerate(graph.nodes):
            adjacency[i] = []
        
        for edge_idx, edge in enumerate(graph.edges):
            src, tgt = edge['source'], edge['target']
            adjacency[src].append((tgt, edge_idx))
            adjacency[tgt].append((src, edge_idx))
        
        # Find nodes with degree 1 (endpoints) - their edges are critical
        for node_id, neighbors in adjacency.items():
            if len(neighbors) == 1:
                critical_edges.append(neighbors[0][1])
        
        return critical_edges
    
    def _find_vessel_paths(self, graph: VascularGraph) -> List[List[int]]:
        """Find linear vessel paths in the graph"""
        # Build adjacency list
        adjacency = {}
        for i, node in enumerate(graph.nodes):
            adjacency[i] = []
        
        for edge in graph.edges:
            src, tgt = edge['source'], edge['target']
            adjacency[src].append(tgt)
            adjacency[tgt].append(src)
        
        visited = set()
        paths = []
        
        # Start from endpoints and nodes with degree != 2
        start_nodes = []
        for node_id, neighbors in adjacency.items():
            if len(neighbors) != 2:  # Endpoint or bifurcation
                start_nodes.append(node_id)
        
        for start_node in start_nodes:
            if start_node in visited:
                continue
            
            # Trace path from this node
            for neighbor in adjacency[start_node]:
                if neighbor in visited:
                    continue
                
                path = [start_node, neighbor]
                visited.add(start_node)
                visited.add(neighbor)
                
                current = neighbor
                while len(adjacency[current]) == 2:
                    # Continue along linear path
                    next_nodes = [n for n in adjacency[current] if n not in visited]
                    if not next_nodes:
                        break
                    
                    next_node = next_nodes[0]
                    path.append(next_node)
                    visited.add(next_node)
                    current = next_node
                
                if len(path) > 2:
                    paths.append(path)
        
        return paths
    
    def _validate_and_fix_graph(self, graph: VascularGraph) -> VascularGraph:
        """Validate and fix common issues in degraded graphs"""
        
        # Remove invalid edges (source or target doesn't exist)
        valid_edges = []
        max_node_id = len(graph.nodes) - 1
        
        for edge in graph.edges:
            src, tgt = edge['source'], edge['target']
            if 0 <= src <= max_node_id and 0 <= tgt <= max_node_id and src != tgt:
                valid_edges.append(edge)
        
        graph.edges = valid_edges
        
        # Update global properties
        graph.global_properties = graph._calculate_global_properties(
            graph.nodes, graph.edges, {}, None
        )
        
        return graph


def create_degradation_pipeline(config: Optional[Dict] = None) -> SyntheticDegradationPipeline:
    """Factory function to create degradation pipeline"""
    if config:
        degradation_config = DegradationConfig(**config)
    else:
        degradation_config = DegradationConfig()
    
    return SyntheticDegradationPipeline(degradation_config)