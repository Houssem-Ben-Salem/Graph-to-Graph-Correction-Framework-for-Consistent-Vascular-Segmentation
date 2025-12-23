"""
Graph Correspondence and Matching Module
Implements Step 2 of the Graph-to-Graph Correction Framework
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any, Set
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree
import logging
from dataclasses import dataclass
from pathlib import Path
import pickle

from src.models.graph_extraction.vascular_graph import VascularGraph


@dataclass
class CorrespondenceResult:
    """Data structure for storing correspondence matching results"""
    
    # Node correspondences
    node_correspondences: Dict[int, int]  # pred_node_id -> gt_node_id
    node_confidences: Dict[int, float]    # pred_node_id -> confidence
    unmatched_pred_nodes: Set[int]        # nodes only in prediction
    unmatched_gt_nodes: Set[int]          # nodes only in ground truth
    
    # Edge correspondences  
    edge_correspondences: Dict[Tuple[int, int], Tuple[int, int]]  # pred_edge -> gt_edge
    edge_confidences: Dict[Tuple[int, int], float]  # pred_edge -> confidence
    unmatched_pred_edges: Set[Tuple[int, int]]
    unmatched_gt_edges: Set[Tuple[int, int]]
    
    # Topology differences
    topology_differences: Dict[str, Any]
    
    # Alignment information
    alignment_transform: Dict[str, Any]
    
    # Quality metrics
    correspondence_quality: Dict[str, float]
    
    # Metadata
    metadata: Dict[str, Any]


class GraphCorrespondenceMatcher:
    """
    Implements sophisticated graph correspondence matching between predicted 
    and ground truth vascular graphs.
    
    Key Features:
    - Multi-level correspondence strategy (spatial -> structural -> contextual)
    - Progressive node matching with confidence quantification
    - Topology difference detection and handling
    - Uncertainty-aware correspondence scoring
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the correspondence matcher
        
        Args:
            config: Configuration parameters for matching
        """
        self.config = self._get_default_config()
        if config:
            self.config.update(config)
        
        self.logger = logging.getLogger(__name__)
        
    def _get_default_config(self) -> Dict:
        """Get default configuration for correspondence matching"""
        return {
            # Spatial alignment
            'spatial_tolerance': 3.0,           # mm tolerance for spatial matching
            'alignment_iterations': 5,          # iterations for iterative alignment
            'robust_alignment': True,           # use robust statistics for alignment
            
            # Node correspondence
            'position_weight': 0.4,             # weight for spatial distance
            'feature_weight': 0.3,              # weight for feature similarity  
            'topology_weight': 0.3,             # weight for topological similarity
            'max_correspondence_distance': 5.0, # maximum distance for valid correspondence
            'confidence_threshold': 0.5,        # minimum confidence for acceptance
            
            # Multi-level strategy
            'use_progressive_matching': True,   # enable progressive refinement
            'refinement_iterations': 3,         # number of refinement passes
            'neighborhood_radius': 8.0,         # radius for local context analysis
            
            # Topology handling
            'detect_topology_differences': True,
            'bifurcation_tolerance': 2.0,       # tolerance for bifurcation matching
            'endpoint_tolerance': 1.5,          # tolerance for endpoint matching
            
            # Quality control
            'min_graph_overlap': 0.3,          # minimum required overlap
            'outlier_rejection_threshold': 2.5, # z-score threshold for outlier rejection
        }
    
    def find_correspondence(self, 
                          pred_graph: VascularGraph, 
                          gt_graph: VascularGraph) -> CorrespondenceResult:
        """
        Find correspondence between predicted and ground truth graphs
        
        Args:
            pred_graph: Predicted vascular graph
            gt_graph: Ground truth vascular graph
            
        Returns:
            CorrespondenceResult with matching information
        """
        self.logger.info("Starting graph correspondence matching...")
        
        # Step 1: Spatial alignment and normalization
        alignment_result = self._perform_spatial_alignment(pred_graph, gt_graph)
        
        # Step 2: Multi-level correspondence matching
        if self.config['use_progressive_matching']:
            correspondence_result = self._progressive_correspondence_matching(
                pred_graph, gt_graph, alignment_result
            )
        else:
            correspondence_result = self._direct_correspondence_matching(
                pred_graph, gt_graph, alignment_result
            )
        
        # Step 3: Topology difference detection
        if self.config['detect_topology_differences']:
            topology_diffs = self._detect_topology_differences(
                pred_graph, gt_graph, correspondence_result
            )
            correspondence_result.topology_differences = topology_diffs
        
        # Step 4: Quality assessment and uncertainty quantification
        quality_metrics = self._assess_correspondence_quality(
            pred_graph, gt_graph, correspondence_result
        )
        correspondence_result.correspondence_quality = quality_metrics
        
        # Store alignment and metadata
        correspondence_result.alignment_transform = alignment_result
        correspondence_result.metadata = {
            'config': self.config,
            'pred_graph_stats': self._get_graph_stats(pred_graph),
            'gt_graph_stats': self._get_graph_stats(gt_graph),
            'matching_timestamp': None  # Could add timestamp
        }
        
        self.logger.info(f"Correspondence matching completed. "
                        f"Matched {len(correspondence_result.node_correspondences)} nodes, "
                        f"{len(correspondence_result.edge_correspondences)} edges")
        
        return correspondence_result
    
    def _perform_spatial_alignment(self, 
                                 pred_graph: VascularGraph, 
                                 gt_graph: VascularGraph) -> Dict:
        """
        Perform spatial alignment between prediction and ground truth graphs
        
        Returns:
            Dictionary with alignment transformation parameters
        """
        self.logger.debug("Performing spatial alignment...")
        
        # Extract positions
        pred_positions = np.array([node['position'] for node in pred_graph.nodes 
                                  if 'position' in node])
        gt_positions = np.array([node['position'] for node in gt_graph.nodes 
                                if 'position' in node])
        
        if len(pred_positions) == 0 or len(gt_positions) == 0:
            return {'translation': np.zeros(3), 'scale': 1.0, 'rotation': np.eye(3)}
        
        # Initial centroid alignment
        pred_centroid = np.mean(pred_positions, axis=0)
        gt_centroid = np.mean(gt_positions, axis=0)
        initial_translation = gt_centroid - pred_centroid
        
        # Apply initial translation
        aligned_pred_positions = pred_positions + initial_translation
        
        # Iterative refinement using robust statistics
        translation = initial_translation.copy()
        scale = 1.0
        rotation = np.eye(3)
        
        if self.config['robust_alignment']:
            for iteration in range(self.config['alignment_iterations']):
                # Find closest point pairs
                distances = cdist(aligned_pred_positions, gt_positions)
                closest_gt_indices = np.argmin(distances, axis=1)
                
                # Filter outliers using distance threshold
                valid_matches = distances[np.arange(len(distances)), closest_gt_indices] < \
                              self.config['max_correspondence_distance']
                
                if np.sum(valid_matches) < 3:  # Need minimum points for alignment
                    break
                
                # Refine alignment using valid matches
                valid_pred = aligned_pred_positions[valid_matches]
                valid_gt = gt_positions[closest_gt_indices[valid_matches]]
                
                # Calculate refined centroid alignment
                refined_translation = np.mean(valid_gt - valid_pred, axis=0)
                translation += refined_translation
                aligned_pred_positions += refined_translation
                
                # Early stopping if translation is small
                if np.linalg.norm(refined_translation) < 0.1:
                    break
        
        return {
            'translation': translation,
            'scale': scale,
            'rotation': rotation,
            'alignment_error': np.mean(distances[np.arange(len(distances)), closest_gt_indices]),
            'num_iterations': iteration + 1 if 'iteration' in locals() else 0
        }
    
    def _progressive_correspondence_matching(self, 
                                          pred_graph: VascularGraph,
                                          gt_graph: VascularGraph,
                                          alignment_result: Dict) -> CorrespondenceResult:
        """
        Progressive multi-level correspondence matching strategy
        
        Strategy:
        1. High-confidence anchor points (bifurcations, endpoints)
        2. Progressive expansion using local neighborhoods
        3. Refinement with topological constraints
        """
        self.logger.debug("Starting progressive correspondence matching...")
        
        # Initialize result structure
        result = CorrespondenceResult(
            node_correspondences={},
            node_confidences={},
            unmatched_pred_nodes=set(range(len(pred_graph.nodes))),
            unmatched_gt_nodes=set(range(len(gt_graph.nodes))),
            edge_correspondences={},
            edge_confidences={},
            unmatched_pred_edges=set(),
            unmatched_gt_edges=set(),
            topology_differences={},
            alignment_transform=alignment_result,
            correspondence_quality={},
            metadata={}
        )
        
        # Apply spatial alignment to prediction graph
        aligned_pred_graph = self._apply_spatial_alignment(pred_graph, alignment_result)
        
        # Level 1: Match high-confidence anchor points
        self._match_anchor_points(aligned_pred_graph, gt_graph, result)
        
        # Level 2: Progressive neighborhood expansion
        for iteration in range(self.config['refinement_iterations']):
            initial_matches = len(result.node_correspondences)
            self._expand_correspondences_locally(aligned_pred_graph, gt_graph, result)
            
            # Check for convergence
            if len(result.node_correspondences) == initial_matches:
                break
        
        # Level 3: Final refinement with topological constraints
        self._refine_with_topology(aligned_pred_graph, gt_graph, result)
        
        # Derive edge correspondences from node correspondences
        self._derive_edge_correspondences(aligned_pred_graph, gt_graph, result)
        
        return result
    
    def _match_anchor_points(self, 
                           pred_graph: VascularGraph,
                           gt_graph: VascularGraph,
                           result: CorrespondenceResult):
        """Match high-confidence anchor points (bifurcations and endpoints)"""
        
        # Separate nodes by type
        pred_anchors = [(i, node) for i, node in enumerate(pred_graph.nodes)
                       if node.get('type') in ['bifurcation', 'endpoint']]
        gt_anchors = [(i, node) for i, node in enumerate(gt_graph.nodes)
                     if node.get('type') in ['bifurcation', 'endpoint']]
        
        if not pred_anchors or not gt_anchors:
            return
        
        # Separate by type for better matching
        for node_type in ['bifurcation', 'endpoint']:
            pred_type_anchors = [(i, node) for i, node in pred_anchors 
                               if node.get('type') == node_type]
            gt_type_anchors = [(i, node) for i, node in gt_anchors 
                             if node.get('type') == node_type]
            
            if not pred_type_anchors or not gt_type_anchors:
                continue
            
            # Calculate distance matrix
            pred_positions = np.array([node['position'] for _, node in pred_type_anchors])
            gt_positions = np.array([node['position'] for _, node in gt_type_anchors])
            
            distance_matrix = cdist(pred_positions, gt_positions)
            
            # Use Hungarian algorithm for optimal assignment
            pred_indices, gt_indices = linear_sum_assignment(distance_matrix)
            
            # Filter by distance threshold
            tolerance = (self.config['bifurcation_tolerance'] if node_type == 'bifurcation' 
                        else self.config['endpoint_tolerance'])
            
            for pi, gi in zip(pred_indices, gt_indices):
                distance = distance_matrix[pi, gi]
                if distance <= tolerance:
                    pred_node_id = pred_type_anchors[pi][0]
                    gt_node_id = gt_type_anchors[gi][0]
                    
                    # Calculate confidence based on distance and feature similarity
                    confidence = self._calculate_node_confidence(
                        pred_type_anchors[pi][1], gt_type_anchors[gi][1], distance
                    )
                    
                    if confidence >= self.config['confidence_threshold']:
                        result.node_correspondences[pred_node_id] = gt_node_id
                        result.node_confidences[pred_node_id] = confidence
                        result.unmatched_pred_nodes.discard(pred_node_id)
                        result.unmatched_gt_nodes.discard(gt_node_id)
    
    def _expand_correspondences_locally(self,
                                      pred_graph: VascularGraph,
                                      gt_graph: VascularGraph,
                                      result: CorrespondenceResult):
        """Expand correspondences using local neighborhood analysis"""
        
        # Build spatial indices for efficient neighborhood queries
        pred_positions = np.array([node['position'] for node in pred_graph.nodes])
        gt_positions = np.array([node['position'] for node in gt_graph.nodes])
        
        pred_tree = KDTree(pred_positions)
        gt_tree = KDTree(gt_positions)
        
        # For each existing correspondence, try to match neighbors
        new_correspondences = {}
        new_confidences = {}
        
        for pred_id, gt_id in result.node_correspondences.items():
            # Find neighbors in both graphs
            pred_neighbors = pred_tree.query_ball_point(
                pred_positions[pred_id], self.config['neighborhood_radius']
            )
            gt_neighbors = gt_tree.query_ball_point(
                gt_positions[gt_id], self.config['neighborhood_radius']
            )
            
            # Filter to unmatched nodes
            pred_neighbors = [n for n in pred_neighbors if n in result.unmatched_pred_nodes]
            gt_neighbors = [n for n in gt_neighbors if n in result.unmatched_gt_nodes]
            
            if not pred_neighbors or not gt_neighbors:
                continue
            
            # Calculate local correspondence matrix
            local_pred_pos = pred_positions[pred_neighbors]
            local_gt_pos = gt_positions[gt_neighbors]
            local_distance_matrix = cdist(local_pred_pos, local_gt_pos)
            
            # Find best matches within distance threshold
            for i, pred_neighbor in enumerate(pred_neighbors):
                distances = local_distance_matrix[i]
                best_gt_idx = np.argmin(distances)
                best_distance = distances[best_gt_idx]
                
                if best_distance <= self.config['max_correspondence_distance']:
                    gt_neighbor = gt_neighbors[best_gt_idx]
                    
                    # Calculate confidence including local context
                    confidence = self._calculate_node_confidence_with_context(
                        pred_graph.nodes[pred_neighbor],
                        gt_graph.nodes[gt_neighbor],
                        best_distance,
                        pred_graph,
                        gt_graph,
                        pred_neighbor,
                        gt_neighbor
                    )
                    
                    if confidence >= self.config['confidence_threshold']:
                        new_correspondences[pred_neighbor] = gt_neighbor
                        new_confidences[pred_neighbor] = confidence
        
        # Add new correspondences
        for pred_id, gt_id in new_correspondences.items():
            result.node_correspondences[pred_id] = gt_id
            result.node_confidences[pred_id] = new_confidences[pred_id]
            result.unmatched_pred_nodes.discard(pred_id)
            result.unmatched_gt_nodes.discard(gt_id)
    
    def _calculate_node_confidence(self, 
                                 pred_node: Dict, 
                                 gt_node: Dict, 
                                 spatial_distance: float) -> float:
        """Calculate confidence score for node correspondence"""
        
        # Spatial component (inverse distance, normalized)
        spatial_score = max(0, 1 - spatial_distance / self.config['max_correspondence_distance'])
        
        # Feature similarity component
        feature_score = self._calculate_feature_similarity(pred_node, gt_node)
        
        # Type consistency bonus
        type_bonus = 1.0 if pred_node.get('type') == gt_node.get('type') else 0.5
        
        # Weighted combination
        confidence = (
            self.config['position_weight'] * spatial_score +
            self.config['feature_weight'] * feature_score
        ) * type_bonus
        
        return np.clip(confidence, 0.0, 1.0)
    
    def _calculate_feature_similarity(self, pred_node: Dict, gt_node: Dict) -> float:
        """Calculate feature similarity between two nodes"""
        
        features_to_compare = ['radius_voxels', 'local_curvature', 'degree']
        similarities = []
        
        for feature in features_to_compare:
            if feature in pred_node and feature in gt_node:
                pred_val = pred_node[feature]
                gt_val = gt_node[feature]
                
                if pred_val == 0 and gt_val == 0:
                    similarity = 1.0
                else:
                    # Normalized absolute difference
                    max_val = max(abs(pred_val), abs(gt_val))
                    if max_val > 0:
                        similarity = 1 - abs(pred_val - gt_val) / max_val
                    else:
                        similarity = 1.0
                
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.5
    
    def _calculate_node_confidence_with_context(self,
                                              pred_node: Dict,
                                              gt_node: Dict,
                                              spatial_distance: float,
                                              pred_graph: VascularGraph,
                                              gt_graph: VascularGraph,
                                              pred_id: int,
                                              gt_id: int) -> float:
        """Calculate confidence with local topological context"""
        
        # Base confidence
        base_confidence = self._calculate_node_confidence(pred_node, gt_node, spatial_distance)
        
        # Topological context score
        topo_score = self._calculate_topological_similarity(
            pred_graph, gt_graph, pred_id, gt_id
        )
        
        # Combine with topological weight
        total_confidence = (
            (self.config['position_weight'] + self.config['feature_weight']) * base_confidence +
            self.config['topology_weight'] * topo_score
        )
        
        return np.clip(total_confidence, 0.0, 1.0)
    
    def _calculate_topological_similarity(self,
                                        pred_graph: VascularGraph,
                                        gt_graph: VascularGraph,
                                        pred_id: int,
                                        gt_id: int) -> float:
        """Calculate topological similarity around two nodes"""
        
        # Convert to NetworkX for easier analysis
        pred_nx = pred_graph.to_networkx()
        gt_nx = gt_graph.to_networkx()
        
        # Compare local neighborhoods
        pred_neighbors = set(pred_nx.neighbors(pred_id))
        gt_neighbors = set(gt_nx.neighbors(gt_id))
        
        # Degree similarity
        degree_sim = 1 - abs(len(pred_neighbors) - len(gt_neighbors)) / max(len(pred_neighbors), len(gt_neighbors), 1)
        
        # TODO: Add more sophisticated topological features
        # - Path lengths to nearest bifurcations
        # - Local clustering coefficient
        # - Betweenness centrality
        
        return degree_sim
    
    def _apply_spatial_alignment(self, 
                               graph: VascularGraph, 
                               alignment: Dict) -> VascularGraph:
        """Apply spatial alignment transformation to graph"""
        
        # Create aligned copy
        aligned_nodes = []
        for node in graph.nodes:
            aligned_node = node.copy()
            if 'position' in node:
                pos = np.array(node['position'])
                # Apply transformation: rotation, scale, translation
                aligned_pos = alignment['rotation'] @ (pos * alignment['scale']) + alignment['translation']
                aligned_node['position'] = aligned_pos.tolist()
            aligned_nodes.append(aligned_node)
        
        # Create new graph with aligned nodes
        aligned_graph = VascularGraph(
            nodes=aligned_nodes,
            edges=graph.edges.copy(),
            global_properties=graph.global_properties.copy(),
            metadata=graph.metadata.copy()
        )
        
        return aligned_graph
    
    def _refine_with_topology(self,
                            pred_graph: VascularGraph,
                            gt_graph: VascularGraph,
                            result: CorrespondenceResult):
        """Refine correspondences using topological constraints"""
        
        # Convert to NetworkX for topological analysis
        pred_nx = pred_graph.to_networkx()
        gt_nx = gt_graph.to_networkx()
        
        # Check consistency of existing correspondences
        inconsistent_correspondences = []
        
        for pred_id, gt_id in result.node_correspondences.items():
            # Check if neighborhood structure is consistent
            pred_neighbors = set(pred_nx.neighbors(pred_id))
            gt_neighbors = set(gt_nx.neighbors(gt_id))
            
            # Map predicted neighbors to ground truth through existing correspondences
            mapped_pred_neighbors = set()
            for pred_neighbor in pred_neighbors:
                if pred_neighbor in result.node_correspondences:
                    mapped_pred_neighbors.add(result.node_correspondences[pred_neighbor])
            
            # Calculate overlap
            if len(gt_neighbors) > 0:
                overlap = len(mapped_pred_neighbors & gt_neighbors) / len(gt_neighbors)
                if overlap < 0.3:  # Low topological consistency
                    inconsistent_correspondences.append((pred_id, gt_id))
        
        # Remove inconsistent correspondences with low confidence
        for pred_id, gt_id in inconsistent_correspondences:
            if result.node_confidences.get(pred_id, 0) < 0.7:
                del result.node_correspondences[pred_id]
                del result.node_confidences[pred_id]
                result.unmatched_pred_nodes.add(pred_id)
                result.unmatched_gt_nodes.add(gt_id)
    
    def _derive_edge_correspondences(self,
                                   pred_graph: VascularGraph,
                                   gt_graph: VascularGraph,
                                   result: CorrespondenceResult):
        """Derive edge correspondences from node correspondences"""
        
        for pred_edge in pred_graph.edges:
            pred_source = pred_edge['source']
            pred_target = pred_edge['target']
            
            # Check if both endpoints have correspondences
            if (pred_source in result.node_correspondences and 
                pred_target in result.node_correspondences):
                
                gt_source = result.node_correspondences[pred_source]
                gt_target = result.node_correspondences[pred_target]
                
                # Check if corresponding edge exists in ground truth
                gt_edge_exists = any(
                    (edge['source'] == gt_source and edge['target'] == gt_target) or
                    (edge['source'] == gt_target and edge['target'] == gt_source)
                    for edge in gt_graph.edges
                )
                
                if gt_edge_exists:
                    pred_edge_key = (pred_source, pred_target)
                    gt_edge_key = (gt_source, gt_target)
                    
                    result.edge_correspondences[pred_edge_key] = gt_edge_key
                    
                    # Calculate edge confidence based on endpoint confidences
                    source_conf = result.node_confidences.get(pred_source, 0)
                    target_conf = result.node_confidences.get(pred_target, 0)
                    edge_confidence = (source_conf + target_conf) / 2
                    
                    result.edge_confidences[pred_edge_key] = edge_confidence
                else:
                    # Edge exists in prediction but not in ground truth
                    result.unmatched_pred_edges.add((pred_source, pred_target))
            else:
                # At least one endpoint is unmatched
                result.unmatched_pred_edges.add((pred_source, pred_target))
        
        # Find unmatched ground truth edges
        for gt_edge in gt_graph.edges:
            gt_source = gt_edge['source']
            gt_target = gt_edge['target']
            gt_edge_key = (gt_source, gt_target)
            
            # Check if this edge is matched
            is_matched = any(
                gt_match == gt_edge_key or gt_match == (gt_target, gt_source)
                for gt_match in result.edge_correspondences.values()
            )
            
            if not is_matched:
                result.unmatched_gt_edges.add(gt_edge_key)
    
    def _detect_topology_differences(self,
                                   pred_graph: VascularGraph,
                                   gt_graph: VascularGraph,
                                   result: CorrespondenceResult) -> Dict:
        """Detect and categorize topology differences"""
        
        differences = {
            'missing_branches': [],      # branches in GT but not in prediction
            'false_branches': [],        # branches in prediction but not in GT
            'connectivity_errors': [],   # incorrect connections
            'bifurcation_differences': [],
            'summary_stats': {}
        }
        
        # Analyze unmatched components
        differences['missing_branches'] = list(result.unmatched_gt_edges)
        differences['false_branches'] = list(result.unmatched_pred_edges)
        
        # Analyze bifurcation differences
        pred_bifurcations = [i for i, node in enumerate(pred_graph.nodes)
                           if node.get('type') == 'bifurcation']
        gt_bifurcations = [i for i, node in enumerate(gt_graph.nodes)
                         if node.get('type') == 'bifurcation']
        
        matched_bifurcations = sum(1 for pred_id in pred_bifurcations
                                 if pred_id in result.node_correspondences)
        
        differences['bifurcation_differences'] = {
            'pred_bifurcations': len(pred_bifurcations),
            'gt_bifurcations': len(gt_bifurcations),
            'matched_bifurcations': matched_bifurcations,
            'missing_bifurcations': len(gt_bifurcations) - matched_bifurcations,
            'false_bifurcations': len(pred_bifurcations) - matched_bifurcations
        }
        
        # Summary statistics
        differences['summary_stats'] = {
            'total_topology_differences': (len(result.unmatched_pred_edges) + 
                                         len(result.unmatched_gt_edges)),
            'correspondence_rate': len(result.node_correspondences) / max(len(pred_graph.nodes), 1),
            'edge_correspondence_rate': len(result.edge_correspondences) / max(len(pred_graph.edges), 1)
        }
        
        return differences
    
    def _assess_correspondence_quality(self,
                                     pred_graph: VascularGraph,
                                     gt_graph: VascularGraph,
                                     result: CorrespondenceResult) -> Dict:
        """Assess overall quality of correspondence matching"""
        
        quality_metrics = {}
        
        # Coverage metrics
        quality_metrics['node_coverage'] = len(result.node_correspondences) / len(pred_graph.nodes)
        quality_metrics['gt_node_coverage'] = len(result.node_correspondences) / len(gt_graph.nodes)
        quality_metrics['edge_coverage'] = len(result.edge_correspondences) / max(len(pred_graph.edges), 1)
        
        # Confidence statistics
        if result.node_confidences:
            confidences = list(result.node_confidences.values())
            quality_metrics['avg_node_confidence'] = np.mean(confidences)
            quality_metrics['min_node_confidence'] = np.min(confidences)
            quality_metrics['high_confidence_ratio'] = np.mean(np.array(confidences) > 0.7)
        
        if result.edge_confidences:
            edge_confidences = list(result.edge_confidences.values())
            quality_metrics['avg_edge_confidence'] = np.mean(edge_confidences)
        
        # Spatial quality
        if result.node_correspondences:
            spatial_errors = []
            for pred_id, gt_id in result.node_correspondences.items():
                pred_pos = np.array(pred_graph.nodes[pred_id]['position'])
                gt_pos = np.array(gt_graph.nodes[gt_id]['position'])
                error = np.linalg.norm(pred_pos - gt_pos)
                spatial_errors.append(error)
            
            quality_metrics['avg_spatial_error'] = np.mean(spatial_errors)
            quality_metrics['max_spatial_error'] = np.max(spatial_errors)
            quality_metrics['spatial_error_std'] = np.std(spatial_errors)
        
        # Overall quality score
        coverage_score = (quality_metrics.get('node_coverage', 0) + 
                         quality_metrics.get('gt_node_coverage', 0)) / 2
        confidence_score = quality_metrics.get('avg_node_confidence', 0)
        spatial_score = max(0, 1 - quality_metrics.get('avg_spatial_error', 10) / 10)
        
        quality_metrics['overall_quality'] = (coverage_score + confidence_score + spatial_score) / 3
        
        return quality_metrics
    
    def _direct_correspondence_matching(self,
                                      pred_graph: VascularGraph,
                                      gt_graph: VascularGraph,
                                      alignment_result: Dict) -> CorrespondenceResult:
        """Direct correspondence matching without progressive refinement"""
        
        # Apply spatial alignment
        aligned_pred_graph = self._apply_spatial_alignment(pred_graph, alignment_result)
        
        # Calculate distance matrix between all nodes
        pred_positions = np.array([node['position'] for node in aligned_pred_graph.nodes])
        gt_positions = np.array([node['position'] for node in gt_graph.nodes])
        
        distance_matrix = cdist(pred_positions, gt_positions)
        
        # Use Hungarian algorithm for optimal assignment
        pred_indices, gt_indices = linear_sum_assignment(distance_matrix)
        
        # Initialize result
        result = CorrespondenceResult(
            node_correspondences={},
            node_confidences={},
            unmatched_pred_nodes=set(range(len(pred_graph.nodes))),
            unmatched_gt_nodes=set(range(len(gt_graph.nodes))),
            edge_correspondences={},
            edge_confidences={},
            unmatched_pred_edges=set(),
            unmatched_gt_edges=set(),
            topology_differences={},
            alignment_transform=alignment_result,
            correspondence_quality={},
            metadata={}
        )
        
        # Filter assignments by distance and confidence
        for pi, gi in zip(pred_indices, gt_indices):
            distance = distance_matrix[pi, gi]
            if distance <= self.config['max_correspondence_distance']:
                confidence = self._calculate_node_confidence(
                    aligned_pred_graph.nodes[pi], gt_graph.nodes[gi], distance
                )
                
                if confidence >= self.config['confidence_threshold']:
                    result.node_correspondences[pi] = gi
                    result.node_confidences[pi] = confidence
                    result.unmatched_pred_nodes.discard(pi)
                    result.unmatched_gt_nodes.discard(gi)
        
        # Derive edge correspondences
        self._derive_edge_correspondences(aligned_pred_graph, gt_graph, result)
        
        return result
    
    def _get_graph_stats(self, graph: VascularGraph) -> Dict:
        """Get basic statistics about a graph"""
        return {
            'num_nodes': len(graph.nodes),
            'num_edges': len(graph.edges),
            'node_types': {node.get('type', 'unknown'): 1 for node in graph.nodes},
            'global_properties': graph.global_properties
        }
    
    def save_correspondence_result(self, 
                                 result: CorrespondenceResult, 
                                 filepath: Path):
        """Save correspondence result to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(result, f)
        self.logger.info(f"Correspondence result saved to {filepath}")
    
    def load_correspondence_result(self, filepath: Path) -> CorrespondenceResult:
        """Load correspondence result from file"""
        with open(filepath, 'rb') as f:
            result = pickle.load(f)
        self.logger.info(f"Correspondence result loaded from {filepath}")
        return result


def create_correspondence_matcher(config: Optional[Dict] = None) -> GraphCorrespondenceMatcher:
    """Factory function to create correspondence matcher"""
    return GraphCorrespondenceMatcher(config)