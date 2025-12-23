"""
Template Placement and Orientation System
Intelligent placement of vessel templates based on corrected graph structure
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
from scipy.spatial import cKDTree
from scipy.optimize import minimize

from src.models.graph_extraction.vascular_graph import VascularGraph
from src.models.reconstruction.vessel_templates import (
    VesselTemplate, CylindricalTemplate, BifurcationTemplate,
    CylinderParameters, BifurcationParameters, TemplateFactory
)


@dataclass
class PlacementConfig:
    """Configuration for template placement"""
    # Spacing and overlap
    min_template_spacing: float = 0.5  # Minimum distance between templates
    max_overlap_ratio: float = 0.3  # Maximum allowed overlap between templates
    
    # Orientation smoothing
    orientation_smoothing: bool = True  # Smooth orientation changes
    smoothing_kernel_size: float = 2.0  # Kernel size for smoothing
    
    # Bifurcation handling
    bifurcation_junction_length: float = 2.0  # Length of bifurcation junction
    murray_law_enforcement: bool = True  # Enforce Murray's Law at bifurcations
    murray_law_tolerance: float = 0.2  # Tolerance for Murray's Law violations
    
    # Template optimization
    optimize_placement: bool = True  # Optimize template positions
    optimization_iterations: int = 50  # Number of optimization iterations
    
    # Quality control
    min_vessel_length: float = 1.0  # Minimum vessel segment length
    min_vessel_radius: float = 0.1  # Minimum vessel radius
    max_vessel_radius: float = 10.0  # Maximum vessel radius


@dataclass
class PlacementResult:
    """Result of template placement"""
    templates: Dict[str, List[VesselTemplate]]  # Placed templates
    placement_quality: float  # Overall placement quality score
    warnings: List[str]  # Placement warnings
    statistics: Dict  # Placement statistics
    
    @property
    def total_templates(self) -> int:
        return sum(len(template_list) for template_list in self.templates.values())


class TemplatePlacer:
    """Intelligent template placement system"""
    
    def __init__(self, config: PlacementConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Internal state
        self.graph = None
        self.node_kdtree = None
        self.edge_vectors = {}
        self.placement_warnings = []
    
    def place_templates(self, graph: VascularGraph, 
                       reference_spacing: Optional[Tuple[float, float, float]] = None) -> PlacementResult:
        """Place templates based on corrected graph structure"""
        self.logger.info(f"Placing templates for graph with {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        
        # Initialize placement
        self.graph = graph
        self.placement_warnings = []
        self._preprocess_graph()
        
        # Create templates
        cylinder_templates = self._create_cylinder_templates()
        bifurcation_templates = self._create_bifurcation_templates()
        
        # Optimize placement if enabled
        if self.config.optimize_placement:
            cylinder_templates = self._optimize_cylinder_placement(cylinder_templates)
            bifurcation_templates = self._optimize_bifurcation_placement(bifurcation_templates)
        
        # Apply quality control
        cylinder_templates = self._filter_templates_by_quality(cylinder_templates)
        bifurcation_templates = self._filter_templates_by_quality(bifurcation_templates)
        
        # Compute placement statistics and quality
        templates = {
            'cylinders': cylinder_templates,
            'bifurcations': bifurcation_templates
        }
        
        placement_quality = self._compute_placement_quality(templates)
        statistics = self._compute_placement_statistics(templates)
        
        result = PlacementResult(
            templates=templates,
            placement_quality=placement_quality,
            warnings=self.placement_warnings.copy(),
            statistics=statistics
        )
        
        self.logger.info(f"Template placement completed: {result.total_templates} templates, "
                        f"quality: {placement_quality:.3f}")
        
        return result
    
    def _preprocess_graph(self):
        """Preprocess graph for efficient template placement"""
        # Build spatial index for nodes
        node_positions = np.array([node['position'][:3] for node in self.graph.nodes])
        self.node_kdtree = cKDTree(node_positions)
        
        # Precompute edge vectors and properties
        self.edge_vectors = {}
        for i, edge in enumerate(self.graph.edges):
            src_pos = np.array(self.graph.nodes[edge['source']]['position'][:3])
            tgt_pos = np.array(self.graph.nodes[edge['target']]['position'][:3])
            
            vector = tgt_pos - src_pos
            length = np.linalg.norm(vector)
            direction = vector / max(length, 1e-6)
            
            self.edge_vectors[i] = {
                'vector': vector,
                'length': length,
                'direction': direction,
                'midpoint': (src_pos + tgt_pos) / 2
            }
    
    def _create_cylinder_templates(self) -> List[CylindricalTemplate]:
        """Create cylindrical templates for vessel segments"""
        cylinder_templates = []
        
        for i, edge in enumerate(self.graph.edges):
            try:
                # Get edge properties
                edge_info = self.edge_vectors[i]
                
                # Check minimum length requirement
                if edge_info['length'] < self.config.min_vessel_length:
                    self.placement_warnings.append(f"Edge {i} too short: {edge_info['length']:.2f}")
                    continue
                
                # Get node properties
                src_node = self.graph.nodes[edge['source']]
                tgt_node = self.graph.nodes[edge['target']]
                
                src_radius = self._get_node_radius(src_node)
                tgt_radius = self._get_node_radius(tgt_node)
                
                # Validate radii
                if src_radius < self.config.min_vessel_radius or tgt_radius < self.config.min_vessel_radius:
                    self.placement_warnings.append(f"Edge {i} radius too small")
                    continue
                
                if src_radius > self.config.max_vessel_radius or tgt_radius > self.config.max_vessel_radius:
                    self.placement_warnings.append(f"Edge {i} radius too large")
                    continue
                
                # Apply orientation smoothing if enabled
                if self.config.orientation_smoothing:
                    src_pos, tgt_pos = self._smooth_edge_positions(i, src_node, tgt_node)
                else:
                    src_pos = np.array(src_node['position'][:3])
                    tgt_pos = np.array(tgt_node['position'][:3])
                
                # Create cylinder parameters
                cylinder_params = CylinderParameters(
                    start_point=src_pos,
                    end_point=tgt_pos,
                    start_radius=src_radius,
                    end_radius=tgt_radius,
                    confidence=edge.get('confidence', 1.0)
                )
                
                # Create template
                cylinder = CylindricalTemplate(cylinder_params)
                cylinder.edge_index = i  # Store reference
                cylinder_templates.append(cylinder)
                
            except Exception as e:
                self.placement_warnings.append(f"Failed to create cylinder for edge {i}: {e}")
                self.logger.warning(f"Cylinder creation failed for edge {i}: {e}")
        
        self.logger.info(f"Created {len(cylinder_templates)} cylinder templates")
        return cylinder_templates
    
    def _create_bifurcation_templates(self) -> List[BifurcationTemplate]:
        """Create bifurcation templates for branch points"""
        bifurcation_templates = []
        
        for i, node in enumerate(self.graph.nodes):
            # Only process bifurcation nodes
            if node.get('type') != 'bifurcation':
                continue
            
            try:
                # Find connected edges
                connected_edges = [
                    j for j, edge in enumerate(self.graph.edges)
                    if edge['source'] == i or edge['target'] == i
                ]
                
                # Bifurcations need exactly 3 connections
                if len(connected_edges) != 3:
                    self.placement_warnings.append(
                        f"Bifurcation node {i} has {len(connected_edges)} connections, expected 3"
                    )
                    continue
                
                # Get connected nodes and their properties
                connections = []
                for edge_idx in connected_edges:
                    edge = self.graph.edges[edge_idx]
                    other_idx = edge['target'] if edge['source'] == i else edge['source']
                    other_node = self.graph.nodes[other_idx]
                    
                    # Direction from bifurcation to connected node
                    node_pos = np.array(node['position'][:3])
                    other_pos = np.array(other_node['position'][:3])
                    
                    direction = other_pos - node_pos
                    distance = np.linalg.norm(direction)
                    direction = direction / max(distance, 1e-6)
                    
                    radius = self._get_node_radius(other_node)
                    
                    connections.append({
                        'node_idx': other_idx,
                        'edge_idx': edge_idx,
                        'direction': direction,
                        'radius': radius,
                        'distance': distance
                    })
                
                # Identify parent and children based on radius (largest = parent)
                radii = [conn['radius'] for conn in connections]
                parent_idx = np.argmax(radii)
                child_indices = [j for j in range(3) if j != parent_idx]
                
                parent_conn = connections[parent_idx]
                child1_conn = connections[child_indices[0]]
                child2_conn = connections[child_indices[1]]
                
                # Validate Murray's Law if enabled
                if self.config.murray_law_enforcement:
                    murray_ratio = (parent_conn['radius']**3 / 
                                  (child1_conn['radius']**3 + child2_conn['radius']**3))
                    
                    if abs(murray_ratio - 1.0) > self.config.murray_law_tolerance:
                        self.placement_warnings.append(
                            f"Bifurcation {i} violates Murray's Law: ratio = {murray_ratio:.3f}"
                        )
                        
                        # Optionally correct radii
                        if self.config.murray_law_enforcement:
                            child1_conn, child2_conn = self._enforce_murray_law(
                                parent_conn, child1_conn, child2_conn
                            )
                
                # Create bifurcation parameters
                bifurcation_params = BifurcationParameters(
                    position=np.array(node['position'][:3]),
                    parent_direction=-parent_conn['direction'],  # Direction toward parent
                    child1_direction=child1_conn['direction'],
                    child2_direction=child2_conn['direction'],
                    parent_radius=parent_conn['radius'],
                    child1_radius=child1_conn['radius'],
                    child2_radius=child2_conn['radius'],
                    junction_length=self.config.bifurcation_junction_length,
                    confidence=node.get('confidence', 1.0)
                )
                
                # Create template
                bifurcation = BifurcationTemplate(bifurcation_params)
                bifurcation.node_index = i  # Store reference
                bifurcation_templates.append(bifurcation)
                
            except Exception as e:
                self.placement_warnings.append(f"Failed to create bifurcation for node {i}: {e}")
                self.logger.warning(f"Bifurcation creation failed for node {i}: {e}")
        
        self.logger.info(f"Created {len(bifurcation_templates)} bifurcation templates")
        return bifurcation_templates
    
    def _smooth_edge_positions(self, edge_idx: int, src_node: Dict, tgt_node: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Apply orientation smoothing to edge endpoints"""
        src_pos = np.array(src_node['position'][:3])
        tgt_pos = np.array(tgt_node['position'][:3])
        
        # Find neighboring edges for smoothing
        src_neighbors = self._find_neighboring_edges(edge_idx, src_node, is_source=True)
        tgt_neighbors = self._find_neighboring_edges(edge_idx, tgt_node, is_source=False)
        
        # Smooth source position
        if src_neighbors:
            neighbor_directions = []
            for neighbor_edge_idx in src_neighbors:
                neighbor_info = self.edge_vectors[neighbor_edge_idx]
                neighbor_directions.append(neighbor_info['direction'])
            
            # Average direction with current edge
            current_direction = self.edge_vectors[edge_idx]['direction']
            all_directions = neighbor_directions + [current_direction]
            
            # Weighted average (current edge gets higher weight)
            weights = [0.3] * len(neighbor_directions) + [0.7]
            avg_direction = np.average(all_directions, axis=0, weights=weights)
            avg_direction = avg_direction / np.linalg.norm(avg_direction)
            
            # Adjust source position slightly
            adjustment = avg_direction * self.config.smoothing_kernel_size * 0.1
            src_pos = src_pos + adjustment
        
        # Smooth target position similarly
        if tgt_neighbors:
            neighbor_directions = []
            for neighbor_edge_idx in tgt_neighbors:
                neighbor_info = self.edge_vectors[neighbor_edge_idx]
                neighbor_directions.append(-neighbor_info['direction'])  # Reverse for target
            
            current_direction = -self.edge_vectors[edge_idx]['direction']
            all_directions = neighbor_directions + [current_direction]
            
            weights = [0.3] * len(neighbor_directions) + [0.7]
            avg_direction = np.average(all_directions, axis=0, weights=weights)
            avg_direction = avg_direction / np.linalg.norm(avg_direction)
            
            adjustment = avg_direction * self.config.smoothing_kernel_size * 0.1
            tgt_pos = tgt_pos + adjustment
        
        return src_pos, tgt_pos
    
    def _find_neighboring_edges(self, edge_idx: int, node: Dict, is_source: bool) -> List[int]:
        """Find edges connected to the same node"""
        node_idx = None
        
        # Find node index
        for i, graph_node in enumerate(self.graph.nodes):
            if graph_node is node:
                node_idx = i
                break
        
        if node_idx is None:
            return []
        
        # Find connected edges (excluding current edge)
        neighbors = []
        for i, edge in enumerate(self.graph.edges):
            if i == edge_idx:
                continue
            
            if edge['source'] == node_idx or edge['target'] == node_idx:
                neighbors.append(i)
        
        return neighbors
    
    def _get_node_radius(self, node: Dict) -> float:
        """Extract node radius with fallback"""
        radius = node.get('radius_voxels', node.get('radius', 1.0))
        return max(radius, self.config.min_vessel_radius)
    
    def _enforce_murray_law(self, parent_conn: Dict, child1_conn: Dict, child2_conn: Dict) -> Tuple[Dict, Dict]:
        """Enforce Murray's Law by adjusting child radii"""
        parent_r3 = parent_conn['radius']**3
        
        # Keep ratio between children, scale to satisfy Murray's Law
        child1_r3 = child1_conn['radius']**3
        child2_r3 = child2_conn['radius']**3
        total_child_r3 = child1_r3 + child2_r3
        
        if total_child_r3 > 0:
            scale = parent_r3 / total_child_r3
            
            child1_conn = child1_conn.copy()
            child2_conn = child2_conn.copy()
            
            child1_conn['radius'] = (child1_r3 * scale)**(1/3)
            child2_conn['radius'] = (child2_r3 * scale)**(1/3)
        
        return child1_conn, child2_conn
    
    def _optimize_cylinder_placement(self, templates: List[CylindricalTemplate]) -> List[CylindricalTemplate]:
        """Optimize cylinder template placement"""
        if not templates:
            return templates
        
        self.logger.info(f"Optimizing placement for {len(templates)} cylinder templates")
        
        # Simple optimization: reduce overlaps
        optimized_templates = []
        
        for i, template in enumerate(templates):
            # Check for overlaps with nearby templates
            overlapping_templates = self._find_overlapping_templates(template, templates[:i])
            
            if overlapping_templates:
                # Adjust template to reduce overlap
                adjusted_template = self._adjust_template_for_overlap(template, overlapping_templates)
                optimized_templates.append(adjusted_template)
            else:
                optimized_templates.append(template)
        
        return optimized_templates
    
    def _optimize_bifurcation_placement(self, templates: List[BifurcationTemplate]) -> List[BifurcationTemplate]:
        """Optimize bifurcation template placement"""
        # For now, return as-is (bifurcations are already well-constrained)
        return templates
    
    def _find_overlapping_templates(self, template: VesselTemplate, 
                                  other_templates: List[VesselTemplate]) -> List[VesselTemplate]:
        """Find templates that overlap with given template"""
        overlapping = []
        
        template_min, template_max = template.get_bounding_box()
        
        for other_template in other_templates:
            other_min, other_max = other_template.get_bounding_box()
            
            # Check bounding box intersection
            if self._bounding_boxes_intersect(template_min, template_max, other_min, other_max):
                # TODO: More sophisticated overlap detection
                overlapping.append(other_template)
        
        return overlapping
    
    def _adjust_template_for_overlap(self, template: VesselTemplate, 
                                   overlapping: List[VesselTemplate]) -> VesselTemplate:
        """Adjust template to reduce overlap (placeholder)"""
        # For now, return original template
        # TODO: Implement sophisticated adjustment
        return template
    
    def _bounding_boxes_intersect(self, min1: np.ndarray, max1: np.ndarray,
                                min2: np.ndarray, max2: np.ndarray) -> bool:
        """Check if two 3D bounding boxes intersect"""
        return np.all(min1 < max2) and np.all(min2 < max1)
    
    def _filter_templates_by_quality(self, templates: List[VesselTemplate]) -> List[VesselTemplate]:
        """Filter templates based on quality criteria"""
        filtered = []
        
        for template in templates:
            # Quality checks
            volume = template.get_volume_estimate()
            
            if volume <= 0:
                self.placement_warnings.append("Template has zero/negative volume")
                continue
            
            # Check template bounds
            min_pt, max_pt = template.get_bounding_box()
            if np.any(np.isnan(min_pt)) or np.any(np.isnan(max_pt)):
                self.placement_warnings.append("Template has invalid bounds")
                continue
            
            filtered.append(template)
        
        removed_count = len(templates) - len(filtered)
        if removed_count > 0:
            self.logger.info(f"Filtered out {removed_count} low-quality templates")
        
        return filtered
    
    def _compute_placement_quality(self, templates: Dict[str, List[VesselTemplate]]) -> float:
        """Compute overall placement quality score"""
        if not templates or not any(template_list for template_list in templates.values()):
            return 0.0
        
        quality_scores = []
        
        # Template coverage quality
        total_templates = sum(len(template_list) for template_list in templates.values())
        expected_templates = len(self.graph.edges) + len([n for n in self.graph.nodes if n.get('type') == 'bifurcation'])
        
        if expected_templates > 0:
            coverage_quality = min(total_templates / expected_templates, 1.0)
            quality_scores.append(coverage_quality)
        
        # Warning penalty
        warning_penalty = max(0, 1.0 - len(self.placement_warnings) * 0.1)
        quality_scores.append(warning_penalty)
        
        # Geometric consistency (placeholder)
        geometric_quality = 0.8  # TODO: Implement proper geometric consistency check
        quality_scores.append(geometric_quality)
        
        return np.mean(quality_scores)
    
    def _compute_placement_statistics(self, templates: Dict[str, List[VesselTemplate]]) -> Dict:
        """Compute placement statistics"""
        stats = {
            'total_templates': sum(len(template_list) for template_list in templates.values()),
            'cylinder_templates': len(templates.get('cylinders', [])),
            'bifurcation_templates': len(templates.get('bifurcations', [])),
            'total_warnings': len(self.placement_warnings),
            'warning_types': {}
        }
        
        # Categorize warnings
        for warning in self.placement_warnings:
            if 'radius' in warning.lower():
                stats['warning_types']['radius_issues'] = stats['warning_types'].get('radius_issues', 0) + 1
            elif 'murray' in warning.lower():
                stats['warning_types']['murray_law_violations'] = stats['warning_types'].get('murray_law_violations', 0) + 1
            elif 'short' in warning.lower():
                stats['warning_types']['length_issues'] = stats['warning_types'].get('length_issues', 0) + 1
            else:
                stats['warning_types']['other'] = stats['warning_types'].get('other', 0) + 1
        
        # Volume statistics
        total_volume = 0
        for template_list in templates.values():
            for template in template_list:
                total_volume += template.get_volume_estimate()
        
        stats['total_estimated_volume'] = total_volume
        
        return stats


def test_template_placement():
    """Test template placement functionality"""
    from src.models.graph_extraction.vascular_graph import VascularGraph
    
    # Create simple test graph
    nodes = [
        {'position': [0, 0, 0], 'radius_voxels': 2.0, 'type': 'normal'},
        {'position': [10, 0, 0], 'radius_voxels': 1.8, 'type': 'bifurcation'},
        {'position': [15, 5, 0], 'radius_voxels': 1.2, 'type': 'normal'},
        {'position': [15, -5, 0], 'radius_voxels': 1.2, 'type': 'normal'},
    ]
    
    edges = [
        {'source': 0, 'target': 1, 'confidence': 0.9},
        {'source': 1, 'target': 2, 'confidence': 0.8},
        {'source': 1, 'target': 3, 'confidence': 0.8},
    ]
    
    test_graph = VascularGraph(nodes=nodes, edges=edges)
    
    # Create placer
    config = PlacementConfig(
        orientation_smoothing=True,
        murray_law_enforcement=True,
        optimize_placement=True
    )
    
    placer = TemplatePlacer(config)
    
    # Place templates
    print("Placing templates...")
    result = placer.place_templates(test_graph)
    
    print(f"Placement completed:")
    print(f"  Total templates: {result.total_templates}")
    print(f"  Cylinders: {len(result.templates['cylinders'])}")
    print(f"  Bifurcations: {len(result.templates['bifurcations'])}")
    print(f"  Quality: {result.placement_quality:.3f}")
    print(f"  Warnings: {len(result.warnings)}")
    
    for warning in result.warnings:
        print(f"    - {warning}")
    
    return result


if __name__ == "__main__":
    # Run tests
    result = test_template_placement()
    print("Template placement tests completed successfully!")