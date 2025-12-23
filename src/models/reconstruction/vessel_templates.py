"""
Parameterized Vessel Templates
Geometric primitives for reconstructing vascular structures from corrected graphs
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

from src.models.graph_extraction.vascular_graph import VascularGraph


@dataclass
class TemplateParameters:
    """Base parameters for vessel templates"""
    position: np.ndarray  # 3D position
    radius: float  # Vessel radius
    orientation: Optional[np.ndarray] = None  # Direction vector
    length: Optional[float] = None  # Template length
    confidence: float = 1.0  # Reconstruction confidence


@dataclass
class CylinderParameters:
    """Parameters for cylindrical vessel segments"""
    start_point: np.ndarray  # Start position
    end_point: np.ndarray  # End position
    start_radius: float  # Radius at start
    end_radius: float  # Radius at end (for tapering)
    curvature: float = 0.0  # Vessel curvature
    confidence: float = 1.0  # Reconstruction confidence
    
    # Derived parameters (computed in __post_init__)
    position: Optional[np.ndarray] = None
    radius: Optional[float] = None
    orientation: Optional[np.ndarray] = None
    length: Optional[float] = None
    
    def __post_init__(self):
        # Compute derived parameters
        self.length = np.linalg.norm(self.end_point - self.start_point)
        self.orientation = (self.end_point - self.start_point) / max(self.length, 1e-6)
        self.position = (self.start_point + self.end_point) / 2
        self.radius = (self.start_radius + self.end_radius) / 2


@dataclass  
class BifurcationParameters:
    """Parameters for vessel bifurcations"""
    position: np.ndarray  # Bifurcation center position
    parent_direction: np.ndarray  # Incoming vessel direction
    child1_direction: np.ndarray  # First branch direction
    child2_direction: np.ndarray  # Second branch direction
    parent_radius: float  # Parent vessel radius
    child1_radius: float  # First branch radius
    child2_radius: float  # Second branch radius
    junction_length: float = 2.0  # Length of bifurcation region
    confidence: float = 1.0  # Reconstruction confidence
    
    # Derived parameters (computed in __post_init__)
    bifurcation_angle: Optional[float] = None  # Angle between branches
    radius: Optional[float] = None  # Average radius
    orientation: Optional[np.ndarray] = None  # Primary orientation
    length: Optional[float] = None  # Junction length
    
    def __post_init__(self):
        # Validate Murray's Law compliance
        murray_ratio = self.parent_radius**3 / (self.child1_radius**3 + self.child2_radius**3)
        if abs(murray_ratio - 1.0) > 0.3:  # Allow 30% deviation
            logging.warning(f"Bifurcation violates Murray's Law: ratio = {murray_ratio:.3f}")
        
        # Compute bifurcation angle
        self.bifurcation_angle = np.arccos(np.clip(
            np.dot(self.child1_direction, self.child2_direction), -1, 1
        ))
        
        # Compute derived parameters
        self.radius = self.parent_radius  # Use parent radius as reference
        self.orientation = -self.parent_direction  # Primary orientation toward parent
        self.length = self.junction_length


class VesselTemplate(ABC):
    """Abstract base class for vessel templates"""
    
    def __init__(self, parameters: Union[TemplateParameters, CylinderParameters, BifurcationParameters]):
        self.parameters = parameters
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def compute_sdf(self, points: np.ndarray) -> np.ndarray:
        """Compute signed distance field at given points"""
        pass
    
    @abstractmethod
    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get template bounding box (min_point, max_point)"""
        pass
    
    @abstractmethod
    def get_volume_estimate(self) -> float:
        """Estimate template volume"""
        pass
    
    def is_inside(self, points: np.ndarray) -> np.ndarray:
        """Check if points are inside the template"""
        return self.compute_sdf(points) <= 0
    
    def get_surface_points(self, num_points: int = 1000) -> np.ndarray:
        """Generate surface points for visualization"""
        # Default implementation using rejection sampling
        min_pt, max_pt = self.get_bounding_box()
        
        surface_points = []
        attempts = 0
        max_attempts = num_points * 10
        
        while len(surface_points) < num_points and attempts < max_attempts:
            # Generate random points in bounding box
            random_points = np.random.uniform(
                min_pt, max_pt, (min(100, num_points), 3)
            )
            
            # Compute SDF
            sdf_values = self.compute_sdf(random_points)
            
            # Find points near surface (|sdf| < threshold)
            surface_mask = np.abs(sdf_values) < 0.1
            surface_points.extend(random_points[surface_mask])
            
            attempts += 100
        
        return np.array(surface_points[:num_points])


class CylindricalTemplate(VesselTemplate):
    """Cylindrical vessel segment template"""
    
    def __init__(self, parameters: CylinderParameters):
        super().__init__(parameters)
        self.params = parameters
    
    def compute_sdf(self, points: np.ndarray) -> np.ndarray:
        """Compute SDF for cylindrical vessel"""
        if points.shape[1] != 3:
            raise ValueError("Points must be Nx3 array")
        
        # Transform points to cylinder coordinate system
        start = self.params.start_point
        end = self.params.end_point
        direction = self.params.orientation
        length = self.params.length
        
        # Vector from start to query points
        to_points = points - start
        
        # Project onto cylinder axis
        axial_distance = np.dot(to_points, direction)
        
        # Clamp to cylinder length
        axial_distance_clamped = np.clip(axial_distance, 0, length)
        
        # Compute radius at each axial position (linear tapering)
        if abs(self.params.start_radius - self.params.end_radius) < 1e-6:
            # Uniform cylinder
            radius_at_pos = np.full_like(axial_distance_clamped, self.params.start_radius)
        else:
            # Tapered cylinder
            t = axial_distance_clamped / max(length, 1e-6)
            radius_at_pos = self.params.start_radius * (1 - t) + self.params.end_radius * t
        
        # Points on cylinder axis at clamped positions
        axis_points = start + axial_distance_clamped.reshape(-1, 1) * direction
        
        # Radial distance from axis
        radial_vectors = points - axis_points
        radial_distances = np.linalg.norm(radial_vectors, axis=1)
        
        # Distance to cylindrical surface
        radial_sdf = radial_distances - radius_at_pos
        
        # Handle end caps
        end_cap_distance = np.maximum(
            -axial_distance,  # Distance to start cap
            axial_distance - length  # Distance to end cap
        )
        
        # Combine radial and axial distances
        sdf = np.where(
            (axial_distance >= 0) & (axial_distance <= length),
            radial_sdf,  # Inside cylinder length
            np.sqrt(radial_sdf**2 + np.maximum(0, end_cap_distance)**2)  # Outside caps
        )
        
        return sdf
    
    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounding box for cylinder"""
        start = self.params.start_point
        end = self.params.end_point
        max_radius = max(self.params.start_radius, self.params.end_radius)
        
        # Bounding box around cylinder
        min_pt = np.minimum(start, end) - max_radius
        max_pt = np.maximum(start, end) + max_radius
        
        return min_pt, max_pt
    
    def get_volume_estimate(self) -> float:
        """Estimate cylinder volume"""
        # Volume of truncated cone
        r1 = self.params.start_radius
        r2 = self.params.end_radius
        h = self.params.length
        
        volume = (np.pi * h / 3) * (r1**2 + r1*r2 + r2**2)
        return volume


class BifurcationTemplate(VesselTemplate):
    """Vessel bifurcation template"""
    
    def __init__(self, parameters: BifurcationParameters):
        super().__init__(parameters)
        self.params = parameters
        
        # Create component templates
        self._create_component_templates()
    
    def _create_component_templates(self):
        """Create cylindrical templates for bifurcation components"""
        junction_length = self.params.junction_length
        
        # Parent segment (leading into bifurcation)
        parent_start = self.params.position - self.params.parent_direction * junction_length
        parent_end = self.params.position
        
        self.parent_template = CylindricalTemplate(CylinderParameters(
            start_point=parent_start,
            end_point=parent_end,
            start_radius=self.params.parent_radius,
            end_radius=self.params.parent_radius
        ))
        
        # Child segments (branching from bifurcation)
        child1_end = self.params.position + self.params.child1_direction * junction_length
        child2_end = self.params.position + self.params.child2_direction * junction_length
        
        self.child1_template = CylindricalTemplate(CylinderParameters(
            start_point=self.params.position,
            end_point=child1_end,
            start_radius=self.params.child1_radius,
            end_radius=self.params.child1_radius
        ))
        
        self.child2_template = CylindricalTemplate(CylinderParameters(
            start_point=self.params.position,
            end_point=child2_end,
            start_radius=self.params.child2_radius,
            end_radius=self.params.child2_radius
        ))
    
    def compute_sdf(self, points: np.ndarray) -> np.ndarray:
        """Compute SDF for bifurcation (union of three cylinders)"""
        # Compute SDF for each component
        parent_sdf = self.parent_template.compute_sdf(points)
        child1_sdf = self.child1_template.compute_sdf(points)
        child2_sdf = self.child2_template.compute_sdf(points)
        
        # Union operation (minimum SDF)
        sdf = np.minimum(parent_sdf, np.minimum(child1_sdf, child2_sdf))
        
        # Smooth blending at junction (optional)
        junction_radius = max(self.params.parent_radius, 
                            self.params.child1_radius, 
                            self.params.child2_radius)
        
        # Distance to junction center
        junction_distance = np.linalg.norm(points - self.params.position, axis=1)
        
        # Apply smooth blending near junction
        blend_mask = junction_distance < junction_radius * 2
        if np.any(blend_mask):
            # Smooth minimum operation
            blend_factor = np.exp(-junction_distance[blend_mask] / junction_radius)
            sdf[blend_mask] *= (1 - blend_factor * 0.1)  # 10% smoothing
        
        return sdf
    
    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounding box for bifurcation"""
        # Union of all component bounding boxes
        parent_min, parent_max = self.parent_template.get_bounding_box()
        child1_min, child1_max = self.child1_template.get_bounding_box()
        child2_min, child2_max = self.child2_template.get_bounding_box()
        
        min_pt = np.minimum(parent_min, np.minimum(child1_min, child2_min))
        max_pt = np.maximum(parent_max, np.maximum(child1_max, child2_max))
        
        return min_pt, max_pt
    
    def get_volume_estimate(self) -> float:
        """Estimate bifurcation volume"""
        # Sum of component volumes (with some overlap correction)
        parent_vol = self.parent_template.get_volume_estimate()
        child1_vol = self.child1_template.get_volume_estimate()
        child2_vol = self.child2_template.get_volume_estimate()
        
        # Approximate overlap correction (reduce by junction volume)
        junction_vol = (4/3) * np.pi * (max(self.params.parent_radius,
                                           self.params.child1_radius,
                                           self.params.child2_radius) ** 3)
        
        total_volume = parent_vol + child1_vol + child2_vol - junction_vol * 0.5
        return max(total_volume, 0)


class TemplateFactory:
    """Factory for creating vessel templates from graph data"""
    
    @staticmethod
    def create_cylinder_from_edge(graph: VascularGraph, edge_idx: int, 
                                 node_radius_scale: float = 1.0) -> CylindricalTemplate:
        """Create cylindrical template from graph edge"""
        edge = graph.edges[edge_idx]
        
        # Get source and target nodes
        src_node = graph.nodes[edge['source']]
        tgt_node = graph.nodes[edge['target']]
        
        # Extract positions and radii
        src_pos = np.array(src_node['position'][:3])
        tgt_pos = np.array(tgt_node['position'][:3])
        
        src_radius = src_node.get('radius_voxels', 1.0) * node_radius_scale
        tgt_radius = tgt_node.get('radius_voxels', 1.0) * node_radius_scale
        
        # Create cylinder parameters
        params = CylinderParameters(
            start_point=src_pos,
            end_point=tgt_pos,
            start_radius=src_radius,
            end_radius=tgt_radius,
            confidence=edge.get('confidence', 1.0)
        )
        
        return CylindricalTemplate(params)
    
    @staticmethod
    def create_bifurcation_from_node(graph: VascularGraph, node_idx: int,
                                   node_radius_scale: float = 1.0) -> Optional[BifurcationTemplate]:
        """Create bifurcation template from graph node"""
        node = graph.nodes[node_idx]
        
        # Only create bifurcation for nodes with exactly 3 connections
        connected_edges = [i for i, edge in enumerate(graph.edges) 
                          if edge['source'] == node_idx or edge['target'] == node_idx]
        
        if len(connected_edges) != 3:
            return None
        
        # Determine parent and child edges based on vessel hierarchy
        node_pos = np.array(node['position'][:3])
        node_radius = node.get('radius_voxels', 1.0) * node_radius_scale
        
        # Get connected nodes and their properties
        connected_nodes = []
        directions = []
        radii = []
        
        for edge_idx in connected_edges:
            edge = graph.edges[edge_idx]
            
            # Find the other node
            other_idx = edge['target'] if edge['source'] == node_idx else edge['source']
            other_node = graph.nodes[other_idx]
            
            other_pos = np.array(other_node['position'][:3])
            other_radius = other_node.get('radius_voxels', 1.0) * node_radius_scale
            
            # Direction from bifurcation to other node
            direction = other_pos - node_pos
            direction = direction / (np.linalg.norm(direction) + 1e-6)
            
            connected_nodes.append(other_idx)
            directions.append(direction)
            radii.append(other_radius)
        
        # Identify parent (largest radius) and children
        radii = np.array(radii)
        parent_idx = np.argmax(radii)
        child_indices = [i for i in range(3) if i != parent_idx]
        
        # Create bifurcation parameters
        params = BifurcationParameters(
            position=node_pos,
            parent_direction=-directions[parent_idx],  # Direction toward parent
            child1_direction=directions[child_indices[0]],
            child2_direction=directions[child_indices[1]],
            parent_radius=radii[parent_idx],
            child1_radius=radii[child_indices[0]],
            child2_radius=radii[child_indices[1]],
            confidence=node.get('confidence', 1.0)
        )
        
        return BifurcationTemplate(params)
    
    @staticmethod
    def create_templates_from_graph(graph: VascularGraph, 
                                  node_radius_scale: float = 1.0) -> Dict[str, List[VesselTemplate]]:
        """Create all templates from a vascular graph"""
        templates = {
            'cylinders': [],
            'bifurcations': []
        }
        
        # Create cylindrical templates for all edges
        for i, edge in enumerate(graph.edges):
            try:
                cylinder = TemplateFactory.create_cylinder_from_edge(
                    graph, i, node_radius_scale
                )
                templates['cylinders'].append(cylinder)
            except Exception as e:
                logging.warning(f"Failed to create cylinder for edge {i}: {e}")
        
        # Create bifurcation templates for branch nodes
        for i, node in enumerate(graph.nodes):
            if node.get('type') == 'bifurcation':
                try:
                    bifurcation = TemplateFactory.create_bifurcation_from_node(
                        graph, i, node_radius_scale
                    )
                    if bifurcation:
                        templates['bifurcations'].append(bifurcation)
                except Exception as e:
                    logging.warning(f"Failed to create bifurcation for node {i}: {e}")
        
        return templates


def test_templates():
    """Test template functionality"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Test cylindrical template
    cylinder_params = CylinderParameters(
        start_point=np.array([0, 0, 0]),
        end_point=np.array([10, 0, 0]),
        start_radius=2.0,
        end_radius=1.0
    )
    
    cylinder = CylindricalTemplate(cylinder_params)
    
    # Test points
    test_points = np.array([
        [5, 0, 0],      # Center
        [5, 1.5, 0],    # Inside
        [5, 2.5, 0],    # Outside
        [5, 0, 1.5],    # Inside
        [5, 0, 2.5],    # Outside
    ])
    
    sdf_values = cylinder.compute_sdf(test_points)
    print("Cylinder SDF values:", sdf_values)
    print("Inside mask:", sdf_values <= 0)
    
    # Test bifurcation template
    bifurcation_params = BifurcationParameters(
        position=np.array([0, 0, 0]),
        parent_direction=np.array([-1, 0, 0]),
        child1_direction=np.array([0.707, 0.707, 0]),
        child2_direction=np.array([0.707, -0.707, 0]),
        parent_radius=2.0,
        child1_radius=1.4,
        child2_radius=1.4
    )
    
    bifurcation = BifurcationTemplate(bifurcation_params)
    
    # Test bifurcation SDF
    bifurcation_sdf = bifurcation.compute_sdf(test_points)
    print("Bifurcation SDF values:", bifurcation_sdf)
    
    return cylinder, bifurcation


if __name__ == "__main__":
    # Run tests
    cylinder, bifurcation = test_templates()
    print("Template tests completed successfully!")