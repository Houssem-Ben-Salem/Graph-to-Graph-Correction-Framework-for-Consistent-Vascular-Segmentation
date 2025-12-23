"""
VascularGraph Data Structure
Implements the main data structure for representing vascular networks as graphs
"""

import numpy as np
import networkx as nx
import json
import pickle
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import logging


class VascularGraph:
    """
    Main data structure for representing vascular networks as graphs
    Compatible with NetworkX, DGL, and PyTorch Geometric
    """
    
    def __init__(self,
                 nodes: Optional[List[Dict]] = None,
                 edges: Optional[List[Dict]] = None,
                 global_properties: Optional[Dict] = None,
                 metadata: Optional[Dict] = None):
        """
        Initialize VascularGraph
        
        Args:
            nodes: List of node dictionaries with attributes
            edges: List of edge dictionaries with attributes
            global_properties: Global graph properties
            metadata: Metadata about graph extraction
        """
        self.nodes = nodes or []
        self.edges = edges or []
        self.global_properties = global_properties or {}
        self.metadata = metadata or {}
        self.logger = logging.getLogger(__name__)
        
        # Internal graph representations
        self._networkx_graph = None
        self._adjacency_matrix = None
        
    @classmethod
    def from_extraction_results(cls,
                              nodes: Dict,
                              node_attributes: List[Dict],
                              edges: Dict,
                              centerline_data: Dict,
                              preprocessing_info: Dict,
                              quality_metrics: Dict,
                              voxel_spacing: Optional[Tuple[float, float, float]] = None,
                              original_mask_shape: Optional[Tuple[int, int, int]] = None) -> 'VascularGraph':
        """
        Create VascularGraph from extraction pipeline results
        
        Args:
            nodes: Node placement results
            node_attributes: Node attributes
            edges: Edge creation results
            centerline_data: Centerline extraction results
            preprocessing_info: Preprocessing information
            quality_metrics: Quality assessment metrics
            voxel_spacing: Voxel spacing in mm
            original_mask_shape: Shape of original mask
            
        Returns:
            VascularGraph instance
        """
        # Prepare node list with IDs and attributes
        node_list = []
        for i, attrs in enumerate(node_attributes):
            node_dict = {
                'id': i,
                'position': nodes['positions'][i] if i < len(nodes['positions']) else None,
                'type': nodes['types'][i] if i < len(nodes['types']) else 'regular',
                **attrs
            }
            node_list.append(node_dict)
        
        # Prepare edge list with attributes
        edge_list = []
        for i, edge in enumerate(edges['edges']):
            edge_attrs = edges['edge_attributes'][i] if i < len(edges['edge_attributes']) else {}
            edge_dict = {
                'source': edge[0],
                'target': edge[1],
                **edge_attrs
            }
            edge_list.append(edge_dict)
        
        # Calculate global properties
        global_props = VascularGraph._calculate_global_properties(
            node_list, edge_list, centerline_data, voxel_spacing
        )
        
        # Prepare metadata
        metadata = {
            'original_mask_shape': original_mask_shape,
            'voxel_spacing': voxel_spacing,
            'extraction_parameters': {
                'preprocessing': preprocessing_info,
                'quality_metrics': quality_metrics,
                'centerline_metrics': centerline_data.get('metrics', {}),
                'critical_points': nodes.get('critical_points', {})
            },
            'extraction_timestamp': None,  # Could add timestamp
            'graph_version': '1.0'
        }
        
        return cls(
            nodes=node_list,
            edges=edge_list,
            global_properties=global_props,
            metadata=metadata
        )
    
    @staticmethod
    def _calculate_global_properties(node_list: List[Dict],
                                   edge_list: List[Dict],
                                   centerline_data: Dict,
                                   voxel_spacing: Optional[Tuple[float, float, float]]) -> Dict:
        """Calculate global graph properties"""
        num_nodes = len(node_list)
        num_edges = len(edge_list)
        
        # Count node types
        node_types = [node.get('type', 'regular') for node in node_list]
        type_counts = {
            'bifurcations': node_types.count('bifurcation'),
            'endpoints': node_types.count('endpoint'),
            'regular': node_types.count('regular'),
            'buffer': node_types.count('buffer')
        }
        
        # Calculate total length
        total_length_voxels = sum(edge.get('euclidean_length', 0) for edge in edge_list)
        total_length_mm = None
        if voxel_spacing is not None:
            total_length_mm = sum(edge.get('physical_length', 0) for edge in edge_list)
        
        # Calculate complexity score
        complexity_score = VascularGraph._calculate_complexity_score(node_list, edge_list)
        
        # Average radius
        radii = [node.get('radius_voxels', 0) for node in node_list if node.get('radius_voxels') is not None]
        average_radius = np.mean(radii) if radii else 0
        
        return {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'node_type_counts': type_counts,
            'total_length_voxels': total_length_voxels,
            'total_length_mm': total_length_mm,
            'average_radius_voxels': average_radius,
            'complexity_score': complexity_score,
            'is_connected': num_nodes > 0 and num_edges >= num_nodes - 1,
            'density': (2 * num_edges) / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        }
    
    @staticmethod
    def _calculate_complexity_score(node_list: List[Dict], edge_list: List[Dict]) -> float:
        """Calculate graph complexity score"""
        if not node_list:
            return 0.0
        
        # Based on number of bifurcations, path length variation, etc.
        num_bifurcations = sum(1 for node in node_list if node.get('type') == 'bifurcation')
        num_nodes = len(node_list)
        
        # Normalize by number of nodes
        bifurcation_ratio = num_bifurcations / num_nodes if num_nodes > 0 else 0
        
        # Add edge length variation
        edge_lengths = [edge.get('euclidean_length', 0) for edge in edge_list]
        length_variation = np.std(edge_lengths) if len(edge_lengths) > 1 else 0
        
        # Simple complexity score
        complexity = bifurcation_ratio + 0.1 * length_variation
        
        return float(complexity)
    
    def to_networkx(self, include_positions: bool = True) -> nx.Graph:
        """Convert to NetworkX graph"""
        if self._networkx_graph is None:
            G = nx.Graph()
            
            # Add nodes with attributes
            for node in self.nodes:
                node_attrs = node.copy()
                node_id = node_attrs.pop('id')
                
                if include_positions and 'position' in node_attrs:
                    # Add position for visualization
                    pos = node_attrs['position']
                    if hasattr(pos, '__len__') and len(pos) >= 3:
                        node_attrs['pos'] = tuple(pos[:3])
                
                G.add_node(node_id, **node_attrs)
            
            # Add edges with attributes
            for edge in self.edges:
                edge_attrs = edge.copy()
                source = edge_attrs.pop('source')
                target = edge_attrs.pop('target')
                G.add_edge(source, target, **edge_attrs)
            
            self._networkx_graph = G
        
        return self._networkx_graph
    
    def to_adjacency_matrix(self) -> np.ndarray:
        """Get adjacency matrix representation"""
        if self._adjacency_matrix is None:
            num_nodes = len(self.nodes)
            adjacency = np.zeros((num_nodes, num_nodes), dtype=bool)
            
            for edge in self.edges:
                i, j = edge['source'], edge['target']
                adjacency[i, j] = True
                adjacency[j, i] = True  # Undirected
            
            self._adjacency_matrix = adjacency
        
        return self._adjacency_matrix
    
    def to_pytorch_geometric(self):
        """Convert to PyTorch Geometric format"""
        try:
            import torch
            from torch_geometric.data import Data
        except ImportError:
            raise ImportError("PyTorch Geometric not available. Install with: pip install torch-geometric")
        
        # Node features
        node_features = []
        for node in self.nodes:
            features = [
                node.get('radius_voxels', 0),
                node.get('local_curvature', 0),
                node.get('degree', 0),
                float(node.get('type') == 'bifurcation'),
                float(node.get('type') == 'endpoint'),
            ]
            node_features.append(features)
        
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Edge indices
        edge_indices = []
        for edge in self.edges:
            edge_indices.append([edge['source'], edge['target']])
            edge_indices.append([edge['target'], edge['source']])  # Undirected
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        
        # Edge features
        edge_features = []
        for edge in self.edges:
            features = [
                edge.get('euclidean_length', 0),
                edge.get('average_radius', 0),
                edge.get('path_curvature', 0),
                edge.get('confidence_score', 1.0),
            ]
            edge_features.extend([features, features])  # Duplicate for undirected
        
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
        
        # Positions
        positions = [node['position'] for node in self.nodes if 'position' in node]
        pos = torch.tensor(positions, dtype=torch.float) if positions else None
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
    
    def save(self, filepath: Path, format: str = 'pickle'):
        """
        Save graph to file
        
        Args:
            filepath: Path to save file
            format: Format ('pickle', 'json', 'graphml')
        """
        filepath = Path(filepath)
        
        if format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
        
        elif format == 'json':
            # Convert numpy arrays to lists for JSON serialization
            data = self._prepare_for_json()
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        
        elif format == 'graphml':
            G = self.to_networkx()
            nx.write_graphml(G, filepath)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Graph saved to {filepath} in {format} format")
    
    @classmethod
    def load(cls, filepath: Path, format: str = 'pickle') -> 'VascularGraph':
        """
        Load graph from file
        
        Args:
            filepath: Path to load file
            format: Format ('pickle', 'json', 'graphml')
            
        Returns:
            VascularGraph instance
        """
        filepath = Path(filepath)
        
        if format == 'pickle':
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        
        elif format == 'json':
            with open(filepath, 'r') as f:
                data = json.load(f)
            return cls._from_json_data(data)
        
        elif format == 'graphml':
            G = nx.read_graphml(filepath)
            return cls._from_networkx(G)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _prepare_for_json(self) -> Dict:
        """Prepare graph data for JSON serialization"""
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        return {
            'nodes': convert_numpy(self.nodes),
            'edges': convert_numpy(self.edges),
            'global_properties': convert_numpy(self.global_properties),
            'metadata': convert_numpy(self.metadata)
        }
    
    @classmethod
    def _from_json_data(cls, data: Dict) -> 'VascularGraph':
        """Create VascularGraph from JSON data"""
        return cls(
            nodes=data.get('nodes', []),
            edges=data.get('edges', []),
            global_properties=data.get('global_properties', {}),
            metadata=data.get('metadata', {})
        )
    
    @classmethod
    def _from_networkx(cls, G: nx.Graph) -> 'VascularGraph':
        """Create VascularGraph from NetworkX graph"""
        nodes = []
        for node_id, attrs in G.nodes(data=True):
            node_dict = {'id': node_id, **attrs}
            nodes.append(node_dict)
        
        edges = []
        for source, target, attrs in G.edges(data=True):
            edge_dict = {'source': source, 'target': target, **attrs}
            edges.append(edge_dict)
        
        return cls(nodes=nodes, edges=edges)
    
    def get_summary(self) -> str:
        """Get human-readable summary of the graph"""
        summary = ["=== Vascular Graph Summary ==="]
        
        # Basic statistics
        summary.append(f"Nodes: {self.global_properties.get('num_nodes', 0)}")
        summary.append(f"Edges: {self.global_properties.get('num_edges', 0)}")
        
        # Node types
        type_counts = self.global_properties.get('node_type_counts', {})
        summary.append("Node types:")
        for node_type, count in type_counts.items():
            summary.append(f"  {node_type}: {count}")
        
        # Physical properties
        total_length = self.global_properties.get('total_length_mm')
        if total_length:
            summary.append(f"Total length: {total_length:.2f} mm")
        
        avg_radius = self.global_properties.get('average_radius_voxels', 0)
        summary.append(f"Average radius: {avg_radius:.2f} voxels")
        
        # Connectivity
        is_connected = self.global_properties.get('is_connected', False)
        summary.append(f"Connected: {is_connected}")
        
        complexity = self.global_properties.get('complexity_score', 0)
        summary.append(f"Complexity score: {complexity:.3f}")
        
        return "\n".join(summary)
    
    def __repr__(self) -> str:
        num_nodes = len(self.nodes)
        num_edges = len(self.edges)
        return f"VascularGraph(nodes={num_nodes}, edges={num_edges})"