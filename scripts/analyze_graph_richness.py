#!/usr/bin/env python
"""
Detailed analysis of graph information richness for GNN learning
"""

import sys
import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.graph_extraction import VascularGraph

def analyze_spatial_coverage(graph: VascularGraph) -> dict:
    """Analyze how well nodes cover the vessel structure spatially"""
    
    # Get node positions
    node_positions = []
    for node in graph.nodes:
        if 'coordinates_voxel' in node:
            pos = node['coordinates_voxel']
            if hasattr(pos, '__len__') and len(pos) >= 3:
                node_positions.append(pos[:3])
    
    if not node_positions:
        return {'error': 'No valid node positions found'}
    
    node_positions = np.array(node_positions)
    
    # Spatial distribution analysis
    coverage = {}
    
    # Volume coverage
    min_coords = np.min(node_positions, axis=0)
    max_coords = np.max(node_positions, axis=0)
    volume_span = np.prod(max_coords - min_coords)
    
    coverage['spatial_span'] = (max_coords - min_coords).tolist()
    coverage['volume_span'] = float(volume_span)
    coverage['node_density_per_volume'] = len(node_positions) / volume_span if volume_span > 0 else 0
    
    # Distance analysis between nodes
    from scipy.spatial.distance import pdist
    distances = pdist(node_positions)
    
    coverage['min_node_distance'] = float(np.min(distances))
    coverage['max_node_distance'] = float(np.max(distances))
    coverage['avg_node_distance'] = float(np.mean(distances))
    coverage['std_node_distance'] = float(np.std(distances))
    
    # Check for spatial gaps (regions without nodes)
    # Grid-based analysis
    grid_size = 8  # 8x8x8 grid
    grid_spans = (max_coords - min_coords) / grid_size
    grid_counts = np.zeros((grid_size, grid_size, grid_size))
    
    for pos in node_positions:
        grid_idx = ((pos - min_coords) / grid_spans).astype(int)
        grid_idx = np.clip(grid_idx, 0, grid_size-1)
        grid_counts[grid_idx[0], grid_idx[1], grid_idx[2]] += 1
    
    empty_cells = np.sum(grid_counts == 0)
    coverage['empty_grid_cells'] = int(empty_cells)
    coverage['empty_cell_ratio'] = float(empty_cells / (grid_size**3))
    
    return coverage

def analyze_vessel_representation(graph: VascularGraph) -> dict:
    """Analyze how well the graph represents vessel characteristics"""
    
    representation = {}
    
    # Node type distribution
    node_types = graph.global_properties.get('node_type_counts', {})
    total_nodes = sum(node_types.values())
    
    if total_nodes > 0:
        representation['node_type_ratios'] = {
            k: v/total_nodes for k, v in node_types.items()
        }
    else:
        representation['node_type_ratios'] = {}
    
    # Vessel radius representation
    radii = []
    radius_positions = []
    
    for node in graph.nodes:
        radius = node.get('radius_voxels', 0)
        if radius > 0:
            radii.append(radius)
            if 'coordinates_voxel' in node:
                radius_positions.append(node['coordinates_voxel'])
    
    if radii:
        representation['radius_stats'] = {
            'min': float(np.min(radii)),
            'max': float(np.max(radii)),
            'mean': float(np.mean(radii)),
            'std': float(np.std(radii)),
            'range_ratio': float(np.max(radii) / np.min(radii)) if np.min(radii) > 0 else 0
        }
        
        # Check for radius diversity (important for vessel hierarchy)
        unique_radii = len(np.unique(np.round(radii, 1)))
        representation['radius_diversity'] = unique_radii
        representation['radius_diversity_ratio'] = unique_radii / len(radii)
    else:
        representation['radius_stats'] = {}
        representation['radius_diversity'] = 0
        representation['radius_diversity_ratio'] = 0
    
    # Edge length distribution (for continuity analysis)
    edge_lengths = []
    for edge in graph.edges:
        length = edge.get('euclidean_length', 0)
        if length > 0:
            edge_lengths.append(length)
    
    if edge_lengths:
        representation['edge_length_stats'] = {
            'min': float(np.min(edge_lengths)),
            'max': float(np.max(edge_lengths)),
            'mean': float(np.mean(edge_lengths)),
            'std': float(np.std(edge_lengths)),
            'variation_coeff': float(np.std(edge_lengths) / np.mean(edge_lengths))
        }
    else:
        representation['edge_length_stats'] = {}
    
    return representation

def analyze_learning_potential(graph: VascularGraph) -> dict:
    """Analyze the graph's potential for GNN learning"""
    
    learning = {}
    
    # Local neighborhood richness
    import networkx as nx
    nx_graph = graph.to_networkx()
    
    # Degree distribution (important for message passing)
    degrees = [nx_graph.degree(node) for node in nx_graph.nodes()]
    if degrees:
        learning['degree_stats'] = {
            'min': int(np.min(degrees)),
            'max': int(np.max(degrees)),
            'mean': float(np.mean(degrees)),
            'std': float(np.std(degrees))
        }
        
        # Degree diversity (variety of local structures)
        unique_degrees = len(np.unique(degrees))
        learning['degree_diversity'] = unique_degrees
        learning['degree_entropy'] = float(-np.sum(np.bincount(degrees)/len(degrees) * 
                                                 np.log(np.bincount(degrees)/len(degrees) + 1e-10)))
    else:
        learning['degree_stats'] = {}
        learning['degree_diversity'] = 0
        learning['degree_entropy'] = 0
    
    # Local clustering (how well connected local neighborhoods are)
    if len(nx_graph.nodes()) > 2:
        try:
            clustering_coeffs = list(nx.clustering(nx_graph).values())
            learning['clustering_stats'] = {
                'mean': float(np.mean(clustering_coeffs)),
                'std': float(np.std(clustering_coeffs))
            }
        except:
            learning['clustering_stats'] = {'mean': 0, 'std': 0}
    else:
        learning['clustering_stats'] = {'mean': 0, 'std': 0}
    
    # Path diversity (variety of path lengths for learning long-range dependencies)
    if nx.is_connected(nx_graph) and len(nx_graph.nodes()) > 1:
        try:
            # Sample path lengths
            nodes = list(nx_graph.nodes())
            sample_size = min(100, len(nodes))
            sampled_nodes = np.random.choice(nodes, sample_size, replace=False)
            
            path_lengths = []
            for i in range(min(20, len(sampled_nodes))):
                for j in range(i+1, min(i+10, len(sampled_nodes))):
                    try:
                        path_len = nx.shortest_path_length(nx_graph, sampled_nodes[i], sampled_nodes[j])
                        path_lengths.append(path_len)
                    except:
                        pass
            
            if path_lengths:
                learning['path_length_stats'] = {
                    'min': int(np.min(path_lengths)),
                    'max': int(np.max(path_lengths)),
                    'mean': float(np.mean(path_lengths)),
                    'diversity': len(np.unique(path_lengths))
                }
            else:
                learning['path_length_stats'] = {}
        except:
            learning['path_length_stats'] = {}
    else:
        learning['path_length_stats'] = {}
    
    return learning

def assess_information_richness():
    """Comprehensive assessment of graph information richness"""
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("=== Graph Information Richness Analysis ===")
    
    # Load the optimal graph
    graph_file = Path("test_output/improved_configs/PA000296_Optimal.pkl")
    
    if not graph_file.exists():
        logger.error(f"Graph file not found: {graph_file}")
        return False
    
    try:
        graph = VascularGraph.load(graph_file)
        logger.info(f"Loaded graph: {graph}")
        
        # Basic statistics
        stats = graph.global_properties
        logger.info(f"Nodes: {stats.get('num_nodes', 0)}")
        logger.info(f"Edges: {stats.get('num_edges', 0)}")
        logger.info(f"Total length: {stats.get('total_length_mm', 0):.1f} mm")
        
        # Detailed analysis
        logger.info("\n--- Spatial Coverage Analysis ---")
        spatial_analysis = analyze_spatial_coverage(graph)
        
        if 'error' not in spatial_analysis:
            logger.info(f"Spatial span: {spatial_analysis['spatial_span']}")
            logger.info(f"Average node distance: {spatial_analysis['avg_node_distance']:.2f} voxels")
            logger.info(f"Min node distance: {spatial_analysis['min_node_distance']:.2f} voxels")
            logger.info(f"Empty grid cells: {spatial_analysis['empty_cell_ratio']:.1%}")
        
        logger.info("\n--- Vessel Representation Analysis ---")
        vessel_analysis = analyze_vessel_representation(graph)
        
        node_ratios = vessel_analysis.get('node_type_ratios', {})
        logger.info(f"Regular node ratio: {node_ratios.get('regular', 0):.3f}")
        logger.info(f"Bifurcation ratio: {node_ratios.get('bifurcations', 0):.3f}")
        
        radius_stats = vessel_analysis.get('radius_stats', {})
        if radius_stats:
            logger.info(f"Radius range: {radius_stats['min']:.1f} - {radius_stats['max']:.1f} voxels")
            logger.info(f"Radius diversity: {vessel_analysis.get('radius_diversity', 0)} unique values")
        
        edge_stats = vessel_analysis.get('edge_length_stats', {})
        if edge_stats:
            logger.info(f"Edge length variation: {edge_stats['variation_coeff']:.3f}")
        
        logger.info("\n--- Learning Potential Analysis ---")
        learning_analysis = analyze_learning_potential(graph)
        
        degree_stats = learning_analysis.get('degree_stats', {})
        if degree_stats:
            logger.info(f"Degree range: {degree_stats['min']} - {degree_stats['max']}")
            logger.info(f"Degree diversity: {learning_analysis.get('degree_diversity', 0)} unique values")
            logger.info(f"Degree entropy: {learning_analysis.get('degree_entropy', 0):.3f}")
        
        clustering_stats = learning_analysis.get('clustering_stats', {})
        logger.info(f"Average clustering: {clustering_stats.get('mean', 0):.3f}")
        
        # Overall assessment
        logger.info("\n=== RICHNESS ASSESSMENT ===")
        
        # Scoring criteria for GNN learning
        scores = {}
        
        # 1. Node density score (0-1)
        # Good range: 0.05-0.2 nodes/mm for vascular structures
        node_density = stats.get('total_length_mm', 0)
        if node_density > 0:
            density_ratio = stats.get('num_nodes', 0) / node_density
            scores['node_density'] = min(1.0, max(0.0, (density_ratio - 0.01) / 0.15))
        else:
            scores['node_density'] = 0
        
        # 2. Regular node representation (0-1)
        regular_ratio = node_ratios.get('regular', 0)
        scores['regular_nodes'] = min(1.0, regular_ratio * 3)  # Target ~33% regular nodes
        
        # 3. Structural diversity (0-1)
        degree_diversity = learning_analysis.get('degree_diversity', 0)
        max_expected_diversity = min(10, stats.get('num_nodes', 0) // 10)
        scores['structural_diversity'] = min(1.0, degree_diversity / max_expected_diversity) if max_expected_diversity > 0 else 0
        
        # 4. Spatial coverage (0-1)
        empty_ratio = spatial_analysis.get('empty_cell_ratio', 1.0)
        scores['spatial_coverage'] = 1.0 - empty_ratio
        
        # Overall richness score
        overall_score = np.mean(list(scores.values()))
        
        logger.info(f"Node density score: {scores['node_density']:.3f}")
        logger.info(f"Regular nodes score: {scores['regular_nodes']:.3f}")
        logger.info(f"Structural diversity score: {scores['structural_diversity']:.3f}")
        logger.info(f"Spatial coverage score: {scores['spatial_coverage']:.3f}")
        logger.info(f"\n🎯 OVERALL RICHNESS SCORE: {overall_score:.3f}")
        
        # Interpretation
        if overall_score >= 0.7:
            logger.info("✅ RICH: Graph has sufficient information for effective GNN learning")
        elif overall_score >= 0.5:
            logger.info("⚠️  MODERATE: Graph may need optimization for better learning")
        else:
            logger.info("❌ SPARSE: Graph likely insufficient for effective GNN learning")
        
        # Recommendations
        logger.info("\n📋 RECOMMENDATIONS:")
        
        if scores['regular_nodes'] < 0.5:
            logger.info("• Increase regular node density along vessel segments")
        
        if scores['spatial_coverage'] < 0.7:
            logger.info("• Improve spatial coverage to reduce gaps")
        
        if scores['structural_diversity'] < 0.5:
            logger.info("• Enhance node placement to capture more structural variety")
        
        if scores['node_density'] < 0.5:
            logger.info("• Consider increasing overall node density")
        
        return overall_score >= 0.6  # Return True if reasonably rich
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = assess_information_richness()
    sys.exit(0 if success else 1)