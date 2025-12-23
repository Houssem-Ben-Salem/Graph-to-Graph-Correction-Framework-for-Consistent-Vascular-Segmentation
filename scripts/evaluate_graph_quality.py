#!/usr/bin/env python
"""
Comprehensive evaluation of graph extraction quality
"""

import sys
import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.graph_extraction import VascularGraph

def analyze_topology_metrics(graph: VascularGraph) -> dict:
    """Analyze topological properties of the graph"""
    nx_graph = graph.to_networkx()
    
    metrics = {}
    
    # Basic connectivity
    metrics['is_connected'] = graph.global_properties.get('is_connected', False)
    metrics['num_components'] = len(list(nx_graph.components)) if hasattr(nx_graph, 'components') else 1
    
    # Node degree analysis
    degrees = [nx_graph.degree(node) for node in nx_graph.nodes()]
    metrics['avg_degree'] = np.mean(degrees) if degrees else 0
    metrics['max_degree'] = np.max(degrees) if degrees else 0
    metrics['degree_distribution'] = np.bincount(degrees).tolist() if degrees else []
    
    # Path analysis
    if metrics['is_connected'] and len(nx_graph.nodes) > 1:
        import networkx as nx
        # Average shortest path length
        try:
            metrics['avg_path_length'] = nx.average_shortest_path_length(nx_graph)
        except:
            metrics['avg_path_length'] = 0
        
        # Diameter
        try:
            metrics['diameter'] = nx.diameter(nx_graph)
        except:
            metrics['diameter'] = 0
    else:
        metrics['avg_path_length'] = 0
        metrics['diameter'] = 0
    
    return metrics

def analyze_anatomical_consistency(graph: VascularGraph) -> dict:
    """Analyze anatomical consistency of the graph"""
    metrics = {}
    
    # Node type distribution
    node_types = graph.global_properties.get('node_type_counts', {})
    total_nodes = sum(node_types.values())
    
    if total_nodes > 0:
        metrics['bifurcation_ratio'] = node_types.get('bifurcations', 0) / total_nodes
        metrics['endpoint_ratio'] = node_types.get('endpoints', 0) / total_nodes
        metrics['regular_ratio'] = node_types.get('regular', 0) / total_nodes
    else:
        metrics['bifurcation_ratio'] = 0
        metrics['endpoint_ratio'] = 0
        metrics['regular_ratio'] = 0
    
    # Vessel radius analysis
    radii = []
    for node in graph.nodes:
        radius = node.get('radius_voxels', 0)
        if radius > 0:
            radii.append(radius)
    
    if radii:
        metrics['avg_radius'] = np.mean(radii)
        metrics['radius_std'] = np.std(radii)
        metrics['radius_range'] = [np.min(radii), np.max(radii)]
        metrics['radius_variation'] = np.std(radii) / np.mean(radii) if np.mean(radii) > 0 else 0
    else:
        metrics['avg_radius'] = 0
        metrics['radius_std'] = 0
        metrics['radius_range'] = [0, 0]
        metrics['radius_variation'] = 0
    
    # Edge length analysis
    edge_lengths = []
    for edge in graph.edges:
        length = edge.get('euclidean_length', 0)
        if length > 0:
            edge_lengths.append(length)
    
    if edge_lengths:
        metrics['avg_edge_length'] = np.mean(edge_lengths)
        metrics['edge_length_std'] = np.std(edge_lengths)
        metrics['edge_length_variation'] = np.std(edge_lengths) / np.mean(edge_lengths) if np.mean(edge_lengths) > 0 else 0
    else:
        metrics['avg_edge_length'] = 0
        metrics['edge_length_std'] = 0
        metrics['edge_length_variation'] = 0
    
    return metrics

def analyze_representation_quality(graph: VascularGraph) -> dict:
    """Analyze how well the graph represents the vascular structure"""
    metrics = {}
    
    # Density and coverage
    num_nodes = graph.global_properties.get('num_nodes', 0)
    num_edges = graph.global_properties.get('num_edges', 0)
    total_length = graph.global_properties.get('total_length_mm', 0)
    
    metrics['node_density'] = num_nodes / total_length if total_length > 0 else 0  # nodes/mm
    metrics['edge_density'] = num_edges / num_nodes if num_nodes > 0 else 0  # edges/node
    metrics['graph_density'] = graph.global_properties.get('density', 0)
    
    # Complexity
    metrics['complexity_score'] = graph.global_properties.get('complexity_score', 0)
    
    # Efficiency metrics
    if num_nodes > 0 and num_edges > 0:
        # How well connected the graph is
        max_possible_edges = num_nodes * (num_nodes - 1) // 2
        metrics['connectivity_efficiency'] = num_edges / max_possible_edges if max_possible_edges > 0 else 0
        
        # Tree-likeness (vascular structures should be tree-like)
        expected_tree_edges = max(0, num_nodes - 1)
        metrics['tree_likeness'] = expected_tree_edges / num_edges if num_edges > 0 else 0
    else:
        metrics['connectivity_efficiency'] = 0
        metrics['tree_likeness'] = 0
    
    return metrics

def evaluate_graph_quality():
    """Comprehensive graph quality evaluation"""
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("=== Graph Quality Evaluation ===")
    
    # Find available graph files
    graph_dirs = [
        Path("test_output/single_sample"),
        Path("test_output/improved_configs"),
        Path("test_output/crops")
    ]
    
    graph_files = []
    for dir_path in graph_dirs:
        if dir_path.exists():
            graph_files.extend(list(dir_path.glob("*.pkl")))
    
    if not graph_files:
        logger.error("No graph files found for evaluation!")
        return False
    
    logger.info(f"Found {len(graph_files)} graph files to evaluate")
    
    # Evaluation results
    evaluation_results = []
    
    for graph_file in graph_files:
        logger.info(f"\n--- Evaluating {graph_file.name} ---")
        
        try:
            # Load graph
            graph = VascularGraph.load(graph_file)
            
            result = {
                'name': graph_file.stem,
                'file': str(graph_file),
            }
            
            # Basic info
            result.update({
                'num_nodes': graph.global_properties.get('num_nodes', 0),
                'num_edges': graph.global_properties.get('num_edges', 0),
                'total_length_mm': graph.global_properties.get('total_length_mm', 0),
                'quality_score': graph.metadata.get('extraction_parameters', {}).get('quality_metrics', {}).get('overall_quality_score', 0)
            })
            
            # Detailed analysis
            topology_metrics = analyze_topology_metrics(graph)
            anatomy_metrics = analyze_anatomical_consistency(graph)
            representation_metrics = analyze_representation_quality(graph)
            
            result.update({
                'topology': topology_metrics,
                'anatomy': anatomy_metrics, 
                'representation': representation_metrics
            })
            
            evaluation_results.append(result)
            
            # Summary for this graph
            logger.info(f"Nodes: {result['num_nodes']}, Edges: {result['num_edges']}")
            logger.info(f"Connected: {topology_metrics['is_connected']}, Components: {topology_metrics['num_components']}")
            logger.info(f"Bifurcation ratio: {anatomy_metrics['bifurcation_ratio']:.3f}")
            logger.info(f"Regular node ratio: {anatomy_metrics['regular_ratio']:.3f}")
            logger.info(f"Node density: {representation_metrics['node_density']:.3f} nodes/mm")
            logger.info(f"Tree-likeness: {representation_metrics['tree_likeness']:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to evaluate {graph_file.name}: {e}")
    
    if not evaluation_results:
        logger.error("No graphs could be evaluated!")
        return False
    
    # Comparative analysis
    logger.info("\n=== COMPARATIVE ANALYSIS ===")
    
    # Create summary table
    summary_data = []
    for result in evaluation_results:
        row = {
            'Name': result['name'],
            'Nodes': result['num_nodes'],
            'Edges': result['num_edges'],
            'Connected': result['topology']['is_connected'],
            'Bifur_Ratio': result['anatomy']['bifurcation_ratio'],
            'Regular_Ratio': result['anatomy']['regular_ratio'],
            'Node_Density': result['representation']['node_density'],
            'Tree_Like': result['representation']['tree_likeness'],
            'Quality': result['quality_score']
        }
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    logger.info("\nSummary Table:")
    logger.info(df.to_string(index=False, float_format='%.3f'))
    
    # Find best configurations
    if len(evaluation_results) > 1:
        logger.info("\n=== RECOMMENDATIONS ===")
        
        # Best for regular nodes
        best_regular = max(evaluation_results, key=lambda x: x['anatomy']['regular_ratio'])
        logger.info(f"Best regular node representation: {best_regular['name']} ({best_regular['anatomy']['regular_ratio']:.3f})")
        
        # Best connectivity
        best_connected = max(evaluation_results, key=lambda x: x['topology']['is_connected'])
        logger.info(f"Best connectivity: {best_connected['name']}")
        
        # Best tree-like structure
        best_tree = max(evaluation_results, key=lambda x: x['representation']['tree_likeness'])
        logger.info(f"Most tree-like: {best_tree['name']} ({best_tree['representation']['tree_likeness']:.3f})")
        
        # Overall recommendation
        # Score based on multiple factors
        for result in evaluation_results:
            score = (
                result['topology']['is_connected'] * 0.3 +
                result['anatomy']['regular_ratio'] * 0.3 +
                result['representation']['tree_likeness'] * 0.2 +
                min(result['quality_score'], 1.0) * 0.2
            )
            result['overall_score'] = score
        
        best_overall = max(evaluation_results, key=lambda x: x['overall_score'])
        logger.info(f"\n🏆 BEST OVERALL: {best_overall['name']} (score: {best_overall['overall_score']:.3f})")
        logger.info("This configuration provides the best balance of connectivity, node representation, and quality.")
    
    # Save detailed results
    output_file = Path("test_output/graph_quality_evaluation.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    import json
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    results_clean = convert_numpy_types(evaluation_results)
    
    with open(output_file, 'w') as f:
        json.dump(results_clean, f, indent=2)
    
    logger.info(f"\nDetailed results saved to: {output_file}")
    
    return True

if __name__ == "__main__":
    success = evaluate_graph_quality()
    sys.exit(0 if success else 1)