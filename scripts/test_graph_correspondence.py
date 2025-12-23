#!/usr/bin/env python
"""
Test script for Graph Correspondence Matching
Tests the correspondence matching implementation on sample extracted graphs
"""

import sys
import numpy as np
import logging
from pathlib import Path
import time
import json
from typing import List, Dict

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.graph_extraction.vascular_graph import VascularGraph
from src.utils.graph_correspondence import GraphCorrespondenceMatcher, create_correspondence_matcher


def setup_logging():
    """Setup logging for the test"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('correspondence_test.log')
        ]
    )
    return logging.getLogger(__name__)


def find_available_samples(extracted_graphs_dir: Path) -> List[str]:
    """Find available patient samples with both GT and PRED graphs"""
    samples = []
    
    if not extracted_graphs_dir.exists():
        return samples
    
    for patient_dir in extracted_graphs_dir.iterdir():
        if patient_dir.is_dir() and patient_dir.name.startswith('PA'):
            gt_file = patient_dir / f"{patient_dir.name}_GT.pkl"
            
            # For now, only check GT files since PRED files might not exist yet
            if gt_file.exists():
                samples.append(patient_dir.name)
    
    return sorted(samples)


def load_graph_safely(filepath: Path, logger: logging.Logger) -> VascularGraph:
    """Safely load a graph file with error handling"""
    try:
        graph = VascularGraph.load(filepath)
        logger.info(f"Loaded graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        return graph
    except Exception as e:
        logger.error(f"Failed to load graph from {filepath}: {e}")
        return None


def create_synthetic_prediction(gt_graph: VascularGraph, degradation_level: float = 0.3) -> VascularGraph:
    """
    Create a synthetic prediction graph by degrading the ground truth
    This simulates typical U-Net prediction errors
    """
    
    # Copy ground truth
    pred_nodes = []
    pred_edges = []
    
    # Add noise to node positions
    for i, node in enumerate(gt_graph.nodes):
        pred_node = node.copy()
        
        if 'position' in node:
            position = np.array(node['position'])
            # Add gaussian noise
            noise = np.random.normal(0, degradation_level, 3)
            pred_node['position'] = (position + noise).tolist()
        
        # Occasionally change node type (simulate detection errors)
        if np.random.random() < degradation_level * 0.3:
            original_type = node.get('type', 'regular')
            if original_type == 'bifurcation':
                pred_node['type'] = 'regular'
            elif original_type == 'regular' and np.random.random() < 0.1:
                pred_node['type'] = 'bifurcation'
        
        # Add noise to radius
        if 'radius_voxels' in node:
            radius = node['radius_voxels']
            noise_factor = 1 + np.random.normal(0, degradation_level * 0.5)
            pred_node['radius_voxels'] = max(0.1, radius * noise_factor)
        
        pred_nodes.append(pred_node)
    
    # Randomly remove some edges (simulate missing connections)
    for edge in gt_graph.edges:
        if np.random.random() > degradation_level * 0.2:  # Keep most edges
            pred_edge = edge.copy()
            
            # Add noise to edge attributes
            if 'euclidean_length' in edge:
                length = edge['euclidean_length']
                noise_factor = 1 + np.random.normal(0, degradation_level * 0.3)
                pred_edge['euclidean_length'] = max(0.1, length * noise_factor)
            
            pred_edges.append(pred_edge)
    
    # Create prediction graph
    pred_graph = VascularGraph(
        nodes=pred_nodes,
        edges=pred_edges,
        global_properties=gt_graph.global_properties.copy(),
        metadata={
            'original_gt': gt_graph.metadata,
            'degradation_level': degradation_level,
            'synthetic_prediction': True
        }
    )
    
    return pred_graph


def test_correspondence_matching(sample_id: str, 
                               gt_graph: VascularGraph,
                               pred_graph: VascularGraph,
                               logger: logging.Logger) -> Dict:
    """Test correspondence matching on a single sample"""
    
    logger.info(f"\n=== Testing Correspondence Matching for {sample_id} ===")
    
    # Create matcher with default configuration
    matcher = create_correspondence_matcher()
    
    logger.info(f"GT Graph: {len(gt_graph.nodes)} nodes, {len(gt_graph.edges)} edges")
    logger.info(f"Pred Graph: {len(pred_graph.nodes)} nodes, {len(pred_graph.edges)} edges")
    
    # Perform correspondence matching
    start_time = time.time()
    try:
        correspondence_result = matcher.find_correspondence(pred_graph, gt_graph)
        matching_time = time.time() - start_time
        
        # Extract results
        num_node_correspondences = len(correspondence_result.node_correspondences)
        num_edge_correspondences = len(correspondence_result.edge_correspondences)
        num_unmatched_pred_nodes = len(correspondence_result.unmatched_pred_nodes)
        num_unmatched_gt_nodes = len(correspondence_result.unmatched_gt_nodes)
        
        # Calculate success metrics
        node_coverage = num_node_correspondences / len(pred_graph.nodes) if len(pred_graph.nodes) > 0 else 0
        gt_node_coverage = num_node_correspondences / len(gt_graph.nodes) if len(gt_graph.nodes) > 0 else 0
        edge_coverage = num_edge_correspondences / len(pred_graph.edges) if len(pred_graph.edges) > 0 else 0
        
        # Average confidence
        avg_node_confidence = np.mean(list(correspondence_result.node_confidences.values())) if correspondence_result.node_confidences else 0
        avg_edge_confidence = np.mean(list(correspondence_result.edge_confidences.values())) if correspondence_result.edge_confidences else 0
        
        logger.info(f"✅ Correspondence matching completed in {matching_time:.2f}s")
        logger.info(f"Node correspondences: {num_node_correspondences} "
                   f"(coverage: {node_coverage:.1%} pred, {gt_node_coverage:.1%} GT)")
        logger.info(f"Edge correspondences: {num_edge_correspondences} "
                   f"(coverage: {edge_coverage:.1%})")
        logger.info(f"Unmatched nodes: {num_unmatched_pred_nodes} pred, {num_unmatched_gt_nodes} GT")
        logger.info(f"Average confidence: nodes {avg_node_confidence:.3f}, edges {avg_edge_confidence:.3f}")
        
        test_result = {
            'sample_id': sample_id,
            'success': True,
            'matching_time': matching_time,
            'node_correspondences': num_node_correspondences,
            'edge_correspondences': num_edge_correspondences,
            'node_coverage': node_coverage,
            'gt_node_coverage': gt_node_coverage,
            'edge_coverage': edge_coverage,
            'avg_node_confidence': avg_node_confidence,
            'avg_edge_confidence': avg_edge_confidence,
            'unmatched_pred_nodes': num_unmatched_pred_nodes,
            'unmatched_gt_nodes': num_unmatched_gt_nodes,
        }
        
        return test_result
        
    except Exception as e:
        logger.error(f"❌ Correspondence matching failed: {e}")
        return {
            'sample_id': sample_id,
            'success': False,
            'error': str(e),
            'matching_time': time.time() - start_time
        }


def run_correspondence_tests():
    """Run correspondence matching tests on available samples"""
    
    logger = setup_logging()
    logger.info("=== Graph Correspondence Matching Tests ===")
    
    # Find available samples
    extracted_graphs_dir = Path("extracted_graphs")
    samples = find_available_samples(extracted_graphs_dir)
    
    if not samples:
        logger.error("No extracted graph samples found!")
        return False
    
    logger.info(f"Found {len(samples)} samples: {samples[:5]}{'...' if len(samples) > 5 else ''}")
    
    # Test parameters
    test_samples = samples[:2]  # Test first 2 samples
    degradation_levels = [0.2, 0.3]  # Different noise levels
    
    all_results = []
    
    # Test each sample with different degradation levels
    for sample_id in test_samples:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing sample: {sample_id}")
        
        # Load ground truth graph
        gt_file = extracted_graphs_dir / sample_id / f"{sample_id}_GT.pkl"
        gt_graph = load_graph_safely(gt_file, logger)
        
        if gt_graph is None:
            logger.error(f"Skipping {sample_id} - could not load GT graph")
            continue
        
        # Test with different degradation levels
        for degradation_level in degradation_levels:
            logger.info(f"\n--- Testing with degradation level {degradation_level} ---")
            
            # Create synthetic prediction
            np.random.seed(42)  # For reproducibility
            pred_graph = create_synthetic_prediction(gt_graph, degradation_level)
            
            # Test correspondence matching
            test_result = test_correspondence_matching(
                f"{sample_id}_deg_{degradation_level}", 
                gt_graph, 
                pred_graph, 
                logger
            )
            
            test_result['degradation_level'] = degradation_level
            all_results.append(test_result)
    
    # Analyze overall results
    logger.info(f"\n{'='*60}")
    logger.info("=== OVERALL RESULTS ===")
    
    successful_tests = [r for r in all_results if r['success']]
    
    if successful_tests:
        # Calculate summary statistics
        avg_node_coverage = np.mean([r['node_coverage'] for r in successful_tests])
        avg_gt_coverage = np.mean([r['gt_node_coverage'] for r in successful_tests])
        avg_edge_coverage = np.mean([r['edge_coverage'] for r in successful_tests])
        avg_node_confidence = np.mean([r['avg_node_confidence'] for r in successful_tests])
        avg_matching_time = np.mean([r['matching_time'] for r in successful_tests])
        
        logger.info(f"Successful tests: {len(successful_tests)}/{len(all_results)}")
        logger.info(f"Average node coverage: {avg_node_coverage:.1%}")
        logger.info(f"Average GT node coverage: {avg_gt_coverage:.1%}")
        logger.info(f"Average edge coverage: {avg_edge_coverage:.1%}")
        logger.info(f"Average node confidence: {avg_node_confidence:.3f}")
        logger.info(f"Average matching time: {avg_matching_time:.2f}s")
        
        # Save results
        results_file = Path("correspondence_test_results.json")
        with open(results_file, 'w') as f:
            json.dump({
                'test_summary': {
                    'total_tests': len(all_results),
                    'successful_tests': len(successful_tests),
                    'success_rate': len(successful_tests) / len(all_results) if all_results else 0,
                    'avg_node_coverage': avg_node_coverage,
                    'avg_gt_coverage': avg_gt_coverage,
                    'avg_confidence': avg_node_confidence,
                    'avg_time': avg_matching_time,
                },
                'individual_results': all_results
            }, f, indent=2)
        
        logger.info(f"Results saved to: {results_file}")
    
    return len(successful_tests) > 0


if __name__ == "__main__":
    success = run_correspondence_tests()
    sys.exit(0 if success else 1)