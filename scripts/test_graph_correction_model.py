#!/usr/bin/env python
"""
Test script for Graph Correction Model
Tests the complete graph correction architecture
"""

import sys
import torch
import logging
from pathlib import Path
import time
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.graph_correction import (
    GraphCorrectionModel, 
    GraphCorrectionLoss,
    create_data_loaders
)
from src.models.graph_extraction.vascular_graph import VascularGraph
from src.utils.graph_correspondence import create_correspondence_matcher


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('graph_correction_test.log')
        ]
    )
    return logging.getLogger(__name__)


def test_model_initialization():
    """Test model initialization and architecture"""
    logger = logging.getLogger(__name__)
    logger.info("Testing model initialization...")
    
    # Test with default configuration
    model = GraphCorrectionModel()
    summary = model.get_model_summary()
    
    logger.info(f"Model initialized successfully")
    logger.info(f"Total parameters: {summary['total_parameters']:,}")
    logger.info(f"Trainable parameters: {summary['trainable_parameters']:,}")
    logger.info(f"Component breakdown:")
    for component, params in summary['components'].items():
        logger.info(f"  {component}: {params:,} parameters")
    
    return model


def test_forward_pass():
    """Test forward pass with synthetic data"""
    logger = logging.getLogger(__name__)
    logger.info("Testing forward pass...")
    
    model = GraphCorrectionModel()
    model.eval()
    
    # Create synthetic data
    batch_size = 1
    num_nodes = 50
    num_edges = 80
    
    # Synthetic prediction graph features
    pred_features = {
        'node_features': torch.randn(num_nodes, 16),
        'edge_features': torch.randn(num_edges, 8),
        'edge_index': torch.randint(0, num_nodes, (2, num_edges)),
        'node_positions': torch.randn(num_nodes, 3),
        'node_radii': torch.abs(torch.randn(num_nodes)) + 0.5,
        'node_types': torch.randint(0, 3, (num_nodes,))
    }
    
    # Synthetic ground truth features
    gt_features = {
        'node_features': torch.randn(num_nodes, 16),
        'edge_features': torch.randn(num_edges, 8),
        'edge_index': torch.randint(0, num_nodes, (2, num_edges)),
        'node_positions': torch.randn(num_nodes, 3),
        'node_radii': torch.abs(torch.randn(num_nodes)) + 0.5,
        'node_types': torch.randint(0, 3, (num_nodes,))
    }
    
    # Synthetic correspondences
    from src.utils.graph_correspondence import CorrespondenceResult
    correspondences = CorrespondenceResult(
        node_correspondences={i: i for i in range(min(40, num_nodes))},
        node_confidences={i: 0.8 for i in range(min(40, num_nodes))},
        unmatched_pred_nodes=set(range(40, num_nodes)),
        unmatched_gt_nodes=set(),
        edge_correspondences={},
        edge_confidences={},
        unmatched_pred_edges=set(),
        unmatched_gt_edges=set(),
        topology_differences={},
        alignment_transform={},
        correspondence_quality={'overall_quality': 0.8},
        metadata={}
    )
    
    # Test forward pass
    with torch.no_grad():
        outputs = model(pred_features, gt_features, correspondences, training_mode=False)
    
    # Verify outputs
    expected_keys = [
        'node_operations', 'node_corrections', 'quality_score',
        'topology_outputs', 'anatomy_outputs', 'encoding_outputs'
    ]
    
    for key in expected_keys:
        if key in outputs:
            logger.info(f"✅ Output '{key}': shape {outputs[key].shape if hasattr(outputs[key], 'shape') else 'N/A'}")
        else:
            logger.warning(f"⚠️ Missing output: {key}")
    
    logger.info("Forward pass completed successfully")
    return outputs


def test_loss_computation():
    """Test loss function computation"""
    logger = logging.getLogger(__name__)
    logger.info("Testing loss computation...")
    
    # Create loss function
    loss_fn = GraphCorrectionLoss()
    
    # Create mock predictions and targets
    num_nodes = 50
    predictions = {
        'node_operations': torch.randn(num_nodes, 4),
        'node_corrections': torch.randn(num_nodes, 7),
        'topology_outputs': {
            'topology_quality_score': torch.tensor(0.7)
        },
        'anatomy_outputs': {
            'murray_analysis': {
                'murray_violations': [0.1, 0.2, 0.15],
                'compliance_scores': [0.9, 0.8, 0.85]
            },
            'tapering_analysis': {
                'avg_tapering_consistency': 0.85,
                'num_tapering_violations': 2
            }
        },
        'corrected_node_features': torch.randn(num_nodes, 128)
    }
    
    targets = {
        'node_op_targets': torch.randint(0, 4, (num_nodes,)),
        'node_correction_targets': torch.randn(num_nodes, 7),
        'edge_index': torch.randint(0, num_nodes, (2, 80))
    }
    
    correspondences = {
        'node_correspondences': {i: i for i in range(40)},
        'edge_correspondences': {}
    }
    
    # Compute loss
    losses = loss_fn(predictions, targets, correspondences)
    
    logger.info("Loss computation results:")
    for loss_name, loss_value in losses.items():
        logger.info(f"  {loss_name}: {loss_value.item():.4f}")
    
    logger.info("Loss computation completed successfully")
    return losses


def test_with_real_data():
    """Test with real extracted graph data"""
    logger = logging.getLogger(__name__)
    logger.info("Testing with real data...")
    
    extracted_graphs_dir = Path("extracted_graphs")
    
    if not extracted_graphs_dir.exists():
        logger.warning("No extracted graphs directory found. Skipping real data test.")
        return None
    
    # Find a sample
    sample_dirs = [d for d in extracted_graphs_dir.iterdir() 
                  if d.is_dir() and d.name.startswith('PA')]
    
    if not sample_dirs:
        logger.warning("No graph samples found. Skipping real data test.")
        return None
    
    sample_dir = sample_dirs[0]
    sample_id = sample_dir.name
    gt_file = sample_dir / f"{sample_id}_GT.pkl"
    
    if not gt_file.exists():
        logger.warning(f"No GT file found for {sample_id}. Skipping real data test.")
        return None
    
    try:
        # Load ground truth graph
        gt_graph = VascularGraph.load(gt_file)
        logger.info(f"Loaded GT graph: {len(gt_graph.nodes)} nodes, {len(gt_graph.edges)} edges")
        
        # Create synthetic prediction (since we may not have real predictions)
        pred_graph = create_synthetic_prediction(gt_graph, degradation_level=0.3)
        logger.info(f"Created pred graph: {len(pred_graph.nodes)} nodes, {len(pred_graph.edges)} edges")
        
        # Compute correspondences
        matcher = create_correspondence_matcher()
        correspondences = matcher.find_correspondence(pred_graph, gt_graph)
        logger.info(f"Found {len(correspondences.node_correspondences)} node correspondences")
        
        # Test model with real data
        model = GraphCorrectionModel()
        model.eval()
        
        with torch.no_grad():
            outputs = model(pred_graph, gt_graph, correspondences, training_mode=False)
        
        logger.info("✅ Real data test completed successfully")
        
        # Test correction application
        corrected_graph = model.apply_corrections(pred_graph, outputs, threshold=0.6)
        logger.info(f"Applied corrections: {len(corrected_graph.nodes)} nodes, {len(corrected_graph.edges)} edges")
        
        return outputs
        
    except Exception as e:
        logger.error(f"Real data test failed: {e}")
        return None


def create_synthetic_prediction(gt_graph, degradation_level=0.3):
    """Create synthetic prediction from ground truth"""
    import numpy as np
    
    # Copy and degrade nodes
    pred_nodes = []
    for node in gt_graph.nodes:
        pred_node = node.copy()
        
        # Add position noise
        if 'position' in node:
            position = np.array(node['position'])
            noise = np.random.normal(0, degradation_level, 3)
            pred_node['position'] = (position + noise).tolist()
        
        # Add radius noise
        if 'radius_voxels' in node:
            radius = node['radius_voxels']
            noise_factor = 1 + np.random.normal(0, degradation_level * 0.5)
            pred_node['radius_voxels'] = max(0.1, radius * noise_factor)
        
        pred_nodes.append(pred_node)
    
    # Keep most edges with some noise
    pred_edges = []
    for edge in gt_graph.edges:
        if np.random.random() > degradation_level * 0.2:
            pred_edge = edge.copy()
            
            # Add length noise
            if 'euclidean_length' in edge:
                length = edge['euclidean_length']
                noise_factor = 1 + np.random.normal(0, degradation_level * 0.3)
                pred_edge['euclidean_length'] = max(0.1, length * noise_factor)
            
            pred_edges.append(pred_edge)
    
    return VascularGraph(
        nodes=pred_nodes,
        edges=pred_edges,
        global_properties=gt_graph.global_properties.copy(),
        metadata={**gt_graph.metadata, 'synthetic_prediction': True}
    )


def test_training_stage_switching():
    """Test multi-stage training functionality"""
    logger = logging.getLogger(__name__)
    logger.info("Testing training stage switching...")
    
    model = GraphCorrectionModel()
    
    # Test each training stage
    for stage in [1, 2, 3]:
        model.set_training_stage(stage)
        logger.info(f"Set training stage to {stage}")
        
        # Check which parameters are trainable
        topology_trainable = any(p.requires_grad for p in model.topology_corrector.parameters())
        anatomy_trainable = any(p.requires_grad for p in model.anatomy_preserver.parameters())
        
        logger.info(f"  Topology trainable: {topology_trainable}")
        logger.info(f"  Anatomy trainable: {anatomy_trainable}")
    
    logger.info("Training stage switching test completed")


def run_all_tests():
    """Run all tests"""
    logger = setup_logging()
    logger.info("=== Graph Correction Model Tests ===")
    
    test_results = {}
    
    # Test 1: Model initialization
    try:
        model = test_model_initialization()
        test_results['initialization'] = True
    except Exception as e:
        logger.error(f"Initialization test failed: {e}")
        test_results['initialization'] = False
    
    # Test 2: Forward pass
    try:
        outputs = test_forward_pass()
        test_results['forward_pass'] = True
    except Exception as e:
        logger.error(f"Forward pass test failed: {e}")
        test_results['forward_pass'] = False
    
    # Test 3: Loss computation
    try:
        losses = test_loss_computation()
        test_results['loss_computation'] = True
    except Exception as e:
        logger.error(f"Loss computation test failed: {e}")
        test_results['loss_computation'] = False
    
    # Test 4: Real data (optional)
    try:
        real_outputs = test_with_real_data()
        test_results['real_data'] = real_outputs is not None
    except Exception as e:
        logger.error(f"Real data test failed: {e}")
        test_results['real_data'] = False
    
    # Test 5: Training stages
    try:
        test_training_stage_switching()
        test_results['training_stages'] = True
    except Exception as e:
        logger.error(f"Training stages test failed: {e}")
        test_results['training_stages'] = False
    
    # Summary
    logger.info("\n=== TEST SUMMARY ===")
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, passed in test_results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    # Save results
    results_file = Path("graph_correction_test_results.json")
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    logger.info(f"Test results saved to: {results_file}")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)