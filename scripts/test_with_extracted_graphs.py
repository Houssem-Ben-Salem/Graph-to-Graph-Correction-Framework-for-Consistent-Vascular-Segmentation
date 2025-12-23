#!/usr/bin/env python
"""
Test Graph Correction Model with Real Extracted Graphs Data
Comprehensive testing with the actual extracted graphs from our dataset
"""

import sys
import torch
import logging
from pathlib import Path
import time
import json
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.graph_correction import GraphCorrectionModel, GraphCorrectionLoss
from src.models.graph_extraction.vascular_graph import VascularGraph
from src.utils.graph_correspondence import create_correspondence_matcher


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('extracted_graphs_test.log')
        ]
    )
    return logging.getLogger(__name__)


def find_available_samples(extracted_graphs_dir):
    """Find all available samples with GT graphs"""
    samples = []
    
    if not extracted_graphs_dir.exists():
        return samples
    
    for patient_dir in extracted_graphs_dir.iterdir():
        if patient_dir.is_dir() and patient_dir.name.startswith('PA'):
            gt_file = patient_dir / f"{patient_dir.name}_GT.pkl"
            if gt_file.exists():
                samples.append(patient_dir.name)
    
    return sorted(samples)


def load_graph_safely(filepath, logger):
    """Safely load a graph with error handling"""
    try:
        graph = VascularGraph.load(filepath)
        logger.info(f"Loaded graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        return graph
    except Exception as e:
        logger.error(f"Failed to load graph from {filepath}: {e}")
        return None


def create_synthetic_prediction_from_gt(gt_graph, degradation_level=0.3, seed=42):
    """Create a realistic synthetic prediction from ground truth"""
    np.random.seed(seed)
    
    pred_nodes = []
    pred_edges = []
    
    # Process nodes with degradation
    for i, node in enumerate(gt_graph.nodes):
        pred_node = node.copy()
        
        # Add spatial noise to position
        if 'position' in node and len(node['position']) >= 3:
            position = np.array(node['position'][:3])
            spatial_noise = np.random.normal(0, degradation_level * 2, 3)
            pred_node['position'] = (position + spatial_noise).tolist()
        
        # Add noise to radius
        if 'radius_voxels' in node:
            radius = float(node['radius_voxels'])
            radius_noise = 1 + np.random.normal(0, degradation_level * 0.4)
            pred_node['radius_voxels'] = max(0.1, radius * radius_noise)
        
        # Modify curvature slightly
        if 'local_curvature' in node:
            curvature = float(node['local_curvature'])
            curvature_noise = 1 + np.random.normal(0, degradation_level * 0.3)
            pred_node['local_curvature'] = curvature * curvature_noise
        
        # Occasionally change node type (simulate detection errors)
        if np.random.random() < degradation_level * 0.2:
            original_type = node.get('type', 'regular')
            if original_type == 'bifurcation':
                pred_node['type'] = 'regular'
            elif original_type == 'regular' and np.random.random() < 0.15:
                pred_node['type'] = 'bifurcation'
        
        pred_nodes.append(pred_node)
    
    # Process edges with some removal and modification
    for edge in gt_graph.edges:
        # Remove some edges to simulate missing connections
        if np.random.random() < degradation_level * 0.1:
            continue  # Skip this edge (simulate missing connection)
        
        pred_edge = edge.copy()
        
        # Add noise to edge length
        if 'euclidean_length' in edge:
            length = float(edge['euclidean_length'])
            length_noise = 1 + np.random.normal(0, degradation_level * 0.25)
            pred_edge['euclidean_length'] = max(0.1, length * length_noise)
        
        # Modify average radius
        if 'average_radius' in edge:
            avg_radius = float(edge['average_radius'])
            radius_noise = 1 + np.random.normal(0, degradation_level * 0.3)
            pred_edge['average_radius'] = max(0.1, avg_radius * radius_noise)
        
        # Add some confidence scores (simulate U-Net predictions)
        pred_edge['confidence_score'] = np.random.uniform(0.6, 0.95)
        
        pred_edges.append(pred_edge)
    
    # Add some confidence scores to nodes
    for node in pred_nodes:
        node['prediction_confidence'] = np.random.uniform(0.5, 0.9)
        node['detection_uncertainty'] = np.random.uniform(0.1, 0.4)
    
    # Create prediction graph
    pred_graph = VascularGraph(
        nodes=pred_nodes,
        edges=pred_edges,
        global_properties=gt_graph.global_properties.copy(),
        metadata={
            **gt_graph.metadata,
            'synthetic_prediction': True,
            'degradation_level': degradation_level,
            'prediction_source': 'synthetic_unet_simulation'
        }
    )
    
    return pred_graph


def test_single_sample(sample_id, extracted_graphs_dir, model, logger):
    """Test the model on a single sample"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing sample: {sample_id}")
    
    # Load ground truth graph
    gt_file = extracted_graphs_dir / sample_id / f"{sample_id}_GT.pkl"
    gt_graph = load_graph_safely(gt_file, logger)
    
    if gt_graph is None:
        logger.error(f"Could not load GT graph for {sample_id}")
        return None
    
    # Create synthetic prediction
    pred_graph = create_synthetic_prediction_from_gt(gt_graph, degradation_level=0.3)
    logger.info(f"Created synthetic prediction: {len(pred_graph.nodes)} nodes, {len(pred_graph.edges)} edges")
    
    # Compute correspondences
    logger.info("Computing correspondences...")
    start_time = time.time()
    
    try:
        matcher = create_correspondence_matcher()
        correspondences = matcher.find_correspondence(pred_graph, gt_graph)
        correspondence_time = time.time() - start_time
        
        logger.info(f"Correspondences computed in {correspondence_time:.2f}s")
        logger.info(f"Found {len(correspondences.node_correspondences)} node correspondences")
        logger.info(f"Found {len(correspondences.edge_correspondences)} edge correspondences")
        
    except Exception as e:
        logger.error(f"Correspondence computation failed: {e}")
        return None
    
    # Test model inference
    logger.info("Testing model inference...")
    model.eval()
    
    try:
        with torch.no_grad():
            start_time = time.time()
            
            # Forward pass
            outputs = model(pred_graph, gt_graph, correspondences, training_mode=False)
            
            inference_time = time.time() - start_time
            logger.info(f"Model inference completed in {inference_time:.2f}s")
            
            # Analyze outputs
            if 'node_operations' in outputs:
                node_ops = outputs['node_operations']
                logger.info(f"Node operations shape: {node_ops.shape}")
                
                # Get operation probabilities
                op_probs = torch.softmax(node_ops, dim=-1)
                op_predictions = torch.argmax(op_probs, dim=-1)
                
                # Count operation types
                op_counts = {
                    'insert': (op_predictions == 0).sum().item(),
                    'delete': (op_predictions == 1).sum().item(), 
                    'keep': (op_predictions == 2).sum().item(),
                    'move': (op_predictions == 3).sum().item()
                }
                logger.info(f"Predicted operations: {op_counts}")
            
            if 'node_corrections' in outputs:
                node_corr = outputs['node_corrections']
                logger.info(f"Node corrections shape: {node_corr.shape}")
                
                # Analyze correction magnitudes
                pos_corrections = node_corr[:, :3]  # position corrections
                pos_magnitudes = torch.norm(pos_corrections, dim=-1)
                logger.info(f"Position correction stats - mean: {pos_magnitudes.mean():.3f}, max: {pos_magnitudes.max():.3f}")
            
            if 'quality_score' in outputs:
                quality = outputs['quality_score']
                logger.info(f"Predicted quality score: {quality.item():.3f}")
            
            # Test correction application
            logger.info("Testing correction application...")
            try:
                corrected_graph = model.apply_corrections(pred_graph, outputs, threshold=0.6)
                logger.info(f"Applied corrections: {len(corrected_graph.nodes)} nodes, {len(corrected_graph.edges)} edges")
                
                # Compare original vs corrected
                improvement = len(corrected_graph.nodes) - len(pred_graph.nodes)
                logger.info(f"Node count change: {improvement:+d}")
                
            except Exception as e:
                logger.warning(f"Correction application failed: {e}")
            
            return {
                'sample_id': sample_id,
                'success': True,
                'correspondence_time': correspondence_time,
                'inference_time': inference_time,
                'gt_nodes': len(gt_graph.nodes),
                'gt_edges': len(gt_graph.edges),
                'pred_nodes': len(pred_graph.nodes),
                'pred_edges': len(pred_graph.edges),
                'node_correspondences': len(correspondences.node_correspondences),
                'edge_correspondences': len(correspondences.edge_correspondences),
                'correspondence_quality': correspondences.correspondence_quality.get('overall_quality', 0.0),
                'predicted_operations': op_counts if 'node_operations' in outputs else {},
                'quality_score': outputs.get('quality_score', torch.tensor(0.0)).item()
            }
            
    except Exception as e:
        logger.error(f"Model inference failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_batch_processing(samples, extracted_graphs_dir, model, logger):
    """Test batch processing of multiple samples"""
    logger.info(f"\n{'='*60}")
    logger.info("Testing batch processing...")
    
    batch_data = []
    
    # Prepare batch data (limit to 3 samples for memory)
    test_samples = samples[:3]
    
    for sample_id in test_samples:
        gt_file = extracted_graphs_dir / sample_id / f"{sample_id}_GT.pkl"
        gt_graph = load_graph_safely(gt_file, logger)
        
        if gt_graph is not None:
            pred_graph = create_synthetic_prediction_from_gt(gt_graph, degradation_level=0.25)
            batch_data.append({
                'sample_id': sample_id,
                'pred_graph': pred_graph,
                'gt_graph': gt_graph
            })
    
    if not batch_data:
        logger.warning("No valid samples for batch processing")
        return None
    
    logger.info(f"Processing batch of {len(batch_data)} samples...")
    
    model.eval()
    batch_results = []
    
    with torch.no_grad():
        for data in batch_data:
            try:
                # Simplified processing for batch test
                matcher = create_correspondence_matcher()
                correspondences = matcher.find_correspondence(data['pred_graph'], data['gt_graph'])
                
                outputs = model(data['pred_graph'], data['gt_graph'], correspondences, training_mode=False)
                
                batch_results.append({
                    'sample_id': data['sample_id'],
                    'success': True,
                    'nodes': len(data['pred_graph'].nodes),
                    'quality': outputs.get('quality_score', torch.tensor(0.0)).item()
                })
                
                logger.info(f"✅ {data['sample_id']}: {len(data['pred_graph'].nodes)} nodes, quality: {outputs.get('quality_score', torch.tensor(0.0)).item():.3f}")
                
            except Exception as e:
                logger.error(f"❌ {data['sample_id']}: {e}")
                batch_results.append({
                    'sample_id': data['sample_id'],
                    'success': False,
                    'error': str(e)
                })
    
    success_rate = sum(1 for r in batch_results if r['success']) / len(batch_results)
    logger.info(f"Batch processing success rate: {success_rate:.1%}")
    
    return batch_results


def test_training_step(sample_data, model, loss_fn, logger):
    """Test a single training step"""
    logger.info("Testing training step...")
    
    model.train()
    
    try:
        # Prepare training data
        pred_graph = sample_data['pred_graph']
        gt_graph = sample_data['gt_graph']
        
        # Compute correspondences
        matcher = create_correspondence_matcher()
        correspondences = matcher.find_correspondence(pred_graph, gt_graph)
        
        # Forward pass
        outputs = model(pred_graph, gt_graph, correspondences, training_mode=True)
        
        # Prepare targets (simplified)
        targets = {
            'node_op_targets': torch.randint(0, 4, (len(pred_graph.nodes),)),
            'node_correction_targets': torch.randn(len(pred_graph.nodes), 7),
            'edge_index': torch.randint(0, len(pred_graph.nodes), (2, len(pred_graph.edges)))
        }
        
        # Compute loss
        losses = loss_fn(outputs, targets, correspondences)
        
        logger.info("Training step completed successfully")
        logger.info(f"Total loss: {losses['total'].item():.4f}")
        for loss_name, loss_value in losses.items():
            if loss_name != 'total':
                logger.info(f"  {loss_name}: {loss_value.item():.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_comprehensive_test():
    """Run comprehensive test with extracted graphs"""
    logger = setup_logging()
    logger.info("=== Comprehensive Graph Correction Test with Extracted Graphs ===")
    
    # Check if extracted graphs directory exists
    extracted_graphs_dir = Path("extracted_graphs")
    if not extracted_graphs_dir.exists():
        logger.error("extracted_graphs directory not found!")
        return False
    
    # Find available samples
    samples = find_available_samples(extracted_graphs_dir)
    if not samples:
        logger.error("No samples found in extracted_graphs directory!")
        return False
    
    logger.info(f"Found {len(samples)} samples: {samples[:10]}{'...' if len(samples) > 10 else ''}")
    
    # Initialize model and loss function
    logger.info("Initializing model...")
    try:
        model = GraphCorrectionModel()
        loss_fn = GraphCorrectionLoss()
        
        summary = model.get_model_summary()
        logger.info(f"Model loaded: {summary['total_parameters']:,} parameters")
        
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        return False
    
    # Test individual samples
    logger.info("\n" + "="*60)
    logger.info("INDIVIDUAL SAMPLE TESTS")
    
    individual_results = []
    test_samples = samples[:5]  # Test first 5 samples
    
    for sample_id in test_samples:
        result = test_single_sample(sample_id, extracted_graphs_dir, model, logger)
        if result:
            individual_results.append(result)
    
    # Test batch processing
    batch_results = test_batch_processing(samples, extracted_graphs_dir, model, logger)
    
    # Test training step
    if individual_results:
        sample_data = {
            'pred_graph': create_synthetic_prediction_from_gt(
                load_graph_safely(extracted_graphs_dir / individual_results[0]['sample_id'] / f"{individual_results[0]['sample_id']}_GT.pkl", logger)
            ),
            'gt_graph': load_graph_safely(extracted_graphs_dir / individual_results[0]['sample_id'] / f"{individual_results[0]['sample_id']}_GT.pkl", logger)
        }
        training_success = test_training_step(sample_data, model, loss_fn, logger)
    else:
        training_success = False
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("COMPREHENSIVE TEST SUMMARY")
    
    success_count = len(individual_results)
    total_tests = len(test_samples)
    
    logger.info(f"Individual tests: {success_count}/{total_tests} passed")
    if individual_results:
        avg_correspondence_time = np.mean([r['correspondence_time'] for r in individual_results])
        avg_inference_time = np.mean([r['inference_time'] for r in individual_results])
        avg_quality = np.mean([r['quality_score'] for r in individual_results])
        
        logger.info(f"Average correspondence time: {avg_correspondence_time:.2f}s")
        logger.info(f"Average inference time: {avg_inference_time:.2f}s")
        logger.info(f"Average quality score: {avg_quality:.3f}")
    
    batch_success = batch_results is not None and any(r['success'] for r in batch_results)
    logger.info(f"Batch processing: {'✅ PASSED' if batch_success else '❌ FAILED'}")
    logger.info(f"Training step: {'✅ PASSED' if training_success else '❌ FAILED'}")
    
    # Save detailed results
    results = {
        'individual_results': individual_results,
        'batch_results': batch_results,
        'training_success': training_success,
        'summary': {
            'total_samples_tested': len(test_samples),
            'successful_samples': success_count,
            'success_rate': success_count / total_tests if total_tests > 0 else 0,
            'batch_processing_success': batch_success,
            'training_step_success': training_success
        }
    }
    
    results_file = Path("extracted_graphs_test_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Detailed results saved to: {results_file}")
    
    overall_success = success_count > 0 and batch_success and training_success
    logger.info(f"\nOVERALL TEST RESULT: {'✅ SUCCESS' if overall_success else '❌ FAILURE'}")
    
    return overall_success


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)