#!/usr/bin/env python3
"""
Evaluate graph correction quality at the graph level
Focus on whether corrected nodes are closer to ground truth positions
"""

import sys
sys.path.append('.')

import torch
import numpy as np
from pathlib import Path
import pickle
import logging
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import time

from src.models.graph_correction.regression_model import GraphCorrectionRegressionModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_trained_model():
    """Load the trained regression model"""
    model_path = Path('experiments/regression_model/best_model.pth')
    
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}")
    
    logger.info("Loading trained model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize model
    model_config = checkpoint['config']['model']
    model = GraphCorrectionRegressionModel(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded (epoch {checkpoint['epoch']}, val_loss: {checkpoint['best_val_loss']:.4f})")
    return model, device


def load_case_graphs(patient_id):
    """Load predicted and ground truth graphs for a patient"""
    
    pred_graph_path = Path(f'extracted_graphs/{patient_id}/{patient_id}_PRED.pkl')
    gt_graph_path = Path(f'extracted_graphs/{patient_id}/{patient_id}_GT.pkl')
    
    if not pred_graph_path.exists() or not gt_graph_path.exists():
        raise FileNotFoundError(f"Graph files not found for {patient_id}")
    
    logger.info(f"Loading graphs for {patient_id}...")
    
    with open(pred_graph_path, 'rb') as f:
        pred_graph = pickle.load(f)
    
    with open(gt_graph_path, 'rb') as f:
        gt_graph = pickle.load(f)
    
    logger.info(f"Loaded: Pred graph {len(pred_graph.nodes)} nodes, GT graph {len(gt_graph.nodes)} nodes")
    
    return pred_graph, gt_graph


def convert_graph_to_pyg_format(graph, device):
    """Convert VascularGraph to PyTorch Geometric format"""
    
    # Extract node features
    nodes = graph.nodes
    node_features = []
    positions = []
    
    for node in nodes:
        pos = np.array(node['position'], dtype=np.float32)
        radius = float(node.get('radius', 1.0))
        confidence = 0.7  # Default confidence for test data
        feat1, feat2 = 0.0, 0.0  # Placeholder features
        
        features = np.concatenate([pos, [radius, confidence, feat1, feat2]])
        node_features.append(features)
        positions.append(pos)
    
    # Convert to tensors
    x = torch.tensor(np.array(node_features), dtype=torch.float32).to(device)
    pos = torch.tensor(np.array(positions), dtype=torch.float32).to(device)
    
    # Extract edges (simplified)
    edge_list = []
    for edge in graph.edges:
        try:
            if hasattr(edge, 'start_node') and hasattr(edge, 'end_node'):
                start, end = edge.start_node, edge.end_node
            elif isinstance(edge, dict):
                if 'start_node' in edge and 'end_node' in edge:
                    start, end = edge['start_node'], edge['end_node']
                elif 'source' in edge and 'target' in edge:
                    start, end = edge['source'], edge['target']
                else:
                    continue
            else:
                continue
                
            edge_list.append([start, end])
            edge_list.append([end, start])  # Undirected
        except:
            continue
    
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().to(device)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long).to(device)
    
    # Create PyG data
    from torch_geometric.data import Data
    data = Data(x=x, edge_index=edge_index, pos=pos)
    data.batch = torch.zeros(x.size(0), dtype=torch.long).to(device)
    
    return data


def apply_graph_correction(model, pred_graph, device):
    """Apply graph correction and return corrected positions"""
    
    # Convert to PyG format
    data = convert_graph_to_pyg_format(pred_graph, device)
    
    # Apply correction
    with torch.no_grad():
        predictions = model(data)
    
    # Extract results
    original_positions = data.pos.cpu().numpy()
    corrected_positions = predictions['predicted_positions'].cpu().numpy()
    correction_magnitudes = predictions['correction_magnitudes'].cpu().numpy()
    node_operations = predictions['node_operations'].cpu().numpy()
    
    return {
        'original_positions': original_positions,
        'corrected_positions': corrected_positions,
        'correction_magnitudes': correction_magnitudes,
        'node_operations': node_operations
    }


def find_nearest_neighbors(pred_positions, gt_positions, max_distance=10.0):
    """Find nearest neighbor correspondences between predicted and GT positions"""
    
    if len(pred_positions) == 0 or len(gt_positions) == 0:
        return [], [], []
    
    # Calculate distance matrix
    distances = cdist(pred_positions, gt_positions)
    
    # Use Hungarian algorithm for optimal assignment
    pred_indices, gt_indices = linear_sum_assignment(distances)
    
    # Filter by maximum distance
    valid_matches = distances[pred_indices, gt_indices] <= max_distance
    pred_indices = pred_indices[valid_matches]
    gt_indices = gt_indices[valid_matches]
    match_distances = distances[pred_indices, gt_indices]
    
    return pred_indices, gt_indices, match_distances


def evaluate_graph_correction_quality(pred_graph, gt_graph, correction_results):
    """Evaluate how much the graph correction improved position accuracy"""
    
    logger.info("Evaluating graph correction quality...")
    
    # Extract positions
    original_positions = correction_results['original_positions']
    corrected_positions = correction_results['corrected_positions']
    correction_magnitudes = correction_results['correction_magnitudes']
    node_operations = correction_results['node_operations']
    
    # GT positions
    gt_positions = np.array([node['position'] for node in gt_graph.nodes])
    
    logger.info(f"Comparing {len(original_positions)} pred nodes with {len(gt_positions)} GT nodes")
    
    # Find correspondences for original positions
    orig_pred_idx, orig_gt_idx, orig_distances = find_nearest_neighbors(
        original_positions, gt_positions, max_distance=15.0
    )
    
    # Find correspondences for corrected positions  
    corr_pred_idx, corr_gt_idx, corr_distances = find_nearest_neighbors(
        corrected_positions, gt_positions, max_distance=15.0
    )
    
    logger.info(f"Found {len(orig_pred_idx)} original matches, {len(corr_pred_idx)} corrected matches")
    
    # Analyze improvements for nodes that have matches in both cases
    improvements = []
    node_analysis = []
    
    for i in range(len(original_positions)):
        # Find if this node has matches in both original and corrected
        orig_match_idx = np.where(orig_pred_idx == i)[0]
        corr_match_idx = np.where(corr_pred_idx == i)[0]
        
        if len(orig_match_idx) > 0 and len(corr_match_idx) > 0:
            orig_dist = orig_distances[orig_match_idx[0]]
            corr_dist = corr_distances[corr_match_idx[0]]
            improvement = orig_dist - corr_dist  # Positive = better
            
            improvements.append(improvement)
            
            node_analysis.append({
                'node_idx': i,
                'original_distance': orig_dist,
                'corrected_distance': corr_dist,
                'improvement': improvement,
                'correction_magnitude': correction_magnitudes[i],
                'operation': node_operations[i]
            })
    
    # Calculate statistics
    if improvements:
        improvements = np.array(improvements)
        
        stats = {
            'num_matched_nodes': len(improvements),
            'mean_improvement': float(np.mean(improvements)),
            'median_improvement': float(np.median(improvements)),
            'std_improvement': float(np.std(improvements)),
            'positive_improvements': int(np.sum(improvements > 0)),
            'negative_improvements': int(np.sum(improvements < 0)),
            'improvement_rate': float(np.sum(improvements > 0) / len(improvements)),
            'mean_original_distance': float(np.mean([n['original_distance'] for n in node_analysis])),
            'mean_corrected_distance': float(np.mean([n['corrected_distance'] for n in node_analysis])),
            'mean_correction_magnitude': float(np.mean(correction_magnitudes)),
        }
        
        # Operation-specific analysis
        modify_nodes = [n for n in node_analysis if n['operation'] == 0]
        remove_nodes = [n for n in node_analysis if n['operation'] == 1]
        
        if modify_nodes:
            stats['modify_nodes'] = {
                'count': len(modify_nodes),
                'mean_improvement': float(np.mean([n['improvement'] for n in modify_nodes])),
                'mean_correction_magnitude': float(np.mean([n['correction_magnitude'] for n in modify_nodes]))
            }
        
        if remove_nodes:
            stats['remove_nodes'] = {
                'count': len(remove_nodes),
                'mean_improvement': float(np.mean([n['improvement'] for n in remove_nodes])),
                'mean_correction_magnitude': float(np.mean([n['correction_magnitude'] for n in remove_nodes]))
            }
    else:
        stats = {
            'num_matched_nodes': 0,
            'error': 'No matching nodes found between predicted and ground truth graphs'
        }
    
    return stats, node_analysis


def evaluate_patient_graph_quality(patient_id):
    """Evaluate graph correction quality for a single patient"""
    
    logger.info("="*80)
    logger.info(f"GRAPH QUALITY EVALUATION: {patient_id}")
    logger.info("="*80)
    
    start_time = time.time()
    
    try:
        # 1. Load trained model
        model, device = load_trained_model()
        
        # 2. Load graphs
        pred_graph, gt_graph = load_case_graphs(patient_id)
        
        # 3. Apply correction
        correction_results = apply_graph_correction(model, pred_graph, device)
        
        # 4. Evaluate quality
        stats, node_analysis = evaluate_graph_correction_quality(
            pred_graph, gt_graph, correction_results
        )
        
        # 5. Report results
        processing_time = time.time() - start_time
        
        logger.info("\n" + "="*60)
        logger.info("GRAPH CORRECTION QUALITY RESULTS")
        logger.info("="*60)
        
        logger.info(f"Patient: {patient_id}")
        logger.info(f"Processing time: {processing_time:.1f} seconds")
        logger.info("")
        
        if 'error' not in stats:
            logger.info("POSITION ACCURACY:")
            logger.info(f"  Matched nodes: {stats['num_matched_nodes']}")
            logger.info(f"  Mean distance improvement: {stats['mean_improvement']:.2f}mm")
            logger.info(f"  Median distance improvement: {stats['median_improvement']:.2f}mm")
            logger.info(f"  Improvement rate: {stats['improvement_rate']:.1%}")
            logger.info(f"  Nodes improved: {stats['positive_improvements']}")
            logger.info(f"  Nodes degraded: {stats['negative_improvements']}")
            logger.info("")
            
            logger.info("DISTANCE COMPARISON:")
            logger.info(f"  Mean original distance to GT: {stats['mean_original_distance']:.2f}mm")
            logger.info(f"  Mean corrected distance to GT: {stats['mean_corrected_distance']:.2f}mm")
            logger.info(f"  Mean correction magnitude: {stats['mean_correction_magnitude']:.2f}mm")
            logger.info("")
            
            # Operation-specific results
            if 'modify_nodes' in stats:
                logger.info(f"MODIFY OPERATION ({stats['modify_nodes']['count']} nodes):")
                logger.info(f"  Mean improvement: {stats['modify_nodes']['mean_improvement']:.2f}mm")
                logger.info(f"  Mean correction: {stats['modify_nodes']['mean_correction_magnitude']:.2f}mm")
            
            if 'remove_nodes' in stats:
                logger.info(f"REMOVE OPERATION ({stats['remove_nodes']['count']} nodes):")
                logger.info(f"  Mean improvement: {stats['remove_nodes']['mean_improvement']:.2f}mm")
                logger.info(f"  Mean correction: {stats['remove_nodes']['mean_correction_magnitude']:.2f}mm")
            
            logger.info("")
            
            # Overall assessment
            if stats['mean_improvement'] > 0.5:
                logger.info("🎉 SUCCESS: Graph correction significantly improved node positions!")
            elif stats['mean_improvement'] > 0:
                logger.info("✅ GOOD: Graph correction improved node positions")
            elif stats['mean_improvement'] > -0.5:
                logger.info("⚖️  NEUTRAL: Graph correction had minimal impact")
            else:
                logger.info("⚠️  PROBLEM: Graph correction degraded node positions")
            
        else:
            logger.info(f"ERROR: {stats['error']}")
        
        logger.info("="*60)
        
        # Save results
        results = {
            'patient_id': patient_id,
            'processing_time': processing_time,
            'correction_stats': stats,
            'node_analysis': node_analysis[:10]  # Save first 10 nodes for inspection
        }
        
        results_path = Path(f'experiments/graph_quality_results_{patient_id}.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"Results saved to {results_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Graph quality evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate graph correction quality')
    parser.add_argument('--patient-id', type=str, default='PA000005',
                       help='Patient ID to evaluate (default: PA000005)')
    
    args = parser.parse_args()
    
    results = evaluate_patient_graph_quality(args.patient_id)
    
    if results:
        logger.info("Graph quality evaluation completed successfully!")
    else:
        logger.error("Graph quality evaluation failed!")


if __name__ == '__main__':
    main()