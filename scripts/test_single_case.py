#!/usr/bin/env python3
"""
Test the full pipeline on a single case
Complete evaluation: Load graphs → Apply correction → Reconstruct → Compare metrics
"""

import sys
sys.path.append('.')

import torch
import numpy as np
from pathlib import Path
import nibabel as nib
import pickle
import logging
import time

from src.models.graph_correction.regression_model import GraphCorrectionRegressionModel
from src.models.reconstruction.volume_reconstructor import VolumeReconstructor
from scipy.spatial.distance import directed_hausdorff

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_dice_score(pred_mask, gt_mask):
    """Calculate Dice similarity coefficient"""
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    total = pred_mask.sum() + gt_mask.sum()
    
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    
    dice = 2.0 * intersection / total
    return float(dice)


def calculate_hausdorff_distance(pred_mask, gt_mask):
    """Calculate Hausdorff distance between two binary masks"""
    try:
        # Get surface points
        pred_points = np.argwhere(pred_mask > 0)
        gt_points = np.argwhere(gt_mask > 0)
        
        if len(pred_points) == 0 or len(gt_points) == 0:
            return 100.0  # Large distance for empty masks
        
        # Calculate directed Hausdorff distances
        hd1 = directed_hausdorff(pred_points, gt_points)[0]
        hd2 = directed_hausdorff(gt_points, pred_points)[0]
        
        # Return symmetric Hausdorff distance
        return float(max(hd1, hd2))
        
    except Exception as e:
        logger.warning(f"Hausdorff calculation failed: {e}")
        return 100.0


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


def load_case_data(patient_id):
    """Load all data for a specific patient"""
    
    # File paths
    pred_mask_path = Path(f'experiments/test_predictions/{patient_id}/binary_mask.nii.gz')
    gt_mask_path = Path(f'DATASET/Parse_dataset/{patient_id}/label/{patient_id}.nii.gz')
    pred_graph_path = Path(f'extracted_graphs/{patient_id}/{patient_id}_PRED.pkl')
    gt_graph_path = Path(f'extracted_graphs/{patient_id}/{patient_id}_GT.pkl')
    
    # Check all files exist
    missing_files = []
    for path in [pred_mask_path, gt_mask_path, pred_graph_path, gt_graph_path]:
        if not path.exists():
            missing_files.append(str(path))
    
    if missing_files:
        raise FileNotFoundError(f"Missing files: {missing_files}")
    
    logger.info(f"Loading data for {patient_id}...")
    
    # Load masks
    pred_mask = nib.load(pred_mask_path).get_fdata()
    gt_mask = nib.load(gt_mask_path).get_fdata()
    
    # Ensure binary
    pred_mask = (pred_mask > 0).astype(np.uint8)
    gt_mask = (gt_mask > 0).astype(np.uint8)
    
    # Load graphs
    with open(pred_graph_path, 'rb') as f:
        pred_graph = pickle.load(f)
    
    with open(gt_graph_path, 'rb') as f:
        gt_graph = pickle.load(f)
    
    logger.info(f"Loaded: Pred mask {pred_mask.shape}, GT mask {gt_mask.shape}")
    logger.info(f"Loaded: Pred graph {len(pred_graph.nodes)} nodes, GT graph {len(gt_graph.nodes)} nodes")
    
    return {
        'pred_mask': pred_mask,
        'gt_mask': gt_mask,
        'pred_graph': pred_graph,
        'gt_graph': gt_graph
    }


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
        # Handle different edge formats
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


def apply_graph_correction(model, graph, device, patient_id):
    """Apply graph correction using trained model"""
    
    logger.info(f"Applying graph correction to {patient_id}...")
    
    # Convert to PyG format
    data = convert_graph_to_pyg_format(graph, device)
    
    # Apply correction
    with torch.no_grad():
        predictions = model(data)
    
    # Extract results
    corrected_positions = predictions['predicted_positions'].cpu().numpy()
    correction_magnitudes = predictions['correction_magnitudes'].cpu().numpy()
    node_operations = predictions['node_operations'].cpu().numpy()
    
    # Create corrected graph
    corrected_nodes = []
    removed_nodes = 0
    
    for i, node in enumerate(graph.nodes):
        if i < len(corrected_positions):
            # Create corrected node
            corrected_node = node.copy()
            corrected_node['position'] = corrected_positions[i].tolist()
            corrected_node['correction_magnitude'] = float(correction_magnitudes[i])
            corrected_node['operation'] = int(node_operations[i])
            
            # Only keep nodes that aren't marked for removal
            if node_operations[i] == 0:  # Modify - keep the node
                corrected_nodes.append(corrected_node)
            else:  # Remove - skip this node
                removed_nodes += 1
        else:
            corrected_nodes.append(node)
    
    # Create corrected graph
    from src.models.graph_extraction.vascular_graph import VascularGraph
    
    corrected_graph = VascularGraph(
        nodes=corrected_nodes,
        edges=graph.edges,  # Keep original edges for simplicity
        global_properties=graph.global_properties.copy(),
        metadata={**graph.metadata, 'corrected': True}
    )
    
    correction_stats = {
        'original_nodes': len(graph.nodes),
        'corrected_nodes': len(corrected_nodes),
        'nodes_removed': removed_nodes,
        'mean_correction_magnitude': float(np.mean(correction_magnitudes)),
        'max_correction_magnitude': float(np.max(correction_magnitudes)),
        'operations': {
            'modify': int(np.sum(node_operations == 0)),
            'remove': int(np.sum(node_operations == 1))
        }
    }
    
    logger.info(f"Correction applied: {correction_stats['nodes_removed']} nodes removed, "
               f"mean correction: {correction_stats['mean_correction_magnitude']:.2f}")
    
    return corrected_graph, correction_stats


def reconstruct_volume_from_graph(graph, target_shape, patient_id):
    """Reconstruct volume from graph (improved version)"""
    
    logger.info(f"Reconstructing volume for {patient_id}...")
    
    # Initialize volume
    reconstructed_volume = np.zeros(target_shape, dtype=np.uint8)
    
    # Better reconstruction: larger spheres + edge connections
    for node in graph.nodes:
        pos = node['position']
        radius = max(3.0, node.get('radius', 3.0))  # Larger default radius
        
        # Convert to voxel coordinates
        x, y, z = int(round(pos[0])), int(round(pos[1])), int(round(pos[2]))
        r = max(2, int(radius))
        
        # Place larger sphere
        for dx in range(-r, r+1):
            for dy in range(-r, r+1):
                for dz in range(-r, r+1):
                    if (dx*dx + dy*dy + dz*dz) <= r*r:
                        nx, ny, nz = x+dx, y+dy, z+dz
                        if (0 <= nx < target_shape[0] and 
                            0 <= ny < target_shape[1] and 
                            0 <= nz < target_shape[2]):
                            reconstructed_volume[nx, ny, nz] = 1
    
    # Connect nodes with edges (draw lines between connected nodes)
    for edge in graph.edges:
        try:
            # Get edge endpoints
            if hasattr(edge, 'start_node') and hasattr(edge, 'end_node'):
                start_idx, end_idx = edge.start_node, edge.end_node
            elif isinstance(edge, dict):
                if 'start_node' in edge and 'end_node' in edge:
                    start_idx, end_idx = edge['start_node'], edge['end_node']
                else:
                    continue
            else:
                continue
            
            # Get node positions
            if start_idx < len(graph.nodes) and end_idx < len(graph.nodes):
                start_pos = graph.nodes[start_idx]['position']
                end_pos = graph.nodes[end_idx]['position']
                
                # Draw line between nodes
                steps = max(1, int(np.linalg.norm(np.array(end_pos) - np.array(start_pos))))
                for i in range(steps + 1):
                    t = i / max(1, steps)
                    pos = np.array(start_pos) * (1-t) + np.array(end_pos) * t
                    
                    x, y, z = int(round(pos[0])), int(round(pos[1])), int(round(pos[2]))
                    
                    # Draw small sphere at each point
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            for dz in range(-1, 2):
                                nx, ny, nz = x+dx, y+dy, z+dz
                                if (0 <= nx < target_shape[0] and 
                                    0 <= ny < target_shape[1] and 
                                    0 <= nz < target_shape[2]):
                                    reconstructed_volume[nx, ny, nz] = 1
        except:
            continue
    
    logger.info(f"Reconstructed volume: {np.sum(reconstructed_volume)} voxels")
    
    return reconstructed_volume


def compute_metrics(pred_mask, corrected_mask, gt_mask):
    """Compute segmentation metrics"""
    
    # Original metrics
    original_dice = calculate_dice_score(pred_mask, gt_mask)
    original_hd = calculate_hausdorff_distance(pred_mask, gt_mask)
    
    # Corrected metrics
    corrected_dice = calculate_dice_score(corrected_mask, gt_mask)
    corrected_hd = calculate_hausdorff_distance(corrected_mask, gt_mask)
    
    # Improvements
    dice_improvement = corrected_dice - original_dice
    hd_improvement = original_hd - corrected_hd  # Lower is better
    
    return {
        'original_dice': original_dice,
        'corrected_dice': corrected_dice,
        'dice_improvement': dice_improvement,
        'original_hd': original_hd,
        'corrected_hd': corrected_hd,
        'hd_improvement': hd_improvement
    }


def evaluate_single_case(patient_id):
    """Run complete pipeline evaluation on a single case"""
    
    logger.info("="*80)
    logger.info(f"FULL PIPELINE EVALUATION: {patient_id}")
    logger.info("="*80)
    
    start_time = time.time()
    
    try:
        # 1. Load trained model
        model, device = load_trained_model()
        
        # 2. Load case data
        data = load_case_data(patient_id)
        
        # 3. Apply graph correction
        corrected_graph, correction_stats = apply_graph_correction(
            model, data['pred_graph'], device, patient_id
        )
        
        # 4. Reconstruct volume from corrected graph
        corrected_mask = reconstruct_volume_from_graph(
            corrected_graph, data['pred_mask'].shape, patient_id
        )
        
        # 5. Compute metrics
        metrics = compute_metrics(
            data['pred_mask'], corrected_mask, data['gt_mask']
        )
        
        # 6. Report results
        processing_time = time.time() - start_time
        
        logger.info("\n" + "="*60)
        logger.info("RESULTS SUMMARY")
        logger.info("="*60)
        
        logger.info(f"Patient: {patient_id}")
        logger.info(f"Processing time: {processing_time:.1f} seconds")
        logger.info("")
        
        logger.info("GRAPH CORRECTION STATS:")
        logger.info(f"  Original nodes: {correction_stats['original_nodes']}")
        logger.info(f"  Final nodes: {correction_stats['corrected_nodes']}")
        logger.info(f"  Nodes removed: {correction_stats['nodes_removed']}")
        logger.info(f"  Operations - Modify: {correction_stats['operations']['modify']}, Remove: {correction_stats['operations']['remove']}")
        logger.info(f"  Mean correction: {correction_stats['mean_correction_magnitude']:.2f}mm")
        logger.info("")
        
        logger.info("SEGMENTATION QUALITY:")
        logger.info(f"  Dice Score:")
        logger.info(f"    Original U-Net:  {metrics['original_dice']:.4f}")
        logger.info(f"    After correction: {metrics['corrected_dice']:.4f}")
        logger.info(f"    Improvement:     {metrics['dice_improvement']:+.4f}")
        logger.info("")
        logger.info(f"  Hausdorff Distance:")
        logger.info(f"    Original U-Net:  {metrics['original_hd']:.2f}mm")
        logger.info(f"    After correction: {metrics['corrected_hd']:.2f}mm")
        logger.info(f"    Improvement:     {metrics['hd_improvement']:+.2f}mm")
        logger.info("")
        
        # Overall assessment
        if metrics['dice_improvement'] > 0:
            logger.info("🎉 SUCCESS: Graph correction IMPROVED segmentation quality!")
        elif metrics['dice_improvement'] > -0.01:
            logger.info("⚖️  NEUTRAL: Graph correction maintained segmentation quality")
        else:
            logger.info("⚠️  DEGRADED: Graph correction reduced segmentation quality")
        
        logger.info("="*60)
        
        # Save results
        results = {
            'patient_id': patient_id,
            'processing_time': processing_time,
            'correction_stats': correction_stats,
            'metrics': metrics
        }
        
        results_path = Path(f'experiments/single_case_results_{patient_id}.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"Results saved to {results_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test full pipeline on single case')
    parser.add_argument('--patient-id', type=str, default='PA000005',
                       help='Patient ID to test (default: PA000005)')
    
    args = parser.parse_args()
    
    results = evaluate_single_case(args.patient_id)
    
    if results:
        logger.info("Single case evaluation completed successfully!")
    else:
        logger.error("Single case evaluation failed!")


if __name__ == '__main__':
    main()