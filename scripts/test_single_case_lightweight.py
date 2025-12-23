#!/usr/bin/env python3
"""
Test the full pipeline with lightweight, memory-efficient reconstruction
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
from scipy import ndimage
from skimage.morphology import dilation, ball

from src.models.graph_correction.regression_model import GraphCorrectionRegressionModel
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
        pred_points = np.argwhere(pred_mask > 0)
        gt_points = np.argwhere(gt_mask > 0)
        
        if len(pred_points) == 0 or len(gt_points) == 0:
            return 100.0
        
        # Sample points if too many (for memory efficiency)
        if len(pred_points) > 5000:
            indices = np.random.choice(len(pred_points), 5000, replace=False)
            pred_points = pred_points[indices]
        if len(gt_points) > 5000:
            indices = np.random.choice(len(gt_points), 5000, replace=False)
            gt_points = gt_points[indices]
        
        hd1 = directed_hausdorff(pred_points, gt_points)[0]
        hd2 = directed_hausdorff(gt_points, pred_points)[0]
        
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
    
    model_config = checkpoint['config']['model']
    model = GraphCorrectionRegressionModel(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded (epoch {checkpoint['epoch']}, val_loss: {checkpoint['best_val_loss']:.4f})")
    return model, device


def load_case_data(patient_id):
    """Load all data for a specific patient"""
    
    pred_mask_path = Path(f'experiments/test_predictions/{patient_id}/binary_mask.nii.gz')
    gt_mask_path = Path(f'DATASET/Parse_dataset/{patient_id}/label/{patient_id}.nii.gz')
    pred_graph_path = Path(f'extracted_graphs/{patient_id}/{patient_id}_PRED.pkl')
    gt_graph_path = Path(f'extracted_graphs/{patient_id}/{patient_id}_GT.pkl')
    
    missing_files = []
    for path in [pred_mask_path, gt_mask_path, pred_graph_path, gt_graph_path]:
        if not path.exists():
            missing_files.append(str(path))
    
    if missing_files:
        raise FileNotFoundError(f"Missing files: {missing_files}")
    
    logger.info(f"Loading data for {patient_id}...")
    
    pred_mask = nib.load(pred_mask_path).get_fdata()
    gt_mask = nib.load(gt_mask_path).get_fdata()
    
    pred_mask = (pred_mask > 0).astype(np.uint8)
    gt_mask = (gt_mask > 0).astype(np.uint8)
    
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
    
    nodes = graph.nodes
    node_features = []
    positions = []
    
    for node in nodes:
        pos = np.array(node['position'], dtype=np.float32)
        radius = float(node.get('radius', 1.0))
        confidence = 0.7
        feat1, feat2 = 0.0, 0.0
        
        features = np.concatenate([pos, [radius, confidence, feat1, feat2]])
        node_features.append(features)
        positions.append(pos)
    
    x = torch.tensor(np.array(node_features), dtype=torch.float32).to(device)
    pos = torch.tensor(np.array(positions), dtype=torch.float32).to(device)
    
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
            edge_list.append([end, start])
        except:
            continue
    
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().to(device)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long).to(device)
    
    from torch_geometric.data import Data
    data = Data(x=x, edge_index=edge_index, pos=pos)
    data.batch = torch.zeros(x.size(0), dtype=torch.long).to(device)
    
    return data


def apply_graph_correction(model, graph, device, patient_id):
    """Apply graph correction using trained model"""
    
    logger.info(f"Applying graph correction to {patient_id}...")
    
    data = convert_graph_to_pyg_format(graph, device)
    
    with torch.no_grad():
        predictions = model(data)
    
    corrected_positions = predictions['predicted_positions'].cpu().numpy()
    correction_magnitudes = predictions['correction_magnitudes'].cpu().numpy()
    node_operations = predictions['node_operations'].cpu().numpy()
    
    # Create corrected graph with proper indexing
    corrected_nodes = []
    removed_nodes = 0
    node_mapping = {}
    
    for i, node in enumerate(graph.nodes):
        if i < len(corrected_positions):
            corrected_node = node.copy()
            corrected_node['position'] = corrected_positions[i].tolist()
            corrected_node['correction_magnitude'] = float(correction_magnitudes[i])
            corrected_node['operation'] = int(node_operations[i])
            
            if node_operations[i] == 0:  # Keep the node
                node_mapping[i] = len(corrected_nodes)
                corrected_nodes.append(corrected_node)
            else:  # Remove
                removed_nodes += 1
        else:
            node_mapping[i] = len(corrected_nodes)
            corrected_nodes.append(node)
    
    # Update edges
    corrected_edges = []
    for edge in graph.edges:
        try:
            if 'source' in edge and 'target' in edge:
                old_source = edge['source']
                old_target = edge['target']
            elif 'start_node' in edge and 'end_node' in edge:
                old_source = edge['start_node']
                old_target = edge['end_node']
            else:
                continue
            
            if old_source in node_mapping and old_target in node_mapping:
                new_edge = edge.copy()
                new_edge['source'] = node_mapping[old_source]
                new_edge['target'] = node_mapping[old_target]
                corrected_edges.append(new_edge)
        except Exception as e:
            logger.warning(f"Error processing edge {edge}: {e}")
            continue
    
    from src.models.graph_extraction.vascular_graph import VascularGraph
    
    corrected_graph = VascularGraph(
        nodes=corrected_nodes,
        edges=corrected_edges,
        global_properties=graph.global_properties.copy(),
        metadata={**graph.metadata, 'corrected': True}
    )
    
    correction_stats = {
        'original_nodes': len(graph.nodes),
        'corrected_nodes': len(corrected_nodes),
        'nodes_removed': removed_nodes,
        'edges_retained': len(corrected_edges),
        'original_edges': len(graph.edges),
        'mean_correction_magnitude': float(np.mean(correction_magnitudes)),
        'max_correction_magnitude': float(np.max(correction_magnitudes)),
        'operations': {
            'modify': int(np.sum(node_operations == 0)),
            'remove': int(np.sum(node_operations == 1))
        }
    }
    
    logger.info(f"Correction applied: {correction_stats['nodes_removed']} nodes removed, "
               f"{correction_stats['edges_retained']}/{correction_stats['original_edges']} edges retained, "
               f"mean correction: {correction_stats['mean_correction_magnitude']:.2f}")
    
    return corrected_graph, correction_stats


def reconstruct_volume_lightweight(corrected_graph, original_prediction, patient_id):
    """Memory-efficient skeleton-based reconstruction"""
    
    logger.info(f"Starting lightweight reconstruction for {patient_id}...")
    
    target_shape = original_prediction.shape
    logger.info(f"Target shape: {target_shape}")
    
    # Step 1: Create skeleton on reduced resolution for memory efficiency
    downsample_factor = 2
    reduced_shape = tuple(s // downsample_factor for s in target_shape)
    logger.info(f"Working on reduced shape: {reduced_shape}")
    
    # Initialize reduced volume
    reduced_volume = np.zeros(reduced_shape, dtype=np.float32)
    
    # Step 2: Place nodes as soft spheres in reduced space
    logger.info("Placing corrected nodes...")
    for i, node in enumerate(corrected_graph.nodes):
        pos = np.array(node['position']) / downsample_factor
        radius = max(1.5, node.get('radius', 1.5) / downsample_factor)
        
        x, y, z = int(round(pos[0])), int(round(pos[1])), int(round(pos[2]))
        r = max(1, int(radius))
        
        # Place soft sphere
        for dx in range(-r*2, r*2+1):
            for dy in range(-r*2, r*2+1):
                for dz in range(-r*2, r*2+1):
                    distance = np.sqrt(dx*dx + dy*dy + dz*dz)
                    if distance <= r*2:
                        nx, ny, nz = x+dx, y+dy, z+dz
                        if (0 <= nx < reduced_shape[0] and 
                            0 <= ny < reduced_shape[1] and 
                            0 <= nz < reduced_shape[2]):
                            
                            weight = max(0, 1.0 - distance / (r*2))
                            reduced_volume[nx, ny, nz] = max(
                                reduced_volume[nx, ny, nz], weight
                            )
    
    # Step 3: Connect nodes with edges in reduced space
    logger.info("Drawing corrected edges...")
    for edge in corrected_graph.edges:
        try:
            if 'source' in edge and 'target' in edge:
                start_idx, end_idx = edge['source'], edge['target']
            else:
                continue
            
            if start_idx < len(corrected_graph.nodes) and end_idx < len(corrected_graph.nodes):
                start_pos = np.array(corrected_graph.nodes[start_idx]['position']) / downsample_factor
                end_pos = np.array(corrected_graph.nodes[end_idx]['position']) / downsample_factor
                
                start_radius = max(1.0, corrected_graph.nodes[start_idx].get('radius', 1.0) / downsample_factor)
                end_radius = max(1.0, corrected_graph.nodes[end_idx].get('radius', 1.0) / downsample_factor)
                
                # Draw line
                steps = max(3, int(np.linalg.norm(end_pos - start_pos)))
                for i in range(steps + 1):
                    t = i / max(1, steps)
                    pos = start_pos * (1-t) + end_pos * t
                    radius = start_radius * (1-t) + end_radius * t
                    
                    x, y, z = int(round(pos[0])), int(round(pos[1])), int(round(pos[2]))
                    r = max(1, int(radius))
                    
                    for dx in range(-r, r+1):
                        for dy in range(-r, r+1):
                            for dz in range(-r, r+1):
                                distance = np.sqrt(dx*dx + dy*dy + dz*dz)
                                if distance <= r:
                                    nx, ny, nz = x+dx, y+dy, z+dz
                                    if (0 <= nx < reduced_shape[0] and 
                                        0 <= ny < reduced_shape[1] and 
                                        0 <= nz < reduced_shape[2]):
                                        
                                        weight = max(0, 1.0 - distance / r)
                                        reduced_volume[nx, ny, nz] = max(
                                            reduced_volume[nx, ny, nz], weight
                                        )
        except:
            continue
    
    # Step 4: Convert to binary and upsample
    logger.info("Processing and upsampling...")
    binary_reduced = (reduced_volume > 0.3).astype(np.uint8)
    
    # Apply morphological operations on reduced volume
    kernel = ball(1)
    binary_reduced = dilation(binary_reduced, kernel)
    binary_reduced = ndimage.binary_closing(binary_reduced, structure=np.ones((2,2,2))).astype(np.uint8)
    
    # Upsample to original resolution
    upsampled = ndimage.zoom(binary_reduced, downsample_factor, order=1)
    
    # Ensure exact target shape
    if upsampled.shape != target_shape:
        # Crop or pad to exact shape
        final_volume = np.zeros(target_shape, dtype=np.uint8)
        
        min_shape = tuple(min(upsampled.shape[i], target_shape[i]) for i in range(3))
        final_volume[:min_shape[0], :min_shape[1], :min_shape[2]] = \
            upsampled[:min_shape[0], :min_shape[1], :min_shape[2]]
        
        upsampled = final_volume
    
    # Step 5: Blend with original prediction
    logger.info("Blending with original prediction...")
    
    # Use original as base, enhance with corrected
    blended = original_prediction.copy().astype(np.float32)
    
    # Where we have high confidence in correction, use corrected
    # Where correction is sparse, keep original
    correction_confidence = ndimage.gaussian_filter(upsampled.astype(float), sigma=1.0)
    
    blend_weight = 0.7  # Give more weight to correction
    final_mask = (blended * (1 - blend_weight) + upsampled * blend_weight * correction_confidence)
    final_mask = (final_mask > 0.5).astype(np.uint8)
    
    # Final cleanup
    final_mask = ndimage.binary_closing(final_mask, structure=np.ones((2,2,2))).astype(np.uint8)
    
    logger.info(f"Lightweight reconstruction completed: {np.sum(final_mask)} voxels")
    
    return final_mask


def compute_metrics(pred_mask, corrected_mask, gt_mask):
    """Compute segmentation metrics"""
    
    original_dice = calculate_dice_score(pred_mask, gt_mask)
    original_hd = calculate_hausdorff_distance(pred_mask, gt_mask)
    
    corrected_dice = calculate_dice_score(corrected_mask, gt_mask)
    corrected_hd = calculate_hausdorff_distance(corrected_mask, gt_mask)
    
    dice_improvement = corrected_dice - original_dice
    hd_improvement = original_hd - corrected_hd
    
    return {
        'original_dice': original_dice,
        'corrected_dice': corrected_dice,
        'dice_improvement': dice_improvement,
        'original_hd': original_hd,
        'corrected_hd': corrected_hd,
        'hd_improvement': hd_improvement
    }


def evaluate_single_case_lightweight(patient_id):
    """Run complete pipeline evaluation with lightweight reconstruction"""
    
    logger.info("="*80)
    logger.info(f"LIGHTWEIGHT PIPELINE EVALUATION: {patient_id}")
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
        
        # 4. Reconstruct volume using lightweight method
        corrected_mask = reconstruct_volume_lightweight(
            corrected_graph, data['pred_mask'], patient_id
        )
        
        # 5. Compute metrics
        metrics = compute_metrics(
            data['pred_mask'], corrected_mask, data['gt_mask']
        )
        
        # 6. Report results
        processing_time = time.time() - start_time
        
        logger.info("\n" + "="*60)
        logger.info("LIGHTWEIGHT RESULTS SUMMARY")
        logger.info("="*60)
        
        logger.info(f"Patient: {patient_id}")
        logger.info(f"Processing time: {processing_time:.1f} seconds")
        logger.info("")
        
        logger.info("GRAPH CORRECTION STATS:")
        logger.info(f"  Original nodes: {correction_stats['original_nodes']}")
        logger.info(f"  Final nodes: {correction_stats['corrected_nodes']}")
        logger.info(f"  Nodes removed: {correction_stats['nodes_removed']}")
        logger.info(f"  Edges: {correction_stats['edges_retained']}/{correction_stats['original_edges']}")
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
        if metrics['dice_improvement'] > 0.01:
            logger.info("🎉 SUCCESS: Graph correction IMPROVED segmentation quality!")
        elif metrics['dice_improvement'] > -0.01:
            logger.info("⚖️  NEUTRAL: Graph correction maintained segmentation quality")
        else:
            logger.info("⚠️  NEEDS WORK: Graph correction reduced segmentation quality")
        
        logger.info("="*60)
        
        # Save results
        results = {
            'patient_id': patient_id,
            'processing_time': processing_time,
            'correction_stats': correction_stats,
            'metrics': metrics
        }
        
        results_path = Path(f'experiments/lightweight_results_{patient_id}.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"Results saved to {results_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Lightweight evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test lightweight pipeline on single case')
    parser.add_argument('--patient-id', type=str, default='PA000005',
                       help='Patient ID to test (default: PA000005)')
    
    args = parser.parse_args()
    
    results = evaluate_single_case_lightweight(args.patient_id)
    
    if results:
        logger.info("Lightweight single case evaluation completed successfully!")
    else:
        logger.error("Lightweight single case evaluation failed!")


if __name__ == '__main__':
    main()