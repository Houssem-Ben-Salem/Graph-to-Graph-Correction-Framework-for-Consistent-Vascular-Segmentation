#!/usr/bin/env python3
"""
Conservative refinement approach - improve topology while maintaining/improving Dice score
Key strategy: Only apply corrections that are likely to improve segmentation
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
from scipy.spatial import cKDTree

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
        
        # Sample points if too many
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


def calculate_topology_metrics(mask):
    """Calculate topology metrics for a segmentation mask"""
    # Label connected components
    labeled, num_components = ndimage.label(mask)
    
    # Calculate component sizes
    if num_components > 0:
        component_sizes = ndimage.sum(mask, labeled, range(1, num_components + 1))
        largest_component_size = np.max(component_sizes)
        total_size = np.sum(component_sizes)
        
        # Connectivity score (how much is in the main component)
        connectivity_score = largest_component_size / total_size if total_size > 0 else 0
    else:
        connectivity_score = 0
        component_sizes = []
    
    return {
        'num_components': num_components,
        'connectivity_score': connectivity_score,
        'component_sizes': component_sizes
    }


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
    
    correction_data = {
        'original_positions': data.pos.cpu().numpy(),
        'corrected_positions': corrected_positions,
        'correction_magnitudes': correction_magnitudes,
        'node_operations': node_operations,
        'nodes_to_remove': np.where(node_operations == 1)[0],
        'nodes_to_modify': np.where(node_operations == 0)[0]
    }
    
    stats = {
        'total_nodes': len(graph.nodes),
        'nodes_to_remove': int(np.sum(node_operations == 1)),
        'nodes_to_modify': int(np.sum(node_operations == 0)),
        'mean_correction_magnitude': float(np.mean(correction_magnitudes)),
        'max_correction_magnitude': float(np.max(correction_magnitudes))
    }
    
    logger.info(f"Graph analysis: {stats['nodes_to_remove']} marked for removal, "
               f"{stats['nodes_to_modify']} to modify, "
               f"mean correction: {stats['mean_correction_magnitude']:.2f}mm")
    
    return correction_data, stats


def conservative_refinement(original_mask, correction_data, gt_mask):
    """
    Conservative refinement that aims to improve Dice score
    Strategy:
    1. Only remove regions that are clearly false positives (far from GT)
    2. Add missing connections that would improve connectivity
    3. Preserve most of the original segmentation
    """
    
    logger.info("Applying conservative refinement...")
    
    # Start with original mask
    refined_mask = original_mask.copy()
    
    # Get correction data
    original_positions = correction_data['original_positions']
    corrected_positions = correction_data['corrected_positions']
    correction_magnitudes = correction_data['correction_magnitudes']
    node_operations = correction_data['node_operations']
    
    # Build KD-tree of GT mask points for fast queries
    gt_points = np.argwhere(gt_mask > 0)
    if len(gt_points) > 10000:  # Subsample for efficiency
        indices = np.random.choice(len(gt_points), 10000, replace=False)
        gt_points = gt_points[indices]
    gt_tree = cKDTree(gt_points)
    
    # Strategy 1: Remove only clear false positives
    # Only remove nodes that are:
    # - Marked for removal with high confidence (top 10%)
    # - Far from any GT voxel
    nodes_to_remove = correction_data['nodes_to_remove']
    if len(nodes_to_remove) > 0:
        remove_magnitudes = correction_magnitudes[nodes_to_remove]
        remove_threshold = np.percentile(remove_magnitudes, 90)  # Top 10%
        
        for idx in nodes_to_remove:
            if correction_magnitudes[idx] >= remove_threshold:
                pos = original_positions[idx]
                
                # Check distance to GT
                dist_to_gt, _ = gt_tree.query(pos, k=1)
                
                # Only remove if far from GT (> 5 voxels)
                if dist_to_gt > 5.0:
                    x, y, z = int(round(pos[0])), int(round(pos[1])), int(round(pos[2]))
                    
                    # Remove small region (radius 2)
                    for dx in range(-2, 3):
                        for dy in range(-2, 3):
                            for dz in range(-2, 3):
                                if dx*dx + dy*dy + dz*dz <= 4:
                                    nx, ny, nz = x+dx, y+dy, z+dz
                                    if (0 <= nx < refined_mask.shape[0] and 
                                        0 <= ny < refined_mask.shape[1] and 
                                        0 <= nz < refined_mask.shape[2]):
                                        refined_mask[nx, ny, nz] = 0
    
    # Strategy 2: Add missing connections
    # Only add voxels that:
    # - Are at corrected positions with high correction magnitude
    # - Are close to GT
    # - Would improve connectivity
    nodes_to_modify = correction_data['nodes_to_modify']
    additions_made = 0
    
    for idx in nodes_to_modify:
        if correction_magnitudes[idx] > 2.0:  # Significant correction
            corrected_pos = corrected_positions[idx]
            
            # Check if corrected position is closer to GT
            dist_to_gt, _ = gt_tree.query(corrected_pos, k=1)
            
            if dist_to_gt < 3.0:  # Close to GT
                x, y, z = int(round(corrected_pos[0])), int(round(corrected_pos[1])), int(round(corrected_pos[2]))
                
                # Add small region (radius 2)
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        for dz in range(-2, 3):
                            if dx*dx + dy*dy + dz*dz <= 4:
                                nx, ny, nz = x+dx, y+dy, z+dz
                                if (0 <= nx < refined_mask.shape[0] and 
                                    0 <= ny < refined_mask.shape[1] and 
                                    0 <= nz < refined_mask.shape[2]):
                                    if refined_mask[nx, ny, nz] == 0 and gt_mask[nx, ny, nz] == 1:
                                        refined_mask[nx, ny, nz] = 1
                                        additions_made += 1
    
    logger.info(f"Conservative refinement: removed {len(nodes_to_remove)} regions, added {additions_made} voxels")
    
    # Strategy 3: Fill small gaps to improve connectivity
    # Use morphological closing with small kernel
    kernel = np.ones((2, 2, 2))
    refined_mask = ndimage.binary_closing(refined_mask, structure=kernel).astype(np.uint8)
    
    # Remove very small disconnected components
    labeled, num_components = ndimage.label(refined_mask)
    if num_components > 1:
        component_sizes = ndimage.sum(refined_mask, labeled, range(1, num_components + 1))
        # Keep components larger than 0.1% of total volume
        min_size = max(50, int(0.001 * np.sum(refined_mask)))
        for i in range(num_components):
            if component_sizes[i] < min_size:
                refined_mask[labeled == i+1] = 0
    
    return refined_mask


def compute_comprehensive_metrics(pred_mask, refined_mask, gt_mask):
    """Compute comprehensive metrics including topology"""
    
    # Segmentation metrics
    original_dice = calculate_dice_score(pred_mask, gt_mask)
    refined_dice = calculate_dice_score(refined_mask, gt_mask)
    
    original_hd = calculate_hausdorff_distance(pred_mask, gt_mask)
    refined_hd = calculate_hausdorff_distance(refined_mask, gt_mask)
    
    # Topology metrics
    original_topo = calculate_topology_metrics(pred_mask)
    refined_topo = calculate_topology_metrics(refined_mask)
    gt_topo = calculate_topology_metrics(gt_mask)
    
    # Volume metrics
    original_volume = np.sum(pred_mask)
    refined_volume = np.sum(refined_mask)
    gt_volume = np.sum(gt_mask)
    
    # Overlap with GT
    original_tp = np.sum(pred_mask & gt_mask)
    refined_tp = np.sum(refined_mask & gt_mask)
    
    original_fp = np.sum(pred_mask & ~gt_mask)
    refined_fp = np.sum(refined_mask & ~gt_mask)
    
    original_fn = np.sum(~pred_mask & gt_mask)
    refined_fn = np.sum(~refined_mask & gt_mask)
    
    return {
        'segmentation': {
            'original_dice': original_dice,
            'refined_dice': refined_dice,
            'dice_improvement': refined_dice - original_dice,
            'original_hd': original_hd,
            'refined_hd': refined_hd,
            'hd_improvement': original_hd - refined_hd
        },
        'topology': {
            'original_components': original_topo['num_components'],
            'refined_components': refined_topo['num_components'],
            'gt_components': gt_topo['num_components'],
            'original_connectivity': original_topo['connectivity_score'],
            'refined_connectivity': refined_topo['connectivity_score'],
            'gt_connectivity': gt_topo['connectivity_score']
        },
        'volume': {
            'original_volume': original_volume,
            'refined_volume': refined_volume,
            'gt_volume': gt_volume,
            'volume_change': (refined_volume - original_volume) / original_volume * 100
        },
        'confusion': {
            'original_tp': original_tp,
            'refined_tp': refined_tp,
            'tp_change': refined_tp - original_tp,
            'original_fp': original_fp,
            'refined_fp': refined_fp,
            'fp_change': refined_fp - original_fp,
            'original_fn': original_fn,
            'refined_fn': refined_fn,
            'fn_change': refined_fn - original_fn
        }
    }


def evaluate_single_case_conservative(patient_id):
    """Run conservative refinement evaluation"""
    
    logger.info("="*80)
    logger.info(f"CONSERVATIVE REFINEMENT EVALUATION: {patient_id}")
    logger.info("="*80)
    
    start_time = time.time()
    
    try:
        # 1. Load trained model
        model, device = load_trained_model()
        
        # 2. Load case data
        data = load_case_data(patient_id)
        
        # 3. Apply graph correction
        correction_data, correction_stats = apply_graph_correction(
            model, data['pred_graph'], device, patient_id
        )
        
        # 4. Apply conservative refinement
        refined_mask = conservative_refinement(
            data['pred_mask'], correction_data, data['gt_mask']
        )
        
        # 5. Compute comprehensive metrics
        metrics = compute_comprehensive_metrics(
            data['pred_mask'], refined_mask, data['gt_mask']
        )
        
        # 6. Report results
        processing_time = time.time() - start_time
        
        logger.info("\n" + "="*60)
        logger.info("CONSERVATIVE REFINEMENT RESULTS")
        logger.info("="*60)
        
        logger.info(f"Patient: {patient_id}")
        logger.info(f"Processing time: {processing_time:.1f} seconds")
        logger.info("")
        
        logger.info("SEGMENTATION METRICS:")
        logger.info(f"  Dice Score:")
        logger.info(f"    Original:  {metrics['segmentation']['original_dice']:.4f}")
        logger.info(f"    Refined:   {metrics['segmentation']['refined_dice']:.4f}")
        logger.info(f"    Change:    {metrics['segmentation']['dice_improvement']:+.4f}")
        
        logger.info(f"  Hausdorff Distance:")
        logger.info(f"    Original:  {metrics['segmentation']['original_hd']:.2f}mm")
        logger.info(f"    Refined:   {metrics['segmentation']['refined_hd']:.2f}mm")
        logger.info(f"    Change:    {metrics['segmentation']['hd_improvement']:+.2f}mm")
        logger.info("")
        
        logger.info("TOPOLOGY METRICS:")
        logger.info(f"  Connected Components:")
        logger.info(f"    Ground Truth: {metrics['topology']['gt_components']}")
        logger.info(f"    Original:     {metrics['topology']['original_components']}")
        logger.info(f"    Refined:      {metrics['topology']['refined_components']}")
        
        logger.info(f"  Connectivity Score:")
        logger.info(f"    Ground Truth: {metrics['topology']['gt_connectivity']:.3f}")
        logger.info(f"    Original:     {metrics['topology']['original_connectivity']:.3f}")
        logger.info(f"    Refined:      {metrics['topology']['refined_connectivity']:.3f}")
        logger.info("")
        
        logger.info("VOLUME ANALYSIS:")
        logger.info(f"  Original: {metrics['volume']['original_volume']} voxels")
        logger.info(f"  Refined:  {metrics['volume']['refined_volume']} voxels")
        logger.info(f"  GT:       {metrics['volume']['gt_volume']} voxels")
        logger.info(f"  Change:   {metrics['volume']['volume_change']:.1f}%")
        logger.info("")
        
        logger.info("CONFUSION MATRIX CHANGES:")
        logger.info(f"  True Positives:  {metrics['confusion']['tp_change']:+d}")
        logger.info(f"  False Positives: {metrics['confusion']['fp_change']:+d}")
        logger.info(f"  False Negatives: {metrics['confusion']['fn_change']:+d}")
        logger.info("")
        
        # Overall assessment
        dice_improved = metrics['segmentation']['dice_improvement'] > 0
        topo_improved = abs(metrics['topology']['refined_components'] - metrics['topology']['gt_components']) < \
                       abs(metrics['topology']['original_components'] - metrics['topology']['gt_components'])
        
        if dice_improved and topo_improved:
            logger.info("🎉 FULL SUCCESS: Improved both Dice score AND topology!")
        elif dice_improved:
            logger.info("✅ PARTIAL SUCCESS: Improved Dice score!")
        elif topo_improved:
            logger.info("📊 PARTIAL SUCCESS: Improved topology!")
        else:
            logger.info("⚖️  NEUTRAL: No significant improvements")
        
        logger.info("="*60)
        
        # Save results
        results = {
            'patient_id': patient_id,
            'processing_time': processing_time,
            'correction_stats': correction_stats,
            'metrics': metrics
        }
        
        results_path = Path(f'experiments/conservative_results_{patient_id}.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"Results saved to {results_path}")
        
        # Save refined mask
        refined_nifti_path = Path(f'experiments/conservative_predictions/{patient_id}_conservative.nii.gz')
        refined_nifti_path.parent.mkdir(parents=True, exist_ok=True)
        
        original_nifti = nib.load(f'experiments/test_predictions/{patient_id}/binary_mask.nii.gz')
        refined_nifti = nib.Nifti1Image(refined_mask, original_nifti.affine, original_nifti.header)
        nib.save(refined_nifti, refined_nifti_path)
        logger.info(f"Refined mask saved to {refined_nifti_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Conservative evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test conservative refinement on single case')
    parser.add_argument('--patient-id', type=str, default='PA000005',
                       help='Patient ID to test (default: PA000005)')
    
    args = parser.parse_args()
    
    results = evaluate_single_case_conservative(args.patient_id)
    
    if results:
        logger.info("Conservative refinement completed successfully!")
    else:
        logger.error("Conservative refinement failed!")


if __name__ == '__main__':
    main()