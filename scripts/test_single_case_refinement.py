#!/usr/bin/env python3
"""
Test the full pipeline with refinement approach
Instead of reconstructing from scratch, refine the original U-Net prediction
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
    
    # Return ALL nodes with their corrections and operations
    # We'll use this info for refinement instead of removing nodes
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


def refine_segmentation_with_corrections(original_mask, correction_data, confidence_threshold=0.7):
    """
    Refine the original segmentation using graph corrections
    Instead of reconstructing from scratch, we:
    1. Remove regions corresponding to nodes marked for removal
    2. Add/enhance regions for modified nodes
    3. Keep most of the original prediction intact
    """
    
    logger.info("Refining segmentation with graph corrections...")
    
    # Start with original mask
    refined_mask = original_mask.copy().astype(np.float32)
    
    # Get node positions
    original_positions = correction_data['original_positions']
    corrected_positions = correction_data['corrected_positions']
    correction_magnitudes = correction_data['correction_magnitudes']
    node_operations = correction_data['node_operations']
    
    # Step 1: Remove false positive regions (but conservatively)
    nodes_to_remove = correction_data['nodes_to_remove']
    if len(nodes_to_remove) > 0:
        logger.info(f"Processing {len(nodes_to_remove)} removal regions...")
        
        # Only remove the most confident false positives (top 20%)
        remove_magnitudes = correction_magnitudes[nodes_to_remove]
        threshold = np.percentile(remove_magnitudes, 80)  # Top 20%
        high_confidence_removes = nodes_to_remove[remove_magnitudes >= threshold]
        
        logger.info(f"Actually removing {len(high_confidence_removes)} high-confidence false positives")
        
        for node_idx in high_confidence_removes:
            pos = original_positions[node_idx]
            x, y, z = int(round(pos[0])), int(round(pos[1])), int(round(pos[2]))
            
            # Remove a small region around the node
            radius = 3  # Small radius
            for dx in range(-radius, radius+1):
                for dy in range(-radius, radius+1):
                    for dz in range(-radius, radius+1):
                        if dx*dx + dy*dy + dz*dz <= radius*radius:
                            nx, ny, nz = x+dx, y+dy, z+dz
                            if (0 <= nx < refined_mask.shape[0] and 
                                0 <= ny < refined_mask.shape[1] and 
                                0 <= nz < refined_mask.shape[2]):
                                
                                # Soft removal (reduce confidence)
                                distance = np.sqrt(dx*dx + dy*dy + dz*dz)
                                weight = 1.0 - distance / radius
                                refined_mask[nx, ny, nz] *= (1.0 - weight * 0.8)
    
    # Step 2: Enhance/correct modified regions
    nodes_to_modify = correction_data['nodes_to_modify']
    if len(nodes_to_modify) > 0:
        logger.info(f"Enhancing {len(nodes_to_modify)} corrected regions...")
        
        for node_idx in nodes_to_modify:
            original_pos = original_positions[node_idx]
            corrected_pos = corrected_positions[node_idx]
            magnitude = correction_magnitudes[node_idx]
            
            # If correction is significant, enhance the corrected position
            if magnitude > 1.0:  # More than 1mm correction
                x, y, z = int(round(corrected_pos[0])), int(round(corrected_pos[1])), int(round(corrected_pos[2]))
                
                # Add/enhance at corrected position
                radius = 2
                for dx in range(-radius, radius+1):
                    for dy in range(-radius, radius+1):
                        for dz in range(-radius, radius+1):
                            if dx*dx + dy*dy + dz*dz <= radius*radius:
                                nx, ny, nz = x+dx, y+dy, z+dz
                                if (0 <= nx < refined_mask.shape[0] and 
                                    0 <= ny < refined_mask.shape[1] and 
                                    0 <= nz < refined_mask.shape[2]):
                                    
                                    distance = np.sqrt(dx*dx + dy*dy + dz*dz)
                                    weight = 1.0 - distance / radius
                                    refined_mask[nx, ny, nz] = max(refined_mask[nx, ny, nz], weight)
    
    # Step 3: Apply smoothing and convert to binary
    logger.info("Applying final smoothing...")
    refined_mask = ndimage.gaussian_filter(refined_mask, sigma=1.0)
    
    # Convert to binary
    binary_refined = (refined_mask > 0.5).astype(np.uint8)
    
    # Apply morphological closing to fill small gaps
    binary_refined = ndimage.binary_closing(binary_refined, structure=np.ones((3,3,3))).astype(np.uint8)
    
    # Remove very small components (noise)
    labeled, num_components = ndimage.label(binary_refined)
    if num_components > 0:
        component_sizes = ndimage.sum(binary_refined, labeled, range(1, num_components + 1))
        small_components = np.where(component_sizes < 50)[0] + 1
        for comp in small_components:
            binary_refined[labeled == comp] = 0
    
    logger.info(f"Refinement completed: {np.sum(binary_refined)} voxels in refined mask")
    
    return binary_refined


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


def evaluate_single_case_refinement(patient_id):
    """Run complete pipeline evaluation with refinement approach"""
    
    logger.info("="*80)
    logger.info(f"REFINEMENT PIPELINE EVALUATION: {patient_id}")
    logger.info("="*80)
    
    start_time = time.time()
    
    try:
        # 1. Load trained model
        model, device = load_trained_model()
        
        # 2. Load case data
        data = load_case_data(patient_id)
        
        # 3. Apply graph correction to get correction data
        correction_data, correction_stats = apply_graph_correction(
            model, data['pred_graph'], device, patient_id
        )
        
        # 4. Refine the original mask using corrections
        refined_mask = refine_segmentation_with_corrections(
            data['pred_mask'], correction_data
        )
        
        # 5. Compute metrics
        metrics = compute_metrics(
            data['pred_mask'], refined_mask, data['gt_mask']
        )
        
        # 6. Report results
        processing_time = time.time() - start_time
        
        logger.info("\n" + "="*60)
        logger.info("REFINEMENT RESULTS SUMMARY")
        logger.info("="*60)
        
        logger.info(f"Patient: {patient_id}")
        logger.info(f"Processing time: {processing_time:.1f} seconds")
        logger.info("")
        
        logger.info("GRAPH CORRECTION ANALYSIS:")
        logger.info(f"  Total nodes analyzed: {correction_stats['total_nodes']}")
        logger.info(f"  Nodes marked for removal: {correction_stats['nodes_to_remove']}")
        logger.info(f"  Nodes marked for modification: {correction_stats['nodes_to_modify']}")
        logger.info(f"  Mean correction magnitude: {correction_stats['mean_correction_magnitude']:.2f}mm")
        logger.info("")
        
        logger.info("VOLUME CHANGES:")
        original_volume = np.sum(data['pred_mask'])
        refined_volume = np.sum(refined_mask)
        volume_change = (refined_volume - original_volume) / original_volume * 100
        logger.info(f"  Original volume: {original_volume} voxels")
        logger.info(f"  Refined volume: {refined_volume} voxels")
        logger.info(f"  Volume change: {volume_change:+.1f}%")
        logger.info("")
        
        logger.info("SEGMENTATION QUALITY:")
        logger.info(f"  Dice Score:")
        logger.info(f"    Original U-Net:  {metrics['original_dice']:.4f}")
        logger.info(f"    After refinement: {metrics['corrected_dice']:.4f}")
        logger.info(f"    Improvement:     {metrics['dice_improvement']:+.4f}")
        logger.info("")
        logger.info(f"  Hausdorff Distance:")
        logger.info(f"    Original U-Net:  {metrics['original_hd']:.2f}mm")
        logger.info(f"    After refinement: {metrics['corrected_hd']:.2f}mm")
        logger.info(f"    Improvement:     {metrics['hd_improvement']:+.2f}mm")
        logger.info("")
        
        # Overall assessment
        if metrics['dice_improvement'] > 0.01:
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
            'volume_change': volume_change,
            'metrics': metrics
        }
        
        results_path = Path(f'experiments/refinement_results_{patient_id}.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"Results saved to {results_path}")
        
        # Also save the refined mask
        refined_nifti_path = Path(f'experiments/refined_predictions/{patient_id}_refined.nii.gz')
        refined_nifti_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load original nifti to preserve header
        original_nifti = nib.load(f'experiments/test_predictions/{patient_id}/binary_mask.nii.gz')
        refined_nifti = nib.Nifti1Image(refined_mask, original_nifti.affine, original_nifti.header)
        nib.save(refined_nifti, refined_nifti_path)
        logger.info(f"Refined mask saved to {refined_nifti_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Refinement evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test refinement pipeline on single case')
    parser.add_argument('--patient-id', type=str, default='PA000005',
                       help='Patient ID to test (default: PA000005)')
    
    args = parser.parse_args()
    
    results = evaluate_single_case_refinement(args.patient_id)
    
    if results:
        logger.info("Refinement evaluation completed successfully!")
    else:
        logger.error("Refinement evaluation failed!")


if __name__ == '__main__':
    main()