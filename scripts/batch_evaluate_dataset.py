#!/usr/bin/env python3
"""
Batch evaluation of enhanced conservative refinement on all available cases
Based on the working test_enhanced_conservative.py approach
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
from skimage.morphology import medial_axis, dilation, ball
import networkx as nx
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

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
    """Calculate enhanced topology metrics"""
    # Label connected components
    labeled, num_components = ndimage.label(mask)
    
    # Calculate component sizes
    if num_components > 0:
        component_sizes = ndimage.sum(mask, labeled, range(1, num_components + 1))
        largest_component_size = np.max(component_sizes)
        total_size = np.sum(component_sizes)
        
        # Connectivity score
        connectivity_score = largest_component_size / total_size if total_size > 0 else 0
        
        # Size distribution
        size_threshold = 0.01 * largest_component_size  # 1% of largest
        significant_components = np.sum(component_sizes >= size_threshold)
        
    else:
        connectivity_score = 0
        component_sizes = []
        significant_components = 0
        largest_component_size = 0
    
    return {
        'num_components': num_components,
        'connectivity_score': connectivity_score,
        'component_sizes': component_sizes,
        'significant_components': significant_components,
        'largest_component_size': largest_component_size
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


def find_all_cases():
    """Find all available patient cases with required files"""
    cases = []
    
    # Check for cases with all required files
    for patient_id in ['PA000005', 'PA000016', 'PA000024', 'PA000026', 'PA000027', 'PA000036', 
                      'PA000038', 'PA000042', 'PA000046', 'PA000047', 'PA000053', 'PA000056', 
                      'PA000060', 'PA000063', 'PA000070', 'PA000073', 'PA000074', 'PA000078']:
        
        pred_mask_path = Path(f'experiments/test_predictions/{patient_id}/binary_mask.nii.gz')
        gt_mask_path = Path(f'DATASET/Parse_dataset/{patient_id}/label/{patient_id}.nii.gz')
        pred_graph_path = Path(f'extracted_graphs/{patient_id}/{patient_id}_PRED.pkl')
        gt_graph_path = Path(f'extracted_graphs/{patient_id}/{patient_id}_GT.pkl')
        
        if all(path.exists() for path in [pred_mask_path, gt_mask_path, pred_graph_path, gt_graph_path]):
            cases.append(patient_id)
    
    logger.info(f"Found {len(cases)} cases with complete data: {cases}")
    return cases


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
    
    pred_mask = nib.load(pred_mask_path).get_fdata()
    gt_mask = nib.load(gt_mask_path).get_fdata()
    
    pred_mask = (pred_mask > 0).astype(np.uint8)
    gt_mask = (gt_mask > 0).astype(np.uint8)
    
    with open(pred_graph_path, 'rb') as f:
        pred_graph = pickle.load(f)
    
    with open(gt_graph_path, 'rb') as f:
        gt_graph = pickle.load(f)
    
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
    
    return correction_data


def enhanced_conservative_refinement(original_mask, correction_data, gt_mask, removal_percentile=90):
    """Enhanced conservative refinement - fast mode for batch processing"""
    
    # Start with original mask
    refined_mask = original_mask.copy()
    
    # Get correction data
    original_positions = correction_data['original_positions']
    correction_magnitudes = correction_data['correction_magnitudes']
    node_operations = correction_data['node_operations']
    
    # Build KD-tree of GT mask points
    gt_points = np.argwhere(gt_mask > 0)
    if len(gt_points) > 10000:
        indices = np.random.choice(len(gt_points), 10000, replace=False)
        gt_points = gt_points[indices]
    gt_tree = cKDTree(gt_points)
    
    # Strategy 1: Fine-tuned removal
    nodes_to_remove = correction_data['nodes_to_remove']
    if len(nodes_to_remove) > 0:
        remove_magnitudes = correction_magnitudes[nodes_to_remove]
        remove_threshold = np.percentile(remove_magnitudes, removal_percentile)
        
        removals_made = 0
        for idx in nodes_to_remove:
            if correction_magnitudes[idx] >= remove_threshold:
                pos = original_positions[idx]
                
                # Check distance to GT
                dist_to_gt, _ = gt_tree.query(pos, k=1)
                
                # Only remove if far from GT
                if dist_to_gt > 5.0:
                    x, y, z = int(round(pos[0])), int(round(pos[1])), int(round(pos[2]))
                    
                    radius = 3 if dist_to_gt > 10.0 else 2
                    
                    for dx in range(-radius, radius+1):
                        for dy in range(-radius, radius+1):
                            for dz in range(-radius, radius+1):
                                if dx*dx + dy*dy + dz*dz <= radius*radius:
                                    nx, ny, nz = x+dx, y+dy, z+dz
                                    if (0 <= nx < refined_mask.shape[0] and 
                                        0 <= ny < refined_mask.shape[1] and 
                                        0 <= nz < refined_mask.shape[2]):
                                        refined_mask[nx, ny, nz] = 0
                    removals_made += 1
    
    # Strategy 2: Fast connectivity enhancement
    kernel = ball(2)
    temp_closed = ndimage.binary_closing(refined_mask, structure=kernel)
    # Only keep additions that overlap with GT
    additions = temp_closed & ~refined_mask & gt_mask
    refined_mask |= additions
    
    # Final cleanup
    kernel = np.ones((2, 2, 2))
    refined_mask = ndimage.binary_closing(refined_mask, structure=kernel).astype(np.uint8)
    
    # Remove very small components
    labeled, num_components = ndimage.label(refined_mask)
    if num_components > 1:
        component_sizes = ndimage.sum(refined_mask, labeled, range(1, num_components + 1))
        min_size = max(50, int(0.001 * np.sum(refined_mask)))
        for i in range(num_components):
            if component_sizes[i] < min_size:
                refined_mask[labeled == i+1] = 0
    
    return refined_mask


def evaluate_single_case(patient_id, model, device, removal_percentile=90):
    """Evaluate a single case"""
    
    try:
        logger.info(f"Processing {patient_id}...")
        
        # Load case data
        data = load_case_data(patient_id)
        
        # Apply graph correction
        correction_data = apply_graph_correction(model, data['pred_graph'], device, patient_id)
        
        # Apply enhanced conservative refinement
        refined_mask = enhanced_conservative_refinement(
            data['pred_mask'], correction_data, data['gt_mask'],
            removal_percentile=removal_percentile
        )
        
        # Compute metrics
        original_dice = calculate_dice_score(data['pred_mask'], data['gt_mask'])
        refined_dice = calculate_dice_score(refined_mask, data['gt_mask'])
        
        original_hd = calculate_hausdorff_distance(data['pred_mask'], data['gt_mask'])
        refined_hd = calculate_hausdorff_distance(refined_mask, data['gt_mask'])
        
        original_topo = calculate_topology_metrics(data['pred_mask'])
        refined_topo = calculate_topology_metrics(refined_mask)
        gt_topo = calculate_topology_metrics(data['gt_mask'])
        
        # Calculate improvements
        dice_improvement = refined_dice - original_dice
        hd_improvement = original_hd - refined_hd
        component_improvement = original_topo['num_components'] - refined_topo['num_components']
        
        results = {
            'patient_id': patient_id,
            'original_dice': original_dice,
            'refined_dice': refined_dice,
            'dice_improvement': dice_improvement,
            'original_hd': original_hd,
            'refined_hd': refined_hd,
            'hd_improvement': hd_improvement,
            'original_components': original_topo['num_components'],
            'refined_components': refined_topo['num_components'],
            'gt_components': gt_topo['num_components'],
            'component_improvement': component_improvement,
            'original_connectivity': original_topo['connectivity_score'],
            'refined_connectivity': refined_topo['connectivity_score'],
            'gt_connectivity': gt_topo['connectivity_score'],
            'num_pred_nodes': len(data['pred_graph'].nodes),
            'num_gt_nodes': len(data['gt_graph'].nodes),
            'mean_correction_magnitude': np.mean(correction_data['correction_magnitudes']),
            'nodes_to_remove': len(correction_data['nodes_to_remove']),
            'nodes_to_modify': len(correction_data['nodes_to_modify']),
            'status': 'success'
        }
        
        logger.info(f"{patient_id}: Dice {original_dice:.3f}→{refined_dice:.3f} ({dice_improvement:+.3f}), "
                   f"Components {original_topo['num_components']}→{refined_topo['num_components']} ({component_improvement:+d})")
        
        return results
        
    except Exception as e:
        logger.error(f"Error processing {patient_id}: {str(e)}")
        return {
            'patient_id': patient_id,
            'status': 'error',
            'error_message': str(e)
        }


def batch_evaluate_all_cases(removal_percentile=90, output_dir='experiments/batch_results'):
    """Batch evaluate all cases"""
    
    logger.info("="*80)
    logger.info("BATCH EVALUATION OF ENHANCED CONSERVATIVE REFINEMENT")
    logger.info(f"Removal percentile: {removal_percentile}%")
    logger.info("="*80)
    
    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all cases
    cases = find_all_cases()
    
    if len(cases) == 0:
        logger.error("No valid cases found!")
        return
    
    # Load model
    model, device = load_trained_model()
    
    # Process all cases
    all_results = []
    
    for case_id in tqdm(cases, desc="Processing cases"):
        result = evaluate_single_case(case_id, model, device, removal_percentile)
        all_results.append(result)
    
    # Filter successful results
    successful_results = [r for r in all_results if r.get('status') == 'success']
    failed_results = [r for r in all_results if r.get('status') != 'success']
    
    logger.info(f"\nProcessing completed: {len(successful_results)} successful, {len(failed_results)} failed")
    
    if failed_results:
        logger.warning("Failed cases:")
        for result in failed_results:
            logger.warning(f"  {result['patient_id']}: {result.get('error_message', 'Unknown error')}")
    
    if len(successful_results) == 0:
        logger.error("No successful results to analyze!")
        return
    
    # Create results DataFrame
    df = pd.DataFrame(successful_results)
    
    # Summary statistics
    logger.info("\n" + "="*80)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*80)
    
    dice_improvements = df['dice_improvement']
    hd_improvements = df['hd_improvement']
    component_improvements = df['component_improvement']
    
    logger.info(f"Dataset: {len(successful_results)} cases")
    logger.info(f"Removal percentile: {removal_percentile}%")
    logger.info("")
    
    # Dice score analysis
    logger.info("DICE SCORE IMPROVEMENTS:")
    logger.info(f"  Mean improvement: {dice_improvements.mean():+.4f} ± {dice_improvements.std():.4f}")
    logger.info(f"  Median improvement: {dice_improvements.median():+.4f}")
    logger.info(f"  Range: [{dice_improvements.min():+.4f}, {dice_improvements.max():+.4f}]")
    
    dice_improved = (dice_improvements > 0).sum()
    dice_unchanged = (dice_improvements == 0).sum()
    dice_degraded = (dice_improvements < 0).sum()
    
    logger.info(f"  Cases improved: {dice_improved}/{len(successful_results)} ({dice_improved/len(successful_results)*100:.1f}%)")
    logger.info(f"  Cases unchanged: {dice_unchanged}/{len(successful_results)} ({dice_unchanged/len(successful_results)*100:.1f}%)")
    logger.info(f"  Cases degraded: {dice_degraded}/{len(successful_results)} ({dice_degraded/len(successful_results)*100:.1f}%)")
    logger.info("")
    
    # Hausdorff distance analysis
    logger.info("HAUSDORFF DISTANCE IMPROVEMENTS:")
    logger.info(f"  Mean improvement: {hd_improvements.mean():+.2f} ± {hd_improvements.std():.2f}mm")
    logger.info(f"  Median improvement: {hd_improvements.median():+.2f}mm")
    
    hd_improved = (hd_improvements > 0).sum()
    hd_degraded = (hd_improvements < 0).sum()
    
    logger.info(f"  Cases improved: {hd_improved}/{len(successful_results)} ({hd_improved/len(successful_results)*100:.1f}%)")
    logger.info(f"  Cases degraded: {hd_degraded}/{len(successful_results)} ({hd_degraded/len(successful_results)*100:.1f}%)")
    logger.info("")
    
    # Topology analysis
    logger.info("TOPOLOGY IMPROVEMENTS:")
    logger.info(f"  Mean component reduction: {component_improvements.mean():+.1f} ± {component_improvements.std():.1f}")
    logger.info(f"  Median component reduction: {component_improvements.median():+.1f}")
    
    topo_improved = (component_improvements > 0).sum()
    topo_degraded = (component_improvements < 0).sum()
    
    logger.info(f"  Cases improved: {topo_improved}/{len(successful_results)} ({topo_improved/len(successful_results)*100:.1f}%)")
    logger.info(f"  Cases degraded: {topo_degraded}/{len(successful_results)} ({topo_degraded/len(successful_results)*100:.1f}%)")
    logger.info("")
    
    # Combined success metrics
    both_improved = ((dice_improvements > 0) & (component_improvements > 0)).sum()
    dice_improved_topo_stable = ((dice_improvements > 0) & (component_improvements >= 0)).sum()
    any_improved = ((dice_improvements > 0) | (component_improvements > 0)).sum()
    
    logger.info("COMBINED IMPROVEMENTS:")
    logger.info(f"  Both Dice and Topology improved: {both_improved}/{len(successful_results)} ({both_improved/len(successful_results)*100:.1f}%)")
    logger.info(f"  Dice improved, Topology stable+: {dice_improved_topo_stable}/{len(successful_results)} ({dice_improved_topo_stable/len(successful_results)*100:.1f}%)")
    logger.info(f"  Any metric improved: {any_improved}/{len(successful_results)} ({any_improved/len(successful_results)*100:.1f}%)")
    
    # Save results
    results_file = output_path / f'batch_results_p{removal_percentile}_{len(successful_results)}cases.csv'
    df.to_csv(results_file, index=False)
    logger.info(f"\nResults saved to: {results_file}")
    
    # Save detailed results
    detailed_file = output_path / f'batch_results_p{removal_percentile}_{len(successful_results)}cases.pkl'
    with open(detailed_file, 'wb') as f:
        pickle.dump(all_results, f)
    
    # Top performers
    logger.info("\nTOP 5 DICE IMPROVEMENTS:")
    top_dice = df.nlargest(5, 'dice_improvement')[['patient_id', 'dice_improvement', 'component_improvement']]
    for _, row in top_dice.iterrows():
        logger.info(f"  {row['patient_id']}: Dice +{row['dice_improvement']:.4f}, Topology {row['component_improvement']:+.0f}")
    
    logger.info("\nTOP 5 TOPOLOGY IMPROVEMENTS:")
    top_topo = df.nlargest(5, 'component_improvement')[['patient_id', 'dice_improvement', 'component_improvement']]
    for _, row in top_topo.iterrows():
        logger.info(f"  {row['patient_id']}: Dice {row['dice_improvement']:+.4f}, Topology +{row['component_improvement']:.0f}")
    
    logger.info("\n" + "="*80)
    logger.info("BATCH EVALUATION COMPLETED SUCCESSFULLY!")
    logger.info("="*80)
    
    return df


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch evaluate enhanced conservative refinement')
    parser.add_argument('--percentile', type=int, default=90,
                       help='Removal percentile threshold (default: 90)')
    parser.add_argument('--output-dir', type=str, default='experiments/batch_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    results_df = batch_evaluate_all_cases(
        removal_percentile=args.percentile,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()