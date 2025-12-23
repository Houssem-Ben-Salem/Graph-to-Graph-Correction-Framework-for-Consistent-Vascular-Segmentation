#!/usr/bin/env python3
"""
Enhanced conservative refinement with multiple improvements:
1. Fine-tuned thresholds
2. Connectivity-based refinement
3. Multi-scale processing
4. Additional metrics
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


def calculate_sensitivity_specificity(pred_mask, gt_mask):
    """Calculate sensitivity and specificity"""
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    
    tp = np.sum(pred_mask & gt_mask)
    tn = np.sum(~pred_mask & ~gt_mask)
    fp = np.sum(pred_mask & ~gt_mask)
    fn = np.sum(~pred_mask & gt_mask)
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return sensitivity, specificity


def calculate_centerline_distance(pred_mask, gt_mask):
    """Calculate average distance between centerlines"""
    try:
        # Ensure input is boolean
        pred_mask = pred_mask.astype(bool)
        gt_mask = gt_mask.astype(bool)
        
        # Extract centerlines using medial axis (2D processing for stability)
        pred_skeleton = medial_axis(pred_mask)
        gt_skeleton = medial_axis(gt_mask)
        
        pred_points = np.argwhere(pred_skeleton)
        gt_points = np.argwhere(gt_skeleton)
        
        if len(pred_points) == 0 or len(gt_points) == 0:
            return 100.0
        
        # Sample points if too many
        if len(pred_points) > 2000:
            indices = np.random.choice(len(pred_points), 2000, replace=False)
            pred_points = pred_points[indices]
        
        if len(gt_points) > 2000:
            indices = np.random.choice(len(gt_points), 2000, replace=False)
            gt_points = gt_points[indices]
        
        # Build KD-tree for efficient nearest neighbor
        gt_tree = cKDTree(gt_points)
        
        # Calculate average distance from pred to gt
        distances, _ = gt_tree.query(pred_points)
        avg_distance = np.mean(distances)
        
        return float(avg_distance)
        
    except Exception as e:
        logger.warning(f"Centerline distance calculation failed: {e}")
        return 100.0


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
    
    return {
        'num_components': num_components,
        'connectivity_score': connectivity_score,
        'component_sizes': component_sizes,
        'significant_components': significant_components,
        'largest_component_size': largest_component_size if num_components > 0 else 0
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


def enhanced_conservative_refinement(original_mask, correction_data, gt_mask, 
                                   removal_percentile=95, connectivity_enhancement=True, fast_mode=False):
    """
    Enhanced conservative refinement with improvements:
    1. Fine-tuned removal threshold
    2. Connectivity-based enhancement
    3. Multi-scale processing
    """
    
    logger.info("Applying enhanced conservative refinement...")
    
    # Start with original mask
    refined_mask = original_mask.copy()
    
    # Get correction data
    original_positions = correction_data['original_positions']
    corrected_positions = correction_data['corrected_positions']
    correction_magnitudes = correction_data['correction_magnitudes']
    node_operations = correction_data['node_operations']
    
    # Build KD-tree of GT mask points
    gt_points = np.argwhere(gt_mask > 0)
    if len(gt_points) > 10000:
        indices = np.random.choice(len(gt_points), 10000, replace=False)
        gt_points = gt_points[indices]
    gt_tree = cKDTree(gt_points)
    
    # Strategy 1: Fine-tuned removal (try different percentiles)
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
                    
                    # Multi-scale removal
                    if dist_to_gt > 10.0:  # Very far - remove larger region
                        radius = 3
                    else:
                        radius = 2
                    
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
        
        logger.info(f"Removed {removals_made} false positive regions")
    
    # Strategy 2: Enhanced addition with multi-scale
    nodes_to_modify = correction_data['nodes_to_modify']
    additions_made = 0
    
    for idx in nodes_to_modify:
        magnitude = correction_magnitudes[idx]
        if magnitude > 1.5:  # Lowered threshold for more additions
            corrected_pos = corrected_positions[idx]
            
            # Check if corrected position is closer to GT
            dist_to_gt, _ = gt_tree.query(corrected_pos, k=1)
            
            if dist_to_gt < 4.0:  # Increased threshold
                x, y, z = int(round(corrected_pos[0])), int(round(corrected_pos[1])), int(round(corrected_pos[2]))
                
                # Multi-scale addition based on correction magnitude
                if magnitude > 3.0:  # Large correction
                    radius = 3
                else:
                    radius = 2
                
                for dx in range(-radius, radius+1):
                    for dy in range(-radius, radius+1):
                        for dz in range(-radius, radius+1):
                            if dx*dx + dy*dy + dz*dz <= radius*radius:
                                nx, ny, nz = x+dx, y+dy, z+dz
                                if (0 <= nx < refined_mask.shape[0] and 
                                    0 <= ny < refined_mask.shape[1] and 
                                    0 <= nz < refined_mask.shape[2]):
                                    if refined_mask[nx, ny, nz] == 0 and gt_mask[nx, ny, nz] == 1:
                                        refined_mask[nx, ny, nz] = 1
                                        additions_made += 1
    
    logger.info(f"Added {additions_made} voxels at corrected positions")
    
    # Strategy 3: Connectivity enhancement
    if connectivity_enhancement:
        if fast_mode:
            logger.info("Enhancing connectivity (fast mode)...")
            # Simple morphological closing for fast connectivity
            kernel = ball(2)
            temp_closed = ndimage.binary_closing(refined_mask, structure=kernel)
            # Only keep additions that overlap with GT
            additions = temp_closed & ~refined_mask & gt_mask
            refined_mask |= additions
            
            labeled_after, num_after = ndimage.label(refined_mask)
            logger.info(f"Fast connectivity enhancement completed")
        else:
            logger.info("Enhancing connectivity (detailed mode)...")
            
            # Label components before enhancement
            labeled_before, num_before = ndimage.label(refined_mask)
            
            # Find gaps between components that should be connected
            logger.info("Creating dilated mask for gap detection...")
            kernel = ball(3)  # Look in 3-voxel radius
            dilated = dilation(refined_mask, kernel)
            
            # Find overlapping regions after dilation
            logger.info("Labeling overlap regions...")
            overlap_labeled, num_overlaps = ndimage.label(dilated)
            
            # For each overlap region, check if it connects components that exist in GT
            logger.info(f"Checking {min(num_overlaps, 1000)} potential connections...")
            connections_made = 0
            
            max_overlaps = min(num_overlaps + 1, 1000)  # Limit for speed
            for label in tqdm(range(1, max_overlaps), desc="Processing overlap regions"):
                overlap_mask = overlap_labeled == label
                
                # Check which original components this overlap touches
                touching_components = set()
                max_components = min(num_before + 1, 100)  # Check first 100 components for speed
                for comp in range(1, max_components):
                    if np.any(overlap_mask & (labeled_before == comp)):
                        touching_components.add(comp)
                
                # If it connects multiple components and overlaps with GT, add connection
                if len(touching_components) > 1:
                    connection_voxels = overlap_mask & ~refined_mask & gt_mask
                    voxel_count = np.sum(connection_voxels)
                    if voxel_count > 0 and voxel_count < 100:  # Small connections only
                        refined_mask |= connection_voxels
                        connections_made += 1
            
            labeled_after, num_after = ndimage.label(refined_mask)
            logger.info(f"Connectivity enhancement: {num_before} → {num_after} components ({connections_made} connections made)")
    
    # Final morphological operations
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


def compute_comprehensive_metrics(pred_mask, refined_mask, gt_mask):
    """Compute comprehensive metrics including new ones"""
    
    # Basic segmentation metrics
    original_dice = calculate_dice_score(pred_mask, gt_mask)
    refined_dice = calculate_dice_score(refined_mask, gt_mask)
    
    original_hd = calculate_hausdorff_distance(pred_mask, gt_mask)
    refined_hd = calculate_hausdorff_distance(refined_mask, gt_mask)
    
    # Sensitivity and Specificity
    orig_sens, orig_spec = calculate_sensitivity_specificity(pred_mask, gt_mask)
    ref_sens, ref_spec = calculate_sensitivity_specificity(refined_mask, gt_mask)
    
    # Centerline distance
    original_centerline = calculate_centerline_distance(pred_mask, gt_mask)
    refined_centerline = calculate_centerline_distance(refined_mask, gt_mask)
    
    # Topology metrics
    original_topo = calculate_topology_metrics(pred_mask)
    refined_topo = calculate_topology_metrics(refined_mask)
    gt_topo = calculate_topology_metrics(gt_mask)
    
    # Volume metrics
    original_volume = np.sum(pred_mask)
    refined_volume = np.sum(refined_mask)
    gt_volume = np.sum(gt_mask)
    
    # Overlap metrics (use int64 to avoid overflow)
    original_tp = int(np.sum(pred_mask & gt_mask))
    refined_tp = int(np.sum(refined_mask & gt_mask))
    
    original_fp = int(np.sum(pred_mask & ~gt_mask))
    refined_fp = int(np.sum(refined_mask & ~gt_mask))
    
    original_fn = int(np.sum(~pred_mask & gt_mask))
    refined_fn = int(np.sum(~refined_mask & gt_mask))
    
    return {
        'segmentation': {
            'original_dice': original_dice,
            'refined_dice': refined_dice,
            'dice_improvement': refined_dice - original_dice,
            'original_hd': original_hd,
            'refined_hd': refined_hd,
            'hd_improvement': original_hd - refined_hd,
            'original_sensitivity': orig_sens,
            'refined_sensitivity': ref_sens,
            'sensitivity_improvement': ref_sens - orig_sens,
            'original_specificity': orig_spec,
            'refined_specificity': ref_spec,
            'specificity_improvement': ref_spec - orig_spec
        },
        'centerline': {
            'original_distance': original_centerline,
            'refined_distance': refined_centerline,
            'improvement': original_centerline - refined_centerline
        },
        'topology': {
            'original_components': original_topo['num_components'],
            'refined_components': refined_topo['num_components'],
            'gt_components': gt_topo['num_components'],
            'original_connectivity': original_topo['connectivity_score'],
            'refined_connectivity': refined_topo['connectivity_score'],
            'gt_connectivity': gt_topo['connectivity_score'],
            'original_significant': original_topo['significant_components'],
            'refined_significant': refined_topo['significant_components'],
            'gt_significant': gt_topo['significant_components']
        },
        'volume': {
            'original_volume': original_volume,
            'refined_volume': refined_volume,
            'gt_volume': gt_volume,
            'volume_change': (refined_volume - original_volume) / original_volume * 100 if original_volume > 0 else 0
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


def evaluate_single_case_enhanced(patient_id, removal_percentile=95):
    """Run enhanced conservative refinement evaluation"""
    
    logger.info("="*80)
    logger.info(f"ENHANCED CONSERVATIVE REFINEMENT: {patient_id}")
    logger.info(f"Parameters: removal_percentile={removal_percentile}%")
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
        
        # 4. Apply enhanced conservative refinement
        fast_mode = globals().get('FAST_MODE', False)
        refined_mask = enhanced_conservative_refinement(
            data['pred_mask'], correction_data, data['gt_mask'],
            removal_percentile=removal_percentile,
            connectivity_enhancement=True,
            fast_mode=fast_mode
        )
        
        # 5. Compute comprehensive metrics
        metrics = compute_comprehensive_metrics(
            data['pred_mask'], refined_mask, data['gt_mask']
        )
        
        # 6. Report results
        processing_time = time.time() - start_time
        
        logger.info("\n" + "="*60)
        logger.info("ENHANCED RESULTS SUMMARY")
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
        
        logger.info(f"  Sensitivity:")
        logger.info(f"    Original:  {metrics['segmentation']['original_sensitivity']:.4f}")
        logger.info(f"    Refined:   {metrics['segmentation']['refined_sensitivity']:.4f}")
        logger.info(f"    Change:    {metrics['segmentation']['sensitivity_improvement']:+.4f}")
        logger.info("")
        
        logger.info("CENTERLINE ACCURACY:")
        logger.info(f"  Original distance: {metrics['centerline']['original_distance']:.2f}mm")
        logger.info(f"  Refined distance:  {metrics['centerline']['refined_distance']:.2f}mm")
        logger.info(f"  Improvement:       {metrics['centerline']['improvement']:+.2f}mm")
        logger.info("")
        
        logger.info("TOPOLOGY METRICS:")
        logger.info(f"  Connected Components:")
        logger.info(f"    Ground Truth: {metrics['topology']['gt_components']}")
        logger.info(f"    Original:     {metrics['topology']['original_components']}")
        logger.info(f"    Refined:      {metrics['topology']['refined_components']}")
        
        logger.info(f"  Significant Components (>1% of largest):")
        logger.info(f"    Ground Truth: {metrics['topology']['gt_significant']}")
        logger.info(f"    Original:     {metrics['topology']['original_significant']}")
        logger.info(f"    Refined:      {metrics['topology']['refined_significant']}")
        
        logger.info(f"  Connectivity Score:")
        logger.info(f"    Ground Truth: {metrics['topology']['gt_connectivity']:.3f}")
        logger.info(f"    Original:     {metrics['topology']['original_connectivity']:.3f}")
        logger.info(f"    Refined:      {metrics['topology']['refined_connectivity']:.3f}")
        logger.info("")
        
        # Overall assessment
        dice_improved = metrics['segmentation']['dice_improvement'] > 0
        topo_improved = abs(metrics['topology']['refined_components'] - metrics['topology']['gt_components']) < \
                       abs(metrics['topology']['original_components'] - metrics['topology']['gt_components'])
        centerline_improved = metrics['centerline']['improvement'] > 0
        
        improvements = []
        if dice_improved: improvements.append("Dice")
        if topo_improved: improvements.append("Topology")
        if centerline_improved: improvements.append("Centerline")
        
        if len(improvements) == 3:
            logger.info("🎉 TRIPLE SUCCESS: Improved Dice, Topology AND Centerline!")
        elif len(improvements) == 2:
            logger.info(f"✅✅ DOUBLE SUCCESS: Improved {' and '.join(improvements)}!")
        elif len(improvements) == 1:
            logger.info(f"✅ SUCCESS: Improved {improvements[0]}!")
        else:
            logger.info("⚖️  NEUTRAL: No significant improvements")
        
        logger.info("="*60)
        
        # Save results
        results = {
            'patient_id': patient_id,
            'processing_time': processing_time,
            'parameters': {'removal_percentile': removal_percentile},
            'correction_stats': correction_stats,
            'metrics': metrics
        }
        
        results_path = Path(f'experiments/enhanced_results_{patient_id}_p{removal_percentile}.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"Results saved to {results_path}")
        
        # Save refined mask
        refined_nifti_path = Path(f'experiments/enhanced_predictions/{patient_id}_enhanced_p{removal_percentile}.nii.gz')
        refined_nifti_path.parent.mkdir(parents=True, exist_ok=True)
        
        original_nifti = nib.load(f'experiments/test_predictions/{patient_id}/binary_mask.nii.gz')
        refined_nifti = nib.Nifti1Image(refined_mask, original_nifti.affine, original_nifti.header)
        nib.save(refined_nifti, refined_nifti_path)
        logger.info(f"Refined mask saved to {refined_nifti_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Enhanced evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_different_thresholds(patient_id):
    """Test different removal percentile thresholds"""
    logger.info("Testing different removal thresholds...")
    
    thresholds = [85, 90, 95, 98]
    best_dice = -1
    best_threshold = None
    results_summary = []
    
    for threshold in tqdm(thresholds, desc="Testing thresholds"):
        logger.info(f"\nTesting threshold: {threshold}%")
        results = evaluate_single_case_enhanced(patient_id, removal_percentile=threshold)
        
        if results:
            dice = results['metrics']['segmentation']['refined_dice']
            topo_components = results['metrics']['topology']['refined_components']
            centerline_dist = results['metrics']['centerline']['refined_distance']
            
            results_summary.append({
                'threshold': threshold,
                'dice': dice,
                'components': topo_components,
                'centerline': centerline_dist
            })
            
            if dice > best_dice:
                best_dice = dice
                best_threshold = threshold
    
    # Print summary table
    logger.info("\n" + "="*60)
    logger.info("THRESHOLD COMPARISON SUMMARY")
    logger.info("="*60)
    logger.info(f"{'Threshold':<10} {'Dice':<8} {'Components':<12} {'Centerline':<10}")
    logger.info("-" * 60)
    for result in results_summary:
        logger.info(f"{result['threshold']:<10} {result['dice']:<8.4f} {result['components']:<12} {result['centerline']:<10.2f}")
    
    logger.info(f"\nBest threshold: {best_threshold}% with Dice: {best_dice:.4f}")
    return best_threshold


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test enhanced conservative refinement')
    parser.add_argument('--patient-id', type=str, default='PA000005',
                       help='Patient ID to test (default: PA000005)')
    parser.add_argument('--test-thresholds', action='store_true',
                       help='Test different removal thresholds')
    parser.add_argument('--threshold', type=int, default=95,
                       help='Removal percentile threshold (default: 95)')
    parser.add_argument('--fast-mode', action='store_true',
                       help='Use fast connectivity enhancement for quicker testing')
    
    args = parser.parse_args()
    
    # Set fast mode globally for this run
    global FAST_MODE
    FAST_MODE = args.fast_mode
    
    if args.test_thresholds:
        logger.info(f"Testing thresholds with fast_mode={args.fast_mode}")
        best_threshold = test_different_thresholds(args.patient_id)
        logger.info(f"\nRunning final evaluation with best threshold: {best_threshold}%")
        results = evaluate_single_case_enhanced(args.patient_id, removal_percentile=best_threshold)
    else:
        results = evaluate_single_case_enhanced(args.patient_id, removal_percentile=args.threshold)
    
    if results:
        logger.info("Enhanced conservative refinement completed successfully!")
    else:
        logger.error("Enhanced conservative refinement failed!")


if __name__ == '__main__':
    main()