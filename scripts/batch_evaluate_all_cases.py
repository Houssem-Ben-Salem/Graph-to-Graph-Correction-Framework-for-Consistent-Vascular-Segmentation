#!/usr/bin/env python3
"""
Batch evaluation of enhanced conservative refinement on all available cases
Test with 90% threshold on the entire dataset
"""

import os
import sys
sys.path.append('.')

import torch
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from pathlib import Path
import logging
from tqdm import tqdm
import pickle
import argparse
from skimage.morphology import remove_small_objects, ball
from scipy.ndimage import binary_dilation
from skimage.measure import label
import pandas as pd
import matplotlib.pyplot as plt

from src.models.graph_correction.regression_model import GraphCorrectionRegressionModel
from src.models.graph_extraction.graph_extractor import GraphExtractor
from src.utils.graph_correspondence import GraphCorrespondenceMatcher
from src.utils.metrics import compute_dice_score, compute_hausdorff_distance

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def count_connected_components(mask):
    """Count connected components in a binary mask"""
    if np.sum(mask > 0) == 0:
        return 0
    labeled_mask = label(mask > 0)
    return labeled_mask.max()

def enhanced_conservative_refinement(original_mask, correction_data, removal_percentile=90):
    """Enhanced conservative refinement - streamlined version"""
    
    node_magnitudes = correction_data['node_magnitudes']
    
    if len(node_magnitudes) == 0:
        logger.warning("No correction data available")
        return original_mask.copy()
    
    # Compute removal threshold
    removal_threshold = np.percentile(node_magnitudes, removal_percentile)
    
    # Start with original mask
    refined_mask = original_mask.copy()
    
    # Identify regions to remove
    remove_positions = []
    for pos, magnitude in zip(correction_data['node_positions'], node_magnitudes):
        if magnitude >= removal_threshold:
            remove_positions.append(pos)
    
    # Fast vectorized removal
    for pos in remove_positions:
        x, y, z = map(int, pos)
        x = max(1, min(x, refined_mask.shape[0] - 2))
        y = max(1, min(y, refined_mask.shape[1] - 2))
        z = max(1, min(z, refined_mask.shape[2] - 2))
        refined_mask[x-1:x+2, y-1:y+2, z-1:z+2] = 0
    
    # Light connectivity enhancement
    refined_mask = remove_small_objects(refined_mask.astype(bool), min_size=50).astype(np.uint8)
    refined_mask = binary_dilation(refined_mask, structure=ball(1)).astype(np.uint8)
    
    return refined_mask

def find_all_cases(data_dir):
    """Find all available patient cases"""
    data_path = Path(data_dir)
    cases = []
    
    for case_dir in data_path.iterdir():
        if case_dir.is_dir() and case_dir.name.startswith('PA'):
            # Check if required files exist
            pred_mask_path = case_dir / "image" / f"{case_dir.name}_pred_mask.nii.gz"
            gt_mask_path = case_dir / "label" / f"{case_dir.name}_label.nii.gz"
            
            if pred_mask_path.exists() and gt_mask_path.exists():
                cases.append(case_dir.name)
    
    return sorted(cases)

def evaluate_single_case(case_id, model, extractor, matcher, data_dir, removal_percentile=90):
    """Evaluate a single case"""
    
    logger.info(f"Processing {case_id}...")
    
    try:
        # Paths
        case_dir = Path(data_dir) / case_id
        pred_mask_path = case_dir / "image" / f"{case_id}_pred_mask.nii.gz"
        gt_mask_path = case_dir / "label" / f"{case_id}_label.nii.gz"
        
        # Load masks
        pred_mask = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_mask_path)))
        gt_mask = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_mask_path)))
        
        # Extract graphs
        pred_graph = extractor.extract_graph(pred_mask)
        gt_graph = extractor.extract_graph(gt_mask)
        
        # Find correspondences
        correspondences = matcher.find_correspondences(pred_graph, gt_graph)
        
        # Compute corrections
        gt_positions = {node['id']: np.array(node['position']) for node in gt_graph.nodes}
        
        correction_magnitudes = []
        correction_positions = []
        
        for node in pred_graph.nodes:
            pred_pos = np.array(node['position'])
            node_id = node['id']
            
            if node_id in correspondences.node_correspondences:
                gt_id = correspondences.node_correspondences[node_id]
                gt_pos = gt_positions[gt_id]
                magnitude = np.linalg.norm(gt_pos - pred_pos)
            else:
                magnitude = 5.0  # High magnitude for unmatched nodes
            
            correction_magnitudes.append(magnitude)
            correction_positions.append(pred_pos)
        
        correction_data = {
            'node_magnitudes': np.array(correction_magnitudes),
            'node_positions': np.array(correction_positions)
        }
        
        # Apply refinement
        refined_mask = enhanced_conservative_refinement(
            pred_mask, correction_data, removal_percentile
        )
        
        # Compute metrics
        original_dice = compute_dice_score(pred_mask, gt_mask)
        refined_dice = compute_dice_score(refined_mask, gt_mask)
        
        try:
            original_hausdorff = compute_hausdorff_distance(pred_mask, gt_mask)
            refined_hausdorff = compute_hausdorff_distance(refined_mask, gt_mask)
        except:
            original_hausdorff = refined_hausdorff = 100.0  # Default for failed calculation
        
        original_components = count_connected_components(pred_mask)
        refined_components = count_connected_components(refined_mask)
        gt_components = count_connected_components(gt_mask)
        
        # Calculate volumes
        original_volume = np.sum(pred_mask > 0)
        refined_volume = np.sum(refined_mask > 0)
        gt_volume = np.sum(gt_mask > 0)
        
        results = {
            'case_id': case_id,
            'original_dice': original_dice,
            'refined_dice': refined_dice,
            'dice_improvement': refined_dice - original_dice,
            'original_hausdorff': original_hausdorff,
            'refined_hausdorff': refined_hausdorff,
            'hausdorff_improvement': original_hausdorff - refined_hausdorff,
            'original_components': original_components,
            'refined_components': refined_components,
            'gt_components': gt_components,
            'topology_improvement': original_components - refined_components,
            'original_volume': original_volume,
            'refined_volume': refined_volume,
            'gt_volume': gt_volume,
            'volume_change_percent': (refined_volume - original_volume) / original_volume * 100 if original_volume > 0 else 0,
            'num_pred_nodes': len(pred_graph.nodes),
            'num_gt_nodes': len(gt_graph.nodes),
            'num_correspondences': len(correspondences.node_correspondences),
            'mean_correction_magnitude': np.mean(correction_magnitudes),
            'removal_threshold': np.percentile(correction_magnitudes, removal_percentile),
            'nodes_removed': np.sum(np.array(correction_magnitudes) >= np.percentile(correction_magnitudes, removal_percentile)),
            'processing_status': 'success'
        }
        
        logger.info(f"{case_id}: Dice {original_dice:.3f}→{refined_dice:.3f} ({refined_dice-original_dice:+.3f}), "
                   f"Components {original_components}→{refined_components} ({original_components-refined_components:+d})")
        
        return results
        
    except Exception as e:
        logger.error(f"Error processing {case_id}: {str(e)}")
        return {
            'case_id': case_id,
            'processing_status': 'error',
            'error_message': str(e)
        }

def batch_evaluate_all_cases(data_dir, model_path, removal_percentile=90, output_dir='experiments/batch_evaluation'):
    """Batch evaluate all cases"""
    
    logger.info("="*80)
    logger.info("BATCH EVALUATION OF ENHANCED CONSERVATIVE REFINEMENT")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Removal percentile: {removal_percentile}%")
    logger.info("="*80)
    
    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all cases
    cases = find_all_cases(data_dir)
    logger.info(f"Found {len(cases)} cases: {cases}")
    
    if len(cases) == 0:
        logger.error("No valid cases found!")
        return
    
    # Load model
    logger.info("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = {
        'hidden_dim': 128,
        'num_heads': 8,
        'num_layers': 4,
        'dropout': 0.1
    }
    
    model = GraphCorrectionRegressionModel(config).to(device)
    
    if Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        logger.warning(f"Model not found: {model_path}")
    
    model.eval()
    
    # Initialize extractors
    extractor = GraphExtractor()
    matcher = GraphCorrespondenceMatcher()
    
    # Process all cases
    all_results = []
    
    for case_id in tqdm(cases, desc="Processing cases"):
        result = evaluate_single_case(
            case_id, model, extractor, matcher, data_dir, removal_percentile
        )
        all_results.append(result)
    
    # Filter successful results
    successful_results = [r for r in all_results if r.get('processing_status') == 'success']
    failed_results = [r for r in all_results if r.get('processing_status') != 'success']
    
    logger.info(f"\nProcessing completed: {len(successful_results)} successful, {len(failed_results)} failed")
    
    if failed_results:
        logger.warning("Failed cases:")
        for result in failed_results:
            logger.warning(f"  {result['case_id']}: {result.get('error_message', 'Unknown error')}")
    
    if len(successful_results) == 0:
        logger.error("No successful results to analyze!")
        return
    
    # Create results DataFrame
    df = pd.DataFrame(successful_results)
    
    # Summary statistics
    logger.info("\n" + "="*80)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*80)
    
    # Overall statistics
    dice_improvements = df['dice_improvement']
    hausdorff_improvements = df['hausdorff_improvement']
    topology_improvements = df['topology_improvement']
    
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
    logger.info(f"  Mean improvement: {hausdorff_improvements.mean():+.2f} ± {hausdorff_improvements.std():.2f}mm")
    logger.info(f"  Median improvement: {hausdorff_improvements.median():+.2f}mm")
    
    hausdorff_improved = (hausdorff_improvements > 0).sum()
    hausdorff_degraded = (hausdorff_improvements < 0).sum()
    
    logger.info(f"  Cases improved: {hausdorff_improved}/{len(successful_results)} ({hausdorff_improved/len(successful_results)*100:.1f}%)")
    logger.info(f"  Cases degraded: {hausdorff_degraded}/{len(successful_results)} ({hausdorff_degraded/len(successful_results)*100:.1f}%)")
    logger.info("")
    
    # Topology analysis
    logger.info("TOPOLOGY IMPROVEMENTS:")
    logger.info(f"  Mean component reduction: {topology_improvements.mean():+.1f} ± {topology_improvements.std():.1f}")
    logger.info(f"  Median component reduction: {topology_improvements.median():+.1f}")
    
    topology_improved = (topology_improvements > 0).sum()
    topology_degraded = (topology_improvements < 0).sum()
    
    logger.info(f"  Cases improved: {topology_improved}/{len(successful_results)} ({topology_improved/len(successful_results)*100:.1f}%)")
    logger.info(f"  Cases degraded: {topology_degraded}/{len(successful_results)} ({topology_degraded/len(successful_results)*100:.1f}%)")
    logger.info("")
    
    # Combined success metrics
    both_improved = ((dice_improvements > 0) & (topology_improvements > 0)).sum()
    dice_improved_topo_stable = ((dice_improvements > 0) & (topology_improvements >= 0)).sum()
    any_improved = ((dice_improvements > 0) | (topology_improvements > 0)).sum()
    
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
    
    # Create summary plots
    logger.info("Creating summary plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Dice improvement histogram
    axes[0, 0].hist(dice_improvements, bins=20, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(x=0, color='red', linestyle='--', label='No change')
    axes[0, 0].axvline(x=dice_improvements.mean(), color='green', linestyle='-', label=f'Mean: {dice_improvements.mean():+.4f}')
    axes[0, 0].set_title('Dice Score Improvements')
    axes[0, 0].set_xlabel('Dice Score Change')
    axes[0, 0].set_ylabel('Number of Cases')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Topology improvement histogram
    axes[0, 1].hist(topology_improvements, bins=20, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(x=0, color='red', linestyle='--', label='No change')
    axes[0, 1].axvline(x=topology_improvements.mean(), color='green', linestyle='-', label=f'Mean: {topology_improvements.mean():+.1f}')
    axes[0, 1].set_title('Topology Improvements (Component Reduction)')
    axes[0, 1].set_xlabel('Component Reduction')
    axes[0, 1].set_ylabel('Number of Cases')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Dice vs Topology scatter
    colors = ['red' if d <= 0 and t <= 0 else 'orange' if d <= 0 or t <= 0 else 'green' 
              for d, t in zip(dice_improvements, topology_improvements)]
    
    axes[1, 0].scatter(dice_improvements, topology_improvements, c=colors, alpha=0.6)
    axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1, 0].axvline(x=0, color='black', linestyle='-', alpha=0.3)
    axes[1, 0].set_title('Dice vs Topology Improvements')
    axes[1, 0].set_xlabel('Dice Score Change')
    axes[1, 0].set_ylabel('Component Reduction')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Success rate pie chart
    success_categories = ['Both Improved', 'Dice Only', 'Topology Only', 'No Improvement']
    both_improved_count = both_improved
    dice_only = dice_improved - both_improved
    topo_only = topology_improved - both_improved
    no_improvement = len(successful_results) - any_improved
    
    success_counts = [both_improved_count, dice_only, topo_only, no_improvement]
    success_colors = ['green', 'lightgreen', 'orange', 'lightcoral']
    
    axes[1, 1].pie(success_counts, labels=success_categories, colors=success_colors, 
                   autopct='%1.1f%%', startangle=90)
    axes[1, 1].set_title('Improvement Categories')
    
    plt.tight_layout()
    plot_file = output_path / f'batch_summary_p{removal_percentile}_{len(successful_results)}cases.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Summary plots saved to: {plot_file}")
    
    # Top and bottom performers
    logger.info("\nTOP 5 DICE IMPROVEMENTS:")
    top_dice = df.nlargest(5, 'dice_improvement')[['case_id', 'dice_improvement', 'topology_improvement']]
    for _, row in top_dice.iterrows():
        logger.info(f"  {row['case_id']}: Dice +{row['dice_improvement']:.4f}, Topology {row['topology_improvement']:+.0f}")
    
    logger.info("\nTOP 5 TOPOLOGY IMPROVEMENTS:")
    top_topo = df.nlargest(5, 'topology_improvement')[['case_id', 'dice_improvement', 'topology_improvement']]
    for _, row in top_topo.iterrows():
        logger.info(f"  {row['case_id']}: Dice {row['dice_improvement']:+.4f}, Topology +{row['topology_improvement']:.0f}")
    
    logger.info("\n" + "="*80)
    logger.info("BATCH EVALUATION COMPLETED SUCCESSFULLY!")
    logger.info("="*80)
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Batch evaluate enhanced conservative refinement')
    parser.add_argument('--data-dir', type=str, default='DATASET/Parse_dataset',
                       help='Directory containing patient data')
    parser.add_argument('--model', type=str, default='experiments/regression_model/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--percentile', type=int, default=90,
                       help='Removal percentile threshold')
    parser.add_argument('--output-dir', type=str, default='experiments/batch_evaluation',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    results_df = batch_evaluate_all_cases(
        data_dir=args.data_dir,
        model_path=args.model,
        removal_percentile=args.percentile,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()