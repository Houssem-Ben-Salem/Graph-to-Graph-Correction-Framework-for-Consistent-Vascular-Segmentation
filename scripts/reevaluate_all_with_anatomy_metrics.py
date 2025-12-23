#!/usr/bin/env python3
"""
Re-evaluate ALL experiments with comprehensive anatomy metrics.
This addresses the critical gap identified by reviewers.

Priority 1 Metrics (MUST HAVE):
- Murray's Law Compliance
- Tapering Consistency
- Branching Angle Statistics

Priority 2 Metrics (STRONGLY RECOMMENDED):
- Radius Estimation Error
- Betti Numbers (β₀, β₁)

Priority 3 Metrics (Nice to Have):
- Endpoint Detection
- Centerline Accuracy
"""

import sys
sys.path.append('.')

import torch
import numpy as np
from pathlib import Path
import nibabel as nib
import logging
import pandas as pd
import json
from tqdm import tqdm
from typing import Dict, List
import warnings

from src.utils.comprehensive_metrics import (
    compute_comprehensive_metrics,
    TopologicalMetrics,
    AnatomicalMetrics
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test cases (from results_impact_table.md)
TEST_CASES = [
    'PA000005', 'PA000016', 'PA000024', 'PA000026', 'PA000027', 'PA000036',
    'PA000038', 'PA000042', 'PA000046', 'PA000047', 'PA000053', 'PA000056',
    'PA000060', 'PA000063', 'PA000070', 'PA000073', 'PA000074', 'PA000078'
]


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def load_nifti_mask(path: Path) -> np.ndarray:
    """Load NIfTI mask and return as binary numpy array"""
    nii = nib.load(str(path))
    mask = nii.get_fdata()
    return (mask > 0.5).astype(np.uint8)


def compute_betti_numbers(mask: np.ndarray) -> Dict[str, int]:
    """
    Compute Betti numbers for topological characterization.

    β₀: Number of connected components
    β₁: Number of loops/cycles
    β₂: Number of voids (typically 0 for vessels)

    Returns:
        Dictionary with Betti numbers
    """
    from scipy import ndimage

    # β₀: Connected components
    labeled, beta_0 = ndimage.label(mask > 0.5)

    # β₁ and β₂ require more sophisticated computation
    # For vessels, β₁ represents loops (anastomoses)
    # Approximation: use Euler characteristic
    # χ = β₀ - β₁ + β₂

    # For vessel trees without voids: β₂ ≈ 0
    # We can estimate β₁ from the skeleton graph structure
    try:
        from skimage.morphology import skeletonize_3d
        skeleton = skeletonize_3d(mask > 0.5)

        # Create graph from skeleton
        topo_metrics = TopologicalMetrics()
        graph = topo_metrics.skeleton_to_graph(skeleton)

        # β₁ ≈ number of independent cycles in the graph
        # For trees: β₁ = 0
        # For graphs with loops: β₁ = E - V + C (where C is # of components)
        num_edges = len(graph.edges)
        num_nodes = len(graph.nodes)
        num_components = len(list(nx.connected_components(graph)))

        # Euler characteristic for graph: V - E + C
        # β₁ = E - V + C
        beta_1 = max(0, num_edges - num_nodes + num_components)

    except Exception as e:
        warnings.warn(f"Error computing β₁: {e}")
        beta_1 = 0

    return {
        'betti_0': int(beta_0),  # Connected components
        'betti_1': int(beta_1),  # Loops/cycles
        'betti_2': 0  # Voids (typically 0 for vessels)
    }


def compute_endpoint_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray, tolerance: float = 3.0) -> Dict[str, float]:
    """
    Compute endpoint detection metrics.

    Endpoints are vessel terminations (degree 1 nodes in skeleton graph).

    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
        tolerance: Distance threshold for matching endpoints (voxels)

    Returns:
        Dictionary with endpoint precision, recall, F1
    """
    try:
        from skimage.morphology import skeletonize_3d

        # Extract skeletons
        pred_skeleton = skeletonize_3d(pred_mask > 0.5)
        gt_skeleton = skeletonize_3d(gt_mask > 0.5)

        # Convert to graphs
        topo_metrics = TopologicalMetrics()
        pred_graph = topo_metrics.skeleton_to_graph(pred_skeleton)
        gt_graph = topo_metrics.skeleton_to_graph(gt_skeleton)

        # Find endpoints (degree 1 nodes)
        pred_endpoints = [n for n, d in pred_graph.degree() if d == 1]
        gt_endpoints = [n for n, d in gt_graph.degree() if d == 1]

        if len(gt_endpoints) == 0:
            precision = 1.0 if len(pred_endpoints) == 0 else 0.0
            recall = 1.0
            f1 = 1.0
        elif len(pred_endpoints) == 0:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        else:
            # Get coordinates
            pred_coords = [pred_graph.nodes[n]['coords'] for n in pred_endpoints]
            gt_coords = [gt_graph.nodes[n]['coords'] for n in gt_endpoints]

            # Match endpoints
            matches = topo_metrics._match_points(pred_coords, gt_coords, tolerance)

            true_positives = len(matches)
            precision = true_positives / len(pred_endpoints)
            recall = true_positives / len(gt_endpoints)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'endpoint_precision': float(precision),
            'endpoint_recall': float(recall),
            'endpoint_f1': float(f1),
            'pred_endpoints': len(pred_endpoints),
            'gt_endpoints': len(gt_endpoints)
        }

    except Exception as e:
        warnings.warn(f"Error computing endpoint metrics: {e}")
        return {
            'endpoint_precision': 0.0,
            'endpoint_recall': 0.0,
            'endpoint_f1': 0.0,
            'pred_endpoints': 0,
            'gt_endpoints': 0
        }


def compute_centerline_accuracy(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    """
    Compute centerline accuracy metrics.

    Measures the geometric accuracy of the predicted vessel centerlines
    compared to ground truth centerlines.

    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask

    Returns:
        Dictionary with centerline distance metrics
    """
    try:
        from skimage.morphology import skeletonize_3d
        from scipy.spatial.distance import directed_hausdorff

        # Extract skeletons (centerlines)
        pred_skeleton = skeletonize_3d(pred_mask > 0.5)
        gt_skeleton = skeletonize_3d(gt_mask > 0.5)

        # Get centerline points
        pred_points = np.argwhere(pred_skeleton)
        gt_points = np.argwhere(gt_skeleton)

        if len(pred_points) == 0 or len(gt_points) == 0:
            return {
                'centerline_mean_distance': 0.0,
                'centerline_hausdorff': 0.0,
                'centerline_overlap': 0.0
            }

        # Sample points if too many
        if len(pred_points) > 5000:
            indices = np.random.choice(len(pred_points), 5000, replace=False)
            pred_points = pred_points[indices]
        if len(gt_points) > 5000:
            indices = np.random.choice(len(gt_points), 5000, replace=False)
            gt_points = gt_points[indices]

        # Compute directed Hausdorff distances
        hd_pred_to_gt = directed_hausdorff(pred_points, gt_points)[0]
        hd_gt_to_pred = directed_hausdorff(gt_points, pred_points)[0]

        # Mean distance (average of both directions)
        mean_distance = (hd_pred_to_gt + hd_gt_to_pred) / 2.0

        # Max Hausdorff
        hausdorff = max(hd_pred_to_gt, hd_gt_to_pred)

        # Centerline overlap (percentage of GT centerline within 2 voxels of pred)
        from scipy.spatial import cKDTree
        tree = cKDTree(pred_points)
        distances, _ = tree.query(gt_points)
        overlap = np.sum(distances <= 2.0) / len(gt_points)

        return {
            'centerline_mean_distance': float(mean_distance),
            'centerline_hausdorff': float(hausdorff),
            'centerline_overlap': float(overlap)
        }

    except Exception as e:
        warnings.warn(f"Error computing centerline accuracy: {e}")
        return {
            'centerline_mean_distance': 0.0,
            'centerline_hausdorff': 0.0,
            'centerline_overlap': 0.0
        }


def evaluate_case_with_full_metrics(pred_path: Path, gt_path: Path, case_id: str) -> Dict:
    """
    Evaluate a single case with ALL metrics including anatomy.

    Args:
        pred_path: Path to predicted mask
        gt_path: Path to ground truth mask
        case_id: Patient case ID

    Returns:
        Dictionary with all metrics
    """
    logger.info(f"Evaluating {case_id} with full metrics...")

    # Load masks
    pred_mask = load_nifti_mask(pred_path)
    gt_mask = load_nifti_mask(gt_path)

    # Compute comprehensive metrics (includes anatomy)
    metrics = compute_comprehensive_metrics(
        pred_mask=pred_mask,
        gt_mask=gt_mask,
        voxel_spacing=(1.0, 1.0, 1.0),
        include_anatomical=True,  # ENABLE ANATOMY METRICS
        include_topological=True
    )

    # Add Betti numbers (Priority 2)
    betti = compute_betti_numbers(pred_mask)
    metrics.update(betti)

    # Add ground truth Betti numbers for comparison
    gt_betti = compute_betti_numbers(gt_mask)
    metrics.update({f'gt_{k}': v for k, v in gt_betti.items()})

    # Add endpoint metrics (Priority 3)
    endpoint_metrics = compute_endpoint_metrics(pred_mask, gt_mask)
    metrics.update(endpoint_metrics)

    # Add centerline accuracy (Priority 3)
    centerline_metrics = compute_centerline_accuracy(pred_mask, gt_mask)
    metrics.update(centerline_metrics)

    # Add case ID
    metrics['case_id'] = case_id

    return convert_to_serializable(metrics)


def reevaluate_main_model():
    """
    Re-evaluate Main Model with anatomy metrics.

    Note: For Main Model, EXP-2, and EXP-3, we need to use the SAME
    approach as the batch evaluation - compute metrics on the fly
    from the graph correction process, since refined predictions
    weren't saved.

    For now, we'll compute metrics on the BASELINE (U-Net only)
    and EXP-1 (which saved predictions).

    TODO: Modify batch_evaluate_dataset.py to SAVE refined predictions
    """
    logger.info("=" * 80)
    logger.info("MAIN MODEL: USING BASELINE U-NET PREDICTIONS")
    logger.info("(Graph-corrected predictions were not saved)")
    logger.info("=" * 80)

    results = []

    for case_id in tqdm(TEST_CASES, desc="Main Model (Baseline)"):
        # Use baseline U-Net predictions (same as EXP-4)
        pred_path = Path(f'experiments/test_predictions/{case_id}_pred.nii.gz')
        gt_path = Path(f'DATASET/Parse_dataset/{case_id}/label/{case_id}.nii.gz')

        if not pred_path.exists():
            logger.warning(f"Skipping {case_id} - prediction not found")
            continue

        metrics = evaluate_case_with_full_metrics(pred_path, gt_path, case_id)
        metrics['note'] = 'Using baseline U-Net (graph-corrected predictions not saved)'
        results.append(metrics)

    # Save results
    output_dir = Path('experiments/main_model_anatomy_metrics')
    output_dir.mkdir(exist_ok=True, parents=True)

    df = pd.DataFrame(results)
    df.to_csv(output_dir / 'baseline_unet_with_anatomy.csv', index=False)

    with open(output_dir / 'baseline_unet_with_anatomy.json', 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved baseline anatomy metrics to {output_dir}")
    logger.info("WARNING: Need to modify batch evaluation to save refined predictions!")
    return results


def reevaluate_exp1_volumetric():
    """Re-evaluate EXP-1 (Volumetric) with anatomy metrics"""
    logger.info("=" * 80)
    logger.info("RE-EVALUATING EXP-1 (VOLUMETRIC) WITH ANATOMY METRICS")
    logger.info("=" * 80)

    results = []

    for case_id in tqdm(TEST_CASES, desc="EXP-1"):
        # Path to EXP-1 refined predictions
        pred_path = Path(f'experiments/ablation/exp1_volumetric/results/predictions/{case_id}_refined.nii.gz')
        gt_path = Path(f'DATASET/Parse_dataset/{case_id}/label/{case_id}.nii.gz')

        if not pred_path.exists():
            logger.warning(f"Skipping {case_id} - prediction not found")
            continue

        metrics = evaluate_case_with_full_metrics(pred_path, gt_path, case_id)
        results.append(metrics)

    # Save results
    output_dir = Path('experiments/ablation/exp1_volumetric/anatomy_metrics')
    output_dir.mkdir(exist_ok=True, parents=True)

    df = pd.DataFrame(results)
    df.to_csv(output_dir / 'exp1_with_anatomy.csv', index=False)

    with open(output_dir / 'exp1_with_anatomy.json', 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved EXP-1 anatomy metrics to {output_dir}")
    return results


def reevaluate_exp2_topology():
    """Re-evaluate EXP-2 (Topology-Only) with anatomy metrics"""
    logger.info("=" * 80)
    logger.info("RE-EVALUATING EXP-2 (TOPOLOGY-ONLY) WITH ANATOMY METRICS")
    logger.info("=" * 80)

    results = []

    for case_id in tqdm(TEST_CASES, desc="EXP-2"):
        # Path to EXP-2 refined predictions (enhanced conservative p90)
        pred_path = Path(f'experiments/ablation/exp2_topology_only/predictions_p90/{case_id}_enhanced_conservative_p90.nii.gz')
        gt_path = Path(f'DATASET/Parse_dataset/{case_id}/label/{case_id}.nii.gz')

        if not pred_path.exists():
            logger.warning(f"Skipping {case_id} - prediction not found")
            continue

        metrics = evaluate_case_with_full_metrics(pred_path, gt_path, case_id)
        results.append(metrics)

    # Save results
    output_dir = Path('experiments/ablation/exp2_topology_only/anatomy_metrics')
    output_dir.mkdir(exist_ok=True, parents=True)

    df = pd.DataFrame(results)
    df.to_csv(output_dir / 'exp2_with_anatomy.csv', index=False)

    with open(output_dir / 'exp2_with_anatomy.json', 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved EXP-2 anatomy metrics to {output_dir}")
    return results


def reevaluate_exp3_anatomy():
    """Re-evaluate EXP-3 (Anatomy-Only) with anatomy metrics"""
    logger.info("=" * 80)
    logger.info("RE-EVALUATING EXP-3 (ANATOMY-ONLY) WITH ANATOMY METRICS")
    logger.info("=" * 80)

    results = []

    for case_id in tqdm(TEST_CASES, desc="EXP-3"):
        # Path to EXP-3 refined predictions
        pred_path = Path(f'experiments/ablation/exp3_anatomy_only/predictions_p90/{case_id}_enhanced_conservative_p90.nii.gz')
        gt_path = Path(f'DATASET/Parse_dataset/{case_id}/label/{case_id}.nii.gz')

        if not pred_path.exists():
            logger.warning(f"Skipping {case_id} - prediction not found")
            continue

        metrics = evaluate_case_with_full_metrics(pred_path, gt_path, case_id)
        results.append(metrics)

    # Save results
    output_dir = Path('experiments/ablation/exp3_anatomy_only/anatomy_metrics')
    output_dir.mkdir(exist_ok=True, parents=True)

    df = pd.DataFrame(results)
    df.to_csv(output_dir / 'exp3_with_anatomy.csv', index=False)

    with open(output_dir / 'exp3_with_anatomy.json', 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved EXP-3 anatomy metrics to {output_dir}")
    return results


def reevaluate_exp4_baseline():
    """Re-evaluate EXP-4 (Baseline) with anatomy metrics"""
    logger.info("=" * 80)
    logger.info("RE-EVALUATING EXP-4 (BASELINE) WITH ANATOMY METRICS")
    logger.info("=" * 80)

    results = []

    for case_id in tqdm(TEST_CASES, desc="EXP-4"):
        # Path to original U-Net predictions (baseline)
        pred_path = Path(f'experiments/test_predictions/{case_id}_pred.nii.gz')
        gt_path = Path(f'DATASET/Parse_dataset/{case_id}/label/{case_id}.nii.gz')

        if not pred_path.exists():
            logger.warning(f"Skipping {case_id} - prediction not found")
            continue

        metrics = evaluate_case_with_full_metrics(pred_path, gt_path, case_id)
        results.append(metrics)

    # Save results
    output_dir = Path('experiments/ablation/exp4_no_correction/anatomy_metrics')
    output_dir.mkdir(exist_ok=True, parents=True)

    df = pd.DataFrame(results)
    df.to_csv(output_dir / 'exp4_with_anatomy.csv', index=False)

    with open(output_dir / 'exp4_with_anatomy.json', 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved EXP-4 anatomy metrics to {output_dir}")
    return results


def main():
    """Re-evaluate all experiments with comprehensive anatomy metrics"""

    logger.info("=" * 80)
    logger.info("COMPREHENSIVE RE-EVALUATION WITH ANATOMY METRICS")
    logger.info("=" * 80)
    logger.info("")
    logger.info("This addresses critical reviewer feedback:")
    logger.info("  ✓ Priority 1: Murray's Law Compliance")
    logger.info("  ✓ Priority 1: Tapering Consistency")
    logger.info("  ✓ Priority 1: Branching Angle Statistics")
    logger.info("  ✓ Priority 2: Betti Numbers (β₀, β₁)")
    logger.info("  ✓ Priority 3: Endpoint Detection")
    logger.info("  ✓ Priority 3: Centerline Accuracy")
    logger.info("")

    # Re-evaluate all experiments
    main_results = reevaluate_main_model()
    exp1_results = reevaluate_exp1_volumetric()
    exp2_results = reevaluate_exp2_topology()
    exp3_results = reevaluate_exp3_anatomy()
    exp4_results = reevaluate_exp4_baseline()

    logger.info("=" * 80)
    logger.info("RE-EVALUATION COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Main Model: {len(main_results)} cases")
    logger.info(f"EXP-1 (Volumetric): {len(exp1_results)} cases")
    logger.info(f"EXP-2 (Topology-Only): {len(exp2_results)} cases")
    logger.info(f"EXP-3 (Anatomy-Only): {len(exp3_results)} cases")
    logger.info(f"EXP-4 (Baseline): {len(exp4_results)} cases")

    return {
        'main': main_results,
        'exp1': exp1_results,
        'exp2': exp2_results,
        'exp3': exp3_results,
        'exp4': exp4_results
    }


if __name__ == '__main__':
    import networkx as nx  # Import here for Betti number computation
    results = main()
