#!/usr/bin/env python3
"""
Evaluation script for traditional post-processing baseline methods.

This script compares traditional post-processing methods against the graph-to-graph
correction framework on pulmonary artery segmentation tasks.
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from typing import Dict, List, Tuple, Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.traditional_postprocessing import (
    MorphologicalPostProcessor,
    LargestConnectedComponentFilter,
    CombinedTraditionalPostProcessor
)
from utils.metrics import compute_metrics


class TraditionalBaselineEvaluator:
    """Evaluator for traditional post-processing baseline methods."""
    
    def __init__(
        self,
        predictions_dir: str,
        ground_truth_dir: str,
        output_dir: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize evaluator.
        
        Args:
            predictions_dir: Directory containing U-Net prediction masks
            ground_truth_dir: Directory containing ground truth masks
            output_dir: Directory to save evaluation results
            device: Device for computations
        """
        self.predictions_dir = Path(predictions_dir)
        self.ground_truth_dir = Path(ground_truth_dir)
        self.output_dir = Path(output_dir)
        self.device = device
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize post-processors with different parameter settings
        self.processors = self._initialize_processors()
    
    def _initialize_processors(self) -> Dict[str, object]:
        """Initialize different traditional post-processing configurations."""
        
        processors = {}
        
        # 1. Morphological post-processing variants
        processors['morph_conservative'] = MorphologicalPostProcessor(
            opening_radius=1,
            closing_radius=2,
            min_object_size=50,
            min_hole_size=25
        )
        
        processors['morph_moderate'] = MorphologicalPostProcessor(
            opening_radius=2,
            closing_radius=3,
            min_object_size=100,
            min_hole_size=50
        )
        
        processors['morph_aggressive'] = MorphologicalPostProcessor(
            opening_radius=3,
            closing_radius=4,
            min_object_size=200,
            min_hole_size=100
        )
        
        # 2. LCC filtering variants
        processors['lcc_single'] = LargestConnectedComponentFilter(
            min_size_ratio=0.0
        )
        
        processors['lcc_multi'] = LargestConnectedComponentFilter(
            min_size_ratio=0.1  # Keep components at least 10% of largest
        )
        
        # 3. Combined approaches
        processors['morph_then_lcc'] = CombinedTraditionalPostProcessor(
            morphological_params={
                'opening_radius': 2,
                'closing_radius': 3,
                'min_object_size': 100,
                'min_hole_size': 50
            },
            lcc_params={'min_size_ratio': 0.05},
            apply_order='morph_then_lcc'
        )
        
        processors['lcc_then_morph'] = CombinedTraditionalPostProcessor(
            morphological_params={
                'opening_radius': 2,
                'closing_radius': 3,
                'min_object_size': 50,
                'min_hole_size': 25
            },
            lcc_params={'min_size_ratio': 0.0},
            apply_order='lcc_then_morph'
        )
        
        return processors
    
    def load_mask(self, mask_path: Path) -> np.ndarray:
        """Load a NIfTI mask file."""
        try:
            nii_img = nib.load(str(mask_path))
            mask = nii_img.get_fdata()
            return (mask > 0.5).astype(np.uint8)
        except Exception as e:
            print(f"Error loading mask {mask_path}: {e}")
            return None
    
    def find_mask_pairs(self) -> List[Tuple[Path, Path]]:
        """Find matching prediction and ground truth mask pairs."""
        pairs = []
        
        # Look for prediction files
        pred_files = list(self.predictions_dir.glob('**/*.nii.gz'))
        
        for pred_file in pred_files:
            # Try to find corresponding ground truth file
            # Assuming similar naming convention
            relative_path = pred_file.relative_to(self.predictions_dir)
            gt_file = self.ground_truth_dir / relative_path
            
            if not gt_file.exists():
                # Try alternative naming patterns
                alt_patterns = [
                    self.ground_truth_dir / relative_path.name,
                    self.ground_truth_dir / relative_path.stem.replace('_pred', '') + '.nii.gz',
                    self.ground_truth_dir / relative_path.stem.replace('prediction', 'label') + '.nii.gz'
                ]
                
                for alt_gt in alt_patterns:
                    if alt_gt.exists():
                        gt_file = alt_gt
                        break
            
            if gt_file.exists():
                pairs.append((pred_file, gt_file))
            else:
                print(f"Warning: No ground truth found for {pred_file}")
        
        print(f"Found {len(pairs)} prediction-ground truth pairs")
        return pairs
    
    def evaluate_single_case(
        self, 
        pred_mask: np.ndarray, 
        gt_mask: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate all post-processing methods on a single case."""
        
        results = {}
        
        # Baseline: Original U-Net prediction
        original_metrics = compute_metrics(pred_mask, gt_mask)
        results['original'] = original_metrics
        
        # Apply each post-processing method
        for method_name, processor in self.processors.items():
            try:
                processed_mask = processor.process(pred_mask)
                processed_metrics = compute_metrics(processed_mask, gt_mask)
                results[method_name] = processed_metrics
                
                # Add improvement metrics
                dice_improvement = processed_metrics['dice'] - original_metrics['dice']
                results[method_name]['dice_improvement'] = dice_improvement
                
            except Exception as e:
                print(f"Error processing with {method_name}: {e}")
                results[method_name] = None
        
        return results
    
    def evaluate_all(self) -> Dict[str, Dict[str, float]]:
        """Evaluate all traditional methods on the entire dataset."""
        
        mask_pairs = self.find_mask_pairs()
        if not mask_pairs:
            raise ValueError("No valid mask pairs found")
        
        # Storage for all results
        all_results = {method: [] for method in ['original'] + list(self.processors.keys())}
        case_names = []
        
        print("Evaluating traditional post-processing methods...")
        
        for pred_path, gt_path in tqdm(mask_pairs, desc="Processing cases"):
            # Load masks
            pred_mask = self.load_mask(pred_path)
            gt_mask = self.load_mask(gt_path)
            
            if pred_mask is None or gt_mask is None:
                continue
            
            case_name = pred_path.stem
            case_names.append(case_name)
            
            # Evaluate this case
            case_results = self.evaluate_single_case(pred_mask, gt_mask)
            
            # Store results
            for method_name, metrics in case_results.items():
                if metrics is not None:
                    all_results[method_name].append(metrics)
                else:
                    # Add empty dict for failed cases
                    all_results[method_name].append({})
        
        # Compute summary statistics
        summary_results = self._compute_summary_statistics(all_results)
        
        # Save detailed results
        self._save_detailed_results(all_results, case_names)
        
        # Save summary results
        self._save_summary_results(summary_results)
        
        return summary_results
    
    def _compute_summary_statistics(self, all_results: Dict) -> Dict[str, Dict[str, float]]:
        """Compute mean and std for each method across all cases."""
        
        summary = {}
        
        for method_name, method_results in all_results.items():
            if not method_results:
                continue
            
            # Get all metric names from non-empty results
            all_metrics = set()
            for result in method_results:
                if result:
                    all_metrics.update(result.keys())
            
            method_summary = {}
            
            for metric_name in all_metrics:
                values = []
                for result in method_results:
                    if result and metric_name in result:
                        values.append(result[metric_name])
                
                if values:
                    method_summary[f'{metric_name}_mean'] = np.mean(values)
                    method_summary[f'{metric_name}_std'] = np.std(values)
                    method_summary[f'{metric_name}_median'] = np.median(values)
                    method_summary[f'{metric_name}_count'] = len(values)
            
            summary[method_name] = method_summary
        
        return summary
    
    def _save_detailed_results(self, all_results: Dict, case_names: List[str]):
        """Save detailed per-case results to CSV."""
        
        detailed_file = self.output_dir / 'detailed_results.csv'
        
        # Prepare data for CSV
        rows = []
        
        for i, case_name in enumerate(case_names):
            base_row = {'case': case_name}
            
            for method_name, method_results in all_results.items():
                if i < len(method_results) and method_results[i]:
                    for metric_name, metric_value in method_results[i].items():
                        column_name = f'{method_name}_{metric_name}'
                        base_row[column_name] = metric_value
            
            rows.append(base_row)
        
        # Create DataFrame and save
        df = pd.DataFrame(rows)
        df.to_csv(detailed_file, index=False)
        print(f"Detailed results saved to {detailed_file}")
    
    def _save_summary_results(self, summary_results: Dict):
        """Save summary statistics to JSON and CSV."""
        
        # Save as JSON
        json_file = self.output_dir / 'summary_results.json'
        with open(json_file, 'w') as f:
            json.dump(summary_results, f, indent=2, default=float)
        
        # Save as CSV for easy viewing
        csv_file = self.output_dir / 'summary_results.csv'
        
        # Convert to DataFrame format
        rows = []
        for method_name, metrics in summary_results.items():
            row = {'method': method_name}
            row.update(metrics)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(csv_file, index=False)
        
        print(f"Summary results saved to {json_file} and {csv_file}")
    
    def print_comparison_table(self, summary_results: Dict):
        """Print a formatted comparison table of key metrics."""
        
        print("\n" + "="*80)
        print("TRADITIONAL POST-PROCESSING BASELINE COMPARISON")
        print("="*80)
        
        # Key metrics to display
        key_metrics = ['dice_mean', 'jaccard_mean', 'hausdorff_mean', 'sensitivity_mean']
        
        print(f"{'Method':<20}", end="")
        for metric in key_metrics:
            print(f"{metric.replace('_mean', '').upper():<12}", end="")
        print()
        print("-" * 80)
        
        for method_name, metrics in summary_results.items():
            print(f"{method_name:<20}", end="")
            for metric in key_metrics:
                if metric in metrics:
                    value = metrics[metric]
                    if 'dice' in metric or 'jaccard' in metric or 'sensitivity' in metric:
                        print(f"{value:.3f}      ", end="")
                    else:
                        print(f"{value:.2f}       ", end="")
                else:
                    print(f"{'N/A':<12}", end="")
            print()


def main():
    parser = argparse.ArgumentParser(description='Evaluate traditional post-processing baselines')
    parser.add_argument('--predictions-dir', required=True, 
                       help='Directory containing U-Net prediction masks')
    parser.add_argument('--ground-truth-dir', required=True,
                       help='Directory containing ground truth masks')
    parser.add_argument('--output-dir', required=True,
                       help='Directory to save evaluation results')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device for computations')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = TraditionalBaselineEvaluator(
        predictions_dir=args.predictions_dir,
        ground_truth_dir=args.ground_truth_dir,
        output_dir=args.output_dir,
        device=args.device
    )
    
    # Run evaluation
    try:
        summary_results = evaluator.evaluate_all()
        
        # Print comparison table
        evaluator.print_comparison_table(summary_results)
        
        print(f"\nEvaluation complete! Results saved in {args.output_dir}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())