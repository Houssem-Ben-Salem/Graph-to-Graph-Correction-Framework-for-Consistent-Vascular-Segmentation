#!/usr/bin/env python3
"""
Comprehensive metrics collection script for all methods comparison.

This script collects all metrics (volumetric, topological, anatomical) 
for traditional post-processing methods and future comparison with 
learning-based models and the graph-to-graph correction framework.
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import nibabel as nib
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Union
import warnings
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.comprehensive_metrics import compute_comprehensive_metrics
from utils.traditional_postprocessing import (
    MorphologicalPostProcessor,
    LargestConnectedComponentFilter,
    CombinedTraditionalPostProcessor
)


class ComprehensiveMetricsCollector:
    """Collect comprehensive metrics for all segmentation methods."""
    
    def __init__(
        self,
        predictions_dir: str,
        ground_truth_dir: str,
        output_dir: str,
        voxel_spacing: tuple = (1.0, 1.0, 1.0),
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize metrics collector.
        
        Args:
            predictions_dir: Directory containing baseline U-Net predictions
            ground_truth_dir: Directory containing ground truth masks
            output_dir: Directory to save comprehensive metrics
            voxel_spacing: Physical voxel spacing (z, y, x)
            device: Device for computations
        """
        self.predictions_dir = Path(predictions_dir)
        self.ground_truth_dir = Path(ground_truth_dir) 
        self.output_dir = Path(output_dir)
        self.voxel_spacing = voxel_spacing
        self.device = device
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize all method processors
        self.method_processors = self._initialize_all_methods()
        
        # Storage for all collected metrics
        self.all_metrics = {}
        
        print(f"Initialized metrics collector with {len(self.method_processors)} methods")
    
    def _initialize_all_methods(self) -> Dict[str, object]:
        """Initialize all segmentation methods to evaluate."""
        methods = {}
        
        # 1. Original U-Net predictions (no post-processing)
        methods['unet_original'] = None
        
        # 2. Traditional morphological methods
        methods['morph_conservative'] = MorphologicalPostProcessor(
            opening_radius=1,
            closing_radius=2,
            min_object_size=50,
            min_hole_size=25
        )
        
        methods['morph_moderate'] = MorphologicalPostProcessor(
            opening_radius=2,
            closing_radius=3,
            min_object_size=100,
            min_hole_size=50
        )
        
        methods['morph_aggressive'] = MorphologicalPostProcessor(
            opening_radius=3,
            closing_radius=4,
            min_object_size=200,
            min_hole_size=100
        )
        
        # 3. LCC filtering methods
        methods['lcc_single'] = LargestConnectedComponentFilter(
            min_size_ratio=0.0
        )
        
        methods['lcc_multi'] = LargestConnectedComponentFilter(
            min_size_ratio=0.1
        )
        
        # 4. Combined methods
        methods['morph_then_lcc_conservative'] = CombinedTraditionalPostProcessor(
            morphological_params={
                'opening_radius': 1,
                'closing_radius': 2,
                'min_object_size': 50,
                'min_hole_size': 25
            },
            lcc_params={'min_size_ratio': 0.05},
            apply_order='morph_then_lcc'
        )
        
        methods['morph_then_lcc_moderate'] = CombinedTraditionalPostProcessor(
            morphological_params={
                'opening_radius': 2,
                'closing_radius': 3,
                'min_object_size': 100,
                'min_hole_size': 50
            },
            lcc_params={'min_size_ratio': 0.05},
            apply_order='morph_then_lcc'
        )
        
        methods['lcc_then_morph'] = CombinedTraditionalPostProcessor(
            morphological_params={
                'opening_radius': 2,
                'closing_radius': 3,
                'min_object_size': 50,
                'min_hole_size': 25
            },
            lcc_params={'min_size_ratio': 0.0},
            apply_order='lcc_then_morph'
        )
        
        # Placeholders for future methods
        methods['unet_cldice'] = 'PLACEHOLDER_FOR_FUTURE'  # Will be implemented later
        methods['unetr'] = 'PLACEHOLDER_FOR_FUTURE'       # Will be implemented later
        methods['graph_correction'] = 'PLACEHOLDER_FOR_FUTURE'  # Your final method
        
        return methods
    
    def load_mask_pair(self, case_id: str) -> tuple:
        """
        Load prediction and ground truth mask pair for a case.
        
        Args:
            case_id: Case identifier (e.g., 'PA000005')
            
        Returns:
            Tuple of (prediction_mask, ground_truth_mask, success_flag)
        """
        # U-Net prediction file patterns:
        # Primary: experiments/test_predictions/PA000303_pred.nii.gz (binary predictions)
        # Alternative: experiments/test_predictions/PA000303/binary_mask.nii.gz
        pred_patterns = [
            self.predictions_dir / f"{case_id}_pred.nii.gz",  # Binary prediction file
            self.predictions_dir / case_id / "binary_mask.nii.gz",  # Binary mask inside folder
            self.predictions_dir / case_id / "prediction.nii.gz",   # Prediction inside folder
            self.predictions_dir / f"{case_id}_prob.nii.gz",  # Probability file (as fallback)
        ]
        
        pred_file = None
        for pattern in pred_patterns:
            if pattern.exists():
                pred_file = pattern
                break
        
        if pred_file is None:
            return None, None, False
        
        # Ground truth file pattern: DATASET/Parse_dataset/PA000005/label/PA000005.nii.gz
        gt_file = self.ground_truth_dir / case_id / "label" / f"{case_id}.nii.gz"
        
        if not gt_file.exists():
            # Try alternative patterns as fallback
            alt_gt_patterns = [
                self.ground_truth_dir / f"{case_id}.nii.gz",
                self.ground_truth_dir / f"{case_id}_gt.nii.gz",
                self.ground_truth_dir / case_id / f"{case_id}.nii.gz"
            ]
            
            for pattern in alt_gt_patterns:
                if pattern.exists():
                    gt_file = pattern
                    break
            else:
                return None, None, False
        
        try:
            # Load prediction
            pred_nii = nib.load(str(pred_file))
            pred_mask = pred_nii.get_fdata()
            pred_mask = (pred_mask > 0.5).astype(np.uint8)
            
            # Load ground truth
            gt_nii = nib.load(str(gt_file))
            gt_mask = gt_nii.get_fdata()
            gt_mask = (gt_mask > 0.5).astype(np.uint8)
            
            return pred_mask, gt_mask, True
            
        except Exception as e:
            print(f"Error loading masks for {case_id}: {e}")
            return None, None, False
    
    def find_all_cases(self) -> List[str]:
        """Find all available cases that have both prediction and ground truth."""
        
        # Get all prediction cases from experiments/test_predictions/
        # Method 1: Find all _pred.nii.gz files
        pred_files = list(self.predictions_dir.glob("*_pred.nii.gz"))
        pred_cases_from_files = []
        for f in pred_files:
            # Remove .nii.gz extension first, then remove _pred
            case_id = f.name.replace('.nii.gz', '').replace('_pred', '')
            pred_cases_from_files.append(case_id)
        
        # Method 2: Find all case directories
        case_dirs = [d.name for d in self.predictions_dir.iterdir() 
                    if d.is_dir() and d.name.startswith('PA')]
        
        # Combine both methods
        pred_cases = list(set(pred_cases_from_files + case_dirs))
        
        print(f"Found {len(pred_cases)} prediction files in {self.predictions_dir}")
        
        # Filter cases that also have ground truth
        available_cases = []
        for case_id in pred_cases:
            pred_mask, gt_mask, success = self.load_mask_pair(case_id)
            if success:
                available_cases.append(case_id)
            else:
                print(f"Skipping {case_id}: ground truth not found")
        
        print(f"Found {len(available_cases)} cases with both prediction and ground truth")
        return sorted(available_cases)
    
    def collect_metrics_for_case(
        self, 
        case_id: str,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
        method_progress_bar: Optional[tqdm] = None
    ) -> Dict[str, Dict[str, Union[float, int, List]]]:
        """
        Collect comprehensive metrics for all methods on a single case.
        
        Args:
            case_id: Case identifier
            pred_mask: Original prediction mask
            gt_mask: Ground truth mask
            method_progress_bar: Optional progress bar for method progress
            
        Returns:
            Dictionary of metrics for each method
        """
        case_results = {}
        
        # Get active methods (exclude placeholders)
        active_methods = {k: v for k, v in self.method_processors.items() 
                         if not (isinstance(v, str) and v == 'PLACEHOLDER_FOR_FUTURE')}
        
        for method_name, processor in active_methods.items():
            
            if method_progress_bar:
                method_progress_bar.set_description(f"Processing {case_id} - {method_name}")
            
            try:
                # Apply method-specific processing
                if method_name == 'unet_original':
                    # Original predictions without post-processing
                    processed_mask = pred_mask.copy()
                else:
                    # Apply post-processing
                    processed_mask = processor.process(pred_mask)
                
                # Compute comprehensive metrics
                metrics = compute_comprehensive_metrics(
                    processed_mask, 
                    gt_mask,
                    voxel_spacing=self.voxel_spacing,
                    include_anatomical=True,
                    include_topological=True
                )
                
                # Add method-specific information
                metrics['method'] = method_name
                metrics['case_id'] = case_id
                metrics['processing_time'] = 0.0  # Could add timing if needed
                
                case_results[method_name] = metrics
                
            except Exception as e:
                print(f"Error processing {case_id} with {method_name}: {e}")
                case_results[method_name] = {
                    'method': method_name,
                    'case_id': case_id,
                    'error': str(e)
                }
            
            if method_progress_bar:
                method_progress_bar.update(1)
        
        return case_results
    
    def collect_all_metrics(self, max_cases: Optional[int] = None) -> Dict:
        """
        Collect comprehensive metrics for all methods on all available cases.
        
        Args:
            max_cases: Maximum number of cases to process (for testing)
            
        Returns:
            Dictionary containing all collected metrics
        """
        print("🔍 Finding available cases...")
        cases = self.find_all_cases()
        
        if max_cases is not None:
            cases = cases[:max_cases]
            print(f"📊 Limiting to first {max_cases} cases for testing")
        
        if not cases:
            raise ValueError("No valid cases found")
        
        # Get active methods count for progress tracking
        active_methods = {k: v for k, v in self.method_processors.items() 
                         if not (isinstance(v, str) and v == 'PLACEHOLDER_FOR_FUTURE')}
        
        all_results = {method: [] for method in self.method_processors.keys()}
        failed_cases = []
        
        print(f"🚀 Starting comprehensive metrics collection...")
        print(f"📋 Cases to process: {len(cases)}")
        print(f"🔧 Methods to evaluate: {len(active_methods)}")
        print(f"📊 Total operations: {len(cases) * len(active_methods)}")
        
        # Main progress bar for cases
        case_progress = tqdm(
            total=len(cases), 
            desc="📁 Processing cases", 
            position=0,
            leave=True,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )
        
        # Secondary progress bar for methods within each case
        method_progress = tqdm(
            total=len(active_methods),
            desc="🔧 Processing methods",
            position=1,
            leave=False,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )
        
        try:
            for i, case_id in enumerate(cases):
                case_progress.set_description(f"📁 Processing case {i+1}/{len(cases)}: {case_id}")
                
                # Reset method progress bar for each case
                method_progress.reset()
                method_progress.set_description(f"🔧 Loading masks for {case_id}")
                
                # Load masks
                pred_mask, gt_mask, success = self.load_mask_pair(case_id)
                
                if not success:
                    failed_cases.append(case_id)
                    case_progress.write(f"❌ Failed to load masks for {case_id}")
                    case_progress.update(1)
                    continue
                
                method_progress.set_description(f"🔧 Computing metrics for {case_id}")
                
                # Collect metrics for all methods
                case_results = self.collect_metrics_for_case(
                    case_id, pred_mask, gt_mask, method_progress
                )
                
                # Store results
                success_count = 0
                error_count = 0
                for method_name, metrics in case_results.items():
                    if 'error' not in metrics:
                        all_results[method_name].append(metrics)
                        success_count += 1
                    else:
                        error_count += 1
                
                # Update case progress
                if error_count > 0:
                    case_progress.write(f"⚠️  {case_id}: {success_count} success, {error_count} errors")
                else:
                    case_progress.write(f"✅ {case_id}: All {success_count} methods completed successfully")
                
                case_progress.update(1)
        
        finally:
            method_progress.close()
            case_progress.close()
        
        if failed_cases:
            print(f"\n❌ Failed to process {len(failed_cases)} cases:")
            for case in failed_cases[:10]:  # Show first 10
                print(f"   - {case}")
            if len(failed_cases) > 10:
                print(f"   ... and {len(failed_cases) - 10} more")
        
        # Store results
        self.all_metrics = all_results
        
        print(f"\n💾 Saving results...")
        # Save results immediately
        self.save_all_results()
        
        successful_cases = len(cases) - len(failed_cases)
        print(f"✅ Metrics collection completed!")
        print(f"📊 Successfully processed: {successful_cases}/{len(cases)} cases")
        
        return all_results
    
    def save_all_results(self):
        """Save all collected metrics to various formats."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        save_progress = tqdm(total=3, desc="💾 Saving results", 
                           bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}')
        
        try:
            # 1. Save raw results as JSON
            save_progress.set_description("💾 Saving JSON results")
            json_file = self.output_dir / f'comprehensive_metrics_{timestamp}.json'
            
            # Convert numpy types for JSON serialization
            json_results = {}
            for method, cases in self.all_metrics.items():
                json_results[method] = []
                for case in cases:
                    json_case = {}
                    for key, value in case.items():
                        if isinstance(value, np.ndarray):
                            json_case[key] = value.tolist()
                        elif isinstance(value, (np.integer, np.floating)):
                            json_case[key] = float(value)
                        else:
                            json_case[key] = value
                    json_results[method].append(json_case)
            
            with open(json_file, 'w') as f:
                json.dump(json_results, f, indent=2, default=str)
            
            save_progress.write(f"📄 Raw results saved to {json_file}")
            save_progress.update(1)
            
            # 2. Save as comprehensive CSV
            save_progress.set_description("💾 Saving CSV results")
            self.save_metrics_csv(timestamp)
            save_progress.update(1)
            
            # 3. Save summary statistics
            save_progress.set_description("💾 Saving summary statistics")
            self.save_summary_statistics(timestamp)
            save_progress.update(1)
            
        finally:
            save_progress.close()
    
    def save_metrics_csv(self, timestamp: str):
        """Save all metrics in CSV format for easy analysis."""
        
        csv_file = self.output_dir / f'comprehensive_metrics_{timestamp}.csv'
        
        # Flatten all metrics into rows
        rows = []
        
        for method_name, cases in self.all_metrics.items():
            if isinstance(self.method_processors[method_name], str):
                continue  # Skip placeholders
                
            for case_data in cases:
                row = {
                    'method': method_name,
                    'case_id': case_data['case_id']
                }
                
                # Add all metrics
                for key, value in case_data.items():
                    if key not in ['method', 'case_id', 'branching_angles']:
                        if isinstance(value, list):
                            # For list metrics, store as string or compute statistics
                            if key == 'component_sizes' and value:
                                row[f'{key}_mean'] = np.mean(value)
                                row[f'{key}_max'] = np.max(value)
                                row[f'{key}_count'] = len(value)
                            else:
                                row[key] = str(value)
                        else:
                            row[key] = value
                
                rows.append(row)
        
        # Create DataFrame and save
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(csv_file, index=False)
            tqdm.write(f"📊 Comprehensive CSV saved to {csv_file}")
        else:
            tqdm.write("⚠️  No data to save to CSV")
        
    def save_summary_statistics(self, timestamp: str):
        """Save summary statistics for each method."""
        
        summary_file = self.output_dir / f'summary_statistics_{timestamp}.csv'
        
        # Define key metrics to summarize
        key_metrics = [
            'dice', 'jaccard', 'hausdorff_95', 'assd', 'sensitivity', 'precision',
            'connectivity_aware_dice', 'num_components', 'component_count_error',
            'tree_isomorphism', 'bifurcation_f1', 
            'murrays_law_score', 'tapering_score', 'angle_physiological_angle_ratio'
        ]
        
        summary_rows = []
        
        for method_name, cases in self.all_metrics.items():
            if isinstance(self.method_processors[method_name], str) or not cases:
                continue  # Skip placeholders or empty results
            
            row = {'method': method_name, 'num_cases': len(cases)}
            
            # Compute statistics for each metric
            for metric in key_metrics:
                values = []
                for case in cases:
                    if metric in case and case[metric] is not None:
                        try:
                            val = float(case[metric])
                            if not np.isnan(val) and not np.isinf(val):
                                values.append(val)
                        except (ValueError, TypeError):
                            pass
                
                if values:
                    row[f'{metric}_mean'] = np.mean(values)
                    row[f'{metric}_std'] = np.std(values)
                    row[f'{metric}_median'] = np.median(values)
                    row[f'{metric}_min'] = np.min(values)
                    row[f'{metric}_max'] = np.max(values)
                    row[f'{metric}_count'] = len(values)
                else:
                    for stat in ['mean', 'std', 'median', 'min', 'max', 'count']:
                        row[f'{metric}_{stat}'] = None
            
            summary_rows.append(row)
        
        # Create DataFrame and save
        if summary_rows:
            df = pd.DataFrame(summary_rows)
            df.to_csv(summary_file, index=False)
            tqdm.write(f"📈 Summary statistics saved to {summary_file}")
        else:
            tqdm.write("⚠️  No summary data to save")
    
    def print_preview_results(self):
        """Print a preview of the collected results."""
        
        if not self.all_metrics:
            print("No metrics collected yet")
            return
        
        print("\n" + "="*80)
        print("COMPREHENSIVE METRICS COLLECTION PREVIEW")
        print("="*80)
        
        for method_name, cases in self.all_metrics.items():
            if isinstance(self.method_processors[method_name], str) or not cases:
                continue
            
            print(f"\n{method_name.upper()}:")
            print("-" * 40)
            
            # Get key metrics
            key_metrics = ['dice', 'connectivity_aware_dice', 'num_components', 
                          'tree_isomorphism', 'murrays_law_score']
            
            for metric in key_metrics:
                values = [case.get(metric) for case in cases if case.get(metric) is not None]
                values = [v for v in values if not (np.isnan(v) if isinstance(v, (int, float)) else False)]
                
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    print(f"  {metric}: {mean_val:.3f} ± {std_val:.3f} (n={len(values)})")
        
        print(f"\nTotal cases processed: {len(next(iter(self.all_metrics.values())))}")


def main():
    parser = argparse.ArgumentParser(description='Collect comprehensive metrics for all methods')
    parser.add_argument('--predictions-dir', required=True,
                       help='Directory containing U-Net prediction masks (e.g., experiments/test_predictions/)')
    parser.add_argument('--ground-truth-dir', required=True,
                       help='Directory containing ground truth masks (e.g., DATASET/Parse_dataset/)')
    parser.add_argument('--output-dir', required=True,
                       help='Directory to save comprehensive metrics')
    parser.add_argument('--voxel-spacing', nargs=3, type=float, default=[1.0, 1.0, 1.0],
                       help='Voxel spacing (z y x)')
    parser.add_argument('--max-cases', type=int, default=None,
                       help='Maximum number of cases to process (for testing)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device for computations')
    
    args = parser.parse_args()
    
    # Initialize collector
    collector = ComprehensiveMetricsCollector(
        predictions_dir=args.predictions_dir,
        ground_truth_dir=args.ground_truth_dir,
        output_dir=args.output_dir,
        voxel_spacing=tuple(args.voxel_spacing),
        device=args.device
    )
    
    try:
        # Collect all metrics
        results = collector.collect_all_metrics(max_cases=args.max_cases)
        
        # Print preview
        collector.print_preview_results()
        
        print(f"\nMetrics collection complete! Results saved in {args.output_dir}")
        print("\nFiles created:")
        print("- comprehensive_metrics_YYYYMMDD_HHMMSS.json: Raw metrics data")
        print("- comprehensive_metrics_YYYYMMDD_HHMMSS.csv: Tabular format")
        print("- summary_statistics_YYYYMMDD_HHMMSS.csv: Summary statistics")
        print("\nExample usage:")
        print("python scripts/collect_comprehensive_metrics.py \\")
        print("  --predictions-dir experiments/test_predictions \\")
        print("  --ground-truth-dir DATASET/Parse_dataset \\")
        print("  --output-dir results/comprehensive_metrics")
        print("\nExpected directory structure:")
        print("experiments/test_predictions/")
        print("├── PA000303_pred.nii.gz     # Binary prediction")
        print("├── PA000303_prob.nii.gz     # Probability map")
        print("└── PA000303/")
        print("    ├── binary_mask.nii.gz   # Binary mask")
        print("    └── prediction.nii.gz    # Prediction")
        
        return 0
        
    except Exception as e:
        print(f"Metrics collection failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())