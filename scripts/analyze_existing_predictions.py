#!/usr/bin/env python
"""Analyze existing predictions and compute per-case metrics"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.metrics import compute_metrics


def compute_per_case_metrics(prediction_prob, prediction_binary, ground_truth, spacing=None, patient_id=""):
    """Compute comprehensive per-case metrics"""
    
    # Convert to numpy if needed
    import torch
    if torch.is_tensor(prediction_prob):
        prediction_prob = prediction_prob.cpu().numpy()
    if torch.is_tensor(prediction_binary):
        prediction_binary = prediction_binary.cpu().numpy()
    if torch.is_tensor(ground_truth):
        ground_truth = ground_truth.cpu().numpy()
    
    # Ensure binary
    prediction_binary = (prediction_binary > 0.5).astype(np.uint8)
    ground_truth = (ground_truth > 0.5).astype(np.uint8)
    
    # Check for empty predictions or ground truth
    pred_sum = prediction_binary.sum()
    gt_sum = ground_truth.sum()
    
    if pred_sum == 0 and gt_sum == 0:
        # Both empty - perfect match
        return {
            'patient_id': patient_id,
            'dice': 1.0,
            'jaccard': 1.0,
            'precision': 1.0,
            'recall': 1.0,
            'specificity': 1.0,
            'hausdorff': 0.0,
            'avg_surface_distance': 0.0,
            'mean_prob_in_pred': 0.0,
            'mean_prob_in_gt': 0.0,
            'max_prob': float(prediction_prob.max()),
            'min_prob_in_pred': 0.0,
            'pred_volume_mm3': 0.0,
            'gt_volume_mm3': 0.0,
            'volume_ratio': 1.0,
            'volume_diff_mm3': 0.0,
            'success': True,
            'empty_case': True
        }
    elif pred_sum == 0:
        # Prediction empty but GT has content - worst case
        voxel_volume = np.prod(spacing) if spacing is not None else 1.0
        return {
            'patient_id': patient_id,
            'dice': 0.0,
            'jaccard': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'specificity': 1.0,  # All true negatives
            'hausdorff': 999.0,  # Max distance
            'avg_surface_distance': 999.0,
            'mean_prob_in_pred': 0.0,
            'mean_prob_in_gt': float(prediction_prob[ground_truth > 0].mean()) if gt_sum > 0 else 0.0,
            'max_prob': float(prediction_prob.max()),
            'min_prob_in_pred': 0.0,
            'pred_volume_mm3': 0.0,
            'gt_volume_mm3': float(gt_sum * voxel_volume),
            'volume_ratio': 0.0,
            'volume_diff_mm3': float(gt_sum * voxel_volume),
            'success': True,
            'prediction_empty': True
        }
    
    try:
        # Use the existing metrics function
        print(f"      Calling compute_metrics for {patient_id}...")
        metrics = compute_metrics(
            torch.from_numpy(prediction_binary), 
            torch.from_numpy(ground_truth),
            spacing=spacing
        )
        print(f"      compute_metrics returned: {metrics}")
        
        # Add additional custom metrics
        voxel_volume = np.prod(spacing) if spacing is not None else 1.0
        
        # Volume metrics
        pred_volume = prediction_binary.sum() * voxel_volume
        gt_volume = ground_truth.sum() * voxel_volume
        
        # Probability statistics
        prob_stats = {
            'mean_prob_in_pred': float(prediction_prob[prediction_binary > 0].mean()) if prediction_binary.sum() > 0 else 0.0,
            'mean_prob_in_gt': float(prediction_prob[ground_truth > 0].mean()) if ground_truth.sum() > 0 else 0.0,
            'max_prob': float(prediction_prob.max()),
            'min_prob_in_pred': float(prediction_prob[prediction_binary > 0].min()) if prediction_binary.sum() > 0 else 0.0,
        }
        
        # Volume metrics
        volume_metrics = {
            'pred_volume_mm3': float(pred_volume),
            'gt_volume_mm3': float(gt_volume),
            'volume_ratio': float(pred_volume / (gt_volume + 1e-8)),
            'volume_diff_mm3': float(abs(pred_volume - gt_volume)),
        }
        
        # Combine all metrics - handle key mapping differences
        all_metrics = {
            'patient_id': patient_id,
            'dice': float(metrics['dice']),
            'jaccard': float(metrics['jaccard']),
            'precision': float(metrics['precision']),
            'recall': float(metrics.get('recall', metrics.get('sensitivity', 0))),  # recall = sensitivity
            'specificity': float(metrics['specificity']),
            'hausdorff': float(metrics.get('hausdorff', metrics.get('hausdorff_distance', 0))),
            'avg_surface_distance': float(metrics.get('avg_surface_distance', 0)),
            **prob_stats,
            **volume_metrics,
            'success': True
        }
        
        return all_metrics
        
    except Exception as e:
        # Return error metrics
        import traceback
        print(f"      ERROR in compute_per_case_metrics for {patient_id}: {e}")
        print(f"      Traceback: {traceback.format_exc()}")
        return {
            'patient_id': patient_id,
            'dice': 0.0,
            'jaccard': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'specificity': 0.0,
            'hausdorff': 0.0,
            'avg_surface_distance': 0.0,
            'mean_prob_in_pred': 0.0,
            'mean_prob_in_gt': 0.0,
            'max_prob': 0.0,
            'min_prob_in_pred': 0.0,
            'pred_volume_mm3': 0.0,
            'gt_volume_mm3': 0.0,
            'volume_ratio': 0.0,
            'volume_diff_mm3': 0.0,
            'success': False,
            'error': str(e)
        }


def analyze_and_filter_cases(metrics_df, output_dir, min_dice=0.5, max_hausdorff=100):
    """Analyze metrics and identify problematic cases"""
    
    output_dir = Path(output_dir)
    
    # Overall statistics
    print(f"\n📊 PERFORMANCE ANALYSIS")
    print(f"=" * 50)
    print(f"Total cases: {len(metrics_df)}")
    print(f"Successful predictions: {metrics_df['success'].sum()}")
    print(f"Failed predictions: {(~metrics_df['success']).sum()}")
    
    successful_df = metrics_df[metrics_df['success']]
    
    if len(successful_df) > 0:
        print(f"\n📈 METRIC STATISTICS:")
        print(f"Dice Score:     {successful_df['dice'].mean():.4f} ± {successful_df['dice'].std():.4f} (range: {successful_df['dice'].min():.4f} - {successful_df['dice'].max():.4f})")
        print(f"Jaccard:        {successful_df['jaccard'].mean():.4f} ± {successful_df['jaccard'].std():.4f}")
        print(f"Precision:      {successful_df['precision'].mean():.4f} ± {successful_df['precision'].std():.4f}")
        print(f"Recall:         {successful_df['recall'].mean():.4f} ± {successful_df['recall'].std():.4f}")
        print(f"Hausdorff:      {successful_df['hausdorff'].mean():.2f} ± {successful_df['hausdorff'].std():.2f}")
        print(f"Volume Ratio:   {successful_df['volume_ratio'].mean():.4f} ± {successful_df['volume_ratio'].std():.4f}")
        
        # Identify problematic cases
        problematic_cases = successful_df[
            (successful_df['dice'] < min_dice) | 
            (successful_df['hausdorff'] > max_hausdorff) |
            (successful_df['volume_ratio'] < 0.3) |
            (successful_df['volume_ratio'] > 3.0)
        ]
        
        good_cases = successful_df[
            (successful_df['dice'] >= min_dice) & 
            (successful_df['hausdorff'] <= max_hausdorff) &
            (successful_df['volume_ratio'] >= 0.3) &
            (successful_df['volume_ratio'] <= 3.0)
        ]
        
        print(f"\n🎯 CASE CLASSIFICATION:")
        print(f"Good cases (Dice≥{min_dice}, HD≤{max_hausdorff}): {len(good_cases)} ({len(good_cases)/len(successful_df)*100:.1f}%)")
        print(f"Problematic cases: {len(problematic_cases)} ({len(problematic_cases)/len(successful_df)*100:.1f}%)")
        
        # Distribution analysis
        print(f"\n📊 DICE SCORE DISTRIBUTION:")
        dice_bins = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        dice_hist = pd.cut(successful_df['dice'], bins=dice_bins)
        for i in range(len(dice_bins)-1):
            count = ((successful_df['dice'] >= dice_bins[i]) & (successful_df['dice'] < dice_bins[i+1])).sum()
            if count > 0:
                print(f"  {dice_bins[i]:.1f}-{dice_bins[i+1]:.1f}: {'█' * int(count/len(successful_df)*50)} ({count} cases)")
        
        # Save lists
        good_cases_file = output_dir / 'good_cases.txt'
        problematic_cases_file = output_dir / 'problematic_cases.txt'
        
        with open(good_cases_file, 'w') as f:
            f.write("# Good performing cases (ready for graph correction)\n")
            f.write(f"# Criteria: Dice≥{min_dice}, Hausdorff≤{max_hausdorff}, Volume ratio 0.3-3.0\n")
            f.write(f"# Total: {len(good_cases)} cases\n\n")
            for _, row in good_cases.sort_values('dice', ascending=False).iterrows():
                f.write(f"{row['patient_id']}\t# Dice: {row['dice']:.4f}, HD: {row['hausdorff']:.2f}\n")
        
        with open(problematic_cases_file, 'w') as f:
            f.write("# Problematic cases (consider excluding)\n")
            f.write(f"# Criteria: Dice<{min_dice} OR Hausdorff>{max_hausdorff} OR Volume ratio <0.3 or >3.0\n")
            f.write(f"# Total: {len(problematic_cases)} cases\n\n")
            for _, row in problematic_cases.sort_values('dice').iterrows():
                reason = []
                if row['dice'] < min_dice:
                    reason.append(f"Low Dice ({row['dice']:.4f})")
                if row['hausdorff'] > max_hausdorff:
                    reason.append(f"High HD ({row['hausdorff']:.2f})")
                if row['volume_ratio'] < 0.3 or row['volume_ratio'] > 3.0:
                    reason.append(f"Volume ratio ({row['volume_ratio']:.2f})")
                
                f.write(f"{row['patient_id']}\t# {', '.join(reason)}\n")
        
        print(f"\n📄 SAVED FILES:")
        print(f"✅ Good cases: {good_cases_file}")
        print(f"⚠️  Problematic cases: {problematic_cases_file}")
        
        # Show top and bottom performers
        print(f"\n🏆 TOP 10 PERFORMERS:")
        top_cases = successful_df.nlargest(10, 'dice')[['patient_id', 'dice', 'jaccard', 'hausdorff']]
        for _, row in top_cases.iterrows():
            print(f"  {row['patient_id']}: Dice={row['dice']:.4f}, Jaccard={row['jaccard']:.4f}, HD={row['hausdorff']:.2f}")
        
        print(f"\n🔴 BOTTOM 10 PERFORMERS:")
        bottom_cases = successful_df.nsmallest(10, 'dice')[['patient_id', 'dice', 'jaccard', 'hausdorff']]
        for _, row in bottom_cases.iterrows():
            print(f"  {row['patient_id']}: Dice={row['dice']:.4f}, Jaccard={row['jaccard']:.4f}, HD={row['hausdorff']:.2f}")
        
        # Volume analysis
        print(f"\n📏 VOLUME ANALYSIS:")
        overestimated = successful_df[successful_df['volume_ratio'] > 1.5]
        underestimated = successful_df[successful_df['volume_ratio'] < 0.5]
        print(f"Overestimated (>1.5x): {len(overestimated)} cases ({len(overestimated)/len(successful_df)*100:.1f}%)")
        print(f"Underestimated (<0.5x): {len(underestimated)} cases ({len(underestimated)/len(successful_df)*100:.1f}%)")
        
        return good_cases['patient_id'].tolist(), problematic_cases['patient_id'].tolist()
    
    else:
        print("❌ No successful predictions to analyze")
        return [], []


def analyze_existing_predictions(predictions_dir, dataset_dir, output_dir=None, 
                               min_dice=0.6, max_hausdorff=50, patient_ids=None):
    """Analyze existing predictions by computing metrics against ground truth"""
    
    predictions_dir = Path(predictions_dir)
    dataset_dir = Path(dataset_dir)
    
    if output_dir is None:
        output_dir = predictions_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🔍 ANALYZING EXISTING PREDICTIONS")
    print(f"=" * 50)
    print(f"Predictions: {predictions_dir}")
    print(f"Dataset: {dataset_dir}")
    print(f"Output: {output_dir}")
    
    # Find all prediction files
    prediction_files = []
    
    # Check for patient directory structure - PRIORITIZE binary_mask.nii.gz
    patient_dirs = [d for d in predictions_dir.iterdir() if d.is_dir() and d.name.startswith('PA')]
    for patient_dir in patient_dirs:
        # Check for binary_mask.nii.gz FIRST (this is the correct prediction)
        binary_file = patient_dir / 'binary_mask.nii.gz'
        if binary_file.exists():
            prediction_files.append(binary_file)
        else:
            # Fallback to prediction.nii.gz if binary doesn't exist
            pred_file = patient_dir / 'prediction.nii.gz'
            if pred_file.exists():
                prediction_files.append(pred_file)
    
    # Also check for direct prediction files - PRIORITIZE _pred.nii.gz (binary)
    # Only use these if we didn't find files in subdirectories
    existing_patients = {pf.parent.name for pf in prediction_files}
    
    # First try binary predictions
    for pred_file in predictions_dir.glob('PA*_pred.nii.gz'):
        patient_id = pred_file.stem.split('_')[0]
        if patient_id not in existing_patients:
            prediction_files.append(pred_file)
            existing_patients.add(patient_id)
    
    # Then probability files as fallback
    for prob_file in predictions_dir.glob('PA*_prob.nii.gz'):
        patient_id = prob_file.stem.split('_')[0]
        if patient_id not in existing_patients:
            prediction_files.append(prob_file)
    
    # Remove duplicates and create final list
    seen_patients = set()
    unique_files = []
    for pred_file in prediction_files:
        # Extract patient ID based on file location
        if pred_file.parent.name.startswith('PA'):
            # File is in patient subdirectory
            patient_id = pred_file.parent.name
        else:
            # File is directly in predictions directory
            patient_id = pred_file.stem.split('_')[0]
        
        if patient_id not in seen_patients:
            seen_patients.add(patient_id)
            unique_files.append((patient_id, pred_file))
    
    # Filter by patient IDs if specified
    if patient_ids:
        unique_files = [(pid, pf) for pid, pf in unique_files if pid in patient_ids]
    
    print(f"Found {len(unique_files)} unique predictions")
    
    # Debug: Show which files are being used
    print(f"\n🔍 Files being analyzed:")
    for patient_id, pred_file in unique_files[:5]:
        print(f"  {patient_id}: {pred_file}")
    if len(unique_files) > 5:
        print(f"  ... and {len(unique_files) - 5} more")
    
    # Prepare output files
    metrics_file = output_dir / 'per_case_metrics.csv'
    excel_file = output_dir / 'per_case_metrics.xlsx'
    
    # Initialize CSV file with headers
    import csv
    csv_headers = ['patient_id', 'dice', 'jaccard', 'precision', 'recall', 'specificity', 
                   'hausdorff', 'avg_surface_distance', 'mean_prob_in_pred', 'mean_prob_in_gt',
                   'max_prob', 'min_prob_in_pred', 'pred_volume_mm3', 'gt_volume_mm3', 
                   'volume_ratio', 'volume_diff_mm3', 'success', 'error']
    
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
    
    print(f"📝 Writing results to: {metrics_file}")
    
    # Compute metrics for each case
    case_metrics = []
    successful_count = 0
    failed_count = 0
    running_dice_sum = 0
    
    # Create progress bar with custom format
    pbar = tqdm(unique_files, desc="Computing metrics", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] Avg Dice: {postfix}')
    
    for patient_id, pred_file in pbar:
        try:
            # Load prediction
            pred_sitk = sitk.ReadImage(str(pred_file))
            pred_array = sitk.GetArrayFromImage(pred_sitk).astype(np.float32)
            
            # Debug info for first few cases
            if successful_count < 3:
                print(f"\n🔍 DEBUG {patient_id}:")
                print(f"   File: {pred_file}")
                print(f"   Shape: {pred_array.shape}")
                print(f"   Min: {pred_array.min()}, Max: {pred_array.max()}")
                print(f"   Non-zero: {np.count_nonzero(pred_array)}")
            
            # Determine if prediction is binary or probability
            unique_values = np.unique(pred_array)
            is_binary = len(unique_values) <= 2 and all(v in [0, 1] for v in unique_values)
            
            if successful_count < 3:
                print(f"   Unique values: {unique_values[:5]}...")
                print(f"   Is binary: {is_binary}")
            
            if is_binary:
                # If already binary, use it directly
                binary_pred = pred_array.astype(np.uint8)
                # Create pseudo-probability for metrics (binary values)
                pred_prob = pred_array
            else:
                # If probability map
                if pred_array.max() > 1.0:
                    # Normalize if needed
                    pred_array = pred_array / pred_array.max()
                pred_prob = pred_array
                binary_pred = (pred_array > 0.5).astype(np.uint8)
            
            # Load ground truth
            gt_path = dataset_dir / patient_id / 'label' / f'{patient_id}.nii.gz'
            if not gt_path.exists():
                print(f"  ⚠️  Skipping {patient_id}: Ground truth not found")
                continue
            
            gt_sitk = sitk.ReadImage(str(gt_path))
            gt_array = sitk.GetArrayFromImage(gt_sitk).astype(np.uint8)
            
            # Get spacing
            spacing = pred_sitk.GetSpacing()[::-1]  # Convert to ZYX
            
            # Compute metrics with detailed error handling
            print(f"   Computing metrics for {patient_id}...")
            metrics_dict = compute_per_case_metrics(
                pred_prob, binary_pred, gt_array,
                spacing=spacing, patient_id=patient_id
            )
            print(f"   Metrics result: success={metrics_dict.get('success', False)}, dice={metrics_dict.get('dice', 'N/A')}")
            case_metrics.append(metrics_dict)
            
            # Write to CSV immediately
            with open(metrics_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=csv_headers)
                writer.writerow({k: metrics_dict.get(k, '') for k in csv_headers})
            
            # Update counters and print progress
            if metrics_dict['success']:
                successful_count += 1
                dice = metrics_dict['dice']
                
                # Debug: Print if dice is 0
                if dice == 0.0:
                    print(f"  ⚠️  {patient_id}: Zero Dice! Pred sum: {binary_pred.sum()}, GT sum: {gt_array.sum()}")
                
                running_dice_sum += dice
                
                # Update progress bar with running average
                avg_dice = running_dice_sum / successful_count if successful_count > 0 else 0
                pbar.set_postfix_str(f'{avg_dice:.3f}')
                
                if dice >= 0.8:
                    print(f"  🟢 {patient_id}: Excellent (Dice={dice:.4f})")
                elif dice < 0.4:
                    print(f"  🔴 {patient_id}: Poor (Dice={dice:.4f})")
            else:
                failed_count += 1
            
        except Exception as e:
            import traceback
            print(f"  ❌ Error processing {patient_id}: {e}")
            print(f"  Traceback: {traceback.format_exc()}")
            error_dict = {
                'patient_id': patient_id,
                'success': False,
                'error': str(e)
            }
            case_metrics.append(error_dict)
            
            # Write error to CSV immediately
            with open(metrics_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=csv_headers)
                writer.writerow({k: error_dict.get(k, '') for k in csv_headers})
            
            failed_count += 1
    
    print(f"\n✅ Metrics computation complete: {successful_count} successful, {failed_count} failed")
    
    # Save and analyze metrics
    if case_metrics:
        metrics_df = pd.DataFrame(case_metrics)
        
        # Save CSV with all metrics
        metrics_file = output_dir / 'per_case_metrics.csv'
        metrics_df.to_csv(metrics_file, index=False)
        print(f"\n📊 Metrics saved to: {metrics_file}")
        
        # Save Excel for easier viewing (optional)
        try:
            excel_file = output_dir / 'per_case_metrics.xlsx'
            metrics_df.to_excel(excel_file, index=False)
            print(f"📊 Excel saved to: {excel_file}")
        except ImportError:
            print("⚠️  Excel export skipped (openpyxl not installed). Install with: pip install openpyxl")
        
        # Analyze and filter cases
        good_cases, problematic_cases = analyze_and_filter_cases(
            metrics_df, output_dir, min_dice=min_dice, max_hausdorff=max_hausdorff
        )
        
        # Save summary - convert all numpy int64 to regular int/float
        def convert_numpy_types(obj):
            """Convert numpy types to JSON serializable types"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            else:
                return obj
        
        summary = {
            'predictions_dir': str(predictions_dir),
            'dataset_dir': str(dataset_dir),
            'timestamp': datetime.now().isoformat(),
            'n_cases_analyzed': int(len(case_metrics)),
            'n_successful': int(metrics_df['success'].sum()),
            'n_failed': int((~metrics_df['success']).sum()),
            'n_good_cases': int(len(good_cases)),
            'n_problematic_cases': int(len(problematic_cases)),
            'performance_stats': {
                'mean_dice': float(metrics_df[metrics_df['success']]['dice'].mean()) if metrics_df['success'].sum() > 0 else 0.0,
                'std_dice': float(metrics_df[metrics_df['success']]['dice'].std()) if metrics_df['success'].sum() > 0 else 0.0,
                'mean_jaccard': float(metrics_df[metrics_df['success']]['jaccard'].mean()) if metrics_df['success'].sum() > 0 else 0.0,
                'mean_hausdorff': float(metrics_df[metrics_df['success']]['hausdorff'].mean()) if metrics_df['success'].sum() > 0 else 0.0,
            },
            'thresholds_used': {
                'min_dice': float(min_dice),
                'max_hausdorff': float(max_hausdorff)
            }
        }
        
        # Convert any remaining numpy types
        summary = convert_numpy_types(summary)
        
        with open(output_dir / 'analysis_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📄 Summary saved to: {output_dir / 'analysis_summary.json'}")
        
    else:
        print("❌ No metrics computed")
    
    print(f"\n✅ Analysis complete!")
    
    if len(good_cases) > 0:
        print(f"\n📋 NEXT STEPS:")
        print(f"For graph extraction with good cases only:")
        print(f"python scripts/extract_predicted_graphs.py \\")
        print(f"    --predictions-dir {predictions_dir} \\")
        print(f"    --graphs-dir extracted_graphs \\")
        print(f"    --patient-list {output_dir}/good_cases.txt")


def main():
    parser = argparse.ArgumentParser(description='Analyze existing predictions and compute metrics')
    parser.add_argument('--predictions-dir', type=str, required=True,
                        help='Directory containing predictions')
    parser.add_argument('--dataset', type=str, default='DATASET/Parse_dataset',
                        help='Path to dataset directory with ground truth')
    parser.add_argument('--output-dir', type=str,
                        help='Output directory for analysis (defaults to predictions dir)')
    parser.add_argument('--min-dice', type=float, default=0.6,
                        help='Minimum Dice score for good cases (default: 0.6)')
    parser.add_argument('--max-hausdorff', type=float, default=50,
                        help='Maximum Hausdorff distance for good cases (default: 50)')
    parser.add_argument('--patient-ids', nargs='+',
                        help='Specific patient IDs to analyze')
    
    args = parser.parse_args()
    
    if not Path(args.predictions_dir).exists():
        print(f"❌ Predictions directory not found: {args.predictions_dir}")
        return
    
    if not Path(args.dataset).exists():
        print(f"❌ Dataset directory not found: {args.dataset}")
        return
    
    # Run analysis
    analyze_existing_predictions(
        args.predictions_dir,
        args.dataset,
        args.output_dir,
        args.min_dice,
        args.max_hausdorff,
        args.patient_ids
    )


if __name__ == '__main__':
    main()