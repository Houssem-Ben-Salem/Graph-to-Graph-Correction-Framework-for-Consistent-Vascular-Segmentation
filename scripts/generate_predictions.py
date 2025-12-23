#!/usr/bin/env python
"""Generate predictions from trained U-Net models for graph correction training"""

import os
import sys
import argparse
import json
import pandas as pd
from pathlib import Path
import torch
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.unet import UNet3D, AttentionUNet3D, UNet2D
from src.data.loaders.dataset import PulmonaryArteryDataset
from src.utils.metrics import compute_metrics


def load_model_from_checkpoint(checkpoint_path, architecture):
    """Load model from checkpoint"""
    # Load with weights_only=False for compatibility with older checkpoints
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    
    # Create model
    if architecture == 'unet3d':
        model = UNet3D(**config['config'])
    elif architecture == 'attention_unet3d':
        model = AttentionUNet3D(**config['config'])
    elif architecture == 'unet2d':
        model = UNet2D(**config['config'])
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def predict_volume(model, image, patch_size=None, overlap=0.5, device='cuda'):
    """Predict full volume using sliding window if necessary"""
    model = model.to(device)
    
    if patch_size is None:
        # Predict entire volume at once
        with torch.no_grad():
            image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)
            output = model(image_tensor)
            prediction = torch.sigmoid(output).cpu().numpy()[0, 0]
        return prediction
    
    # Use sliding window prediction for large volumes
    return sliding_window_prediction(model, image, patch_size, overlap, device)


def sliding_window_prediction(model, image, patch_size, overlap, device):
    """Sliding window prediction for large volumes"""
    image_shape = image.shape
    prediction = np.zeros(image_shape, dtype=np.float32)
    count_map = np.zeros(image_shape, dtype=np.float32)
    
    # Calculate step size
    step = [int(p * (1 - overlap)) for p in patch_size]
    
    # Generate all patch positions
    positions = []
    for z in range(0, image_shape[0] - patch_size[0] + 1, step[0]):
        for y in range(0, image_shape[1] - patch_size[1] + 1, step[1]):
            for x in range(0, image_shape[2] - patch_size[2] + 1, step[2]):
                positions.append((z, y, x))
    
    # Add edge patches to ensure full coverage
    for z in [max(0, image_shape[0] - patch_size[0])]:
        for y in [max(0, image_shape[1] - patch_size[1])]:
            for x in [max(0, image_shape[2] - patch_size[2])]:
                if (z, y, x) not in positions:
                    positions.append((z, y, x))
    
    # Process each patch
    with torch.no_grad():
        for z, y, x in tqdm(positions, desc="Processing patches"):
            # Extract patch
            patch = image[
                z:z+patch_size[0],
                y:y+patch_size[1],
                x:x+patch_size[2]
            ]
            
            # Predict patch
            patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
            output = model(patch_tensor)
            patch_pred = torch.sigmoid(output).cpu().numpy()[0, 0]
            
            # Add to prediction
            prediction[
                z:z+patch_size[0],
                y:y+patch_size[1],
                x:x+patch_size[2]
            ] += patch_pred
            
            count_map[
                z:z+patch_size[0],
                y:y+patch_size[1],
                x:x+patch_size[2]
            ] += 1
    
    # Average overlapping predictions
    prediction = prediction / np.maximum(count_map, 1)
    
    return prediction


def normalize_image(image):
    """Normalize image using percentile normalization"""
    p1, p99 = np.percentile(image, [1, 99])
    image = np.clip(image, p1, p99)
    image = (image - p1) / (p99 - p1 + 1e-8)
    return image


def compute_per_case_metrics(prediction_prob, prediction_binary, ground_truth, spacing=None, patient_id=""):
    """Compute comprehensive per-case metrics"""
    
    # Convert to numpy if needed
    if torch.is_tensor(prediction_prob):
        prediction_prob = prediction_prob.cpu().numpy()
    if torch.is_tensor(prediction_binary):
        prediction_binary = prediction_binary.cpu().numpy()
    if torch.is_tensor(ground_truth):
        ground_truth = ground_truth.cpu().numpy()
    
    # Ensure binary
    prediction_binary = (prediction_binary > 0.5).astype(np.uint8)
    ground_truth = (ground_truth > 0.5).astype(np.uint8)
    
    try:
        # Use the existing metrics function
        metrics = compute_metrics(
            torch.from_numpy(prediction_binary), 
            torch.from_numpy(ground_truth),
            spacing=spacing
        )
        
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
        
        # Combine all metrics
        all_metrics = {
            'patient_id': patient_id,
            'dice': float(metrics['dice']),
            'jaccard': float(metrics['jaccard']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'specificity': float(metrics['specificity']),
            'hausdorff': float(metrics.get('hausdorff', 0)),
            'avg_surface_distance': float(metrics.get('avg_surface_distance', 0)),
            **prob_stats,
            **volume_metrics,
            'success': True
        }
        
        return all_metrics
        
    except Exception as e:
        # Return error metrics
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
        
        return good_cases['patient_id'].tolist(), problematic_cases['patient_id'].tolist()
    
    else:
        print("❌ No successful predictions to analyze")
        return [], []


def generate_predictions_from_manual_model(model_path, architecture, dataset_path, output_dir, 
                                         device='cuda', min_dice=0.6, max_hausdorff=50, patient_ids=None, skip_analysis=False):
    """Generate predictions from a single manually specified model"""
    
    model_path = Path(model_path)
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 GENERATING PREDICTIONS FROM MANUAL MODEL")
    print(f"=" * 50)
    print(f"Model: {model_path}")
    print(f"Architecture: {architecture}")
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_dir}")
    
    # Check if model file exists
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return
    
    # Load model
    model = load_model_from_checkpoint(model_path, architecture)
    model = model.to(device)
    
    # Get patient list
    all_patient_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    
    if patient_ids:
        patient_dirs = [d for d in all_patient_dirs if d.name in patient_ids]
        print(f"Processing {len(patient_dirs)} specified patients (out of {len(all_patient_dirs)} total)")
    else:
        patient_dirs = all_patient_dirs
        print(f"Processing {len(patient_dirs)} patients")
    
    # Generate predictions and compute metrics for each patient
    success_count = 0
    case_metrics = []
    
    for patient_dir in tqdm(patient_dirs, desc=f"Predicting {architecture}"):
        patient_id = patient_dir.name
        
        try:
            # Load image
            image_path = patient_dir / 'image' / f'{patient_id}.nii.gz'
            if not image_path.exists():
                print(f"    ⚠️  Skipping {patient_id}: Image not found")
                continue
            
            image_sitk = sitk.ReadImage(str(image_path))
            image_array = sitk.GetArrayFromImage(image_sitk).astype(np.float32)
            
            # Load ground truth and spacing only if analysis is needed
            if not skip_analysis:
                gt_path = patient_dir / 'label' / f'{patient_id}.nii.gz'
                if not gt_path.exists():
                    print(f"    ⚠️  Skipping {patient_id}: Ground truth not found")
                    continue
                    
                gt_sitk = sitk.ReadImage(str(gt_path))
                gt_array = sitk.GetArrayFromImage(gt_sitk).astype(np.uint8)
                
                # Get spacing for metric calculations
                spacing = image_sitk.GetSpacing()[::-1]  # Convert to ZYX
            else:
                gt_array = None
                spacing = None
            
            # Normalize image
            normalized_image = normalize_image(image_array)
            
            # Generate prediction
            if '2d' in architecture:
                # For 2D models, process slice by slice
                prediction = predict_2d_volume(model, normalized_image, device)
            else:
                # For 3D models, use sliding window if needed
                patch_size = [128, 128, 128] if max(normalized_image.shape) > 256 else None
                prediction = predict_volume(model, normalized_image, patch_size, device=device)
            
            binary_prediction = (prediction > 0.5).astype(np.uint8)
            
            # Compute per-case metrics only if analysis is not skipped
            if not skip_analysis and gt_array is not None:
                case_metrics_dict = compute_per_case_metrics(
                    prediction, binary_prediction, gt_array, 
                    spacing=spacing, patient_id=patient_id
                )
                case_metrics.append(case_metrics_dict)
            else:
                # Just add basic info without metrics
                case_metrics.append({
                    'patient_id': patient_id,
                    'success': True,
                    'skipped_analysis': True
                })
            
            # Create patient output directory (COMPATIBLE WITH GRAPH EXTRACTION)
            patient_output_dir = output_dir / patient_id
            patient_output_dir.mkdir(exist_ok=True)
            
            # Save probability map
            prob_output = patient_output_dir / 'prediction.nii.gz'
            prob_sitk = sitk.GetImageFromArray(prediction.astype(np.float32))
            prob_sitk.CopyInformation(image_sitk)
            sitk.WriteImage(prob_sitk, str(prob_output))
            
            # Also save in alternative format for compatibility
            alt_prob_output = output_dir / f'{patient_id}_prob.nii.gz'
            sitk.WriteImage(prob_sitk, str(alt_prob_output))
            
            # Binary mask in patient directory
            binary_output = patient_output_dir / 'binary_mask.nii.gz'
            pred_sitk = sitk.GetImageFromArray(binary_prediction)
            pred_sitk.CopyInformation(image_sitk)
            sitk.WriteImage(pred_sitk, str(binary_output))
            
            # Also save in alternative format
            alt_pred_output = output_dir / f'{patient_id}_pred.nii.gz'
            sitk.WriteImage(pred_sitk, str(alt_pred_output))
            
            success_count += 1
            
            # Print progress for top/bottom cases (only if analysis is enabled)
            if not skip_analysis and not case_metrics[-1].get('skipped_analysis', False):
                if case_metrics[-1]['success'] and 'dice' in case_metrics[-1]:
                    dice = case_metrics[-1]['dice']
                    if dice >= 0.8:
                        print(f"    🟢 {patient_id}: Excellent (Dice={dice:.4f})")
                    elif dice < 0.4:
                        print(f"    🔴 {patient_id}: Poor (Dice={dice:.4f})")
            
        except Exception as e:
            print(f"    ❌ Error processing {patient_id}: {e}")
            case_metrics.append({
                'patient_id': patient_id,
                'success': False,
                'error': str(e)
            })
    
    print(f"\n✅ Successfully processed {success_count}/{len(patient_dirs)} patients")
    
    # Save detailed metrics and analyze only if not skipped
    if not skip_analysis and case_metrics and not case_metrics[0].get('skipped_analysis', False):
        metrics_df = pd.DataFrame(case_metrics)
        
        # Save CSV with all metrics
        metrics_file = output_dir / 'per_case_metrics.csv'
        metrics_df.to_csv(metrics_file, index=False)
        print(f"📊 Metrics saved to: {metrics_file}")
        
        # Analyze and filter cases
        print(f"\n🔍 ANALYZING {architecture.upper()} PERFORMANCE:")
        good_cases, problematic_cases = analyze_and_filter_cases(
            metrics_df, output_dir, min_dice=min_dice, max_hausdorff=max_hausdorff
        )
    else:
        if skip_analysis:
            print(f"\n⏭️  Analysis skipped as requested")
        good_cases, problematic_cases = [], []
    
    # Save prediction summary
    successful_metrics = pd.DataFrame()
    if not skip_analysis and case_metrics and not case_metrics[0].get('skipped_analysis', False):
        metrics_df = pd.DataFrame(case_metrics)
        successful_metrics = metrics_df[metrics_df['success']]
    
    summary = {
        'model_path': str(model_path),
        'architecture': architecture,
        'dataset_path': str(dataset_path),
        'timestamp': datetime.now().isoformat(),
        'n_patients_processed': success_count,
        'n_good_cases': len(good_cases),
        'n_problematic_cases': len(problematic_cases),
        'performance_stats': {
            'mean_dice': float(successful_metrics['dice'].mean()) if len(successful_metrics) > 0 else 0.0,
            'std_dice': float(successful_metrics['dice'].std()) if len(successful_metrics) > 0 else 0.0,
            'mean_jaccard': float(successful_metrics['jaccard'].mean()) if len(successful_metrics) > 0 else 0.0,
            'mean_hausdorff': float(successful_metrics['hausdorff'].mean()) if len(successful_metrics) > 0 else 0.0,
        },
        'output_structure': {
            'patient_dirs': f'{{patient_id}}/prediction.nii.gz',
            'alternative_format': f'{{patient_id}}_pred.nii.gz'
        }
    }
    
    with open(output_dir / 'prediction_info.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Prediction generation completed!")
    print(f"📁 Output directory: {output_dir}")
    print(f"🔗 Ready for graph extraction!")
    
    # Show usage instructions
    print(f"\n📋 NEXT STEPS:")
    print(f"python scripts/extract_predicted_graphs.py \\")
    print(f"    --predictions-dir {output_dir} \\")
    print(f"    --graphs-dir extracted_graphs \\")
    print(f"    --threshold 0.5")
    
    if good_cases:
        print(f"\nFor good cases only:")
        print(f"python scripts/extract_predicted_graphs.py \\")
        print(f"    --predictions-dir {output_dir} \\")
        print(f"    --graphs-dir extracted_graphs \\")
        print(f"    --patient-list {output_dir}/good_cases.txt")


def generate_predictions_from_cv_results(results_dir, dataset_path, output_dir, device='cuda', best_only=False, 
                                       min_dice=0.6, max_hausdorff=50, patient_ids=None):
    """Generate predictions from cross-validation results"""
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results summary
    with open(results_dir / 'all_results.json', 'r') as f:
        all_results = json.load(f)
    
    # Get list of all patients
    dataset_path = Path(dataset_path)
    all_patient_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    
    # Filter by patient IDs if specified
    if patient_ids:
        patient_dirs = [d for d in all_patient_dirs if d.name in patient_ids]
        print(f"Generating predictions for {len(patient_dirs)} specified patients (out of {len(all_patient_dirs)} total)")
    else:
        patient_dirs = all_patient_dirs
        print(f"Generating predictions for {len(patient_dirs)} patients")
    
    # If best_only, find the best architecture
    if best_only:
        best_arch = None
        best_dice = 0
        for arch_name, arch_results in all_results.items():
            if arch_results and 'average_dice' in arch_results:
                if arch_results['average_dice'] > best_dice:
                    best_dice = arch_results['average_dice']
                    best_arch = arch_name
        
        if best_arch:
            print(f"Using best architecture: {best_arch} (Dice: {best_dice:.4f})")
            architectures_to_process = {best_arch: all_results[best_arch]}
        else:
            print("No successful architecture found!")
            return
    else:
        architectures_to_process = all_results
    
    # For each architecture
    for arch_name, arch_results in architectures_to_process.items():
        if arch_results is None or 'fold_results' not in arch_results:
            continue
            
        print(f"\nProcessing {arch_name} (Dice: {arch_results.get('average_dice', 'unknown'):.4f})...")
        
        # Create output directories (compatible with graph extraction)
        arch_output_dir = output_dir / arch_name
        arch_output_dir.mkdir(exist_ok=True)
        
        # Load all fold models
        fold_models = []
        for fold_result in arch_results['fold_results']:
            model_path = fold_result['model_path']
            try:
                model = load_model_from_checkpoint(model_path, arch_name)
                fold_models.append(model)
                print(f"  Loaded fold {len(fold_models)} (Dice: {fold_result['val_dice']:.4f})")
            except Exception as e:
                print(f"  ⚠️  Failed to load fold {len(fold_models) + 1}: {e}")
        
        if not fold_models:
            print(f"  ❌ No models loaded for {arch_name}")
            continue
        
        print(f"  Using {len(fold_models)} fold models for ensemble prediction")
        
        # Generate predictions and compute metrics for each patient
        success_count = 0
        case_metrics = []
        
        for patient_dir in tqdm(patient_dirs, desc=f"Predicting {arch_name}"):
            patient_id = patient_dir.name
            
            try:
                # Load image
                image_path = patient_dir / 'image' / f'{patient_id}.nii.gz'
                if not image_path.exists():
                    print(f"    ⚠️  Skipping {patient_id}: Image not found")
                    continue
                
                # Load ground truth
                gt_path = patient_dir / 'label' / f'{patient_id}.nii.gz'
                if not gt_path.exists():
                    print(f"    ⚠️  Skipping {patient_id}: Ground truth not found")
                    continue
                    
                image_sitk = sitk.ReadImage(str(image_path))
                gt_sitk = sitk.ReadImage(str(gt_path))
                
                image_array = sitk.GetArrayFromImage(image_sitk).astype(np.float32)
                gt_array = sitk.GetArrayFromImage(gt_sitk).astype(np.uint8)
                
                # Get spacing for metric calculations
                spacing = image_sitk.GetSpacing()[::-1]  # Convert to ZYX
                
                # Normalize image
                normalized_image = normalize_image(image_array)
                
                # Generate predictions from all folds and average
                fold_predictions = []
                
                for fold_idx, model in enumerate(fold_models):
                    # Use appropriate patch size for architecture
                    if '2d' in arch_name:
                        # For 2D models, process slice by slice
                        prediction = predict_2d_volume(model, normalized_image, device)
                    else:
                        # For 3D models, use sliding window if needed
                        patch_size = [128, 128, 128] if max(normalized_image.shape) > 256 else None
                        prediction = predict_volume(model, normalized_image, patch_size, device=device)
                    
                    fold_predictions.append(prediction)
                
                # Average predictions across folds
                avg_prediction = np.mean(fold_predictions, axis=0)
                binary_prediction = (avg_prediction > 0.5).astype(np.uint8)
                
                # Compute per-case metrics
                case_metrics_dict = compute_per_case_metrics(
                    avg_prediction, binary_prediction, gt_array, 
                    spacing=spacing, patient_id=patient_id
                )
                case_metrics.append(case_metrics_dict)
                
                # Create patient output directory (COMPATIBLE WITH GRAPH EXTRACTION)
                patient_output_dir = arch_output_dir / patient_id
                patient_output_dir.mkdir(exist_ok=True)
                
                # Save probability map
                prob_output = patient_output_dir / 'prediction.nii.gz'
                prob_sitk = sitk.GetImageFromArray(avg_prediction.astype(np.float32))
                prob_sitk.CopyInformation(image_sitk)
                sitk.WriteImage(prob_sitk, str(prob_output))
                
                # Also save in alternative format for compatibility
                alt_prob_output = arch_output_dir / f'{patient_id}_prob.nii.gz'
                sitk.WriteImage(prob_sitk, str(alt_prob_output))
                
                # Binary mask in patient directory
                binary_output = patient_output_dir / 'binary_mask.nii.gz'
                pred_sitk = sitk.GetImageFromArray(binary_prediction)
                pred_sitk.CopyInformation(image_sitk)
                sitk.WriteImage(pred_sitk, str(binary_output))
                
                # Also save in alternative format
                alt_pred_output = arch_output_dir / f'{patient_id}_pred.nii.gz'
                sitk.WriteImage(pred_sitk, str(alt_pred_output))
                
                success_count += 1
                
                # Print progress for top/bottom cases
                if case_metrics_dict['success']:
                    dice = case_metrics_dict['dice']
                    if dice >= 0.8:
                        print(f"    🟢 {patient_id}: Excellent (Dice={dice:.4f})")
                    elif dice < 0.4:
                        print(f"    🔴 {patient_id}: Poor (Dice={dice:.4f})")
                
            except Exception as e:
                print(f"    ❌ Error processing {patient_id}: {e}")
                case_metrics.append({
                    'patient_id': patient_id,
                    'success': False,
                    'error': str(e)
                })
        
        print(f"  ✅ Successfully processed {success_count}/{len(patient_dirs)} patients")
        
        # Save detailed metrics
        if case_metrics:
            metrics_df = pd.DataFrame(case_metrics)
            
            # Save CSV with all metrics
            metrics_file = arch_output_dir / 'per_case_metrics.csv'
            metrics_df.to_csv(metrics_file, index=False)
            print(f"  📊 Metrics saved to: {metrics_file}")
            
            # Analyze and filter cases
            print(f"\n🔍 ANALYZING {arch_name.upper()} PERFORMANCE:")
            good_cases, problematic_cases = analyze_and_filter_cases(
                metrics_df, arch_output_dir, min_dice=min_dice, max_hausdorff=max_hausdorff
            )
        else:
            good_cases, problematic_cases = [], []
        
        # Save prediction summary
        successful_metrics = metrics_df[metrics_df['success']] if case_metrics else pd.DataFrame()
        
        summary = {
            'architecture': arch_name,
            'training_dice': arch_results.get('average_dice', 0),
            'n_folds_used': len(fold_models),
            'n_patients_processed': success_count,
            'n_good_cases': len(good_cases),
            'n_problematic_cases': len(problematic_cases),
            'output_structure': {
                'patient_dirs': f'{arch_name}/{{patient_id}}/prediction.nii.gz',
                'alternative_format': f'{arch_name}/{{patient_id}}_pred.nii.gz'
            },
            'performance_stats': {
                'mean_dice': float(successful_metrics['dice'].mean()) if len(successful_metrics) > 0 else 0.0,
                'std_dice': float(successful_metrics['dice'].std()) if len(successful_metrics) > 0 else 0.0,
                'mean_jaccard': float(successful_metrics['jaccard'].mean()) if len(successful_metrics) > 0 else 0.0,
                'mean_hausdorff': float(successful_metrics['hausdorff'].mean()) if len(successful_metrics) > 0 else 0.0,
            }
        }
        
        with open(arch_output_dir / 'prediction_info.json', 'w') as f:
            json.dump(summary, f, indent=2)
    
    print(f"\n✅ Prediction generation completed!")
    print(f"📁 Output directory: {output_dir}")
    print(f"🔗 Ready for graph extraction!")
    
    # Show usage instructions
    print("\n📋 NEXT STEPS:")
    if best_only and best_arch:
        print(f"python scripts/extract_predicted_graphs.py \\")
        print(f"    --predictions-dir {output_dir}/{best_arch} \\")
        print(f"    --graphs-dir extracted_graphs \\")
        print(f"    --threshold 0.5")
    else:
        print("Choose the best architecture and run:")
        print("python scripts/extract_predicted_graphs.py --predictions-dir <output_dir>/<arch_name> --graphs-dir extracted_graphs")


def predict_2d_volume(model, volume, device):
    """Predict 3D volume using 2D model slice by slice"""
    prediction = np.zeros_like(volume)
    
    with torch.no_grad():
        for z in range(volume.shape[0]):
            slice_2d = volume[z]
            slice_tensor = torch.from_numpy(slice_2d).unsqueeze(0).unsqueeze(0).float().to(device)
            output = model(slice_tensor)
            pred_slice = torch.sigmoid(output).cpu().numpy()[0, 0]
            prediction[z] = pred_slice
    
    return prediction


def main():
    parser = argparse.ArgumentParser(description='Generate predictions from trained U-Net models with per-case analysis')
    
    # Mode selection: either use results directory OR manual model path
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--results-dir', type=str,
                           help='Directory containing training results from train_multiple_unets.py')
    mode_group.add_argument('--model-path', type=str,
                           help='Path to specific model checkpoint (.pth file)')
    
    # Required when using manual model path
    parser.add_argument('--architecture', type=str, 
                        choices=['unet3d', 'attention_unet3d', 'unet2d'],
                        help='Model architecture (required when using --model-path)')
    
    parser.add_argument('--dataset', type=str, default='DATASET/Parse_dataset',
                        help='Path to dataset directory')
    parser.add_argument('--output-dir', type=str, default='experiments/predictions',
                        help='Output directory for predictions')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for inference')
    parser.add_argument('--best-only', action='store_true',
                        help='Only generate predictions from the best performing architecture (only for --results-dir mode)')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='GPU ID to use (0 or 1)')
    
    # Filtering criteria
    parser.add_argument('--min-dice', type=float, default=0.6,
                        help='Minimum Dice score for good cases (default: 0.6)')
    parser.add_argument('--max-hausdorff', type=float, default=50,
                        help='Maximum Hausdorff distance for good cases (default: 50)')
    parser.add_argument('--patient-ids', nargs='+',
                        help='Specific patient IDs to process')
    parser.add_argument('--skip-analysis', action='store_true',
                        help='Skip per-case metrics analysis (only generate predictions)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.model_path and not args.architecture:
        parser.error("--architecture is required when using --model-path")
    
    if args.best_only and args.model_path:
        parser.error("--best-only can only be used with --results-dir mode")
    
    # Set device with specific GPU ID
    if torch.cuda.is_available():
        device = f'cuda:{args.gpu_id}'
        print(f"Using GPU {args.gpu_id}")
    else:
        device = 'cpu'
        print("CUDA not available, using CPU")
    
    # Generate predictions based on mode
    if args.model_path:
        # Manual model mode
        generate_predictions_from_manual_model(
            args.model_path,
            args.architecture,
            args.dataset,
            args.output_dir,
            device,
            args.min_dice,
            args.max_hausdorff,
            args.patient_ids,
            args.skip_analysis
        )
    else:
        # Results directory mode
        generate_predictions_from_cv_results(
            args.results_dir,
            args.dataset,
            args.output_dir,
            device,
            args.best_only,
            args.min_dice,
            args.max_hausdorff,
            args.patient_ids
        )


if __name__ == '__main__':
    main()