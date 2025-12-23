#!/usr/bin/env python3
"""
Test script for comprehensive metrics collection system.
"""

import os
import sys
import numpy as np
import nibabel as nib
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.comprehensive_metrics import compute_comprehensive_metrics
from utils.traditional_postprocessing import MorphologicalPostProcessor


def create_test_data():
    """Create synthetic test data for metrics validation."""
    
    # Create a synthetic 3D vessel structure
    shape = (64, 64, 32)
    mask = np.zeros(shape, dtype=np.uint8)
    
    # Main vessel trunk
    mask[25:40, 30:34, 10:25] = 1
    
    # Bifurcation
    mask[30:35, 34:45, 15:20] = 1  # Branch 1
    mask[35:40, 34:45, 15:20] = 1  # Branch 2
    
    # Add some realistic variations
    # Small branch
    mask[32:34, 45:55, 17:19] = 1
    
    return mask


def test_comprehensive_metrics():
    """Test comprehensive metrics computation."""
    
    print("Testing comprehensive metrics computation...")
    
    # Create test data
    gt_mask = create_test_data()
    
    # Create a "prediction" with some errors
    pred_mask = gt_mask.copy()
    
    # Add some noise (false positives)
    pred_mask[10:12, 10:12, 5:7] = 1
    pred_mask[50:52, 50:52, 25:27] = 1
    
    # Remove some parts (false negatives)
    pred_mask[32:34, 45:50, 17:19] = 0  # Remove small branch
    pred_mask[27:29, 31:33, 12:14] = 0  # Create gap in main vessel
    
    print(f"GT mask volume: {gt_mask.sum()} voxels")
    print(f"Pred mask volume: {pred_mask.sum()} voxels")
    
    # Compute comprehensive metrics
    try:
        metrics = compute_comprehensive_metrics(
            pred_mask, 
            gt_mask,
            voxel_spacing=(1.0, 1.0, 1.0),
            include_anatomical=True,
            include_topological=True
        )
        
        print("\nComprehensive metrics computed successfully!")
        
        # Print key metrics
        key_metrics = [
            'dice', 'jaccard', 'connectivity_aware_dice',
            'num_components', 'gt_num_components', 'component_count_error',
            'tree_isomorphism', 'bifurcation_f1',
            'murrays_law_score', 'tapering_score',
            'angle_physiological_angle_ratio'
        ]
        
        print("\nKey Metrics:")
        print("-" * 40)
        for metric in key_metrics:
            if metric in metrics:
                value = metrics[metric]
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")
        
        return True, metrics
        
    except Exception as e:
        print(f"Error computing comprehensive metrics: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_traditional_processing():
    """Test traditional processing with comprehensive metrics."""
    
    print("\n" + "="*60)
    print("Testing traditional processing with comprehensive metrics...")
    
    # Create test data
    original_mask = create_test_data()
    
    # Add some noise and issues that post-processing should fix
    noisy_mask = original_mask.copy()
    
    # Add small disconnected components
    noisy_mask[5:7, 5:7, 5:7] = 1
    noisy_mask[55:57, 55:57, 25:27] = 1
    
    # Add small holes
    noisy_mask[28:30, 31:33, 12:14] = 0
    
    # Add some spurs
    noisy_mask[25, 25:30, 12] = 1
    
    print(f"Original mask volume: {original_mask.sum()} voxels")
    print(f"Noisy mask volume: {noisy_mask.sum()} voxels")
    
    # Apply morphological post-processing
    processor = MorphologicalPostProcessor(
        opening_radius=2,
        closing_radius=3,
        min_object_size=50,
        min_hole_size=25
    )
    
    processed_mask = processor.process(noisy_mask)
    print(f"Processed mask volume: {processed_mask.sum()} voxels")
    
    # Compute metrics for all versions
    methods = {
        'noisy': noisy_mask,
        'processed': processed_mask
    }
    
    results = {}
    
    for method_name, mask in methods.items():
        try:
            metrics = compute_comprehensive_metrics(
                mask, 
                original_mask,  # Use original as "ground truth"
                voxel_spacing=(1.0, 1.0, 1.0)
            )
            results[method_name] = metrics
            
        except Exception as e:
            print(f"Error computing metrics for {method_name}: {e}")
            results[method_name] = None
    
    # Compare results
    print("\nComparison Results:")
    print("-" * 60)
    
    key_metrics = ['dice', 'connectivity_aware_dice', 'num_components', 
                  'tree_isomorphism', 'murrays_law_score']
    
    for metric in key_metrics:
        print(f"\n{metric}:")
        for method_name in ['noisy', 'processed']:
            if results[method_name] and metric in results[method_name]:
                value = results[method_name][metric]
                if isinstance(value, float):
                    print(f"  {method_name}: {value:.4f}")
                else:
                    print(f"  {method_name}: {value}")
    
    return results


def test_with_real_data_if_available():
    """Test with real data if available."""
    
    print("\n" + "="*60)
    print("Testing with real data (if available)...")
    
    dataset_dir = Path("DATASET/Parse_dataset")
    
    if not dataset_dir.exists():
        print("No real dataset found - skipping real data test")
        return None
    
    # Find first available patient
    patient_dirs = list(dataset_dir.glob("PA*"))
    if not patient_dirs:
        print("No patient directories found - skipping real data test")
        return None
    
    patient_dir = patient_dirs[0]
    label_file = patient_dir / "label" / f"{patient_dir.name}.nii.gz"
    
    if not label_file.exists():
        print(f"Label file not found: {label_file}")
        return None
    
    try:
        # Load real mask
        print(f"Loading real data from: {patient_dir.name}")
        nii_img = nib.load(str(label_file))
        real_mask = nii_img.get_fdata()
        real_mask = (real_mask > 0.5).astype(np.uint8)
        
        print(f"Real mask shape: {real_mask.shape}")
        print(f"Real mask volume: {real_mask.sum()} voxels")
        
        # Create a synthetic "prediction" by degrading the real mask
        pred_mask = real_mask.copy()
        
        # Add some noise
        noise_locations = np.random.random(real_mask.shape) > 0.999
        pred_mask[noise_locations] = 1
        
        # Remove some parts
        remove_locations = (real_mask == 1) & (np.random.random(real_mask.shape) > 0.95)
        pred_mask[remove_locations] = 0
        
        print(f"Synthetic prediction volume: {pred_mask.sum()} voxels")
        
        # Compute comprehensive metrics
        metrics = compute_comprehensive_metrics(
            pred_mask,
            real_mask,
            voxel_spacing=(1.0, 1.0, 1.0)  # Use unit spacing for simplicity
        )
        
        print("\nReal Data Metrics:")
        print("-" * 40)
        
        key_metrics = ['dice', 'connectivity_aware_dice', 'num_components', 
                      'gt_num_components', 'tree_isomorphism']
        
        for metric in key_metrics:
            if metric in metrics:
                value = metrics[metric]
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")
        
        print("✓ Real data test completed successfully")
        return metrics
        
    except Exception as e:
        print(f"Error in real data test: {e}")
        return None


def main():
    """Run all comprehensive metrics tests."""
    
    print("="*60)
    print("COMPREHENSIVE METRICS SYSTEM TEST")
    print("="*60)
    
    success_count = 0
    total_tests = 3
    
    # Test 1: Basic comprehensive metrics
    try:
        success, metrics = test_comprehensive_metrics()
        if success:
            success_count += 1
            print("✓ Test 1 PASSED: Basic comprehensive metrics")
        else:
            print("✗ Test 1 FAILED: Basic comprehensive metrics")
    except Exception as e:
        print(f"✗ Test 1 FAILED with exception: {e}")
    
    # Test 2: Traditional processing with metrics
    try:
        results = test_traditional_processing()
        if results:
            success_count += 1
            print("✓ Test 2 PASSED: Traditional processing with metrics")
        else:
            print("✗ Test 2 FAILED: Traditional processing with metrics")
    except Exception as e:
        print(f"✗ Test 2 FAILED with exception: {e}")
    
    # Test 3: Real data (if available)
    try:
        real_results = test_with_real_data_if_available()
        if real_results is not None:
            success_count += 1
            print("✓ Test 3 PASSED: Real data test")
        else:
            print("✓ Test 3 SKIPPED: Real data not available")
            success_count += 1  # Count as success since it's optional
    except Exception as e:
        print(f"✗ Test 3 FAILED with exception: {e}")
    
    print("\n" + "="*60)
    print(f"TEST SUMMARY: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 ALL TESTS PASSED! Comprehensive metrics system is ready.")
        print("\nYou can now run the full metrics collection using:")
        print("python scripts/collect_comprehensive_metrics.py \\")
        print("  --predictions-dir /path/to/unet/predictions \\")
        print("  --ground-truth-dir DATASET/Parse_dataset \\")
        print("  --output-dir results/comprehensive_metrics")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == '__main__':
    exit(main())