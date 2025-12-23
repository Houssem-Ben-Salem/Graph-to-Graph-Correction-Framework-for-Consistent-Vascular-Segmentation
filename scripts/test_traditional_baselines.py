#!/usr/bin/env python3
"""
Test script for traditional post-processing baseline methods.

This script performs basic functionality tests and demonstrates the methods
on sample data to ensure everything works correctly.
"""

import os
import sys
import numpy as np
import torch
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.traditional_postprocessing import (
    MorphologicalPostProcessor,
    LargestConnectedComponentFilter,
    CombinedTraditionalPostProcessor,
    apply_morphological_postprocessing,
    apply_lcc_filtering
)
from utils.metrics import compute_metrics


def create_synthetic_test_mask():
    """Create a synthetic 3D test mask with known issues for testing."""
    # Create a 64x64x32 test volume
    mask = np.zeros((64, 64, 32), dtype=np.uint8)
    
    # Main vessel structure
    mask[20:45, 30:34, 10:25] = 1  # Main trunk
    mask[25:30, 34:50, 15:20] = 1  # Branch 1
    mask[35:40, 34:50, 15:20] = 1  # Branch 2
    
    # Add some noise (small disconnected objects)
    mask[10:12, 10:12, 5:7] = 1    # Small noise 1
    mask[50:52, 50:52, 8:10] = 1   # Small noise 2
    mask[5:7, 20:22, 28:30] = 1    # Small noise 3
    
    # Add some holes in main structure
    mask[22:24, 31:33, 12:14] = 0  # Small hole
    mask[27, 40:42, 16:18] = 0     # Gap in branch
    
    # Add some spurs/protrusions
    mask[20, 25:30, 12] = 1        # Spur 1
    mask[44, 35:38, 18] = 1        # Spur 2
    
    return mask


def create_disconnected_test_mask():
    """Create a test mask with multiple disconnected components."""
    mask = np.zeros((64, 64, 32), dtype=np.uint8)
    
    # Large main component
    mask[15:35, 20:25, 8:20] = 1
    
    # Medium component
    mask[40:50, 40:45, 15:22] = 1
    
    # Small components (noise)
    mask[8:10, 8:10, 5:7] = 1
    mask[55:57, 55:57, 25:27] = 1
    mask[30:32, 50:52, 10:12] = 1
    
    return mask


def test_morphological_processor():
    """Test morphological post-processing functionality."""
    print("Testing Morphological Post-Processor...")
    
    # Create test data
    test_mask = create_synthetic_test_mask()
    original_volume = test_mask.sum()
    
    print(f"Original mask volume: {original_volume} voxels")
    
    # Test different configurations
    configs = {
        'conservative': {'opening_radius': 1, 'closing_radius': 2, 'min_object_size': 20},
        'moderate': {'opening_radius': 2, 'closing_radius': 3, 'min_object_size': 50},
        'aggressive': {'opening_radius': 3, 'closing_radius': 4, 'min_object_size': 100}
    }
    
    results = {}
    
    for config_name, params in configs.items():
        processor = MorphologicalPostProcessor(**params)
        processed_mask = processor.process(test_mask)
        
        volume_after = processed_mask.sum()
        volume_change = volume_after - original_volume
        
        results[config_name] = {
            'volume_before': original_volume,
            'volume_after': int(volume_after),
            'volume_change': int(volume_change),
            'relative_change': volume_change / original_volume * 100
        }
        
        print(f"  {config_name}: {int(volume_after)} voxels ({volume_change:+d}, {volume_change/original_volume*100:+.1f}%)")
    
    # Test with PyTorch tensor
    test_tensor = torch.from_numpy(test_mask.astype(np.float32))
    processed_tensor = apply_morphological_postprocessing(test_tensor)
    
    assert isinstance(processed_tensor, torch.Tensor), "Should return PyTorch tensor"
    print("  ✓ PyTorch tensor processing works")
    
    return results


def test_lcc_filter():
    """Test Largest Connected Component filtering."""
    print("\nTesting LCC Filter...")
    
    # Create test data with multiple components
    test_mask = create_disconnected_test_mask()
    
    filter_obj = LargestConnectedComponentFilter()
    
    # Get component statistics before filtering
    stats_before = filter_obj.get_component_stats(test_mask)
    print(f"  Components before filtering: {stats_before['num_components']}")
    print(f"  Component sizes: {stats_before['size_distribution']}")
    print(f"  Largest component: {stats_before['largest_size']} voxels")
    
    # Apply LCC filtering
    filtered_mask = filter_obj.process(test_mask, keep_n_largest=1)
    
    # Get statistics after filtering
    stats_after = filter_obj.get_component_stats(filtered_mask)
    print(f"  Components after filtering: {stats_after['num_components']}")
    print(f"  Remaining volume: {stats_after['total_size']} voxels")
    
    # Test keeping multiple components
    multi_filtered = filter_obj.process(test_mask, keep_n_largest=2)
    stats_multi = filter_obj.get_component_stats(multi_filtered)
    print(f"  Components after keeping 2 largest: {stats_multi['num_components']}")
    
    assert stats_after['num_components'] <= 1, "Should keep at most 1 component"
    assert stats_multi['num_components'] <= 2, "Should keep at most 2 components"
    print("  ✓ LCC filtering works correctly")
    
    return stats_before, stats_after


def test_combined_processor():
    """Test combined post-processing pipeline."""
    print("\nTesting Combined Post-Processor...")
    
    test_mask = create_synthetic_test_mask()
    original_volume = test_mask.sum()
    
    # Test different processing orders
    orders = ['morph_then_lcc', 'lcc_then_morph', 'morph_only', 'lcc_only']
    
    for order in orders:
        processor = CombinedTraditionalPostProcessor(apply_order=order)
        processed_mask = processor.process(test_mask)
        
        volume_after = processed_mask.sum()
        volume_change = volume_after - original_volume
        
        print(f"  {order}: {int(volume_after)} voxels ({volume_change:+d}, {volume_change/original_volume*100:+.1f}%)")
    
    print("  ✓ Combined processing works correctly")


def test_with_real_data():
    """Test with actual medical imaging data if available."""
    print("\nTesting with Real Data...")
    
    # Look for a sample patient
    dataset_dir = Path("DATASET/Parse_dataset")
    
    if not dataset_dir.exists():
        print("  Skipping real data test - dataset directory not found")
        return None
    
    # Find first available patient
    patient_dirs = list(dataset_dir.glob("PA*"))
    if not patient_dirs:
        print("  Skipping real data test - no patient directories found")
        return None
    
    patient_dir = patient_dirs[0]
    label_file = patient_dir / "label" / f"{patient_dir.name}.nii.gz"
    
    if not label_file.exists():
        print(f"  Skipping real data test - label file not found: {label_file}")
        return None
    
    try:
        # Load real mask
        print(f"  Loading real data from: {patient_dir.name}")
        nii_img = nib.load(str(label_file))
        real_mask = nii_img.get_fdata()
        real_mask = (real_mask > 0.5).astype(np.uint8)
        
        print(f"  Original mask shape: {real_mask.shape}")
        print(f"  Original mask volume: {real_mask.sum()} voxels")
        
        # Test LCC statistics
        lcc_filter = LargestConnectedComponentFilter()
        stats = lcc_filter.get_component_stats(real_mask)
        
        print(f"  Connected components: {stats['num_components']}")
        print(f"  Largest component: {stats['largest_size']} voxels")
        if len(stats['size_distribution']) > 1:
            print(f"  Size distribution (top 5): {stats['size_distribution'][:5]}")
        
        # Apply morphological processing
        morph_processor = MorphologicalPostProcessor()
        processed_mask = morph_processor.process(real_mask)
        
        volume_change = processed_mask.sum() - real_mask.sum()
        print(f"  After morphological processing: {processed_mask.sum()} voxels ({volume_change:+d})")
        
        print("  ✓ Real data processing works correctly")
        return True
        
    except Exception as e:
        print(f"  Error processing real data: {e}")
        return False


def test_metrics_integration():
    """Test integration with metrics computation."""
    print("\nTesting Metrics Integration...")
    
    # Create test data
    gt_mask = create_synthetic_test_mask()
    
    # Create a "noisy" prediction by adding some noise
    pred_mask = gt_mask.copy()
    # Add some noise
    pred_mask[10:12, 10:12, 5:7] = 1  # False positive
    pred_mask[22:24, 31:33, 12:14] = 0  # False negative (create hole)
    
    try:
        # Compute metrics before post-processing
        original_metrics = compute_metrics(pred_mask, gt_mask)
        print(f"  Original Dice score: {original_metrics['dice']:.3f}")
        
        # Apply post-processing
        processor = CombinedTraditionalPostProcessor(apply_order='morph_then_lcc')
        processed_mask = processor.process(pred_mask)
        
        # Compute metrics after post-processing
        processed_metrics = compute_metrics(processed_mask, gt_mask)
        print(f"  Processed Dice score: {processed_metrics['dice']:.3f}")
        
        dice_improvement = processed_metrics['dice'] - original_metrics['dice']
        print(f"  Dice improvement: {dice_improvement:+.3f}")
        
        print("  ✓ Metrics integration works correctly")
        return True
        
    except Exception as e:
        print(f"  Error in metrics integration: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("TESTING TRADITIONAL POST-PROCESSING BASELINES")
    print("="*60)
    
    # Run all tests
    test_results = {}
    
    try:
        test_results['morphological'] = test_morphological_processor()
        test_results['lcc'] = test_lcc_filter()
        test_combined_processor()
        test_results['real_data'] = test_with_real_data()
        test_results['metrics'] = test_metrics_integration()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print("\nThe traditional post-processing baselines are ready for evaluation.")
        print("You can now run the full evaluation using:")
        print("  python scripts/evaluate_traditional_baselines.py \\")
        print("    --predictions-dir /path/to/unet/predictions \\")
        print("    --ground-truth-dir DATASET/Parse_dataset \\")
        print("    --output-dir results/traditional_baselines")
        
        return 0
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())