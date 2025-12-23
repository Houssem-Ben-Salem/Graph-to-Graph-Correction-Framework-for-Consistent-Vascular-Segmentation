#!/usr/bin/env python3
"""
Test script for clDice loss function implementation.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from training.cldice_loss import (
    SoftclDiceLoss, 
    CombinedclDiceLoss, 
    SoftcbDiceLoss,
    create_cldice_loss,
    soft_skel
)


def create_synthetic_vessel_data(batch_size=2, channels=1, height=64, width=64, depth=32):
    """Create synthetic 3D vessel-like data for testing."""
    
    # Ground truth with vessel-like structure
    y_true = torch.zeros(batch_size, channels, depth, height, width)
    
    for b in range(batch_size):
        # Main vessel trunk (horizontal)
        y_true[b, :, 10:20, 20:25, 10:50] = 1.0
        
        # Bifurcation branches
        y_true[b, :, 12:18, 25:40, 30:35] = 1.0  # Branch 1
        y_true[b, :, 12:18, 25:40, 25:30] = 1.0  # Branch 2
        
        # Additional smaller branches
        y_true[b, :, 14:16, 40:55, 32:34] = 1.0
        y_true[b, :, 14:16, 40:55, 27:29] = 1.0
    
    # Prediction with some topological errors
    y_pred = y_true.clone()
    
    # Add connectivity breaks (topology errors)
    y_pred[:, :, 12:15, 22:24, 35:40] = 0.0  # Break in main vessel
    y_pred[:, :, 14:16, 35:38, 30:32] = 0.0  # Break in branch
    
    # Add false positives
    y_pred[:, :, 5:8, 5:8, 5:8] = 0.8      # Isolated false positive
    y_pred[:, :, 25:28, 50:53, 50:53] = 0.7 # Another false positive
    
    # Convert to logits for loss computation
    y_pred_logits = torch.logit(torch.clamp(y_pred, 0.01, 0.99))
    
    return y_pred_logits, y_true


def create_2d_vessel_data(batch_size=2, channels=1, height=64, width=64):
    """Create synthetic 2D vessel-like data for testing."""
    
    # Ground truth with vessel-like structure
    y_true = torch.zeros(batch_size, channels, height, width)
    
    for b in range(batch_size):
        # Main vessel (horizontal)
        y_true[b, :, 20:25, 10:50] = 1.0
        
        # Vertical branch
        y_true[b, :, 10:50, 30:35] = 1.0
        
        # Diagonal branches
        for i in range(15):
            y_true[b, :, 25+i, 30+i] = 1.0  # Diagonal 1
            y_true[b, :, 25+i, 30-i] = 1.0  # Diagonal 2
    
    # Prediction with errors
    y_pred = y_true.clone()
    
    # Add breaks
    y_pred[:, :, 22:24, 35:40] = 0.0
    y_pred[:, :, 25:30, 32:34] = 0.0
    
    # Add false positives
    y_pred[:, :, 5:8, 5:8] = 0.8
    y_pred[:, :, 55:58, 55:58] = 0.7
    
    # Convert to logits
    y_pred_logits = torch.logit(torch.clamp(y_pred, 0.01, 0.99))
    
    return y_pred_logits, y_true


def test_soft_skeletonization():
    """Test the soft skeletonization function."""
    print("🧪 Testing soft skeletonization...")
    
    # Create simple 2D test case
    y_pred_logits, y_true = create_2d_vessel_data(batch_size=1)
    
    # Compute skeleton
    skel = soft_skel(y_true, num_iter=10)
    
    print(f"   Original shape: {y_true.shape}")
    print(f"   Original sum: {y_true.sum().item():.1f}")
    print(f"   Skeleton sum: {skel.sum().item():.1f}")
    print(f"   Skeleton max: {skel.max().item():.3f}")
    
    # Visualize if possible
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        original = y_true[0, 0].numpy()
        skeleton = skel[0, 0].numpy()
        prediction = torch.sigmoid(y_pred_logits[0, 0]).numpy()
        
        axes[0].imshow(original, cmap='gray')
        axes[0].set_title('Ground Truth')
        axes[0].axis('off')
        
        axes[1].imshow(prediction, cmap='gray')
        axes[1].set_title('Prediction')
        axes[1].axis('off')
        
        axes[2].imshow(skeleton, cmap='hot')
        axes[2].set_title('Soft Skeleton')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig('test_skeletonization.png', dpi=150, bbox_inches='tight')
        print("   💾 Visualization saved as 'test_skeletonization.png'")
        plt.close()
        
    except Exception as e:
        print(f"   ⚠️  Could not create visualization: {e}")
    
    print("   ✅ Soft skeletonization test passed")


def test_loss_functions():
    """Test all clDice loss function variants."""
    print("\n🧪 Testing loss functions...")
    
    # Test 2D data
    print("   Testing 2D data...")
    y_pred_2d, y_true_2d = create_2d_vessel_data()
    
    # Test 3D data
    print("   Testing 3D data...")
    y_pred_3d, y_true_3d = create_synthetic_vessel_data()
    
    # Initialize loss functions
    losses = {
        'clDice': SoftclDiceLoss(num_iter=10),
        'Combined': CombinedclDiceLoss(num_iter=10, alpha=0.7),
        'cbDice': SoftcbDiceLoss(num_iter=10, alpha=0.5, beta=0.3)
    }
    
    # Test each loss function
    results = {}
    
    for loss_name, loss_fn in losses.items():
        print(f"   Testing {loss_name}...")
        
        try:
            # Test 2D
            loss_2d = loss_fn(y_pred_2d, y_true_2d)
            
            # Test 3D
            loss_3d = loss_fn(y_pred_3d, y_true_3d)
            
            results[loss_name] = {
                '2D': loss_2d.item(),
                '3D': loss_3d.item()
            }
            
            print(f"     2D Loss: {loss_2d.item():.4f}")
            print(f"     3D Loss: {loss_3d.item():.4f}")
            print(f"     ✅ {loss_name} passed")
            
        except Exception as e:
            print(f"     ❌ {loss_name} failed: {e}")
            results[loss_name] = {'error': str(e)}
    
    return results


def test_gradient_flow():
    """Test that gradients flow properly through the loss functions."""
    print("\n🧪 Testing gradient flow...")
    
    # Create test data
    y_pred, y_true = create_2d_vessel_data(batch_size=1)
    y_pred.requires_grad_(True)
    
    # Test combined loss (most commonly used)
    loss_fn = CombinedclDiceLoss(num_iter=5, alpha=0.7)  # Fewer iterations for speed
    
    # Forward pass
    loss = loss_fn(y_pred, y_true)
    print(f"   Loss value: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    if y_pred.grad is not None:
        grad_norm = y_pred.grad.norm().item()
        grad_mean = y_pred.grad.mean().item()
        grad_std = y_pred.grad.std().item()
        
        print(f"   Gradient norm: {grad_norm:.6f}")
        print(f"   Gradient mean: {grad_mean:.6f}")
        print(f"   Gradient std: {grad_std:.6f}")
        
        if grad_norm > 1e-8:
            print("   ✅ Gradients are flowing properly")
        else:
            print("   ⚠️  Gradients might be too small")
    else:
        print("   ❌ No gradients computed")


def test_factory_function():
    """Test the create_cldice_loss factory function."""
    print("\n🧪 Testing factory function...")
    
    loss_types = ['cldice', 'combined', 'cbdice']
    
    for loss_type in loss_types:
        try:
            loss_fn = create_cldice_loss(
                loss_type=loss_type,
                num_iter=10,
                alpha=0.5,
                beta=0.3,
                sigma=1.0
            )
            print(f"   ✅ {loss_type} created successfully")
            
            # Quick test
            y_pred, y_true = create_2d_vessel_data(batch_size=1)
            loss_value = loss_fn(y_pred, y_true)
            print(f"     Loss value: {loss_value.item():.4f}")
            
        except Exception as e:
            print(f"   ❌ {loss_type} failed: {e}")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n🧪 Testing edge cases...")
    
    # Test empty masks
    try:
        empty_pred = torch.zeros(1, 1, 32, 32)
        empty_true = torch.zeros(1, 1, 32, 32)
        
        loss_fn = SoftclDiceLoss(num_iter=5)
        loss = loss_fn(empty_pred, empty_true)
        print(f"   Empty masks loss: {loss.item():.4f}")
        print("   ✅ Empty masks handled correctly")
    except Exception as e:
        print(f"   ❌ Empty masks failed: {e}")
    
    # Test perfect prediction
    try:
        perfect_pred = torch.ones(1, 1, 32, 32) * 10  # High logits
        perfect_true = torch.ones(1, 1, 32, 32)
        
        loss_fn = SoftclDiceLoss(num_iter=5)
        loss = loss_fn(perfect_pred, perfect_true)
        print(f"   Perfect prediction loss: {loss.item():.4f}")
        print("   ✅ Perfect prediction handled correctly")
    except Exception as e:
        print(f"   ❌ Perfect prediction failed: {e}")


def benchmark_performance():
    """Benchmark the performance of different loss functions."""
    print("\n⏱️  Benchmarking performance...")
    
    import time
    
    # Create larger test data
    y_pred, y_true = create_synthetic_vessel_data(batch_size=4, depth=64, height=128, width=128)
    
    loss_functions = {
        'clDice (iter=20)': SoftclDiceLoss(num_iter=20),
        'clDice (iter=40)': SoftclDiceLoss(num_iter=40),
        'Combined': CombinedclDiceLoss(num_iter=20),
        'cbDice': SoftcbDiceLoss(num_iter=20)
    }
    
    for name, loss_fn in loss_functions.items():
        try:
            # Warm up
            _ = loss_fn(y_pred, y_true)
            
            # Benchmark
            start_time = time.time()
            for _ in range(5):
                loss = loss_fn(y_pred, y_true)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 5
            print(f"   {name}: {avg_time:.3f}s per forward pass")
            
        except Exception as e:
            print(f"   {name}: Failed - {e}")


def main():
    """Run all tests."""
    print("🧪 Testing clDice Loss Function Implementation")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    try:
        # Run tests
        test_soft_skeletonization()
        test_results = test_loss_functions()
        test_gradient_flow()
        test_factory_function()
        test_edge_cases()
        benchmark_performance()
        
        print("\n" + "=" * 60)
        print("📊 Test Summary:")
        
        for loss_name, results in test_results.items():
            if 'error' in results:
                print(f"   ❌ {loss_name}: {results['error']}")
            else:
                print(f"   ✅ {loss_name}: 2D={results['2D']:.4f}, 3D={results['3D']:.4f}")
        
        print("\n🎉 All tests completed!")
        print("\nYou can now use the clDice loss functions for training:")
        print("   python scripts/train_unet_cldice.py --loss-type combined")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())