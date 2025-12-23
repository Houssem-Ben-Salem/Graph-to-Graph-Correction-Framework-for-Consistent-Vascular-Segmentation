#!/usr/bin/env python
"""Quick check of training results"""

import json
import argparse
from pathlib import Path


def quick_check(results_dir):
    """Quick check of training results"""
    results_dir = Path(results_dir)
    
    # Check if results exist
    all_results_file = results_dir / 'all_results.json'
    if not all_results_file.exists():
        print(f"❌ No results found at {results_dir}")
        print(f"Expected file: {all_results_file}")
        return
    
    # Load results
    with open(all_results_file, 'r') as f:
        results = json.load(f)
    
    print(f"📊 Training Results from: {results_dir}")
    print("=" * 50)
    
    best_dice = 0
    best_arch = None
    total_trained = 0
    
    for arch_name, result in results.items():
        if result is None:
            print(f"❌ {arch_name}: FAILED")
            continue
            
        if 'average_dice' not in result:
            print(f"⚠️  {arch_name}: INCOMPLETE")
            continue
        
        dice = result['average_dice']
        std = result['std_dice']
        total_trained += 1
        
        if dice > best_dice:
            best_dice = dice
            best_arch = arch_name
        
        # Simple status
        if dice >= 0.80:
            status = "🟢 EXCELLENT"
        elif dice >= 0.70:
            status = "🟡 GOOD"
        else:
            status = "🔴 NEEDS WORK"
        
        print(f"{status} {arch_name}: {dice:.4f} ± {std:.4f}")
    
    print("=" * 50)
    
    if total_trained == 0:
        print("❌ No architectures completed successfully")
        return
    
    print(f"🏆 BEST: {best_arch} ({best_dice:.4f})")
    
    # Simple recommendation
    if best_dice >= 0.80:
        print("✅ READY FOR GRAPH CORRECTION!")
        print(f"Use predictions from: {results_dir}/cv_predictions/{best_arch}/")
    elif best_dice >= 0.70:
        print("⚠️  Consider improvements or proceed carefully")
    else:
        print("🔴 Improve segmentation before proceeding")
    
    # Check if predictions exist
    cv_pred_dir = results_dir / 'cv_predictions'
    if cv_pred_dir.exists() and best_arch:
        arch_pred_dir = cv_pred_dir / best_arch
        if arch_pred_dir.exists():
            pred_files = list(arch_pred_dir.glob('*.npy'))
            print(f"📁 Predictions ready: {len(pred_files)} files")
        else:
            print("❌ Predictions not generated yet")
    else:
        print("❌ No predictions directory found")


def main():
    parser = argparse.ArgumentParser(description='Quick check of training results')
    parser.add_argument('results_dir', type=str, nargs='?', 
                        default='experiments/multi_unet',
                        help='Path to results directory')
    
    args = parser.parse_args()
    quick_check(args.results_dir)


if __name__ == '__main__':
    main()