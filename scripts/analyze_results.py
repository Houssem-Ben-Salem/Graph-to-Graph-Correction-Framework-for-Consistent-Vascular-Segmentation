#!/usr/bin/env python
"""Analyze training results and provide recommendations"""

import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


def load_results(results_dir):
    """Load all results from experiment directory"""
    results_dir = Path(results_dir)
    
    # Load main results
    all_results_file = results_dir / 'all_results.json'
    if not all_results_file.exists():
        print(f"❌ No results found at {all_results_file}")
        return None
    
    with open(all_results_file, 'r') as f:
        results = json.load(f)
    
    return results, results_dir


def analyze_architecture_performance(results):
    """Analyze performance of each architecture"""
    print("\n" + "="*60)
    print("ARCHITECTURE PERFORMANCE ANALYSIS")
    print("="*60)
    
    performance_data = []
    
    for arch_name, result in results.items():
        if result is None:
            print(f"❌ {arch_name}: FAILED - No results")
            continue
            
        if 'average_dice' not in result:
            print(f"⚠️  {arch_name}: INCOMPLETE - No Dice score")
            continue
        
        dice = result['average_dice']
        std = result['std_dice']
        
        performance_data.append({
            'Architecture': arch_name,
            'Dice': dice,
            'Std': std,
            'Status': 'Success'
        })
        
        # Performance evaluation
        if dice >= 0.85:
            status = "🟢 EXCELLENT"
        elif dice >= 0.75:
            status = "🟡 GOOD"
        elif dice >= 0.65:
            status = "🟠 FAIR"
        else:
            status = "🔴 POOR"
        
        print(f"{status} {arch_name:15}: {dice:.4f} ± {std:.4f}")
    
    return performance_data


def analyze_fold_consistency(results, results_dir):
    """Analyze consistency across folds"""
    print("\n" + "="*60)
    print("FOLD CONSISTENCY ANALYSIS")
    print("="*60)
    
    for arch_name, result in results.items():
        if result is None or 'fold_results' not in result:
            continue
            
        print(f"\n📊 {arch_name} Fold Analysis:")
        
        fold_scores = [fold['val_dice'] for fold in result['fold_results']]
        
        print(f"  Fold Scores: {[f'{score:.4f}' for score in fold_scores]}")
        print(f"  Mean:        {np.mean(fold_scores):.4f}")
        print(f"  Std:         {np.std(fold_scores):.4f}")
        print(f"  Min:         {np.min(fold_scores):.4f}")
        print(f"  Max:         {np.max(fold_scores):.4f}")
        print(f"  Range:       {np.max(fold_scores) - np.min(fold_scores):.4f}")
        
        # Consistency evaluation
        std_dev = np.std(fold_scores)
        if std_dev <= 0.02:
            consistency = "🟢 VERY CONSISTENT"
        elif std_dev <= 0.05:
            consistency = "🟡 MODERATELY CONSISTENT"
        else:
            consistency = "🔴 INCONSISTENT"
        
        print(f"  Consistency: {consistency}")


def provide_recommendations(results):
    """Provide recommendations based on results"""
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    performance_data = []
    for arch_name, result in results.items():
        if result and 'average_dice' in result:
            performance_data.append({
                'name': arch_name,
                'dice': result['average_dice'],
                'std': result['std_dice']
            })
    
    if not performance_data:
        print("❌ No successful architectures to analyze")
        return
    
    # Sort by performance
    performance_data.sort(key=lambda x: x['dice'], reverse=True)
    
    best_arch = performance_data[0]
    
    print(f"\n🏆 BEST PERFORMING ARCHITECTURE: {best_arch['name']}")
    print(f"   Dice Score: {best_arch['dice']:.4f} ± {best_arch['std']:.4f}")
    
    # Decision recommendations
    print(f"\n🎯 DECISION RECOMMENDATIONS:")
    
    if best_arch['dice'] >= 0.80:
        print("✅ PROCEED TO GRAPH CORRECTION")
        print("   Your best model shows excellent segmentation performance.")
        print("   You can confidently move to the graph correction pipeline.")
        print(f"   Use {best_arch['name']} predictions for graph correction training.")
        
    elif best_arch['dice'] >= 0.70:
        print("⚠️  CONSIDER IMPROVEMENTS BEFORE PROCEEDING")
        print("   Performance is decent but could be better.")
        print("   Options:")
        print("   1. Proceed with current best model")
        print("   2. Try hyperparameter tuning")
        print("   3. Data augmentation")
        print("   4. Longer training")
        
    else:
        print("🔴 IMPROVE SEGMENTATION BEFORE PROCEEDING")
        print("   Performance is below recommended threshold.")
        print("   Suggested actions:")
        print("   1. Check data quality and labels")
        print("   2. Adjust hyperparameters")
        print("   3. Try different architectures")
        print("   4. Increase training epochs")
        print("   5. Add data augmentation")
    
    # Architecture-specific recommendations
    print(f"\n📋 ARCHITECTURE-SPECIFIC INSIGHTS:")
    for arch_data in performance_data:
        name = arch_data['name']
        dice = arch_data['dice']
        
        if 'unet3d' in name.lower():
            if dice < 0.75:
                print(f"  • {name}: Consider reducing batch size or patch size")
        elif 'attention' in name.lower():
            if dice < 0.75:
                print(f"  • {name}: Attention might need more training epochs")
        elif 'unet2d' in name.lower():
            if dice < 0.70:
                print(f"  • {name}: 2D might miss 3D context, consider 3D models")


def check_training_files(results_dir):
    """Check what files were generated"""
    print("\n" + "="*60)
    print("GENERATED FILES")
    print("="*60)
    
    results_dir = Path(results_dir)
    
    # Check architecture directories
    for arch_dir in results_dir.iterdir():
        if arch_dir.is_dir() and arch_dir.name != 'cv_predictions':
            print(f"\n📁 {arch_dir.name}/")
            
            # Check config file
            config_file = arch_dir / 'config.yaml'
            print(f"  ✅ config.yaml" if config_file.exists() else "  ❌ config.yaml")
            
            # Check results file
            results_file = arch_dir / 'results.json'
            print(f"  ✅ results.json" if results_file.exists() else "  ❌ results.json")
            
            # Check fold directories
            fold_count = 0
            for fold_dir in arch_dir.iterdir():
                if fold_dir.is_dir() and fold_dir.name.startswith('fold_'):
                    fold_count += 1
                    model_file = fold_dir / 'best_model.pth'
                    if model_file.exists():
                        size_mb = model_file.stat().st_size / (1024*1024)
                        print(f"  ✅ {fold_dir.name}/best_model.pth ({size_mb:.1f} MB)")
                    else:
                        print(f"  ❌ {fold_dir.name}/best_model.pth")
            
            print(f"  📊 Total folds: {fold_count}/5")
    
    # Check CV predictions
    cv_pred_dir = results_dir / 'cv_predictions'
    if cv_pred_dir.exists():
        print(f"\n📁 cv_predictions/")
        for arch_pred_dir in cv_pred_dir.iterdir():
            if arch_pred_dir.is_dir():
                pred_files = list(arch_pred_dir.glob('*.npy'))
                print(f"  📊 {arch_pred_dir.name}: {len(pred_files)} prediction files")
    else:
        print(f"\n❌ cv_predictions/ directory not found")


def generate_summary_report(results, results_dir, output_file=None):
    """Generate a summary report"""
    if output_file is None:
        output_file = results_dir / 'training_summary.txt'
    
    with open(output_file, 'w') as f:
        f.write("PULMONARY ARTERY SEGMENTATION TRAINING SUMMARY\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Results Directory: {results_dir}\n\n")
        
        # Performance summary
        f.write("PERFORMANCE SUMMARY:\n")
        f.write("-" * 20 + "\n")
        
        best_dice = 0
        best_arch = None
        
        for arch_name, result in results.items():
            if result and 'average_dice' in result:
                dice = result['average_dice']
                std = result['std_dice']
                f.write(f"{arch_name:15}: {dice:.4f} ± {std:.4f}\n")
                
                if dice > best_dice:
                    best_dice = dice
                    best_arch = arch_name
        
        f.write(f"\nBest Architecture: {best_arch} ({best_dice:.4f})\n")
        
        # Recommendation
        f.write(f"\nRECOMMENDAION:\n")
        f.write("-" * 15 + "\n")
        if best_dice >= 0.80:
            f.write("✅ PROCEED to graph correction pipeline\n")
        elif best_dice >= 0.70:
            f.write("⚠️  CONSIDER improvements before proceeding\n")
        else:
            f.write("🔴 IMPROVE segmentation before proceeding\n")
    
    print(f"\n📄 Summary report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze U-Net training results')
    parser.add_argument('--results-dir', type=str, required=True,
                        help='Path to results directory (e.g., experiments/multi_unet)')
    parser.add_argument('--save-report', action='store_true',
                        help='Save summary report to file')
    
    args = parser.parse_args()
    
    # Load results
    data = load_results(args.results_dir)
    if data is None:
        return
    
    results, results_dir = data
    
    # Perform analysis
    print(f"📊 Analyzing results from: {results_dir}")
    
    performance_data = analyze_architecture_performance(results)
    analyze_fold_consistency(results, results_dir)
    provide_recommendations(results)
    check_training_files(results_dir)
    
    # Generate report if requested
    if args.save_report:
        generate_summary_report(results, results_dir)
    
    print(f"\n✅ Analysis complete!")
    print(f"\nNext steps:")
    print(f"1. Review recommendations above")
    print(f"2. Check individual fold models in architecture subdirectories")
    print(f"3. Use cv_predictions/ for graph correction if performance is good")


if __name__ == '__main__':
    main()