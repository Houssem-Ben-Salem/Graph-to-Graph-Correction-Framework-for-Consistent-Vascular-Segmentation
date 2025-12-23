#!/usr/bin/env python
"""Analyze per-case prediction metrics and filter cases"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path


def analyze_metrics_file(metrics_file, output_dir=None, min_dice=0.6):
    """Analyze metrics CSV file"""
    
    # Load metrics
    df = pd.read_csv(metrics_file)
    output_dir = Path(output_dir) if output_dir else Path(metrics_file).parent
    
    print(f"📊 ANALYZING: {metrics_file}")
    print(f"=" * 60)
    
    # Basic statistics
    successful = df[df['success'] == True]
    failed = df[df['success'] == False]
    
    print(f"Total cases: {len(df)}")
    print(f"Successful: {len(successful)} ({len(successful)/len(df)*100:.1f}%)")
    print(f"Failed: {len(failed)} ({len(failed)/len(df)*100:.1f}%)")
    
    if len(successful) == 0:
        print("❌ No successful cases to analyze")
        return
    
    # Performance statistics
    print(f"\n📈 PERFORMANCE METRICS:")
    metrics_cols = ['dice', 'jaccard', 'precision', 'recall', 'hausdorff', 'volume_ratio']
    
    for metric in metrics_cols:
        if metric in successful.columns:
            mean_val = successful[metric].mean()
            std_val = successful[metric].std()
            min_val = successful[metric].min()
            max_val = successful[metric].max()
            print(f"{metric:15}: {mean_val:.4f} ± {std_val:.4f} (range: {min_val:.4f} - {max_val:.4f})")
    
    # Case classification
    good_cases = successful[successful['dice'] >= min_dice]
    poor_cases = successful[successful['dice'] < min_dice]
    
    print(f"\n🎯 CASE CLASSIFICATION (Dice threshold: {min_dice}):")
    print(f"Good cases:  {len(good_cases)} ({len(good_cases)/len(successful)*100:.1f}%)")
    print(f"Poor cases:  {len(poor_cases)} ({len(poor_cases)/len(successful)*100:.1f}%)")
    
    # Top and bottom performers
    print(f"\n🏆 TOP 10 PERFORMERS:")
    top_10 = successful.nlargest(10, 'dice')
    for _, row in top_10.iterrows():
        print(f"  {row['patient_id']}: Dice={row['dice']:.4f}, Volume={row['pred_volume_mm3']:.0f}mm³")
    
    print(f"\n🔴 BOTTOM 10 PERFORMERS:")
    bottom_10 = successful.nsmallest(10, 'dice')
    for _, row in bottom_10.iterrows():
        print(f"  {row['patient_id']}: Dice={row['dice']:.4f}, Volume={row['pred_volume_mm3']:.0f}mm³")
    
    # Volume analysis
    print(f"\n📏 VOLUME ANALYSIS:")
    print(f"Predicted volumes: {successful['pred_volume_mm3'].mean():.0f} ± {successful['pred_volume_mm3'].std():.0f} mm³")
    print(f"Ground truth volumes: {successful['gt_volume_mm3'].mean():.0f} ± {successful['gt_volume_mm3'].std():.0f} mm³")
    print(f"Volume ratio: {successful['volume_ratio'].mean():.3f} ± {successful['volume_ratio'].std():.3f}")
    
    # Outlier detection
    outliers = successful[
        (successful['volume_ratio'] < 0.3) | 
        (successful['volume_ratio'] > 3.0) |
        (successful['dice'] < 0.2) |
        (successful['hausdorff'] > 100)
    ]
    
    if len(outliers) > 0:
        print(f"\n⚠️  OUTLIERS DETECTED ({len(outliers)} cases):")
        for _, row in outliers.iterrows():
            reasons = []
            if row['volume_ratio'] < 0.3 or row['volume_ratio'] > 3.0:
                reasons.append(f"Volume ratio: {row['volume_ratio']:.2f}")
            if row['dice'] < 0.2:
                reasons.append(f"Very low Dice: {row['dice']:.3f}")
            if row['hausdorff'] > 100:
                reasons.append(f"High Hausdorff: {row['hausdorff']:.1f}")
            
            print(f"  {row['patient_id']}: {', '.join(reasons)}")
    
    # Save filtered lists
    good_cases_file = output_dir / 'good_cases_analysis.txt'
    poor_cases_file = output_dir / 'poor_cases_analysis.txt'
    outliers_file = output_dir / 'outliers_analysis.txt'
    
    # Good cases
    with open(good_cases_file, 'w') as f:
        f.write(f"# Good cases (Dice >= {min_dice})\n")
        f.write(f"# Total: {len(good_cases)}/{len(successful)} successful cases\n\n")
        for _, row in good_cases.sort_values('dice', ascending=False).iterrows():
            f.write(f"{row['patient_id']}\n")
    
    # Poor cases
    with open(poor_cases_file, 'w') as f:
        f.write(f"# Poor cases (Dice < {min_dice})\n") 
        f.write(f"# Total: {len(poor_cases)}/{len(successful)} successful cases\n\n")
        for _, row in poor_cases.sort_values('dice').iterrows():
            f.write(f"{row['patient_id']}\t# Dice: {row['dice']:.4f}\n")
    
    # Outliers
    with open(outliers_file, 'w') as f:
        f.write(f"# Outlier cases (extreme values)\n")
        f.write(f"# Total: {len(outliers)} cases\n\n")
        for _, row in outliers.iterrows():
            f.write(f"{row['patient_id']}\t# Dice: {row['dice']:.4f}, Volume ratio: {row['volume_ratio']:.2f}\n")
    
    print(f"\n📄 SAVED FILES:")
    print(f"✅ Good cases: {good_cases_file}")
    print(f"⚠️  Poor cases: {poor_cases_file}")
    print(f"🚨 Outliers: {outliers_file}")
    
    # Create summary statistics
    summary_stats = {
        'total_cases': len(df),
        'successful_cases': len(successful),
        'failed_cases': len(failed),
        'good_cases': len(good_cases),
        'poor_cases': len(poor_cases),
        'outliers': len(outliers),
        'mean_dice': float(successful['dice'].mean()),
        'std_dice': float(successful['dice'].std()),
        'min_dice_threshold': min_dice,
        'success_rate': float(len(successful)/len(df)),
        'good_case_rate': float(len(good_cases)/len(successful)) if len(successful) > 0 else 0.0
    }
    
    # Save summary
    summary_file = output_dir / 'analysis_summary.json'
    import json
    with open(summary_file, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    print(f"📊 Summary: {summary_file}")
    
    return summary_stats


def plot_metrics_distribution(metrics_file, output_dir=None):
    """Create distribution plots for key metrics"""
    
    df = pd.read_csv(metrics_file)
    successful = df[df['success'] == True]
    
    if len(successful) == 0:
        print("No successful cases for plotting")
        return
    
    output_dir = Path(output_dir) if output_dir else Path(metrics_file).parent
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Per-Case Prediction Metrics Distribution', fontsize=16)
    
    # Dice score
    axes[0, 0].hist(successful['dice'], bins=20, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(successful['dice'].mean(), color='red', linestyle='--', label='Mean')
    axes[0, 0].set_xlabel('Dice Score')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Dice Score Distribution')
    axes[0, 0].legend()
    
    # Jaccard
    axes[0, 1].hist(successful['jaccard'], bins=20, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(successful['jaccard'].mean(), color='red', linestyle='--', label='Mean')
    axes[0, 1].set_xlabel('Jaccard Index')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Jaccard Distribution')
    axes[0, 1].legend()
    
    # Hausdorff distance
    axes[0, 2].hist(successful['hausdorff'], bins=20, alpha=0.7, edgecolor='black')
    axes[0, 2].axvline(successful['hausdorff'].mean(), color='red', linestyle='--', label='Mean')
    axes[0, 2].set_xlabel('Hausdorff Distance')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].set_title('Hausdorff Distribution')
    axes[0, 2].legend()
    
    # Volume ratio
    axes[1, 0].hist(successful['volume_ratio'], bins=20, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(successful['volume_ratio'].mean(), color='red', linestyle='--', label='Mean')
    axes[1, 0].axvline(1.0, color='green', linestyle='-', alpha=0.5, label='Perfect')
    axes[1, 0].set_xlabel('Volume Ratio (Pred/GT)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Volume Ratio Distribution')
    axes[1, 0].legend()
    
    # Predicted vs GT volume
    axes[1, 1].scatter(successful['gt_volume_mm3'], successful['pred_volume_mm3'], alpha=0.6)
    axes[1, 1].plot([successful['gt_volume_mm3'].min(), successful['gt_volume_mm3'].max()], 
                    [successful['gt_volume_mm3'].min(), successful['gt_volume_mm3'].max()], 
                    'r--', label='Perfect correlation')
    axes[1, 1].set_xlabel('Ground Truth Volume (mm³)')
    axes[1, 1].set_ylabel('Predicted Volume (mm³)')
    axes[1, 1].set_title('Volume Correlation')
    axes[1, 1].legend()
    
    # Mean probability in predictions
    axes[1, 2].hist(successful['mean_prob_in_pred'], bins=20, alpha=0.7, edgecolor='black')
    axes[1, 2].axvline(successful['mean_prob_in_pred'].mean(), color='red', linestyle='--', label='Mean')
    axes[1, 2].set_xlabel('Mean Probability in Predictions')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].set_title('Prediction Confidence')
    axes[1, 2].legend()
    
    plt.tight_layout()
    
    plot_file = output_dir / 'metrics_distribution.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"📈 Distribution plot saved: {plot_file}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze per-case prediction metrics')
    parser.add_argument('--metrics-file', type=str, required=True,
                        help='Path to per_case_metrics.csv file')
    parser.add_argument('--output-dir', type=str,
                        help='Output directory (defaults to metrics file directory)')
    parser.add_argument('--min-dice', type=float, default=0.6,
                        help='Minimum Dice score for good cases')
    parser.add_argument('--plot', action='store_true',
                        help='Generate distribution plots')
    
    args = parser.parse_args()
    
    if not Path(args.metrics_file).exists():
        print(f"❌ Metrics file not found: {args.metrics_file}")
        return
    
    # Analyze metrics
    summary = analyze_metrics_file(args.metrics_file, args.output_dir, args.min_dice)
    
    # Generate plots if requested
    if args.plot:
        plot_metrics_distribution(args.metrics_file, args.output_dir)
    
    print(f"\n✅ Analysis complete!")
    print(f"🎯 Recommendation: Use good cases for graph extraction pipeline")


if __name__ == '__main__':
    main()