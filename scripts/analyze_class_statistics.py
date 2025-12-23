#!/usr/bin/env python3
"""
Comprehensive statistical analysis of graph correction classes
to understand the true nature of the classification problem
"""

import sys
sys.path.append('.')  # Add current directory to path

import numpy as np
import pickle
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mutual_info_score
from collections import defaultdict
import pandas as pd
import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in divide')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
warnings.filterwarnings('ignore', message='Each of the input arrays is constant')

def load_dataset(dataset_path):
    """Load the training dataset"""
    print(f"Loading dataset from {dataset_path}")
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    return data

def extract_features_and_labels(dataset):
    """Extract all features and labels from dataset"""
    features = defaultdict(list)
    labels = []
    corrections = defaultdict(list)
    
    # Dataset is a list of sample dictionaries
    samples = dataset
    
    for sample in samples:
        pred_graph = sample['degraded_graph']
        gt_graph = sample['gt_graph']
        correspondences = sample['correspondences']
        
        # For each correspondence
        for pred_idx, gt_idx in correspondences.node_correspondences.items():
            # Get confidence
            confidence = correspondences.node_confidences.get(pred_idx, 0.5)
            
            # Calculate spatial error
            pred_pos = np.array(pred_graph.nodes[pred_idx]['position'])
            gt_pos = np.array(gt_graph.nodes[gt_idx]['position'])
            spatial_error = np.linalg.norm(gt_pos - pred_pos)
            
            # Calculate radius error (handle missing radius data)
            pred_radius = pred_graph.nodes[pred_idx].get('radius', 1.0)
            gt_radius = gt_graph.nodes[gt_idx].get('radius', 1.0)
            radius_error = abs(gt_radius - pred_radius) if pred_radius and gt_radius else 0.0
            radius_ratio = gt_radius / (pred_radius + 1e-6) if pred_radius > 0 else 1.0
            
            # Store features
            features['confidence'].append(confidence)
            features['spatial_error'].append(spatial_error)
            features['radius_error'].append(radius_error)
            features['radius_ratio'].append(radius_ratio)
            
            # Store corrections (for regression analysis)
            corrections['position_correction'].append(gt_pos - pred_pos)
            corrections['radius_correction'].append(gt_radius - pred_radius)
            
            # Determine class using current logic
            if confidence > 0.77 and spatial_error < 0.5:
                label = 0  # Keep
            elif confidence < 0.52 or spatial_error > 2.0:
                label = 1  # Remove
            else:
                label = 2  # Modify
            labels.append(label)
    
    return features, labels, corrections

def analyze_class_separability(features, labels):
    """Analyze how separable the classes are"""
    print("\n=== Class Separability Analysis ===")
    
    # Convert to numpy arrays
    X = np.column_stack([features['confidence'], features['spatial_error'], 
                         features['radius_error'], features['radius_ratio']])
    y = np.array(labels)
    
    # 1. Statistical tests for each feature
    print("\n1. ANOVA F-statistic for each feature:")
    feature_names = ['confidence', 'spatial_error', 'radius_error', 'radius_ratio']
    
    for i, fname in enumerate(feature_names):
        # Get feature values for each class
        class_values = [X[y == c, i] for c in range(3)]
        f_stat, p_value = stats.f_oneway(*class_values)
        print(f"   {fname}: F={f_stat:.3f}, p={p_value:.3e}")
    
    # 2. Mutual information
    print("\n2. Mutual Information with class labels:")
    for i, fname in enumerate(feature_names):
        try:
            # Handle constant features
            if np.std(X[:, i]) < 1e-10:
                print(f"   {fname}: MI=0.000 (constant feature)")
            else:
                mi = mutual_info_score(y, pd.qcut(X[:, i], q=10, labels=False, duplicates='drop'))
                print(f"   {fname}: MI={mi:.3f}")
        except Exception as e:
            print(f"   {fname}: MI=N/A (error: {str(e)})")
    
    # 3. Class overlap analysis
    print("\n3. Class Overlap Analysis:")
    for i, fname in enumerate(feature_names):
        print(f"\n   {fname}:")
        for c in range(3):
            class_data = X[y == c, i]
            print(f"     Class {c}: mean={np.mean(class_data):.3f}, "
                  f"std={np.std(class_data):.3f}, "
                  f"[{np.percentile(class_data, 5):.3f}, {np.percentile(class_data, 95):.3f}]")

def analyze_correction_magnitudes(corrections, labels):
    """Analyze if correction magnitudes naturally cluster"""
    print("\n=== Correction Magnitude Analysis ===")
    
    position_corrections = np.array(corrections['position_correction'])
    radius_corrections = np.array(corrections['radius_correction'])
    labels = np.array(labels)
    
    # Position correction magnitudes
    pos_magnitudes = np.linalg.norm(position_corrections, axis=1)
    
    print("\n1. Position Correction Magnitudes by Class:")
    for c in range(3):
        class_mags = pos_magnitudes[labels == c]
        if len(class_mags) > 0:
            print(f"   Class {c}: mean={np.mean(class_mags):.3f}, "
                  f"median={np.median(class_mags):.3f}, "
                  f"std={np.std(class_mags):.3f}")
            print(f"            [{np.percentile(class_mags, 5):.3f}, "
                  f"{np.percentile(class_mags, 95):.3f}]")
    
    print("\n2. Radius Correction Magnitudes by Class:")
    for c in range(3):
        class_corrections = radius_corrections[labels == c]
        if len(class_corrections) > 0:
            print(f"   Class {c}: mean={np.mean(class_corrections):.3f}, "
                  f"median={np.median(class_corrections):.3f}, "
                  f"std={np.std(class_corrections):.3f}")

def suggest_better_thresholds(features, corrections):
    """Suggest data-driven thresholds"""
    print("\n=== Data-Driven Threshold Suggestions ===")
    
    # Combine features
    confidences = np.array(features['confidence'])
    spatial_errors = np.array(features['spatial_error'])
    position_corrections = np.array(corrections['position_correction'])
    pos_magnitudes = np.linalg.norm(position_corrections, axis=1)
    
    # Natural clustering based on correction magnitude
    print("\n1. Natural clusters based on correction magnitude:")
    
    # Define natural thresholds
    small_correction = np.percentile(pos_magnitudes, 33)
    large_correction = np.percentile(pos_magnitudes, 67)
    
    print(f"   Small corrections: < {small_correction:.3f}")
    print(f"   Medium corrections: {small_correction:.3f} - {large_correction:.3f}")
    print(f"   Large corrections: > {large_correction:.3f}")
    
    # Analyze confidence/error for each correction group
    print("\n2. Feature statistics for each correction group:")
    
    small_mask = pos_magnitudes < small_correction
    medium_mask = (pos_magnitudes >= small_correction) & (pos_magnitudes < large_correction)
    large_mask = pos_magnitudes >= large_correction
    
    for mask, name in [(small_mask, "Small"), (medium_mask, "Medium"), (large_mask, "Large")]:
        print(f"\n   {name} corrections:")
        if np.sum(mask) > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                print(f"     Confidence: {np.mean(confidences[mask]):.3f} ± {np.std(confidences[mask]):.3f}")
                print(f"     Spatial Error: {np.mean(spatial_errors[mask]):.3f} ± {np.std(spatial_errors[mask]):.3f}")
                print(f"     Count: {np.sum(mask)} samples")
        else:
            print(f"     No samples in this category")

def create_visualization(features, labels, corrections):
    """Create visualization of the data distribution"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Confidence vs Spatial Error scatter
    ax = axes[0, 0]
    scatter = ax.scatter(features['confidence'], features['spatial_error'], 
                        c=labels, cmap='viridis', alpha=0.5, s=10)
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Spatial Error')
    ax.set_title('Current Class Distribution')
    plt.colorbar(scatter, ax=ax, label='Class')
    
    # 2. Position correction magnitude distribution
    ax = axes[0, 1]
    pos_mags = np.linalg.norm(np.array(corrections['position_correction']), axis=1)
    for c in range(3):
        class_mags = pos_mags[np.array(labels) == c]
        if len(class_mags) > 0:
            ax.hist(class_mags, bins=30, alpha=0.5, label=f'Class {c}', density=True)
    ax.set_xlabel('Position Correction Magnitude')
    ax.set_ylabel('Density')
    ax.set_title('Correction Magnitude by Class')
    ax.legend()
    
    # 3. Feature correlation heatmap
    ax = axes[0, 2]
    feature_matrix = np.column_stack([
        features['confidence'], features['spatial_error'],
        features['radius_error'], features['radius_ratio']
    ])
    
    # Handle constant features in correlation
    try:
        # Check if features have variance
        feature_std = np.std(feature_matrix, axis=0)
        non_constant_mask = feature_std > 1e-10
        
        if np.sum(non_constant_mask) > 1:
            # Compute correlation only for non-constant features
            non_constant_features = feature_matrix[:, non_constant_mask]
            non_constant_labels = np.array(['Conf', 'Spatial', 'Radius Err', 'Radius Ratio'])[non_constant_mask]
            
            corr = np.corrcoef(non_constant_features.T)
            sns.heatmap(corr, 
                        xticklabels=non_constant_labels,
                        yticklabels=non_constant_labels,
                        annot=True, fmt='.2f', ax=ax, cbar=True)
        else:
            ax.text(0.5, 0.5, 'Insufficient feature variance\nfor correlation', 
                   ha='center', va='center', transform=ax.transAxes)
    except Exception as e:
        ax.text(0.5, 0.5, f'Error computing correlations:\n{str(e)}', 
               ha='center', va='center', transform=ax.transAxes)
    
    ax.set_title('Feature Correlations')
    
    # 4. Class balance
    ax = axes[1, 0]
    class_counts = [np.sum(np.array(labels) == c) for c in range(3)]
    ax.bar(['Keep', 'Remove', 'Modify'], class_counts)
    ax.set_ylabel('Count')
    ax.set_title('Class Distribution')
    
    # 5. Confidence distribution by class
    ax = axes[1, 1]
    for c in range(3):
        class_conf = np.array(features['confidence'])[np.array(labels) == c]
        if len(class_conf) > 0:
            ax.hist(class_conf, bins=30, alpha=0.5, label=f'Class {c}', density=True)
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Density')
    ax.set_title('Confidence Distribution by Class')
    ax.legend()
    
    # 6. Spatial error distribution by class
    ax = axes[1, 2]
    for c in range(3):
        class_errors = np.array(features['spatial_error'])[np.array(labels) == c]
        if len(class_errors) > 0:
            ax.hist(class_errors, bins=30, alpha=0.5, label=f'Class {c}', density=True)
    ax.set_xlabel('Spatial Error')
    ax.set_ylabel('Density')
    ax.set_title('Spatial Error Distribution by Class')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('experiments/graph_correction/class_analysis.png', dpi=150)
    print("\nVisualization saved to experiments/graph_correction/class_analysis.png")

def analyze_dataset(dataset_path, dataset_name):
    """Analyze a single dataset"""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {dataset_name}")
    print(f"{'='*60}")
    
    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}")
        return None, None, None
    
    dataset = load_dataset(dataset_path)
    # Handle different dataset formats
    if isinstance(dataset, list):
        print(f"Loaded {len(dataset)} samples")
    elif isinstance(dataset, dict):
        print(f"Loaded {len(dataset.get('samples', dataset))} samples")
    else:
        print(f"Loaded dataset with unknown structure")
    
    # Extract features and labels
    features, labels, corrections = extract_features_and_labels(dataset)
    print(f"Extracted {len(labels)} node correspondences")
    
    # Analyze class separability
    analyze_class_separability(features, labels)
    
    # Analyze correction magnitudes
    analyze_correction_magnitudes(corrections, labels)
    
    # Suggest better thresholds
    suggest_better_thresholds(features, corrections)
    
    return features, labels, corrections

def main():
    # Analyze both synthetic and real datasets
    datasets = [
        (Path("training_data/train_easy_dataset.pkl"), "Synthetic Easy Data"),
        (Path("training_data/train_real_dataset.pkl"), "Real U-Net Predictions")
    ]
    
    all_results = []
    
    for dataset_path, dataset_name in datasets:
        features, labels, corrections = analyze_dataset(dataset_path, dataset_name)
        if features is not None:
            all_results.append((dataset_name, features, labels, corrections))
    
    # Create comparative visualizations if we have results
    if len(all_results) > 0:
        print("\n" + "="*60)
        print("CREATING COMPARATIVE VISUALIZATIONS")
        print("="*60)
        
        # Create separate visualizations for each dataset
        for i, (name, features, labels, corrections) in enumerate(all_results):
            create_visualization(features, labels, corrections)
            output_path = f'experiments/graph_correction/class_analysis_{name.replace(" ", "_").lower()}.png'
            plt.savefig(output_path, dpi=150)
            print(f"\nVisualization saved to {output_path}")
            plt.close()
    
    print("\n=== FINAL RECOMMENDATIONS ===")
    print("Based on the analysis, consider:")
    print("1. If synthetic data separates well but real doesn't -> Need better features")
    print("2. If both have high overlap -> Fundamental reformulation needed")
    print("3. If correction magnitudes cluster naturally -> Use regression approach")
    print("4. If confidence/spatial error aren't discriminative -> Add graph topology features")

if __name__ == "__main__":
    main()