#!/usr/bin/env python3
"""
Analyze the training data class distribution to understand the imbalance issue
"""

import sys
sys.path.append('.')

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_training_data():
    """Analyze class distribution in training data"""
    
    # Load training data
    train_path = Path('training_data/train_real_dataset.pkl')
    val_path = Path('training_data/val_real_dataset.pkl')
    
    if not train_path.exists():
        logger.error("Training data not found!")
        return
    
    logger.info("Loading training data...")
    with open(train_path, 'rb') as f:
        train_samples = pickle.load(f)
    
    logger.info("Loading validation data...")
    with open(val_path, 'rb') as f:
        val_samples = pickle.load(f)
    
    logger.info(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")
    
    # Analyze class distribution
    def analyze_samples(samples, name):
        all_magnitudes = []
        all_classes = []
        modify_magnitudes = []
        remove_magnitudes = []
        
        for sample in samples:
            # Get correspondences
            correspondences = sample['correspondences']
            pred_graph = sample['degraded_graph']
            gt_graph = sample['gt_graph']
            
            # Extract positions
            pred_positions = {node['id']: np.array(node['position']) for node in pred_graph.nodes}
            gt_positions = {node['id']: np.array(node['position']) for node in gt_graph.nodes}
            
            # Calculate magnitudes and classes
            for pred_id, pred_pos in pred_positions.items():
                if pred_id in correspondences.node_correspondences:
                    # Matched node - compute correction magnitude
                    gt_id = correspondences.node_correspondences[pred_id]
                    gt_pos = gt_positions[gt_id]
                    magnitude = np.linalg.norm(gt_pos - pred_pos)
                    
                    # Current threshold logic
                    if magnitude >= 2.5:
                        class_label = 1  # Remove
                        remove_magnitudes.append(magnitude)
                    else:
                        class_label = 0  # Modify
                        modify_magnitudes.append(magnitude)
                else:
                    # Unmatched node - should be removed
                    magnitude = 5.0  # Large magnitude for removal
                    class_label = 1  # Remove
                    remove_magnitudes.append(magnitude)
                
                all_magnitudes.append(magnitude)
                all_classes.append(class_label)
        
        # Statistics
        class_counts = Counter(all_classes)
        total = len(all_classes)
        
        logger.info(f"\n{name} Analysis:")
        logger.info(f"Total nodes: {total}")
        logger.info(f"Modify (0): {class_counts[0]} ({class_counts[0]/total*100:.1f}%)")
        logger.info(f"Remove (1): {class_counts[1]} ({class_counts[1]/total*100:.1f}%)")
        
        if modify_magnitudes:
            logger.info(f"Modify magnitudes - Mean: {np.mean(modify_magnitudes):.3f}, "
                       f"Std: {np.std(modify_magnitudes):.3f}, "
                       f"Range: [{np.min(modify_magnitudes):.3f}, {np.max(modify_magnitudes):.3f}]")
        
        if remove_magnitudes:
            logger.info(f"Remove magnitudes - Mean: {np.mean(remove_magnitudes):.3f}, "
                       f"Std: {np.std(remove_magnitudes):.3f}, "
                       f"Range: [{np.min(remove_magnitudes):.3f}, {np.max(remove_magnitudes):.3f}]")
        
        return {
            'magnitudes': all_magnitudes,
            'classes': all_classes,
            'modify_magnitudes': modify_magnitudes,
            'remove_magnitudes': remove_magnitudes,
            'class_counts': class_counts
        }
    
    # Analyze both datasets
    train_stats = analyze_samples(train_samples, "TRAINING")
    val_stats = analyze_samples(val_samples, "VALIDATION")
    
    # Plot distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training data histogram
    axes[0, 0].hist(train_stats['magnitudes'], bins=50, alpha=0.7, label='All')
    if train_stats['modify_magnitudes']:
        axes[0, 0].hist(train_stats['modify_magnitudes'], bins=30, alpha=0.7, label='Modify')
    if train_stats['remove_magnitudes']:
        axes[0, 0].hist(train_stats['remove_magnitudes'], bins=30, alpha=0.7, label='Remove')
    axes[0, 0].axvline(x=2.5, color='red', linestyle='--', label='Threshold=2.5')
    axes[0, 0].set_title('Training Data - Magnitude Distribution')
    axes[0, 0].set_xlabel('Correction Magnitude')
    axes[0, 0].legend()
    
    # Validation data histogram
    axes[0, 1].hist(val_stats['magnitudes'], bins=50, alpha=0.7, label='All')
    if val_stats['modify_magnitudes']:
        axes[0, 1].hist(val_stats['modify_magnitudes'], bins=30, alpha=0.7, label='Modify')
    if val_stats['remove_magnitudes']:
        axes[0, 1].hist(val_stats['remove_magnitudes'], bins=30, alpha=0.7, label='Remove')
    axes[0, 1].axvline(x=2.5, color='red', linestyle='--', label='Threshold=2.5')
    axes[0, 1].set_title('Validation Data - Magnitude Distribution')
    axes[0, 1].set_xlabel('Correction Magnitude')
    axes[0, 1].legend()
    
    # Class distribution pie charts
    train_labels = ['Modify', 'Remove']
    train_sizes = [train_stats['class_counts'][0], train_stats['class_counts'][1]]
    axes[1, 0].pie(train_sizes, labels=train_labels, autopct='%1.1f%%', startangle=90)
    axes[1, 0].set_title('Training Data - Class Distribution')
    
    val_sizes = [val_stats['class_counts'][0], val_stats['class_counts'][1]]
    axes[1, 1].pie(val_sizes, labels=train_labels, autopct='%1.1f%%', startangle=90)
    axes[1, 1].set_title('Validation Data - Class Distribution')
    
    plt.tight_layout()
    plt.savefig('training_data_analysis.png', dpi=150, bbox_inches='tight')
    logger.info("Analysis plot saved as 'training_data_analysis.png'")
    
    # Threshold analysis
    logger.info("\n=== THRESHOLD ANALYSIS ===")
    thresholds = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    
    for threshold in thresholds:
        modify_count = sum(1 for mag in train_stats['magnitudes'] if mag < threshold)
        remove_count = sum(1 for mag in train_stats['magnitudes'] if mag >= threshold)
        total = len(train_stats['magnitudes'])
        
        modify_pct = modify_count / total * 100
        remove_pct = remove_count / total * 100
        
        logger.info(f"Threshold {threshold:.1f}: Modify={modify_count} ({modify_pct:.1f}%), "
                   f"Remove={remove_count} ({remove_pct:.1f}%)")
    
    # Recommendations
    logger.info("\n=== RECOMMENDATIONS ===")
    
    current_modify_pct = train_stats['class_counts'][0] / len(train_stats['classes']) * 100
    current_remove_pct = train_stats['class_counts'][1] / len(train_stats['classes']) * 100
    
    if current_modify_pct < 20:
        logger.info("❌ SEVERE class imbalance detected!")
        logger.info(f"Current: {current_modify_pct:.1f}% Modify, {current_remove_pct:.1f}% Remove")
        
        # Find better threshold
        best_threshold = 2.5
        best_balance = abs(50 - current_modify_pct)
        
        for threshold in thresholds:
            modify_count = sum(1 for mag in train_stats['magnitudes'] if mag < threshold)
            modify_pct = modify_count / len(train_stats['magnitudes']) * 100
            balance = abs(50 - modify_pct)
            
            if balance < best_balance and 20 <= modify_pct <= 80:
                best_balance = balance
                best_threshold = threshold
        
        logger.info(f"✅ RECOMMENDED: Change threshold from 2.5 to {best_threshold}")
        logger.info("✅ RECOMMENDED: Add class weights [4.0, 1.0] to penalize modify errors")
        logger.info("✅ RECOMMENDED: Use focal loss to focus on hard examples")
    else:
        logger.info("✅ Class distribution is reasonable")

if __name__ == "__main__":
    analyze_training_data()