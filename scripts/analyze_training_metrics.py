#!/usr/bin/env python
"""Analyze training metrics to understand accuracy issue"""

import torch
import json
from pathlib import Path
import numpy as np

def main():
    print("🔍 Analyzing Training Metrics\n")
    
    # Check latest checkpoint
    checkpoint_dir = Path("experiments/graph_correction")
    
    # Find latest checkpoint
    checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pth"))
    if not checkpoints:
        print("❌ No checkpoints found!")
        return
    
    latest = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
    print(f"📄 Loading {latest.name}...")
    
    try:
        checkpoint = torch.load(latest, map_location='cpu', weights_only=False)
        
        print("\n📊 Training History:")
        if 'train_history' in checkpoint:
            history = checkpoint['train_history']
            
            # Show last 5 epochs
            for metric in ['loss', 'node_accuracy', 'edge_accuracy']:
                if metric in history:
                    values = history[metric][-5:]
                    print(f"\n{metric.title()}:")
                    for i, v in enumerate(values):
                        print(f"  Epoch {len(history[metric])-5+i+1}: {v:.4f}")
        
        print("\n🎯 Model State:")
        print(f"  Current epoch: {checkpoint.get('epoch', 'Unknown')}")
        print(f"  Best val loss: {checkpoint.get('best_val_loss', 'Unknown')}")
        
        # Check loss components
        if 'val_history' in checkpoint and 'loss_components' in checkpoint['val_history']:
            print("\n📈 Loss Components (Latest):")
            components = checkpoint['val_history']['loss_components']
            if components and len(components) > 0:
                latest_components = components[-1] if isinstance(components, list) else components
                for name, value in latest_components.items():
                    print(f"  {name}: {value:.4f}")
        
        # Analyze why accuracy might be 0
        print("\n💡 Possible Reasons for 0 Accuracy:")
        print("  1. Model predicting same class for all nodes (conservative)")
        print("  2. Class imbalance in training data")
        print("  3. Learning rate too high/low")
        print("  4. Still in early training phase")
        
        print("\n🔧 Suggestions:")
        print("  - Continue training (accuracy often improves after 50+ epochs)")
        print("  - Check class distribution in your training data")
        print("  - Consider reducing learning rate if loss plateaus")
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")

if __name__ == "__main__":
    main()