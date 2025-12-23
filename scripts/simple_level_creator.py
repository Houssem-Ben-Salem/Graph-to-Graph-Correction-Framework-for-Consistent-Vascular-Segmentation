#!/usr/bin/env python
"""Create one dataset file at a time to avoid memory issues"""

import sys
import gzip
import pickle
from pathlib import Path
import gc

sys.path.append(str(Path(__file__).parent.parent))

def create_single_dataset(cache_dir, training_dir, split_name, level_name):
    """Create a single dataset file"""
    
    print(f"📦 Creating {split_name}_{level_name}_dataset.pkl...")
    
    cache_files = list(cache_dir.glob("*.pkl.gz"))
    samples = []
    
    for i, cache_file in enumerate(cache_files):
        if i % 100 == 0 and i > 0:
            print(f"  Scanned {i}/{len(cache_files)} files, found {len(samples)} samples...")
        
        try:
            with gzip.open(cache_file, 'rb') as f:
                sample = pickle.load(f)
                
            split = sample.get('split', 'unknown')
            level = sample.get('curriculum_level', 'unknown')
            
            if split == split_name and level == level_name:
                samples.append(sample)
                
        except Exception as e:
            continue  # Skip corrupted files
    
    if samples:
        output_file = training_dir / f"{split_name}_{level_name}_dataset.pkl"
        
        try:
            with open(output_file, 'wb') as f:
                pickle.dump(samples, f)
            
            size_mb = output_file.stat().st_size / (1024 * 1024)
            print(f"  ✅ Saved {len(samples)} samples ({size_mb:.1f} MB)")
            return True
            
        except Exception as e:
            print(f"  ❌ Error saving: {e}")
            return False
    else:
        print(f"  ⚠️  No samples found")
        return False

def main():
    print("🎯 Simple Level-by-Level Dataset Creation\n")
    
    cache_dir = Path("training_data/cache")
    training_dir = Path("training_data")
    
    if not cache_dir.exists():
        print("❌ Cache directory not found!")
        return
    
    # Create each dataset one at a time
    splits = ['train', 'val', 'test']
    levels = ['easy', 'medium', 'hard', 'expert']
    
    success_count = 0
    total_count = 0
    
    for split in splits:
        print(f"\n=== {split.upper()} SPLIT ===")
        
        for level in levels:
            total_count += 1
            if create_single_dataset(cache_dir, training_dir, split, level):
                success_count += 1
            
            # Force cleanup after each dataset
            gc.collect()
    
    print(f"\n🎉 Creation complete!")
    print(f"✅ Successfully created {success_count}/{total_count} datasets")
    
    print(f"\n📁 Created files:")
    for f in sorted(training_dir.glob("*_*_dataset.pkl")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.1f} MB")
    
    print(f"\n🚀 Ready to train!")

if __name__ == "__main__":
    main()