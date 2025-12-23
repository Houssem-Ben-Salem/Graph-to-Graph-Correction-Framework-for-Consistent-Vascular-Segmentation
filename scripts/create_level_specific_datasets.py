#!/usr/bin/env python
"""Create separate dataset files for each curriculum level"""

import sys
import gzip
import pickle
from pathlib import Path
from collections import defaultdict
import gc

sys.path.append(str(Path(__file__).parent.parent))

def main():
    print("🔧 Creating Level-Specific Datasets\n")
    
    cache_dir = Path("training_data/cache")
    training_dir = Path("training_data")
    
    if not cache_dir.exists():
        print("❌ Cache directory not found!")
        return
    
    cache_files = list(cache_dir.glob("*.pkl.gz"))
    print(f"📦 Found {len(cache_files)} cache files")
    
    # Organize by split and level
    datasets = defaultdict(lambda: defaultdict(list))
    
    print("🔍 Organizing samples...")
    for i, cache_file in enumerate(cache_files):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(cache_files)}...")
        
        try:
            with gzip.open(cache_file, 'rb') as f:
                sample = pickle.load(f)
                split = sample.get('split', 'unknown')
                level = sample.get('curriculum_level', 'unknown')
                
                if split in ['train', 'val', 'test'] and level != 'unknown':
                    datasets[split][level].append(sample)
                    
        except Exception as e:
            print(f"  ⚠️  Skipped {cache_file.name}: {e}")
        
        # Periodic cleanup
        if i % 200 == 0:
            gc.collect()
    
    # Save each level separately
    print(f"\n💾 Saving level-specific datasets...")
    
    for split, levels in datasets.items():
        print(f"\n{split.upper()} split:")
        
        # Save individual levels
        for level, samples in levels.items():
            if samples:
                filename = f"{split}_{level}_dataset.pkl"
                filepath = training_dir / filename
                
                try:
                    with open(filepath, 'wb') as f:
                        pickle.dump(samples, f)
                    
                    size_mb = filepath.stat().st_size / (1024 * 1024)
                    print(f"  ✅ {filename}: {len(samples)} samples ({size_mb:.1f} MB)")
                    
                except Exception as e:
                    print(f"  ❌ Error saving {filename}: {e}")
                
                # Clear from memory
                del samples
                gc.collect()
        
        # Also create a combined file in smaller chunks
        print(f"  📦 Creating combined {split} dataset...")
        
        all_samples = []
        for level in ['easy', 'medium', 'hard', 'expert']:
            level_file = training_dir / f"{split}_{level}_dataset.pkl"
            if level_file.exists():
                try:
                    with open(level_file, 'rb') as f:
                        level_samples = pickle.load(f)
                        all_samples.extend(level_samples)
                        del level_samples
                        gc.collect()
                except Exception as e:
                    print(f"    ⚠️  Couldn't load {level_file}: {e}")
        
        if all_samples:
            # Try to save combined file
            combined_file = training_dir / f"{split}_dataset.pkl"
            try:
                # Save in chunks to avoid memory issues
                chunk_size = 100
                with open(combined_file, 'wb') as f:
                    # Use protocol 4 for better memory efficiency
                    pickler = pickle.Pickler(f, protocol=4)
                    pickler.dump(all_samples)
                
                size_mb = combined_file.stat().st_size / (1024 * 1024)
                print(f"  ✅ {split}_dataset.pkl: {len(all_samples)} samples ({size_mb:.1f} MB)")
                
            except Exception as e:
                print(f"  ⚠️  Combined file failed: {e}")
                print(f"      Training script can use individual level files")
            
            del all_samples
            gc.collect()
    
    print(f"\n🎉 Dataset creation complete!")
    print(f"\n📁 Available files:")
    for f in sorted(training_dir.glob("*_dataset.pkl")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.1f} MB")
    
    print(f"\n🚀 Ready to train!")

if __name__ == "__main__":
    main()