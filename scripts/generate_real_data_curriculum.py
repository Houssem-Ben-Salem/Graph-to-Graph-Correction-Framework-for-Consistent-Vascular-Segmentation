#!/usr/bin/env python
"""Generate real GT/PRED training data for curriculum learning"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.training.dataset_generator import TrainingDatasetGenerator

def main():
    print("🔄 Generating Real GT/PRED Training Data\n")
    
    # Configuration
    config = {
        'samples_per_graph': 1,  # One sample per GT/PRED pair
        'max_graphs_to_process': None,
        'parallel_workers': 4,
        'min_correspondence_quality': 0.3,
        'min_node_coverage': 0.5,
        'max_degradation_rate': 0.8,  # More lenient for real data
        'curriculum_enabled': True,
        'adaptive_sampling': True,
        'cache_enabled': True,
        'cache_max_size_gb': 10.0,
        'cache_compress': True,
        'cache_preload': False,
        'cache_auto_cleanup': True,
        'save_metadata': True,
        'compress_output': True,
        'validate_generated_data': True,
        'validation_sample_rate': 0.1
    }
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create generator
    generator = TrainingDatasetGenerator(
        extracted_graphs_dir=Path("extracted_graphs"),
        output_dir=Path("training_data"),
        config=config
    )
    
    print("📊 Generating real data for each split...")
    
    # Generate for each split
    for split in ['train', 'val', 'test']:
        print(f"\n=== {split.upper()} SPLIT ===")
        
        try:
            # Generate only "real" curriculum level
            samples = generator._generate_curriculum_level_samples(split, "real")
            
            if samples:
                # Save the real data samples
                output_file = Path("training_data") / f"{split}_real_dataset.pkl"
                generator._save_dataset(samples, output_file)
                
                print(f"✅ {split} real data: {len(samples)} samples")
                
                # Show sample stats
                if samples:
                    patient_ids = set(s['patient_id'] for s in samples)
                    print(f"   Patients: {len(patient_ids)}")
                    print(f"   Example patients: {list(patient_ids)[:5]}")
            else:
                print(f"⚠️  No real data samples generated for {split}")
                
        except Exception as e:
            print(f"❌ Error generating {split} real data: {e}")
    
    print(f"\n🎉 Real data generation complete!")
    print(f"\n📁 Generated files:")
    
    for split in ['train', 'val', 'test']:
        for ext in ['.pkl', '.pkl.gz']:
            file_path = Path("training_data") / f"{split}_real_dataset{ext}"
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"  {file_path.name}: {size_mb:.1f} MB")
    
    print(f"\n🚀 Ready to use real data in training!")
    print("Update your training configuration to include 'real' curriculum stage.")

if __name__ == "__main__":
    main()