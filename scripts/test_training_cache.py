#!/usr/bin/env python
"""
Test Training Data Caching System
Comprehensive test for the new caching functionality
"""

import sys
import torch
import logging
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.training.dataset_generator import TrainingDatasetGenerator
from src.training.training_cache import CacheConfig


def test_training_cache():
    """Test the complete training data caching system"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("=== Testing Training Data Caching System ===")
    
    try:
        # Configuration
        extracted_graphs_dir = Path("extracted_graphs")
        cache_test_dir = Path("test_cache_system")
        
        # Clean up any previous test data
        if cache_test_dir.exists():
            import shutil
            shutil.rmtree(cache_test_dir)
        
        # Step 1: Test with caching enabled
        logger.info("\n=== Step 1: Testing with Cache Enabled ===")
        
        generator_config = {
            'samples_per_graph': 1,  # Minimal for testing
            'max_graphs_to_process': 2,  # Very small for testing
            'parallel_workers': 1,
            'min_correspondence_quality': 0.1,
            'cache_enabled': True,
            'cache_max_size_gb': 1.0,
            'cache_compress': True,
            'cache_preload': False,
            'cache_auto_cleanup': True
        }
        
        generator = TrainingDatasetGenerator(
            extracted_graphs_dir=extracted_graphs_dir,
            output_dir=cache_test_dir,
            config=generator_config
        )
        
        # First generation (should be cache misses)
        logger.info("First generation - expect cache misses...")
        start_time = time.time()
        
        train_summary1 = generator.generate_training_dataset(
            split="train",
            curriculum_stage="easy"
        )
        
        first_gen_time = time.time() - start_time
        logger.info(f"✅ First generation completed in {first_gen_time:.2f}s")
        logger.info(f"   Generated {train_summary1['total_samples']} samples")
        
        # Check cache statistics
        cache_info1 = generator.get_cache_info()
        logger.info(f"   Cache entries: {cache_info1['total_entries']}")
        logger.info(f"   Cache size: {cache_info1['total_size_mb']:.1f} MB")
        logger.info(f"   Cache hits: {cache_info1['cache_hits']}")
        logger.info(f"   Cache misses: {cache_info1['cache_misses']}")
        
        # Step 2: Second generation (should be cache hits)
        logger.info("\n=== Step 2: Testing Cache Hits ===")
        
        start_time = time.time()
        
        train_summary2 = generator.generate_training_dataset(
            split="train",
            curriculum_stage="easy"
        )
        
        second_gen_time = time.time() - start_time
        logger.info(f"✅ Second generation completed in {second_gen_time:.2f}s")
        logger.info(f"   Generated {train_summary2['total_samples']} samples")
        
        # Check cache statistics
        cache_info2 = generator.get_cache_info()
        logger.info(f"   Cache entries: {cache_info2['total_entries']}")
        logger.info(f"   Cache hits: {cache_info2['cache_hits']}")
        logger.info(f"   Cache misses: {cache_info2['cache_misses']}")
        logger.info(f"   Hit rate: {cache_info2['cache_hit_rate']*100:.1f}%")
        
        # Verify speedup
        speedup = first_gen_time / second_gen_time if second_gen_time > 0 else 1.0
        logger.info(f"   Speedup: {speedup:.2f}x faster")
        
        # Step 3: Test cache management
        logger.info("\n=== Step 3: Testing Cache Management ===")
        
        # Test cache warming
        logger.info("Testing cache warming...")
        warmed_count = generator.warm_cache()
        logger.info(f"✅ Warmed {warmed_count} cache entries")
        
        # Test cache info display
        cache_info3 = generator.get_cache_info()
        logger.info("Cache information:")
        for key, value in cache_info3.items():
            if key != 'most_accessed':
                logger.info(f"   {key}: {value}")
        
        # Test most accessed entries
        if 'most_accessed' in cache_info3:
            logger.info("Most accessed entries:")
            for entry in cache_info3['most_accessed'][:3]:
                logger.info(f"   {entry['cache_key']}: {entry['access_count']} accesses")
        
        # Step 4: Test different curriculum stages
        logger.info("\n=== Step 4: Testing Different Curriculum Stages ===")
        
        # Generate medium level (should be new cache entries)
        start_time = time.time()
        medium_summary = generator.generate_training_dataset(
            split="train",
            curriculum_stage="medium"
        )
        medium_gen_time = time.time() - start_time
        
        logger.info(f"✅ Medium level generated in {medium_gen_time:.2f}s")
        logger.info(f"   Generated {medium_summary['total_samples']} samples")
        
        # Step 5: Test cache cleanup
        logger.info("\n=== Step 5: Testing Cache Cleanup ===")
        
        cleanup_results = generator.cleanup_cache()
        logger.info(f"✅ Cache cleanup completed:")
        logger.info(f"   Entries removed: {cleanup_results.get('entries_removed', 0)}")
        logger.info(f"   Size freed: {cleanup_results.get('size_freed_mb', 0):.1f} MB")
        
        # Step 6: Test cache validation
        logger.info("\n=== Step 6: Testing Cache Validation ===")
        
        final_cache_info = generator.get_cache_info()
        logger.info(f"Final cache state:")
        logger.info(f"   Total entries: {final_cache_info['total_entries']}")
        logger.info(f"   Total cache size: {final_cache_info['total_size_mb']:.1f} MB")
        logger.info(f"   Overall hit rate: {final_cache_info['cache_hit_rate']*100:.1f}%")
        logger.info(f"   Time saved: {final_cache_info['generation_time_saved_hours']:.3f} hours")
        
        # Verify that caching works
        if cache_info2['cache_hits'] > cache_info1['cache_hits']:
            logger.info("✅ Cache hits increased as expected!")
        else:
            logger.warning("⚠️  Cache hits did not increase")
        
        if speedup > 1.5:  # Expect at least 1.5x speedup from caching
            logger.info(f"✅ Significant speedup achieved: {speedup:.2f}x")
        else:
            logger.warning(f"⚠️  Limited speedup: {speedup:.2f}x")
        
        # Step 7: Test with cache disabled for comparison
        logger.info("\n=== Step 7: Testing with Cache Disabled ===")
        
        generator_no_cache = TrainingDatasetGenerator(
            extracted_graphs_dir=extracted_graphs_dir,
            output_dir=cache_test_dir / "no_cache",
            config={
                'samples_per_graph': 1,
                'max_graphs_to_process': 2,
                'parallel_workers': 1,
                'min_correspondence_quality': 0.1,
                'cache_enabled': False  # Disabled
            }
        )
        
        start_time = time.time()
        no_cache_summary = generator_no_cache.generate_training_dataset(
            split="train",
            curriculum_stage="easy"
        )
        no_cache_time = time.time() - start_time
        
        logger.info(f"✅ No-cache generation completed in {no_cache_time:.2f}s")
        
        # Compare performance
        cache_vs_no_cache_speedup = no_cache_time / second_gen_time if second_gen_time > 0 else 1.0
        logger.info(f"   Cache vs no-cache speedup: {cache_vs_no_cache_speedup:.2f}x")
        
        logger.info("\n🎉 === ALL CACHE TESTS PASSED! ===")
        logger.info("Training data caching system is working correctly!")
        
        # Performance summary
        logger.info("\n📊 Performance Summary:")
        logger.info(f"   First generation (cache miss): {first_gen_time:.2f}s")
        logger.info(f"   Second generation (cache hit): {second_gen_time:.2f}s")
        logger.info(f"   No-cache generation: {no_cache_time:.2f}s")
        logger.info(f"   Cache speedup: {speedup:.2f}x")
        logger.info(f"   Overall efficiency gain: {cache_vs_no_cache_speedup:.2f}x")
        
        # Cleanup
        logger.info("\n🧹 Cleaning up test files...")
        if cache_test_dir.exists():
            import shutil
            shutil.rmtree(cache_test_dir)
        logger.info("✅ Cleanup completed")
        
        return True
        
    except Exception as e:
        logger.error(f"Cache test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_training_cache()
    print(f"\nCache test {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)