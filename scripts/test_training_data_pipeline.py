#!/usr/bin/env python
"""
Test Training Data Pipeline
Validates the complete Step 4 training data generation pipeline
"""

import sys
import torch
import logging
import numpy as np
from pathlib import Path
from typing import Dict
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.training.dataset_generator import TrainingDatasetGenerator
from src.training.graph_correction_dataloader import CurriculumDataManager, DataLoaderConfig
from src.models.graph_extraction.vascular_graph import VascularGraph

def test_training_data_pipeline():
    """Test the complete training data generation and loading pipeline"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("=== Testing Training Data Pipeline ===")
    
    # Configuration
    extracted_graphs_dir = Path("extracted_graphs")
    output_dir = Path("test_training_data")
    
    # Verify extracted graphs exist
    if not extracted_graphs_dir.exists():
        logger.error(f"Extracted graphs directory not found: {extracted_graphs_dir}")
        return False
    
    # Find available graphs
    available_graphs = list(extracted_graphs_dir.glob("PA*/PA*_GT.pkl"))
    if len(available_graphs) < 3:
        logger.error(f"Need at least 3 graphs for testing, found {len(available_graphs)}")
        return False
    
    logger.info(f"Found {len(available_graphs)} extracted graphs")
    
    try:
        # Test 1: Dataset Generator Initialization
        logger.info("\n=== Test 1: Dataset Generator Initialization ===")
        
        generator_config = {
            'samples_per_graph': 2,  # Small for testing
            'max_graphs_to_process': 3,  # Limit for testing
            'parallel_workers': 2,
            'min_correspondence_quality': 0.2,  # Lower threshold for testing
            'compress_output': False  # Faster for testing
        }
        
        generator = TrainingDatasetGenerator(
            extracted_graphs_dir=extracted_graphs_dir,
            output_dir=output_dir,
            config=generator_config
        )
        
        logger.info("✅ Dataset generator initialized successfully")
        
        # Test 2: Generate Small Training Dataset
        logger.info("\n=== Test 2: Generate Small Training Dataset ===")
        
        start_time = time.time()
        train_summary = generator.generate_training_dataset(
            split="train",
            curriculum_stage="easy"  # Start with easy level
        )
        generation_time = time.time() - start_time
        
        logger.info(f"✅ Generated training dataset in {generation_time:.2f}s")
        logger.info(f"   Total samples: {train_summary['total_samples']}")
        logger.info(f"   Dataset path: {train_summary['dataset_path']}")
        
        # Test 3: Validate Generated Dataset
        logger.info("\n=== Test 3: Validate Generated Dataset ===")
        
        dataset_path = Path(train_summary['dataset_path'])
        validation_results = generator.validate_generated_dataset(dataset_path)
        
        logger.info(f"✅ Dataset validation completed")
        logger.info(f"   Valid samples: {validation_results['valid_samples']}")
        logger.info(f"   Invalid samples: {validation_results['invalid_samples']}")
        logger.info(f"   Validation success: {validation_results['validation_success']}")
        
        if not validation_results['validation_success']:
            logger.error("Dataset validation failed!")
            return False
        
        # Test 4: Curriculum Data Manager
        logger.info("\n=== Test 4: Curriculum Data Manager ===")
        
        # Create train and val datasets
        val_summary = generator.generate_training_dataset(
            split="val", 
            curriculum_stage="easy"
        )
        
        # Initialize curriculum manager
        train_dir = output_dir
        val_dir = output_dir
        
        dataloader_config = DataLoaderConfig(
            batch_size=2,
            num_workers=0,  # Avoid multiprocessing issues in tests
            curriculum_enabled=True
        )
        
        curriculum_manager = CurriculumDataManager(
            train_dataset_dir=train_dir,
            val_dataset_dir=val_dir,
            config=dataloader_config
        )
        
        logger.info("✅ Curriculum manager initialized successfully")
        
        # Test 5: Data Loader Creation
        logger.info("\n=== Test 5: Data Loader Creation ===")
        
        train_loader, val_loader = curriculum_manager.get_dataloaders_for_stage("easy")
        
        logger.info(f"✅ Data loaders created successfully")
        logger.info(f"   Train batches: {len(train_loader)}")
        logger.info(f"   Val batches: {len(val_loader)}")
        
        # Test 6: Sample Data Loading
        logger.info("\n=== Test 6: Sample Data Loading ===")
        
        # Test loading a single batch
        train_iter = iter(train_loader)
        try:
            batch = next(train_iter)
            
            logger.info(f"✅ Successfully loaded training batch")
            logger.info(f"   Batch size: {batch['batch_stats']['batch_size']}")
            logger.info(f"   GT graphs: {len(batch['gt_graphs'])}")
            logger.info(f"   Degraded graphs: {len(batch['degraded_graphs'])}")
            logger.info(f"   Correspondences: {len(batch['correspondences'])}")
            
            # Inspect first sample
            gt_graph = batch['gt_graphs'][0]
            degraded_graph = batch['degraded_graphs'][0]
            correspondences = batch['correspondences'][0]
            
            logger.info(f"   Sample 0 - GT: {len(gt_graph.nodes)} nodes, {len(gt_graph.edges)} edges")
            logger.info(f"   Sample 0 - Degraded: {len(degraded_graph.nodes)} nodes, {len(degraded_graph.edges)} edges")
            logger.info(f"   Sample 0 - Node correspondences: {len(correspondences.node_correspondences)}")
            
        except Exception as e:
            logger.error(f"Failed to load training batch: {e}")
            return False
        
        # Test 7: Curriculum Progression
        logger.info("\n=== Test 7: Curriculum Progression ===")
        
        current_stage = curriculum_manager.get_current_stage()
        logger.info(f"   Current stage: {current_stage}")
        
        # Test advancement logic
        should_advance = curriculum_manager.should_advance_curriculum(
            epoch=15, 
            performance_metric=0.85,
            min_epochs=10
        )
        logger.info(f"   Should advance curriculum: {should_advance}")
        
        if should_advance:
            advanced = curriculum_manager.advance_curriculum()
            logger.info(f"   Advanced to: {curriculum_manager.get_current_stage()}")
        
        # Test 8: Quality Statistics
        logger.info("\n=== Test 8: Quality Statistics ===")
        
        if 'quality_stats' in validation_results:
            stats = validation_results['quality_stats']
            logger.info(f"   Average quality: {stats.get('avg_quality', 0):.3f}")
            logger.info(f"   Average coverage: {stats.get('avg_coverage', 0):.3f}")
            logger.info(f"   Quality range: {stats.get('min_quality', 0):.3f} - {stats.get('max_quality', 0):.3f}")
        
        # Test 9: Synthetic Degradation Validation
        logger.info("\n=== Test 9: Synthetic Degradation Validation ===")
        
        # Load a sample and check degradation metadata
        sample_batch = batch
        degradation_metadata = sample_batch['degradation_metadata'][0]
        
        logger.info(f"   Degradation level: {degradation_metadata.get('degradation_level', 'unknown')}")
        logger.info(f"   Applied degradations: {len(degradation_metadata.get('applied_degradations', []))}")
        
        if 'degradation_summary' in degradation_metadata:
            summary = degradation_metadata['degradation_summary']
            logger.info(f"   Node loss rate: {summary.get('node_loss_rate', 0):.3f}")
            logger.info(f"   Edge loss rate: {summary.get('edge_loss_rate', 0):.3f}")
        
        logger.info("\n🎉 === ALL TESTS PASSED! ===")
        logger.info("Training data pipeline is working correctly and ready for model training")
        
        # Cleanup test files
        logger.info("\nCleaning up test files...")
        import shutil
        if output_dir.exists():
            shutil.rmtree(output_dir)
        logger.info("✅ Cleanup completed")
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_training_data_pipeline()
    sys.exit(0 if success else 1)