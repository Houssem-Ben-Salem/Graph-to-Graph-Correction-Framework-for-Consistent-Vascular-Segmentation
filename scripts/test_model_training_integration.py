#!/usr/bin/env python
"""
Test Model Training Integration
Quick test to verify the model can train with our Step 4 training data
"""

import sys
import torch
import logging
from pathlib import Path
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.graph_correction import GraphCorrectionModel
from src.training.graph_correction_dataloader import CurriculumDataManager, DataLoaderConfig
from src.training.dataset_generator import TrainingDatasetGenerator

def test_model_training_integration():
    """Test that the model can train with our Step 4 data pipeline"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("=== Testing Model Training Integration ===")
    
    try:
        # Step 1: Generate minimal training data
        logger.info("Step 1: Generating minimal training data...")
        
        extracted_graphs_dir = Path("extracted_graphs")
        training_data_dir = Path("test_integration_data")
        
        generator_config = {
            'samples_per_graph': 1,  # Minimal
            'max_graphs_to_process': 2,  # Very small for testing
            'parallel_workers': 1,
            'min_correspondence_quality': 0.1,  # Very low threshold
            'compress_output': False
        }
        
        generator = TrainingDatasetGenerator(
            extracted_graphs_dir=extracted_graphs_dir,
            output_dir=training_data_dir,
            config=generator_config
        )
        
        # Generate just easy level data
        train_summary = generator.generate_training_dataset(
            split="train",
            curriculum_stage="easy"
        )
        
        val_summary = generator.generate_training_dataset(
            split="val", 
            curriculum_stage="easy"
        )
        
        logger.info(f"✅ Generated {train_summary['total_samples']} train samples")
        logger.info(f"✅ Generated {val_summary['total_samples']} val samples")
        
        # Step 2: Create data manager
        logger.info("Step 2: Creating data manager...")
        
        dataloader_config = DataLoaderConfig(
            batch_size=1,  # Single sample
            num_workers=0,  # Avoid multiprocessing in test
            curriculum_enabled=True,
            shuffle=False
        )
        
        data_manager = CurriculumDataManager(
            train_dataset_dir=training_data_dir,
            val_dataset_dir=training_data_dir,
            config=dataloader_config
        )
        
        train_loader, val_loader = data_manager.get_dataloaders_for_stage("easy")
        logger.info(f"✅ Created data loaders: {len(train_loader)} train, {len(val_loader)} val batches")
        
        # Step 3: Initialize model
        logger.info("Step 3: Initializing model...")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        model = GraphCorrectionModel()
        model.to(device)
        model.train()
        
        # Create simple optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        summary = model.get_model_summary()
        logger.info(f"✅ Model initialized: {summary['total_parameters']:,} parameters")
        
        # Step 4: Test single forward pass
        logger.info("Step 4: Testing single forward pass...")
        
        train_iter = iter(train_loader)
        batch = next(train_iter)
        
        # Extract sample data
        pred_graph = batch['degraded_graphs'][0]
        gt_graph = batch['gt_graphs'][0]
        correspondences = batch['correspondences'][0]
        
        logger.info(f"Sample data loaded:")
        logger.info(f"  - GT graph: {len(gt_graph.nodes)} nodes, {len(gt_graph.edges)} edges")
        logger.info(f"  - Degraded graph: {len(pred_graph.nodes)} nodes, {len(pred_graph.edges)} edges")
        logger.info(f"  - Correspondences: {len(correspondences.node_correspondences)} node matches")
        
        # Forward pass
        start_time = time.time()
        outputs = model(pred_graph, gt_graph, correspondences, training_mode=True)
        forward_time = time.time() - start_time
        
        logger.info(f"✅ Forward pass completed in {forward_time:.3f}s")
        logger.info(f"  - Node operations shape: {outputs['node_operations'].shape}")
        logger.info(f"  - Quality score: {outputs['quality_score'].item():.3f}")
        
        # Step 5: Test backward pass
        logger.info("Step 5: Testing backward pass...")
        
        # Simple loss computation
        node_ops = outputs['node_operations']
        dummy_targets = torch.zeros(node_ops.size(0), dtype=torch.long, device=device)
        loss = torch.nn.functional.cross_entropy(node_ops, dummy_targets)
        
        # Backward pass
        optimizer.zero_grad()
        start_time = time.time()
        loss.backward()
        backward_time = time.time() - start_time
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Optimizer step
        optimizer.step()
        
        logger.info(f"✅ Backward pass completed in {backward_time:.3f}s")
        logger.info(f"  - Loss: {loss.item():.4f}")
        
        # Step 6: Test mini training loop
        logger.info("Step 6: Testing mini training loop...")
        
        model.train()
        total_loss = 0.0
        
        for i, batch in enumerate(train_loader):
            if i >= 3:  # Only test first 3 batches
                break
                
            pred_graph = batch['degraded_graphs'][0]
            gt_graph = batch['gt_graphs'][0]
            correspondences = batch['correspondences'][0]
            
            # Forward
            outputs = model(pred_graph, gt_graph, correspondences, training_mode=True)
            
            # Loss
            node_ops = outputs['node_operations']
            dummy_targets = torch.zeros(node_ops.size(0), dtype=torch.long, device=device)
            loss = torch.nn.functional.cross_entropy(node_ops, dummy_targets)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            logger.info(f"  Batch {i+1}: Loss = {loss.item():.4f}")
        
        avg_loss = total_loss / min(3, len(train_loader))
        logger.info(f"✅ Mini training loop completed: Average loss = {avg_loss:.4f}")
        
        # Step 7: Test validation mode
        logger.info("Step 7: Testing validation mode...")
        
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= 2:  # Only test first 2 validation batches
                    break
                    
                pred_graph = batch['degraded_graphs'][0]
                gt_graph = batch['gt_graphs'][0]
                correspondences = batch['correspondences'][0]
                
                outputs = model(pred_graph, gt_graph, correspondences, training_mode=False)
                
                node_ops = outputs['node_operations']
                dummy_targets = torch.zeros(node_ops.size(0), dtype=torch.long, device=device)
                loss = torch.nn.functional.cross_entropy(node_ops, dummy_targets)
                
                val_loss += loss.item()
                logger.info(f"  Val Batch {i+1}: Loss = {loss.item():.4f}")
        
        avg_val_loss = val_loss / min(2, len(val_loader))
        logger.info(f"✅ Validation completed: Average loss = {avg_val_loss:.4f}")
        
        logger.info("\n🎉 === INTEGRATION TEST PASSED! ===")
        logger.info("The model successfully integrates with Step 4 training data pipeline")
        logger.info("Ready for full training with the complete framework!")
        
        # Cleanup
        logger.info("\nCleaning up test files...")
        import shutil
        if training_data_dir.exists():
            shutil.rmtree(training_data_dir)
        logger.info("✅ Cleanup completed")
        
        return True
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_training_integration()
    sys.exit(0 if success else 1)