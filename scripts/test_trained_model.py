#!/usr/bin/env python3
"""
Test the trained regression model on real data
Quick validation to see if the model works as expected
"""

import sys
sys.path.append('.')

import torch
import numpy as np
from pathlib import Path
import pickle
import logging

from src.models.graph_correction.regression_model import GraphCorrectionRegressionModel
from src.training.regression_dataloader import RegressionGraphDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_trained_model():
    """Test the trained model on some real data"""
    
    # Load trained model
    model_path = Path('experiments/regression_model/best_model.pth')
    
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return
    
    logger.info("Loading trained model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize model
    model_config = checkpoint['config']['model']
    model = GraphCorrectionRegressionModel(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully (trained for {checkpoint['epoch']} epochs)")
    logger.info(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
    
    # Load some test data
    test_data_path = Path('training_data/val_real_dataset.pkl')
    
    if not test_data_path.exists():
        logger.error(f"Test data not found at {test_data_path}")
        return
    
    logger.info("Loading test data...")
    dataset = RegressionGraphDataset(test_data_path)
    
    logger.info(f"Test dataset has {len(dataset)} samples")
    
    # Test on a few samples
    num_test_samples = min(5, len(dataset))
    
    results = []
    
    with torch.no_grad():
        for i in range(num_test_samples):
            logger.info(f"Testing sample {i+1}/{num_test_samples}")
            
            # Get sample
            sample = dataset[i]
            
            # Move to device
            sample = sample.to(device)
            
            # Add batch dimension
            sample.batch = torch.zeros(sample.x.size(0), dtype=torch.long).to(device)
            
            # Forward pass
            predictions = model(sample)
            
            # Extract results
            position_corrections = predictions['position_corrections'].cpu().numpy()
            correction_magnitudes = predictions['correction_magnitudes'].cpu().numpy()
            node_operations = predictions['node_operations'].cpu().numpy()
            predicted_positions = predictions['predicted_positions'].cpu().numpy()
            
            # Compute targets for comparison
            targets = model.compute_targets(
                sample.pred_features,
                sample.gt_features,
                sample.correspondences
            )
            
            true_corrections = targets['position_corrections'].cpu().numpy()
            true_magnitudes = targets['correction_magnitudes'].cpu().numpy()
            true_operations = targets['node_operations'].cpu().numpy()
            
            # Compute metrics
            magnitude_mae = np.mean(np.abs(correction_magnitudes - true_magnitudes))
            position_mae = np.mean(np.abs(position_corrections - true_corrections))
            operation_accuracy = np.mean(node_operations == true_operations)
            
            sample_result = {
                'sample_id': sample.sample_id,
                'patient_id': sample.patient_id,
                'num_nodes': len(position_corrections),
                'magnitude_mae': magnitude_mae,
                'position_mae': position_mae,
                'operation_accuracy': operation_accuracy,
                'predicted_operations': {
                    'modify': int(np.sum(node_operations == 0)),
                    'remove': int(np.sum(node_operations == 1))
                },
                'true_operations': {
                    'modify': int(np.sum(true_operations == 0)),
                    'remove': int(np.sum(true_operations == 1))
                },
                'correction_stats': {
                    'mean_predicted_magnitude': float(np.mean(correction_magnitudes)),
                    'mean_true_magnitude': float(np.mean(true_magnitudes)),
                    'max_predicted_magnitude': float(np.max(correction_magnitudes)),
                    'max_true_magnitude': float(np.max(true_magnitudes))
                }
            }
            
            results.append(sample_result)
            
            logger.info(f"  Sample {sample.patient_id}:")
            logger.info(f"    Nodes: {sample_result['num_nodes']}")
            logger.info(f"    Magnitude MAE: {magnitude_mae:.3f}")
            logger.info(f"    Position MAE: {position_mae:.3f}")
            logger.info(f"    Operation Accuracy: {operation_accuracy:.3f}")
            logger.info(f"    Predicted ops - Modify: {sample_result['predicted_operations']['modify']}, Remove: {sample_result['predicted_operations']['remove']}")
            logger.info(f"    True ops - Modify: {sample_result['true_operations']['modify']}, Remove: {sample_result['true_operations']['remove']}")
    
    # Overall statistics
    logger.info("\n" + "="*60)
    logger.info("OVERALL TEST RESULTS")
    logger.info("="*60)
    
    avg_magnitude_mae = np.mean([r['magnitude_mae'] for r in results])
    avg_position_mae = np.mean([r['position_mae'] for r in results])
    avg_operation_accuracy = np.mean([r['operation_accuracy'] for r in results])
    
    total_nodes = sum(r['num_nodes'] for r in results)
    total_predicted_modify = sum(r['predicted_operations']['modify'] for r in results)
    total_predicted_remove = sum(r['predicted_operations']['remove'] for r in results)
    total_true_modify = sum(r['true_operations']['modify'] for r in results)
    total_true_remove = sum(r['true_operations']['remove'] for r in results)
    
    logger.info(f"Samples tested: {len(results)}")
    logger.info(f"Total nodes: {total_nodes}")
    logger.info(f"")
    logger.info(f"Average Magnitude MAE: {avg_magnitude_mae:.3f}")
    logger.info(f"Average Position MAE: {avg_position_mae:.3f}")
    logger.info(f"Average Operation Accuracy: {avg_operation_accuracy:.3f} ({100*avg_operation_accuracy:.1f}%)")
    logger.info(f"")
    logger.info(f"Operation Distribution:")
    logger.info(f"  Predicted - Modify: {total_predicted_modify} ({100*total_predicted_modify/total_nodes:.1f}%), Remove: {total_predicted_remove} ({100*total_predicted_remove/total_nodes:.1f}%)")
    logger.info(f"  True      - Modify: {total_true_modify} ({100*total_true_modify/total_nodes:.1f}%), Remove: {total_true_remove} ({100*total_true_remove/total_nodes:.1f}%)")
    
    logger.info("="*60)
    
    # Save results
    results_path = Path('experiments/regression_model/test_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    logger.info(f"Test results saved to {results_path}")
    
    return results


if __name__ == '__main__':
    test_trained_model()