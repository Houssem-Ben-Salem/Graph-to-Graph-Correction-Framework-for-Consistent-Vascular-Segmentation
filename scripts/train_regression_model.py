#!/usr/bin/env python3
"""
Train the regression-based graph correction model
Using real U-Net prediction data only
"""

import os
import sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import logging
import yaml
from tqdm import tqdm
import pickle
from datetime import datetime

from src.models.graph_correction.regression_model import GraphCorrectionRegressionModel
from src.training.regression_dataloader import RegressionGraphDataset, create_regression_dataloader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RegressionTrainer:
    def __init__(self, config_path):
        """Initialize trainer with config"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Setup model
        self.model = GraphCorrectionRegressionModel(self.config['model']).to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )
        
        # Setup directories
        self.experiment_dir = Path('experiments/regression_model')
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Tracking
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
    def load_real_data(self):
        """Load only real U-Net prediction data"""
        logger.info("Loading real U-Net prediction data...")
        
        # Load datasets
        train_path = Path('training_data/train_real_dataset.pkl')
        val_path = Path('training_data/val_real_dataset.pkl')
        
        if not train_path.exists() or not val_path.exists():
            raise FileNotFoundError("Real datasets not found. Please generate them first.")
        
        # Create data loaders using regression dataloader
        self.train_loader = create_regression_dataloader(
            train_path,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['training']['num_workers']
        )
        
        self.val_loader = create_regression_dataloader(
            val_path,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['training']['num_workers']
        )
        
        # Store dataset sizes
        self.train_dataset = self.train_loader.dataset
        self.val_dataset = self.val_loader.dataset
        
        logger.info(f"Loaded {len(self.train_dataset)} training samples")
        logger.info(f"Loaded {len(self.val_dataset)} validation samples")
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Track metrics
        all_magnitudes = []
        all_predicted_magnitudes = []
        all_true_classes = []
        all_pred_classes = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(self.device)
            
            # Forward pass
            predictions = self.model(batch)
            
            # Handle variable batch sizes
            if len(batch.correspondences) == 1:
                # Single sample
                targets = self.model.compute_targets(
                    batch.pred_features,
                    batch.gt_features,
                    batch.correspondences[0]
                )
            else:
                # Multiple samples - process each and concatenate
                batch_targets = {'position_corrections': [], 'correction_magnitudes': [], 
                               'node_operations': [], 'has_correspondence': []}
                
                node_offset = 0
                for i, correspondence in enumerate(batch.correspondences):
                    # Extract features for this sample
                    sample_mask = batch.batch == i
                    sample_pred_features = {
                        'node_positions': batch.pred_features['node_positions'][sample_mask],
                        'node_radii': batch.pred_features['node_radii'][sample_mask],
                        'num_nodes': sample_mask.sum().item()
                    }
                    
                    # GT features for this sample (assuming they're stored per sample)
                    if isinstance(batch.gt_features['node_positions'], list):
                        sample_gt_features = {
                            'node_positions': batch.gt_features['node_positions'][i],
                            'node_radii': batch.gt_features['node_radii'][i],
                            'num_nodes': len(batch.gt_features['node_positions'][i])
                        }
                    else:
                        # Handle tensor format
                        sample_gt_features = batch.gt_features
                    
                    sample_targets = self.model.compute_targets(
                        sample_pred_features, sample_gt_features, correspondence
                    )
                    
                    for key in batch_targets:
                        batch_targets[key].append(sample_targets[key])
                
                # Concatenate all samples
                targets = {key: torch.cat(values, dim=0) for key, values in batch_targets.items()}
            
            # Compute loss
            losses = self.model.compute_loss(
                predictions,
                targets,
                self.config['loss']
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['gradient_clip']
            )
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += losses['total'].item()
            num_batches += 1
            
            # Collect predictions for metrics
            all_magnitudes.extend(targets['correction_magnitudes'].cpu().numpy())
            all_predicted_magnitudes.extend(predictions['correction_magnitudes'].detach().cpu().numpy())
            all_true_classes.extend(targets['node_operations'].cpu().numpy())
            all_pred_classes.extend(predictions['node_operations'].detach().cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'pos_loss': f"{losses['position'].item():.4f}",
                'mag_loss': f"{losses.get('magnitude', 0):.4f}"
            })
        
        # Compute epoch metrics
        avg_loss = total_loss / num_batches
        
        # Magnitude error
        mag_mae = np.mean(np.abs(np.array(all_magnitudes) - np.array(all_predicted_magnitudes)))
        
        # Classification accuracy (derived from magnitudes)
        accuracy = np.mean(np.array(all_true_classes) == np.array(all_pred_classes))
        
        # Per-class accuracy for binary classification
        class_accuracies = {}
        for c in range(2):  # Only 2 classes now
            mask = np.array(all_true_classes) == c
            if mask.sum() > 0:
                class_accuracies[c] = np.mean(
                    np.array(all_pred_classes)[mask] == c
                )
            else:
                class_accuracies[c] = 0.0
        
        logger.info(f"Epoch {self.current_epoch} - "
                   f"Loss: {avg_loss:.4f} - "
                   f"Mag MAE: {mag_mae:.4f} - "
                   f"Accuracy: {accuracy:.4f}")
        logger.info(f"Class accuracies - "
                   f"Modify: {class_accuracies.get(0, 0):.3f}, "
                   f"Remove: {class_accuracies.get(1, 0):.3f}")
        
        return avg_loss
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        all_magnitudes = []
        all_predicted_magnitudes = []
        all_true_classes = []
        all_pred_classes = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                batch = batch.to(self.device)
                
                # Forward pass
                predictions = self.model(batch)
                
                # Handle variable batch sizes (same as training)
                if len(batch.correspondences) == 1:
                    targets = self.model.compute_targets(
                        batch.pred_features, batch.gt_features, batch.correspondences[0]
                    )
                else:
                    # Handle multiple samples (same logic as training)
                    batch_targets = {'position_corrections': [], 'correction_magnitudes': [], 
                                   'node_operations': [], 'has_correspondence': []}
                    
                    for i, correspondence in enumerate(batch.correspondences):
                        sample_mask = batch.batch == i
                        sample_pred_features = {
                            'node_positions': batch.pred_features['node_positions'][sample_mask],
                            'node_radii': batch.pred_features['node_radii'][sample_mask],
                            'num_nodes': sample_mask.sum().item()
                        }
                        
                        if isinstance(batch.gt_features['node_positions'], list):
                            sample_gt_features = {
                                'node_positions': batch.gt_features['node_positions'][i],
                                'node_radii': batch.gt_features['node_radii'][i],
                                'num_nodes': len(batch.gt_features['node_positions'][i])
                            }
                        else:
                            sample_gt_features = batch.gt_features
                        
                        sample_targets = self.model.compute_targets(
                            sample_pred_features, sample_gt_features, correspondence
                        )
                        
                        for key in batch_targets:
                            batch_targets[key].append(sample_targets[key])
                    
                    targets = {key: torch.cat(values, dim=0) for key, values in batch_targets.items()}
                
                # Compute loss
                losses = self.model.compute_loss(
                    predictions,
                    targets,
                    self.config['loss']
                )
                
                total_loss += losses['total'].item()
                num_batches += 1
                
                # Collect predictions
                all_magnitudes.extend(targets['correction_magnitudes'].cpu().numpy())
                all_predicted_magnitudes.extend(predictions['correction_magnitudes'].cpu().numpy())
                all_true_classes.extend(targets['node_operations'].cpu().numpy())
                all_pred_classes.extend(predictions['node_operations'].cpu().numpy())
        
        # Compute metrics
        avg_loss = total_loss / num_batches
        mag_mae = np.mean(np.abs(np.array(all_magnitudes) - np.array(all_predicted_magnitudes)))
        accuracy = np.mean(np.array(all_true_classes) == np.array(all_pred_classes))
        
        # Per-class metrics
        from sklearn.metrics import classification_report
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            class_report = classification_report(
                all_true_classes,
                all_pred_classes,
                target_names=['Modify', 'Remove'],  # Binary classification
                output_dict=True,
                zero_division=0
            )
        
        logger.info(f"Validation - Loss: {avg_loss:.4f} - "
                   f"Mag MAE: {mag_mae:.4f} - "
                   f"Accuracy: {accuracy:.4f}")
        
        # Log detailed metrics for binary classification
        for class_name in ['Modify', 'Remove']:
            if class_name.lower() in class_report:
                metrics = class_report[class_name.lower()]
                logger.info(f"{class_name}: "
                           f"Precision={metrics['precision']:.3f}, "
                           f"Recall={metrics['recall']:.3f}, "
                           f"F1={metrics['f1-score']:.3f}")
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.experiment_dir / 'checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.experiment_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model with val loss: {self.best_val_loss:.4f}")
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        # Load data
        self.load_real_data()
        
        # Optional: Update thresholds based on data
        if self.config.get('update_thresholds', True):
            logger.info("Updating magnitude thresholds from data...")
            self.model.update_thresholds_from_data(self.train_loader)
        
        # Training loop
        for epoch in range(self.config['training']['num_epochs']):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            if epoch % self.config['validation']['val_interval'] == 0:
                val_loss, val_acc = self.validate()
                self.val_losses.append(val_loss)
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(is_best=True)
            
            # Regular checkpoint
            if epoch % self.config['validation']['checkpoint_interval'] == 0:
                self.save_checkpoint()
        
        logger.info("Training completed!")


def main():
    """Main function"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, 
                       default='configs/regression_model_config.yaml',
                       help='Path to config file')
    args = parser.parse_args()
    
    # Create trainer and train
    trainer = RegressionTrainer(args.config)
    trainer.train()


if __name__ == '__main__':
    main()