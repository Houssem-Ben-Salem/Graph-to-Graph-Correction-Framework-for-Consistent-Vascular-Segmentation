#!/usr/bin/env python3
"""
Train the enhanced regression-based graph correction model
Addresses training plateaus with:
1. Better loss functions
2. Learning rate scheduling
3. Advanced regularization
4. Improved monitoring
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
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from src.models.graph_correction.enhanced_regression_model import EnhancedGraphCorrectionModel
from src.training.regression_dataloader import RegressionGraphDataset, create_regression_dataloader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedRegressionTrainer:
    def __init__(self, config_path):
        """Initialize enhanced trainer with config"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Setup enhanced model
        self.model = EnhancedGraphCorrectionModel(self.config['model']).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Enhanced optimizer with different learning rates for different components
        param_groups = [
            # Main model parameters
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(x in n for x in ['threshold', 'scale'])],
                'lr': self.config['training']['learning_rate'],
                'weight_decay': self.config['training']['weight_decay']
            },
            # Threshold and scale parameters (slower learning)
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(x in n for x in ['threshold', 'scale'])],
                'lr': self.config['training']['learning_rate'] * 0.1,
                'weight_decay': 0.0  # No weight decay for these
            }
        ]
        
        self.optimizer = optim.AdamW(param_groups)
        
        # Learning rate scheduler
        self.scheduler = self.model.get_learning_rate_schedule(
            self.optimizer, self.config['training']['num_epochs']
        )
        
        # Gradient scaler for mixed precision
        self.scaler = torch.amp.GradScaler('cuda') if self.device.type == 'cuda' else None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [], 'val_loss': [], 'accuracy': [], 
            'lr': [], 'threshold': [], 'scale': []
        }
        
        # Setup experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = Path(f'experiments/enhanced_model_{timestamp}')
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup tensorboard
        self.writer = SummaryWriter(self.experiment_dir / 'tensorboard')
        
        # Save config
        with open(self.experiment_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.config, f)
        
        logger.info(f"Experiment directory: {self.experiment_dir}")
    
    def load_data(self):
        """Load training and validation data"""
        logger.info("Loading data...")
        
        # Use existing dataset paths from the working regression model
        train_path = Path('training_data/train_real_dataset.pkl')
        val_path = Path('training_data/val_real_dataset.pkl')
        
        if not train_path.exists() or not val_path.exists():
            raise FileNotFoundError("Real datasets not found. Please generate them first.")
        
        # Create data loaders using the same interface as working model
        self.train_loader = create_regression_dataloader(
            train_path,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['training'].get('num_workers', 4)
        )
        
        self.val_loader = create_regression_dataloader(
            val_path,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['training'].get('num_workers', 4)
        )
        
        logger.info(f"Train samples: {len(self.train_loader.dataset)}, Val samples: {len(self.val_loader.dataset)}")
        logger.info("Data loading completed")
    
    def train_epoch(self):
        """Train for one epoch with enhanced monitoring"""
        self.model.train()
        
        total_loss = 0
        total_samples = 0
        component_losses = {'position': 0, 'magnitude': 0, 'confidence': 0, 
                           'classification': 0, 'consistency': 0}
        
        # Metrics tracking
        correct_predictions = 0
        total_predictions = 0
        magnitude_errors = []
        all_true_classes = []
        all_pred_classes = []
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            batch = batch.to(self.device)
            
            # Forward pass with mixed precision
            with torch.amp.autocast('cuda', enabled=self.scaler is not None):
                predictions = self.model(batch)
                
                # Handle batch correspondences (same logic as working model)
                if len(batch.correspondences) == 1:
                    # Single sample
                    targets = self.model.compute_targets(
                        batch.pred_features, batch.gt_features, batch.correspondences[0]
                    )
                else:
                    # Multiple samples - process each and concatenate (this shouldn't happen with batch_size=1)
                    raise NotImplementedError("Batch size > 1 not implemented for enhanced model yet")
                
                # Compute enhanced loss
                losses = self.model.compute_enhanced_loss(
                    predictions, targets, self.config['loss']
                )
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                self.scaler.scale(losses['total']).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training']['gradient_clip']
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses['total'].backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training']['gradient_clip']
                )
                self.optimizer.step()
            
            # Update learning rate
            self.scheduler.step()
            
            # Track metrics
            batch_size = batch.x.size(0)
            total_loss += losses['total'].item() * batch_size
            total_samples += batch_size
            
            # Component losses
            for key in component_losses:
                if key in losses:
                    component_losses[key] += losses[key].item() * batch_size
            
            # Accuracy calculation
            with torch.no_grad():
                pred_classes = predictions['node_operations']
                true_classes = targets['classification_targets']
                correct_predictions += (pred_classes == true_classes).sum().item()
                total_predictions += true_classes.size(0)
                
                # Collect for per-class accuracy
                all_true_classes.extend(true_classes.cpu().numpy())
                all_pred_classes.extend(pred_classes.cpu().numpy())
                
                # Magnitude errors
                mag_error = torch.abs(predictions['predicted_magnitudes'] - targets['magnitude_targets'])
                magnitude_errors.extend(mag_error.cpu().numpy())
            
            # Update progress bar
            avg_loss = total_loss / total_samples
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            progress_bar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Acc': f'{accuracy:.3f}',
                'LR': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Log to tensorboard (every 100 batches)
            if batch_idx % 100 == 0:
                global_step = self.current_epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss_Batch', losses['total'].item(), global_step)
                self.writer.add_scalar('Train/Accuracy_Batch', accuracy, global_step)
                self.writer.add_scalar('Train/LR', self.scheduler.get_last_lr()[0], global_step)
                
                # Log component losses
                for key, value in losses.items():
                    if key != 'total':
                        self.writer.add_scalar(f'Train/Loss_{key.title()}', value.item(), global_step)
        
        # Compute epoch metrics
        avg_loss = total_loss / total_samples
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        avg_magnitude_error = np.mean(magnitude_errors) if magnitude_errors else 0
        
        # Per-class accuracy for binary classification
        class_accuracies = {}
        all_true_classes_np = np.array(all_true_classes)
        all_pred_classes_np = np.array(all_pred_classes)
        
        for c in range(2):  # Binary classification: 0=Modify, 1=Remove
            mask = all_true_classes_np == c
            if mask.sum() > 0:
                class_accuracies[c] = np.mean(all_pred_classes_np[mask] == c)
            else:
                class_accuracies[c] = 0.0
        
        # Average component losses
        for key in component_losses:
            component_losses[key] /= total_samples
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'magnitude_mae': avg_magnitude_error,
            'component_losses': component_losses,
            'class_accuracies': class_accuracies
        }
    
    def validate(self):
        """Validate model with enhanced metrics"""
        self.model.eval()
        
        total_loss = 0
        total_samples = 0
        correct_predictions = 0
        total_predictions = 0
        magnitude_errors = []
        all_true_classes = []
        all_pred_classes = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                batch = batch.to(self.device)
                
                # Forward pass
                predictions = self.model(batch)
                
                # Handle batch correspondences
                if len(batch.correspondences) == 1:
                    targets = self.model.compute_targets(
                        batch.pred_features, batch.gt_features, batch.correspondences[0]
                    )
                else:
                    raise NotImplementedError("Batch size > 1 not implemented for enhanced model yet")
                
                # Compute loss
                losses = self.model.compute_enhanced_loss(
                    predictions, targets, self.config['loss']
                )
                
                # Track metrics
                batch_size = batch.x.size(0)
                total_loss += losses['total'].item() * batch_size
                total_samples += batch_size
                
                # Accuracy
                pred_classes = predictions['node_operations']
                true_classes = targets['classification_targets']
                correct_predictions += (pred_classes == true_classes).sum().item()
                total_predictions += true_classes.size(0)
                
                # Collect for per-class accuracy
                all_true_classes.extend(true_classes.cpu().numpy())
                all_pred_classes.extend(pred_classes.cpu().numpy())
                
                # Magnitude errors
                mag_error = torch.abs(predictions['predicted_magnitudes'] - targets['magnitude_targets'])
                magnitude_errors.extend(mag_error.cpu().numpy())
        
        avg_loss = total_loss / total_samples
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        avg_magnitude_error = np.mean(magnitude_errors) if magnitude_errors else 0
        
        # Per-class accuracy for binary classification
        class_accuracies = {}
        all_true_classes_np = np.array(all_true_classes)
        all_pred_classes_np = np.array(all_pred_classes)
        
        for c in range(2):  # Binary classification: 0=Modify, 1=Remove
            mask = all_true_classes_np == c
            if mask.sum() > 0:
                class_accuracies[c] = np.mean(all_pred_classes_np[mask] == c)
            else:
                class_accuracies[c] = 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'magnitude_mae': avg_magnitude_error,
            'class_accuracies': class_accuracies
        }
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'training_history': self.training_history
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.experiment_dir / 'latest_checkpoint.pth')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.experiment_dir / 'best_model.pth')
            logger.info(f"New best model saved with validation loss: {self.best_val_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint['training_history']
        
        logger.info(f"Checkpoint loaded from epoch {self.current_epoch}")
    
    def plot_training_history(self):
        """Plot training history"""
        if not self.training_history['train_loss']:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.training_history['train_loss'], label='Train')
        axes[0, 0].plot(self.training_history['val_loss'], label='Validation')
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(self.training_history['accuracy'])
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].grid(True)
        
        # Learning Rate
        axes[0, 2].plot(self.training_history['lr'])
        axes[0, 2].set_title('Learning Rate')
        axes[0, 2].set_yscale('log')
        axes[0, 2].grid(True)
        
        # Learnable Threshold
        axes[1, 0].plot(self.training_history['threshold'])
        axes[1, 0].set_title('Learnable Threshold')
        axes[1, 0].grid(True)
        
        # Correction Scale
        axes[1, 1].plot(self.training_history['scale'])
        axes[1, 1].set_title('Correction Scale')
        axes[1, 1].grid(True)
        
        # Loss vs Accuracy
        axes[1, 2].scatter(self.training_history['train_loss'], self.training_history['accuracy'], alpha=0.6)
        axes[1, 2].set_xlabel('Train Loss')
        axes[1, 2].set_ylabel('Accuracy')
        axes[1, 2].set_title('Loss vs Accuracy')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.experiment_dir / 'training_history.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def train(self):
        """Main training loop"""
        logger.info("Starting enhanced training...")
        
        # Load data
        self.load_data()
        
        # Training loop
        for epoch in range(self.current_epoch, self.config['training']['num_epochs']):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate
            if epoch % self.config['validation']['val_interval'] == 0:
                val_metrics = self.validate()
                
                # Log metrics
                logger.info(
                    f"Epoch {epoch:3d} - "
                    f"Train Loss: {train_metrics['loss']:.4f} - "
                    f"Val Loss: {val_metrics['loss']:.4f} - "
                    f"Accuracy: {val_metrics['accuracy']:.4f} - "
                    f"Mag MAE: {val_metrics['magnitude_mae']:.4f} - "
                    f"Threshold: {self.model.magnitude_threshold.item():.3f} - "
                    f"Scale: {self.model.correction_scale.item():.3f}"
                )
                
                # Log per-class accuracies
                train_class_acc = train_metrics.get('class_accuracies', {})
                val_class_acc = val_metrics.get('class_accuracies', {})
                logger.info(
                    f"Class accuracies - Train: Modify={train_class_acc.get(0, 0):.3f}, Remove={train_class_acc.get(1, 0):.3f} - "
                    f"Val: Modify={val_class_acc.get(0, 0):.3f}, Remove={val_class_acc.get(1, 0):.3f}"
                )
                
                # Update history
                self.training_history['train_loss'].append(train_metrics['loss'])
                self.training_history['val_loss'].append(val_metrics['loss'])
                self.training_history['accuracy'].append(val_metrics['accuracy'])
                self.training_history['lr'].append(self.scheduler.get_last_lr()[0])
                self.training_history['threshold'].append(self.model.magnitude_threshold.item())
                self.training_history['scale'].append(self.model.correction_scale.item())
                
                # Tensorboard logging
                self.writer.add_scalar('Epoch/Train_Loss', train_metrics['loss'], epoch)
                self.writer.add_scalar('Epoch/Val_Loss', val_metrics['loss'], epoch)
                self.writer.add_scalar('Epoch/Accuracy', val_metrics['accuracy'], epoch)
                self.writer.add_scalar('Epoch/Magnitude_MAE', val_metrics['magnitude_mae'], epoch)
                self.writer.add_scalar('Epoch/Threshold', self.model.magnitude_threshold.item(), epoch)
                self.writer.add_scalar('Epoch/Scale', self.model.correction_scale.item(), epoch)
                
                # Log component losses
                for key, value in train_metrics['component_losses'].items():
                    self.writer.add_scalar(f'Epoch/Loss_{key.title()}', value, epoch)
                
                # Check for improvement
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.patience_counter = 0
                    self.save_checkpoint(is_best=True)
                else:
                    self.patience_counter += 1
                
                # Early stopping
                if self.patience_counter >= self.config['validation']['early_stopping_patience']:
                    logger.info(f"Early stopping triggered after {epoch} epochs")
                    break
            
            # Save checkpoint periodically
            if epoch % self.config['validation']['checkpoint_interval'] == 0:
                self.save_checkpoint()
            
            # Plot history periodically
            if epoch % 20 == 0 and epoch > 0:
                self.plot_training_history()
        
        # Final operations
        self.plot_training_history()
        self.writer.close()
        logger.info("Enhanced training completed!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train enhanced graph correction model')
    parser.add_argument('--config', type=str, default='configs/enhanced_model_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = EnhancedRegressionTrainer(args.config)
    
    # Resume if requested
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()