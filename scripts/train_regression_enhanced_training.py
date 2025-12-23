#!/usr/bin/env python3
"""
Enhanced Training for the Original Regression Model
Focus on breaking through the loss plateau with better training techniques
"""

import os
import sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import logging
import yaml
from tqdm import tqdm
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import math

from src.models.graph_correction.regression_model import GraphCorrectionRegressionModel
from src.training.regression_dataloader import RegressionGraphDataset, create_regression_dataloader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedRegressionTrainer:
    def __init__(self, config_path):
        """Initialize enhanced trainer with better training techniques"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Setup model (using original working model)
        self.model = GraphCorrectionRegressionModel(self.config['model']).to(self.device)
        
        # Enhanced optimizer - AdamW with better parameters
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8,
            amsgrad=True  # Better convergence for hard problems
        )
        
        # Enhanced learning rate scheduler - Cosine Annealing with Warm Restarts
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config['training'].get('restart_period', 20),  # Initial restart period
            T_mult=2,  # Period multiplier
            eta_min=self.config['training']['learning_rate'] * 0.01  # Minimum LR
        )
        
        # Learning rate warmup
        self.warmup_epochs = self.config['training'].get('warmup_epochs', 5)
        self.warmup_scheduler = None
        if self.warmup_epochs > 0:
            self.warmup_scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.warmup_epochs
            )
        
        # Setup directories with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = Path(f'experiments/enhanced_training_{timestamp}')
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Enhanced tracking
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        
        # Plateau detection
        self.plateau_patience = 15
        self.plateau_counter = 0
        self.min_improvement = 0.001
        
        logger.info(f"Experiment directory: {self.experiment_dir}")
        
        # Save config
        with open(self.experiment_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.config, f)
    
    def enhanced_loss_function(self, predictions, targets):
        """Enhanced loss function to break through plateau"""
        losses = {}
        
        # Use Smooth L1 Loss instead of MSE (more robust to outliers)
        position_loss = F.smooth_l1_loss(
            predictions['position_corrections'],
            targets['position_corrections'],
            beta=1.0  # Controls transition from L2 to L1
        )
        losses['position'] = position_loss
        
        # Enhanced magnitude loss with better scaling
        magnitude_loss = F.smooth_l1_loss(
            predictions['correction_magnitudes'],
            targets['correction_magnitudes'],
            beta=0.5
        )
        losses['magnitude'] = magnitude_loss * self.config['loss']['magnitude_weight']
        
        # Confidence regularization (encourage learning)
        confidence_reg = torch.mean(predictions['correction_confidence'])  # Encourage confidence
        losses['confidence_reg'] = confidence_reg * 0.01
        
        # Classification loss as auxiliary supervision on magnitudes
        if self.config['loss'].get('use_classification_loss', False):
            # Use magnitude predictions to create better class boundaries
            pred_magnitudes = predictions['correction_magnitudes']
            target_classes = targets['node_operations'].float()
            
            # Convert to binary classification problem using threshold
            threshold = self.model.magnitude_threshold
            pred_probs = torch.sigmoid((pred_magnitudes - threshold) * 2.0)  # Scale for sharpness
            
            # Binary cross entropy with class weights
            modify_weight = 3.0 if hasattr(self, 'last_modify_acc') and self.last_modify_acc < 0.3 else 1.5
            
            # Weight the loss per sample
            pos_weight = torch.tensor([modify_weight], device=pred_magnitudes.device)
            classification_loss = F.binary_cross_entropy_with_logits(
                (pred_magnitudes - threshold) * 2.0,  # logits
                target_classes,
                pos_weight=pos_weight
            )
            losses['classification'] = classification_loss * self.config['loss']['classification_weight']
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total'] = total_loss
        
        return losses
    
    def load_data(self):
        """Load training and validation data"""
        logger.info("Loading real U-Net prediction data...")
        
        # Load datasets
        train_path = Path('training_data/train_real_dataset.pkl')
        val_path = Path('training_data/val_real_dataset.pkl')
        
        if not train_path.exists() or not val_path.exists():
            raise FileNotFoundError("Real datasets not found. Please generate them first.")
        
        # Create data loaders
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
        
        logger.info(f"Loaded {len(self.train_loader.dataset)} training samples")
        logger.info(f"Loaded {len(self.val_loader.dataset)} validation samples")
        
    def get_current_lr(self):
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']
    
    def train_epoch(self):
        """Enhanced training epoch"""
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
            
            # Handle batch correspondences
            if len(batch.correspondences) == 1:
                targets = self.model.compute_targets(
                    batch.pred_features,
                    batch.gt_features,
                    batch.correspondences[0]
                )
            else:
                # Handle multiple samples if needed
                raise NotImplementedError("Batch size > 1 needs implementation")
            
            # Enhanced loss computation
            losses = self.enhanced_loss_function(predictions, targets)
            
            # Backward pass with gradient clipping
            self.optimizer.zero_grad()
            losses['total'].backward()
            
            # Enhanced gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['gradient_clip']
            )
            
            self.optimizer.step()
            
            # Learning rate scheduling during warmup
            if self.current_epoch < self.warmup_epochs and self.warmup_scheduler:
                self.warmup_scheduler.step()
            
            # Track metrics
            total_loss += losses['total'].item()
            num_batches += 1
            
            # Collect predictions for metrics
            all_magnitudes.extend(targets['correction_magnitudes'].cpu().numpy())
            all_predicted_magnitudes.extend(predictions['correction_magnitudes'].detach().cpu().numpy())
            all_true_classes.extend(targets['node_operations'].cpu().numpy())
            all_pred_classes.extend(predictions['node_operations'].detach().cpu().numpy())
            
            # Update progress bar
            current_lr = self.get_current_lr()
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'pos_loss': f"{losses['position'].item():.4f}",
                'lr': f"{current_lr:.2e}"
            })
        
        # Compute epoch metrics
        avg_loss = total_loss / num_batches
        mag_mae = np.mean(np.abs(np.array(all_magnitudes) - np.array(all_predicted_magnitudes)))
        accuracy = np.mean(np.array(all_true_classes) == np.array(all_pred_classes))
        
        # Per-class accuracy
        class_accuracies = {}
        for c in range(2):
            mask = np.array(all_true_classes) == c
            if mask.sum() > 0:
                class_accuracies[c] = np.mean(np.array(all_pred_classes)[mask] == c)
            else:
                class_accuracies[c] = 0.0
        
        # Store for adaptive weighting
        self.last_modify_acc = class_accuracies[0]
        
        logger.info(f"Epoch {self.current_epoch} - "
                   f"Loss: {avg_loss:.4f} - "
                   f"Mag MAE: {mag_mae:.4f} - "
                   f"Accuracy: {accuracy:.4f} - "
                   f"LR: {self.get_current_lr():.2e}")
        logger.info(f"Class accuracies - "
                   f"Modify: {class_accuracies.get(0, 0):.3f}, "
                   f"Remove: {class_accuracies.get(1, 0):.3f}")
        
        return avg_loss, accuracy, class_accuracies
    
    def validate(self):
        """Enhanced validation"""
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
                
                predictions = self.model(batch)
                
                if len(batch.correspondences) == 1:
                    targets = self.model.compute_targets(
                        batch.pred_features, batch.gt_features, batch.correspondences[0]
                    )
                else:
                    raise NotImplementedError("Batch size > 1 needs implementation")
                
                losses = self.enhanced_loss_function(predictions, targets)
                
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
        
        # Per-class accuracy
        class_accuracies = {}
        for c in range(2):
            mask = np.array(all_true_classes) == c
            if mask.sum() > 0:
                class_accuracies[c] = np.mean(np.array(all_pred_classes)[mask] == c)
            else:
                class_accuracies[c] = 0.0
        
        logger.info(f"Validation - Loss: {avg_loss:.4f} - "
                   f"Mag MAE: {mag_mae:.4f} - "
                   f"Accuracy: {accuracy:.4f}")
        logger.info(f"Val Class accuracies - "
                   f"Modify: {class_accuracies.get(0, 0):.3f}, "
                   f"Remove: {class_accuracies.get(1, 0):.3f}")
        
        return avg_loss, accuracy, class_accuracies
    
    def save_checkpoint(self, is_best=False, is_best_acc=False):
        """Enhanced checkpoint saving"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates,
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy,
            'config': self.config
        }
        
        # Save regular checkpoint
        torch.save(checkpoint, self.experiment_dir / 'latest_checkpoint.pth')
        
        # Save best loss model
        if is_best:
            torch.save(checkpoint, self.experiment_dir / 'best_loss_model.pth')
            logger.info(f"Saved best loss model: {self.best_val_loss:.4f}")
        
        # Save best accuracy model
        if is_best_acc:
            torch.save(checkpoint, self.experiment_dir / 'best_accuracy_model.pth')
            logger.info(f"Saved best accuracy model: {self.best_val_accuracy:.4f}")
    
    def plot_training_curves(self):
        """Plot enhanced training curves"""
        if len(self.train_losses) < 2:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.train_losses, label='Train')
        axes[0, 0].plot(self.val_losses, label='Validation')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves
        axes[0, 1].plot(self.train_accuracies, label='Train')
        axes[0, 1].plot(self.val_accuracies, label='Validation')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate
        axes[1, 0].plot(self.learning_rates)
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        # Loss vs Accuracy scatter
        axes[1, 1].scatter(self.val_losses, self.val_accuracies, alpha=0.6)
        axes[1, 1].set_title('Validation Loss vs Accuracy')
        axes[1, 1].set_xlabel('Validation Loss')
        axes[1, 1].set_ylabel('Validation Accuracy')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.experiment_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def detect_plateau(self, current_loss):
        """Enhanced plateau detection"""
        if len(self.val_losses) < 2:
            return False
        
        # Check if improvement is significant
        best_recent = min(self.val_losses[-self.plateau_patience:]) if len(self.val_losses) >= self.plateau_patience else min(self.val_losses)
        improvement = best_recent - current_loss
        
        if improvement < self.min_improvement:
            self.plateau_counter += 1
        else:
            self.plateau_counter = 0
        
        return self.plateau_counter >= self.plateau_patience
    
    def train(self):
        """Enhanced main training loop"""
        logger.info("Starting enhanced training...")
        
        # Load data
        self.load_data()
        
        # Training loop
        for epoch in range(self.config['training']['num_epochs']):
            self.current_epoch = epoch
            
            # Train epoch
            train_loss, train_acc, train_class_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.learning_rates.append(self.get_current_lr())
            
            # Validate
            if epoch % self.config['validation']['val_interval'] == 0:
                val_loss, val_acc, val_class_acc = self.validate()
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_acc)
                
                # Enhanced learning rate scheduling (after warmup)
                if epoch >= self.warmup_epochs:
                    self.scheduler.step()
                
                # Save best models
                is_best_loss = val_loss < self.best_val_loss
                is_best_acc = val_acc > self.best_val_accuracy
                
                if is_best_loss:
                    self.best_val_loss = val_loss
                
                if is_best_acc:
                    self.best_val_accuracy = val_acc
                
                self.save_checkpoint(is_best=is_best_loss, is_best_acc=is_best_acc)
                
                # Plateau detection and handling
                if self.detect_plateau(val_loss):
                    logger.info(f"Plateau detected at epoch {epoch}. Reducing learning rate.")
                    # Additional LR reduction for plateau
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= 0.5
                    self.plateau_counter = 0
                
                # Plot curves periodically
                if epoch % 10 == 0 and epoch > 0:
                    self.plot_training_curves()
        
        # Final operations
        self.plot_training_curves()
        logger.info(f"Training completed!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info(f"Best validation accuracy: {self.best_val_accuracy:.4f}")


def main():
    """Main function"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, 
                       default='configs/enhanced_training_config.yaml',
                       help='Path to config file')
    args = parser.parse_args()
    
    # Create trainer and train
    trainer = EnhancedRegressionTrainer(args.config)
    trainer.train()


if __name__ == '__main__':
    main()