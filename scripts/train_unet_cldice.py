#!/usr/bin/env python3
"""
Training script for U-Net with clDice loss function.

This script trains a U-Net model using topology-preserving clDice loss
for comparison with traditional post-processing methods and the 
graph-to-graph correction framework.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import wandb
from datetime import datetime
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.unet.unet_3d import UNet3D
from data.loaders.dataset import PulmonaryArteryDataset
from training.cldice_loss import create_cldice_loss
from utils.metrics import compute_dice_score, compute_jaccard_score
from utils.comprehensive_metrics import compute_comprehensive_metrics


class UNetclDiceTrainer:
    """Trainer for U-Net with clDice loss function."""
    
    def __init__(self, config: dict):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Setup directories
        self.setup_directories()
        
        # Initialize model
        self.model = self.create_model()
        
        # Initialize loss function
        self.criterion = self.create_loss_function()
        
        # Initialize optimizer and scheduler
        self.optimizer = self.create_optimizer()
        self.scheduler = self.create_scheduler()
        
        # Initialize data loaders
        self.train_loader, self.val_loader = self.create_data_loaders()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_dice = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        
        print(f"🚀 Initialized U-Net clDice trainer")
        print(f"📱 Device: {self.device}")
        print(f"🏗️  Model parameters: {self.count_parameters():,}")
        print(f"🎯 Loss function: {config['loss']['type']}")
    
    def setup_directories(self):
        """Setup output directories."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"unet_cldice_{self.config['loss']['type']}_{timestamp}"
        
        self.output_dir = Path(self.config['output_dir']) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.output_dir / 'predictions').mkdir(exist_ok=True)
        (self.output_dir / 'logs').mkdir(exist_ok=True)
        
        print(f"📁 Output directory: {self.output_dir}")
    
    def create_model(self) -> nn.Module:
        """Create U-Net model."""
        model_config = self.config['model']
        
        # Extract features configuration
        base_features = model_config.get('base_features', 32)
        features = model_config.get('features', [base_features, base_features*2, base_features*4, base_features*8, base_features*16])
        
        model = UNet3D(
            in_channels=model_config.get('in_channels', 1),
            out_channels=model_config.get('out_channels', 1),
            features=features,
            trilinear=model_config.get('trilinear', True),
            dropout=model_config.get('dropout', 0.1)
        )
        
        return model.to(self.device)
    
    def create_loss_function(self) -> nn.Module:
        """Create clDice loss function."""
        loss_config = self.config['loss']
        
        return create_cldice_loss(
            loss_type=loss_config['type'],
            num_iter=loss_config.get('num_iter', 40),
            smooth=loss_config.get('smooth', 1.0),
            alpha=loss_config.get('alpha', 0.5),
            beta=loss_config.get('beta', 0.3),
            sigma=loss_config.get('sigma', 1.0)
        )
    
    def create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        opt_config = self.config['optimizer']
        
        if opt_config['type'].lower() == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config.get('weight_decay', 1e-4)
            )
        elif opt_config['type'].lower() == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config.get('weight_decay', 1e-4)
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config['type']}")
    
    def create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        sched_config = self.config.get('scheduler', {})
        
        if sched_config.get('type') == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs'],
                eta_min=sched_config.get('eta_min', 1e-6)
            )
        elif sched_config.get('type') == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config.get('step_size', 30),
                gamma=sched_config.get('gamma', 0.1)
            )
        else:
            return optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)
    
    def create_data_loaders(self) -> tuple:
        """Create train and validation data loaders."""
        data_config = self.config['data']
        
        # Training dataset
        train_dataset = PulmonaryArteryDataset(
            data_dir=data_config['train_dir'],
            mode='train',
            patch_size=tuple(data_config['patch_size']),
            patch_overlap=data_config.get('patch_overlap', 0.5),
            cache_num=data_config.get('cache_num', 0)
        )
        
        # Limit dataset size for faster training if specified
        max_train_samples = data_config.get('max_train_samples', None)
        if max_train_samples and len(train_dataset) > max_train_samples:
            train_dataset.patches = train_dataset.patches[:max_train_samples]
            print(f"🔥 Limited training samples to {max_train_samples} for faster training")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=data_config['batch_size'],
            shuffle=True,
            num_workers=data_config.get('num_workers', 4),
            pin_memory=True,
            drop_last=True
        )
        
        # Validation dataset - also use patches for memory efficiency
        val_dataset = PulmonaryArteryDataset(
            data_dir=data_config['val_dir'],
            mode='train',  # Use 'train' mode to get patches instead of full volumes
            patch_size=tuple(data_config['patch_size']),
            patch_overlap=data_config.get('patch_overlap', 0.25),  # Less overlap for validation
            cache_num=data_config.get('cache_num', 0)
        )
        
        # Limit validation dataset size
        max_val_samples = data_config.get('max_val_samples', 500)
        if max_val_samples and len(val_dataset) > max_val_samples:
            val_dataset.patches = val_dataset.patches[:max_val_samples]
            print(f"🔥 Limited validation samples to {max_val_samples} for faster validation")
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=data_config.get('val_batch_size', 2),  # Smaller batch size for validation
            shuffle=False,
            num_workers=data_config.get('num_workers', 4),
            pin_memory=True
        )
        
        print(f"📊 Training samples: {len(train_dataset)}")
        print(f"📊 Validation samples: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"🏋️ Training Epoch {self.current_epoch + 1}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(self.device, non_blocking=True)
            masks = batch['mask'].to(self.device, non_blocking=True)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Compute loss
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config['training'].get('grad_clip_norm'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training']['grad_clip_norm']
                )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / num_batches:.4f}"
            })
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate_epoch(self) -> dict:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        dice_scores = []
        jaccard_scores = []
        num_batches = 0
        
        progress_bar = tqdm(
            self.val_loader,
            desc=f"🧪 Validation Epoch {self.current_epoch + 1}",
            leave=False
        )
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                images = batch['image'].to(self.device, non_blocking=True)
                masks = batch['mask'].to(self.device, non_blocking=True)
                
                # Forward pass
                outputs = self.model(images)
                
                # Compute loss
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()
                
                # Compute metrics
                predictions = torch.sigmoid(outputs) > 0.5
                
                # For validation, we process full volumes (batch_size=1)
                for i in range(predictions.shape[0]):
                    pred_np = predictions[i].cpu().numpy().astype(np.uint8)
                    mask_np = masks[i].cpu().numpy().astype(np.uint8)
                    
                    # Skip empty predictions/masks to avoid NaN in metrics
                    if pred_np.sum() > 0 or mask_np.sum() > 0:
                        dice = compute_dice_score(pred_np, mask_np)
                        jaccard = compute_jaccard_score(pred_np, mask_np)
                        
                        dice_scores.append(dice)
                        jaccard_scores.append(jaccard)
                
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'dice': f"{np.mean(dice_scores):.4f}"
                })
        
        avg_loss = total_loss / num_batches
        avg_dice = np.mean(dice_scores)
        avg_jaccard = np.mean(jaccard_scores)
        
        return {
            'loss': avg_loss,
            'dice': avg_dice,
            'jaccard': avg_jaccard,
            'dice_std': np.std(dice_scores),
            'jaccard_std': np.std(jaccard_scores)
        }
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_dice': self.best_val_dice,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics
        }
        
        # Save latest checkpoint
        latest_path = self.output_dir / 'checkpoints' / 'latest.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / 'checkpoints' / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"💾 Saved best model with dice: {self.best_val_dice:.4f}")
    
    def train(self):
        """Main training loop."""
        print(f"🚀 Starting training for {self.config['training']['epochs']} epochs")
        
        for epoch in range(self.config['training']['epochs']):
            self.current_epoch = epoch
            
            # Train epoch
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate epoch
            val_metrics = self.validate_epoch()
            val_loss = val_metrics['loss']
            val_dice = val_metrics['dice']
            
            self.val_losses.append(val_loss)
            self.val_metrics.append(val_metrics)
            
            # Update learning rate
            self.scheduler.step()
            
            # Check for best model
            is_best = val_dice > self.best_val_dice
            if is_best:
                self.best_val_dice = val_dice
                self.best_val_loss = val_loss
            
            # Save checkpoint
            if (epoch + 1) % self.config['training'].get('save_freq', 10) == 0 or is_best:
                self.save_checkpoint(is_best)
            
            # Print epoch summary
            print(f"📊 Epoch {epoch + 1}/{self.config['training']['epochs']}:")
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss: {val_loss:.4f}")
            print(f"   Val Dice: {val_dice:.4f} ± {val_metrics['dice_std']:.4f}")
            print(f"   Val Jaccard: {val_metrics['jaccard']:.4f} ± {val_metrics['jaccard_std']:.4f}")
            print(f"   Best Val Dice: {self.best_val_dice:.4f}")
            print(f"   LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Early stopping
            if self.config['training'].get('early_stopping'):
                patience = self.config['training']['early_stopping']['patience']
                if epoch >= patience:
                    recent_dice = [m['dice'] for m in self.val_metrics[-patience:]]
                    if max(recent_dice) <= self.best_val_dice:
                        print(f"🛑 Early stopping triggered after {epoch + 1} epochs")
                        break
        
        # Save final results
        self.save_training_results()
        print(f"✅ Training completed!")
        print(f"🏆 Best validation Dice: {self.best_val_dice:.4f}")
    
    def save_training_results(self):
        """Save training results and configuration."""
        results = {
            'config': self.config,
            'final_epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'best_val_dice': self.best_val_dice,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics,
            'model_parameters': self.count_parameters()
        }
        
        # Save as JSON
        results_file = self.output_dir / 'training_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save config
        config_file = self.output_dir / 'config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(self.config, f, indent=2)
        
        print(f"📄 Results saved to {results_file}")


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_default_config() -> dict:
    """Create default configuration for clDice training."""
    return {
        'model': {
            'in_channels': 1,
            'out_channels': 1,
            'base_features': 32,
            'features': [32, 64, 128, 256, 512],
            'trilinear': True,
            'dropout': 0.1
        },
        'loss': {
            'type': 'combined',  # 'cldice', 'combined', 'cbdice'
            'num_iter': 40,
            'smooth': 1.0,
            'alpha': 0.7,  # Weight for clDice
            'beta': 0.2,   # Weight for boundary (cbdice only)
            'sigma': 1.0   # Gaussian sigma (cbdice only)
        },
        'optimizer': {
            'type': 'AdamW',
            'lr': 1e-4,
            'weight_decay': 1e-4
        },
        'scheduler': {
            'type': 'cosine',
            'eta_min': 1e-6
        },
        'data': {
            'train_dir': 'DATASET/Parse_dataset',
            'val_dir': 'DATASET/Parse_dataset',
            'patch_size': [64, 64, 64],
            'patch_overlap': 0.5,
            'batch_size': 4,
            'val_batch_size': 2,
            'num_workers': 4,
            'cache_num': 0,
            'max_train_samples': 5000,  # Limit for faster training
            'max_val_samples': 500      # Limit for faster validation
        },
        'training': {
            'epochs': 100,
            'save_freq': 10,
            'grad_clip_norm': 1.0,
            'early_stopping': {
                'patience': 20
            }
        },
        'output_dir': 'experiments/unet_cldice',
        'device': 'cuda'
    }


def main():
    parser = argparse.ArgumentParser(description='Train U-Net with clDice loss')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration YAML file')
    parser.add_argument('--loss-type', type=str, default='combined',
                       choices=['cldice', 'combined', 'cbdice'],
                       help='Type of clDice loss to use')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--output-dir', type=str, default='experiments/unet_cldice',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Load or create configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = create_default_config()
        
        # Override with command line arguments
        config['loss']['type'] = args.loss_type
        config['training']['epochs'] = args.epochs
        config['optimizer']['lr'] = args.lr
        config['data']['batch_size'] = args.batch_size
        config['output_dir'] = args.output_dir
    
    # Initialize and start training
    trainer = UNetclDiceTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()