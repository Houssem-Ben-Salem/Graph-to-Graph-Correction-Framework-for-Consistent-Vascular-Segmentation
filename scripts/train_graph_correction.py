#!/usr/bin/env python
"""Train Graph Correction Model - Updated for Step 4 Training Pipeline"""

import os
import sys
import argparse
import yaml
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime
import time
from dataclasses import dataclass
from typing import Dict, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.graph_correction import GraphCorrectionModel
from src.models.graph_correction.loss_functions import GraphCorrectionLoss
from src.training.graph_correction_dataloader import CurriculumDataManager, DataLoaderConfig
from src.training.dataset_generator import TrainingDatasetGenerator


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Model
    model_config: Dict = None
    
    # Training
    num_epochs: int = 300
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    
    # Curriculum Learning
    curriculum_enabled: bool = True
    curriculum_stages: Dict = None
    advancement_threshold: float = 0.85
    min_epochs_per_stage: int = 20
    
    # Data
    batch_size: int = 4
    num_workers: int = 4
    
    # Logging
    log_interval: int = 10
    val_interval: int = 5
    checkpoint_interval: int = 10
    
    # Paths
    extracted_graphs_dir: Path = Path("extracted_graphs")
    training_data_dir: Path = Path("training_data")
    checkpoint_dir: Path = Path("experiments/graph_correction")
    
    def __post_init__(self):
        if self.curriculum_stages is None:
            self.curriculum_stages = {
                "easy": {"epochs": 50, "lr_scale": 1.0},
                "medium": {"epochs": 50, "lr_scale": 0.5},
                "hard": {"epochs": 50, "lr_scale": 0.3},
                "expert": {"epochs": 50, "lr_scale": 0.1},
                "real": {"epochs": 100, "lr_scale": 0.05}
            }


def setup_logging(output_dir):
    """Setup logging configuration"""
    log_file = output_dir / f'graph_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class GraphCorrectionTrainer:
    """Main trainer for graph correction model"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Setup directories
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Initialize loss with config
        loss_config = self.config.model_config.get('loss_config', {})
        if hasattr(self.config, 'node_class_weights'):
            loss_config['node_class_weights'] = self.config.node_class_weights
        elif self.config.model_config and 'node_class_weights' in self.config.model_config:
            loss_config['node_class_weights'] = self.config.model_config['node_class_weights']
        self.loss_fn = GraphCorrectionLoss(loss_config)
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Initialize data manager
        self.data_manager = self._create_data_manager()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.curriculum_stage = "easy"
        
    def _create_model(self) -> GraphCorrectionModel:
        """Create and initialize model"""
        model_config = self.config.model_config or {}
        model = GraphCorrectionModel(model_config)
        
        # Log model summary
        summary = model.get_model_summary()
        self.logger.info(f"Model initialized with {summary['total_parameters']:,} parameters")
        
        return model
    
    def _create_optimizer(self):
        """Create optimizer"""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.num_epochs,
            eta_min=1e-6
        )
    
    def _create_data_manager(self) -> CurriculumDataManager:
        """Create curriculum data manager"""
        dataloader_config = DataLoaderConfig(
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            curriculum_enabled=self.config.curriculum_enabled,
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )
        
        return CurriculumDataManager(
            train_dataset_dir=self.config.training_data_dir,
            val_dataset_dir=self.config.training_data_dir,
            config=dataloader_config
        )


    def train(self):
        """Main training loop"""
        self.logger.info("=== Starting Training ===")
        
        # Get initial data loaders
        train_loader, val_loader = self.data_manager.get_dataloaders_for_stage(self.curriculum_stage)
        
        # Training loop - start from current_epoch if resuming
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            
            # Check for curriculum advancement (epoch-based)
            if self.config.curriculum_enabled:
                new_stage = self._get_stage_for_epoch(epoch)
                if new_stage != self.curriculum_stage:
                    self.logger.info(f"🎯 Curriculum advancement: {self.curriculum_stage} → {new_stage} at epoch {epoch+1}")
                    self.curriculum_stage = new_stage
                    train_loader, val_loader = self.data_manager.get_dataloaders_for_stage(new_stage)
                    self._adjust_training_for_stage(new_stage)
            
            # Train one epoch
            train_metrics = self._train_epoch(train_loader, epoch)
            
            # Validation
            if (epoch + 1) % self.config.val_interval == 0:
                val_metrics = self._validate(val_loader, epoch)
                
                # Check for best model
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self._save_checkpoint('best_model.pth', is_best=True)
            
            # Regular checkpoint
            if (epoch + 1) % self.config.checkpoint_interval == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
        
        self.logger.info("Training completed!")
    
    def _train_epoch(self, train_loader, epoch: int) -> Dict:
        """Train one epoch"""
        self.model.train()
        
        epoch_loss = 0.0
        epoch_metrics = {
            'topology_loss': 0.0,
            'anatomy_loss': 0.0,
            'consistency_loss': 0.0,
            'node_op_accuracy': 0.0,
            'samples_processed': 0,
            # Class-specific metrics
            'keep_precision': 0.0,
            'keep_recall': 0.0,
            'keep_f1': 0.0,
            'remove_precision': 0.0,
            'remove_recall': 0.0,
            'remove_f1': 0.0,
            'modify_precision': 0.0,
            'modify_recall': 0.0,
            'modify_f1': 0.0,
            'balanced_accuracy': 0.0,
            # Class distribution
            'keep_distribution': 0.0,
            'remove_distribution': 0.0,
            'modify_distribution': 0.0
        }
        
        # Collect all predictions and targets for epoch-level metrics
        all_predictions = []
        all_targets = []
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            if batch_idx == 0:
                self.logger.info(f"Processing first batch of epoch {epoch+1}...")
            # Forward pass
            outputs = self._forward_batch(batch)
            
            # Compute loss
            loss, loss_components = self._compute_loss(outputs, batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.gradient_clip
                )
            
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            for key, value in loss_components.items():
                if key in epoch_metrics:
                    epoch_metrics[key] += value
            
            # Compute node operation metrics if available
            if 'node_operations' in outputs and 'training_signals' in outputs:
                if 'node_op_targets' in outputs['training_signals']:
                    # Simple accuracy
                    node_acc = self._compute_node_accuracy(
                        outputs['node_operations'], 
                        outputs['training_signals']['node_op_targets']
                    )
                    epoch_metrics['node_op_accuracy'] += node_acc
                    
                    # Collect predictions and targets for epoch-level metrics
                    if outputs['node_operations'].dim() == 2:
                        pred_classes = torch.argmax(outputs['node_operations'], dim=1)
                    else:
                        pred_classes = outputs['node_operations']
                    
                    all_predictions.extend(pred_classes.cpu().numpy())
                    all_targets.extend(outputs['training_signals']['node_op_targets'].cpu().numpy())
            
            epoch_metrics['samples_processed'] += batch['batch_stats']['batch_size']
            
            self.global_step += 1
        
        # Compute epoch averages
        num_batches = len(train_loader)
        epoch_loss /= num_batches
        for key in epoch_metrics:
            if key != 'samples_processed' and not key.endswith('_precision') and not key.endswith('_recall') and not key.endswith('_f1') and not key.endswith('_distribution') and key != 'balanced_accuracy':
                epoch_metrics[key] /= num_batches
        
        # Compute comprehensive metrics on all collected predictions
        if len(all_predictions) > 0 and len(all_targets) > 0:
            from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
            import numpy as np
            
            all_predictions = np.array(all_predictions)
            all_targets = np.array(all_targets)
            
            # Remap classes from [1,2,3] to [0,1,2] if needed
            unique_targets = np.unique(all_targets)
            unique_preds = np.unique(all_predictions)
            
            if 0 not in unique_targets and min(unique_targets) > 0:
                # Remap targets: [1,2,3] -> [0,1,2]
                all_targets = all_targets - 1
                all_predictions = all_predictions - 1
                self.logger.info(f"DEBUG: Remapped classes from {unique_targets} to {np.unique(all_targets)}")
            
            # Calculate precision, recall, f1 for each class
            precision, recall, f1, support = precision_recall_fscore_support(
                all_targets, all_predictions, labels=[0, 1, 2], zero_division=0
            )
            
            # Class distribution
            unique, counts = np.unique(all_targets, return_counts=True)
            total = len(all_targets)
            class_dist = {int(cls): cnt / total for cls, cnt in zip(unique, counts)}
            
            # Log unique classes found (temporary for debugging)
            self.logger.info(f"DEBUG: Unique classes in targets: {unique}")
            self.logger.info(f"DEBUG: Class counts: {counts}")
            self.logger.info(f"DEBUG: Unique classes in predictions: {np.unique(all_predictions)}")
            
            # Update metrics
            class_names = ['keep', 'remove', 'modify']
            for i, name in enumerate(class_names):
                epoch_metrics[f'{name}_precision'] = precision[i]
                epoch_metrics[f'{name}_recall'] = recall[i]
                epoch_metrics[f'{name}_f1'] = f1[i]
                epoch_metrics[f'{name}_distribution'] = class_dist.get(i, 0.0)
            
            # Balanced accuracy
            epoch_metrics['balanced_accuracy'] = recall.mean()
        
        epoch_time = time.time() - start_time
        
        # Log epoch summary with key metrics
        self.logger.info(
            f"Epoch {epoch+1}/{self.config.num_epochs} - "
            f"Loss: {epoch_loss:.4f} - "
            f"Node Acc: {epoch_metrics['node_op_accuracy']:.3f} - "
            f"Balanced Acc: {epoch_metrics['balanced_accuracy']:.3f} - "
            f"Time: {epoch_time:.1f}s - "
            f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
        )
        
        # Log detailed class metrics every 5 epochs
        if (epoch + 1) % 5 == 0:
            self._log_detailed_metrics(epoch_metrics, epoch)
        
        return {'loss': epoch_loss, **epoch_metrics}


    def _forward_batch(self, batch) -> Dict:
        """Forward pass for a batch"""
        # For simplicity, process single sample (can be extended for batching)
        pred_graph = batch['degraded_graphs'][0]
        gt_graph = batch['gt_graphs'][0]
        correspondences = batch['correspondences'][0]
        
        # Forward pass
        outputs = self.model(
            pred_graph,
            gt_graph,
            correspondences,
            training_mode=True
        )
        
        return outputs
    
    def _compute_loss(self, outputs, batch) -> tuple:
        """Compute loss for outputs"""
        # Get training signals from outputs
        if 'training_signals' in outputs:
            training_signals = outputs['training_signals']
        else:
            # Generate default training signals
            training_signals = self._generate_training_signals(outputs, batch)
        
        # Compute loss
        loss_dict = self.loss_fn(
            outputs,
            training_signals,
            batch['correspondences'],
            metadata={'stage': self.curriculum_stage}
        )
        
        # Extract total loss and components
        total_loss = loss_dict['total']
        loss_components = {k: v for k, v in loss_dict.items() if k != 'total'}
        
        return total_loss, loss_components


    def _validate(self, val_loader, epoch: int) -> Dict:
        """Validate model"""
        self.model.eval()
        
        val_loss = 0.0
        all_predictions = []
        all_targets = []
        val_metrics = {
            'topology_loss': 0.0,
            'anatomy_loss': 0.0,
            'consistency_loss': 0.0,
            'node_op_accuracy': 0.0,
            'quality_score': 0.0,
            # Class-specific metrics
            'keep_precision': 0.0,
            'keep_recall': 0.0,
            'keep_f1': 0.0,
            'remove_precision': 0.0,
            'remove_recall': 0.0,
            'remove_f1': 0.0,
            'modify_precision': 0.0,
            'modify_recall': 0.0,
            'modify_f1': 0.0,
            'balanced_accuracy': 0.0,
            # Class distribution
            'keep_distribution': 0.0,
            'remove_distribution': 0.0,
            'modify_distribution': 0.0
        }
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                outputs = self._forward_batch(batch)
                loss, loss_components = self._compute_loss(outputs, batch)
                
                val_loss += loss.item()
                for key, value in loss_components.items():
                    if key in val_metrics:
                        val_metrics[key] += value
                
                # Compute node operation metrics if available
                if 'node_operations' in outputs and 'training_signals' in outputs:
                    if 'node_op_targets' in outputs['training_signals']:
                        # Simple accuracy
                        node_acc = self._compute_node_accuracy(
                            outputs['node_operations'], 
                            outputs['training_signals']['node_op_targets']
                        )
                        val_metrics['node_op_accuracy'] += node_acc
                        
                        # Collect predictions and targets for epoch-level metrics
                        if outputs['node_operations'].dim() == 2:
                            pred_classes = torch.argmax(outputs['node_operations'], dim=1)
                        else:
                            pred_classes = outputs['node_operations']
                        
                        all_predictions.extend(pred_classes.cpu().numpy())
                        all_targets.extend(outputs['training_signals']['node_op_targets'].cpu().numpy())
                
                # Collect quality scores
                if 'quality_score' in outputs:
                    val_metrics['quality_score'] += outputs['quality_score'].mean().item()
        
        # Compute averages
        num_batches = len(val_loader)
        val_loss /= num_batches
        for key in val_metrics:
            if not key.endswith('_precision') and not key.endswith('_recall') and not key.endswith('_f1') and not key.endswith('_distribution') and key != 'balanced_accuracy':
                val_metrics[key] /= num_batches
        
        # Compute comprehensive metrics on all collected predictions
        if len(all_predictions) > 0 and len(all_targets) > 0:
            from sklearn.metrics import precision_recall_fscore_support
            import numpy as np
            
            all_predictions = np.array(all_predictions)
            all_targets = np.array(all_targets)
            
            # Remap classes from [1,2,3] to [0,1,2] if needed
            unique_targets = np.unique(all_targets)
            unique_preds = np.unique(all_predictions)
            
            if 0 not in unique_targets and min(unique_targets) > 0:
                # Remap targets: [1,2,3] -> [0,1,2]
                all_targets = all_targets - 1
                all_predictions = all_predictions - 1
                self.logger.info(f"DEBUG: Validation remapped classes from {unique_targets} to {np.unique(all_targets)}")
            
            # Calculate precision, recall, f1 for each class
            precision, recall, f1, support = precision_recall_fscore_support(
                all_targets, all_predictions, labels=[0, 1, 2], zero_division=0
            )
            
            # Class distribution
            unique, counts = np.unique(all_targets, return_counts=True)
            total = len(all_targets)
            class_dist = {int(cls): cnt / total for cls, cnt in zip(unique, counts)}
            
            # Update metrics
            class_names = ['keep', 'remove', 'modify']
            for i, name in enumerate(class_names):
                val_metrics[f'{name}_precision'] = precision[i]
                val_metrics[f'{name}_recall'] = recall[i]
                val_metrics[f'{name}_f1'] = f1[i]
                val_metrics[f'{name}_distribution'] = class_dist.get(i, 0.0)
            
            # Balanced accuracy
            val_metrics['balanced_accuracy'] = recall.mean()
        
        # Log validation results
        self.logger.info(
            f"Validation - Loss: {val_loss:.4f} - "
            f"Quality: {val_metrics['quality_score']:.3f} - "
            f"Node Acc: {val_metrics['node_op_accuracy']:.3f} - "
            f"Balanced Acc: {val_metrics['balanced_accuracy']:.3f}"
        )
        
        # Log detailed validation metrics too
        if (epoch + 1) % 5 == 0:
            self.logger.info("\nValidation Class Metrics:")
            self._log_detailed_metrics(val_metrics, epoch)
        
        return {'loss': val_loss, **val_metrics}


    def _get_stage_for_epoch(self, epoch: int) -> str:
        """Get curriculum stage based on epoch number"""
        # Simple epoch-based curriculum progression
        if epoch < 50:
            return "easy"      # Pure synthetic, easy degradations
        elif epoch < 100:
            return "medium"    # 50% real + 50% synthetic, medium degradations  
        elif epoch < 150:
            return "hard"      # 70% real + 30% synthetic, hard degradations
        elif epoch < 200:
            return "expert"    # 80% real + 20% synthetic, expert degradations
        else:
            return "real"      # 100% real data
    
    def _check_curriculum_advancement(self) -> bool:
        """Check if curriculum should advance"""
        if not self.config.curriculum_enabled:
            return False
        
        # Simplified advancement logic - can be made more sophisticated
        return self.data_manager.should_advance_curriculum(
            epoch=self.current_epoch,
            performance_metric=0.8,  # Placeholder metric
            min_epochs=self.config.min_epochs_per_stage,
            performance_threshold=self.config.advancement_threshold
        )
    
    def _log_detailed_metrics(self, metrics: Dict, epoch: int):
        """Log detailed class-specific metrics"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Detailed Metrics - Epoch {epoch+1}")
        self.logger.info(f"{'='*60}")
        
        # Class distribution
        self.logger.info("\nClass Distribution in Training Data:")
        self.logger.info(f"  Keep (0):   {metrics['keep_distribution']*100:.1f}%")
        self.logger.info(f"  Remove (1): {metrics['remove_distribution']*100:.1f}%")
        self.logger.info(f"  Modify (2): {metrics['modify_distribution']*100:.1f}%")
        
        # Show actual distribution if different
        total_dist = metrics['keep_distribution'] + metrics['remove_distribution'] + metrics['modify_distribution']
        if abs(total_dist - 1.0) > 0.01:
            self.logger.info(f"\n  Note: Total distribution = {total_dist*100:.1f}% (should be 100%)")
            self.logger.info("  This suggests the class indices might be different than expected")
        
        # Per-class performance
        self.logger.info("\nPer-Class Performance:")
        self.logger.info("  Class  | Precision | Recall | F1-Score")
        self.logger.info("  -------|-----------|--------|----------")
        self.logger.info(f"  Keep   |   {metrics['keep_precision']:.3f}   | {metrics['keep_recall']:.3f}  |  {metrics['keep_f1']:.3f}")
        self.logger.info(f"  Remove |   {metrics['remove_precision']:.3f}   | {metrics['remove_recall']:.3f}  |  {metrics['remove_f1']:.3f}")
        self.logger.info(f"  Modify |   {metrics['modify_precision']:.3f}   | {metrics['modify_recall']:.3f}  |  {metrics['modify_f1']:.3f}")
        
        self.logger.info(f"\nBalanced Accuracy: {metrics['balanced_accuracy']:.3f}")
        self.logger.info(f"Overall Accuracy:  {metrics['node_op_accuracy']:.3f}")
        self.logger.info(f"{'='*60}\n")
    
    def _adjust_training_for_stage(self, stage: str):
        """Adjust training parameters for curriculum stage"""
        # Update model training stage
        stage_to_model_stage = {
            "easy": 1,      # Topology focus
            "medium": 1,    # Still topology
            "hard": 2,      # Anatomy focus
            "expert": 3,    # Joint optimization
            "real": 3       # Full optimization for real data
        }
        self.model.set_training_stage(stage_to_model_stage.get(stage, 3))
        
        # Adjust learning rate
        if stage in self.config.curriculum_stages:
            lr_scale = self.config.curriculum_stages[stage].get('lr_scale', 1.0)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.config.learning_rate * lr_scale
        
        self.logger.info(f"Adjusted training for {stage} stage")
    
    def _save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'curriculum_stage': self.curriculum_stage,
            'config': self.config,
            'global_step': self.global_step
        }
        
        path = self.config.checkpoint_dir / filename
        torch.save(checkpoint, path)
        
        if is_best:
            self.logger.info(f"Saved best model checkpoint: {path}")
        else:
            self.logger.info(f"Saved checkpoint: {path}")
    
    def _compute_node_accuracy(self, predictions, targets):
        """Compute node operation accuracy"""
        if predictions.dim() == 2:  # [N, 3] for 3 classes: keep, remove, modify
            pred_classes = torch.argmax(predictions, dim=1)
        else:
            pred_classes = predictions
        
        correct = (pred_classes == targets).float().sum()
        total = targets.numel()
        accuracy = correct / total if total > 0 else 0.0
        
        return accuracy.item() if hasattr(accuracy, 'item') else accuracy
    
    def _compute_node_metrics(self, predictions, targets):
        """Compute comprehensive node operation metrics"""
        if predictions.dim() == 2:  # [N, 3] for 3 classes
            pred_classes = torch.argmax(predictions, dim=1)
        else:
            pred_classes = predictions
        
        # Move to CPU for sklearn metrics
        pred_cpu = pred_classes.cpu().numpy()
        target_cpu = targets.cpu().numpy()
        
        metrics = {}
        
        # Overall accuracy
        metrics['accuracy'] = (pred_classes == targets).float().mean().item()
        
        # Class distribution in targets
        unique, counts = torch.unique(targets, return_counts=True)
        total = targets.numel()
        class_dist = {}
        for i, (cls, cnt) in enumerate(zip(unique.cpu().numpy(), counts.cpu().numpy())):
            class_dist[int(cls)] = cnt / total
        
        # Per-class metrics
        from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
        
        # Calculate precision, recall, f1 for each class
        precision, recall, f1, support = precision_recall_fscore_support(
            target_cpu, pred_cpu, labels=[0, 1, 2], zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(target_cpu, pred_cpu, labels=[0, 1, 2])
        
        # Store metrics
        class_names = ['keep', 'remove', 'modify']
        for i, name in enumerate(class_names):
            metrics[f'{name}_precision'] = precision[i]
            metrics[f'{name}_recall'] = recall[i]
            metrics[f'{name}_f1'] = f1[i]
            metrics[f'{name}_support'] = support[i]
            metrics[f'{name}_distribution'] = class_dist.get(i, 0.0)
        
        # Balanced accuracy (average recall across classes)
        metrics['balanced_accuracy'] = recall.mean()
        
        # Confusion matrix as list for logging
        metrics['confusion_matrix'] = cm.tolist()
        
        return metrics
    
    def _generate_training_signals(self, outputs, batch):
        """Generate default training signals if not provided"""
        # Simplified version - would need proper implementation
        return {
            'node_op_targets': torch.zeros(1, dtype=torch.long),
            'node_correction_targets': torch.zeros(1, 7),
            'edge_op_targets': torch.zeros(1, dtype=torch.long)
        }


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train Graph Correction Model")
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--generate-data', action='store_true', help='Generate training data first')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--num-epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create training config
    config = TrainingConfig()
    
    # Load config file if provided
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        # Update model config
        if 'model' in config_dict:
            config.model_config = config_dict['model']
        
        # Update training params with proper type conversion
        if 'training' in config_dict:
            for key, value in config_dict['training'].items():
                if hasattr(config, key):
                    # Convert types to match TrainingConfig expected types
                    if key in ['learning_rate', 'weight_decay', 'gradient_clip']:
                        value = float(value)
                    elif key in ['num_epochs', 'batch_size', 'num_workers', 'log_interval', 'val_interval', 'checkpoint_interval', 'min_epochs_per_stage']:
                        value = int(value)
                    elif key in ['curriculum_enabled', 'adaptive_sampling']:
                        value = bool(value)
                    setattr(config, key, value)
        
        # Update loss weights
        if 'loss_weights' in config_dict:
            if not config.model_config:
                config.model_config = {}
            config.model_config['loss_config'] = config_dict['loss_weights']
    
    # Override with command line args
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    
    # Generate training data if requested
    if args.generate_data:
        logging.info("Generating training data...")
        generator = TrainingDatasetGenerator(
            extracted_graphs_dir=config.extracted_graphs_dir,
            output_dir=config.training_data_dir
        )
        generator.generate_curriculum_datasets()
    
    # Create trainer
    trainer = GraphCorrectionTrainer(config)
    
    # Resume from checkpoint if requested
    if args.resume:
        # Allow loading our custom TrainingConfig class
        torch.serialization.add_safe_globals([TrainingConfig])
        checkpoint = torch.load(args.resume, weights_only=False)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint.get('scheduler_state_dict') and trainer.scheduler:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.current_epoch = checkpoint['epoch'] + 1
        trainer.best_val_loss = checkpoint['best_val_loss']
        trainer.global_step = checkpoint.get('global_step', 0)
        trainer.curriculum_stage = checkpoint.get('curriculum_stage', 'easy')
        logging.info(f"Resumed from epoch {trainer.current_epoch}, stage: {trainer.curriculum_stage}")
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()