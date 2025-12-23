#!/usr/bin/env python
"""Train multiple U-Net variants for pulmonary artery segmentation"""

import os
import sys
import argparse
import yaml
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime
from sklearn.model_selection import KFold
import json
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# WandB for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.unet import UNet3D, AttentionUNet3D, UNet2D
from src.models.unet.nnunet_wrapper import nnUNetWrapper, check_nnunet_installation
from src.data.loaders.dataset import PulmonaryArteryDataset
from src.training.losses import DiceLoss, FocalLoss, CombinedLoss
from src.utils.metrics import compute_metrics


class MultiUNetTrainer:
    """Train multiple U-Net architectures with cross-validation"""
    
    def __init__(self, config, enable_architectures=None, use_wandb=False, wandb_project="pulmonary-artery-seg", 
                 parallel_training=False, gpu_ids=None):
        self.config = config
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.parallel_training = parallel_training
        
        # Setup GPU configuration
        if torch.cuda.is_available():
            self.available_gpus = list(range(torch.cuda.device_count()))
            self.gpu_ids = gpu_ids if gpu_ids else self.available_gpus
            print(f"Available GPUs: {self.available_gpus}")
            print(f"Using GPUs: {self.gpu_ids}")
            
            if parallel_training and len(self.gpu_ids) > 1:
                self.device = None  # Will be set per worker
                self.gpu_queue = queue.Queue()
                for gpu_id in self.gpu_ids:
                    self.gpu_queue.put(gpu_id)
            else:
                self.device = torch.device(f'cuda:{self.gpu_ids[0]}')
        else:
            self.device = torch.device('cpu')
            self.parallel_training = False
            print("CUDA not available, using CPU")
        
        # Get architecture selection from config or override
        if enable_architectures is not None:
            # Use command line override
            arch_selection = {
                'unet3d': 'unet3d' in enable_architectures,
                'attention_unet3d': 'attention_unet3d' in enable_architectures,
                'unet2d': 'unet2d' in enable_architectures,
                'nnunet': 'nnunet' in enable_architectures
            }
        else:
            # Use config file settings
            multi_unet_config = config.get('multi_unet', {})
            arch_selection = {
                'unet3d': multi_unet_config.get('train_unet3d', True),
                'attention_unet3d': multi_unet_config.get('train_attention_unet3d', True),
                'unet2d': multi_unet_config.get('train_unet2d', True),
                'nnunet': multi_unet_config.get('train_nnunet', False)
            }
        
        # Define all available architectures
        all_architectures = {
            'unet3d': {
                'model_class': UNet3D,
                'config': {
                    'in_channels': 1,
                    'out_channels': 1,
                    'features': [32, 64, 128, 256, 512],
                    'dropout': 0.1
                },
                'training': {
                    'batch_size': 2,
                    'patch_size': [128, 128, 128],
                    'loss': 'combined'
                }
            },
            'attention_unet3d': {
                'model_class': AttentionUNet3D,
                'config': {
                    'in_channels': 1,
                    'out_channels': 1,
                    'features': [32, 64, 128, 256, 512],
                    'dropout': 0.1
                },
                'training': {
                    'batch_size': 2,
                    'patch_size': [128, 128, 128],
                    'loss': 'combined'
                }
            },
            'unet2d': {
                'model_class': UNet2D,
                'config': {
                    'in_channels': 1,
                    'out_channels': 1,
                    'features': [64, 128, 256, 512, 1024],
                    'dropout': 0.1
                },
                'training': {
                    'batch_size': 8,
                    'patch_size': [512, 512],  # 2D patches
                    'loss': 'dice'
                }
            }
        }
        
        # Filter architectures based on selection
        self.architectures = {
            name: config for name, config in all_architectures.items() 
            if arch_selection.get(name, False)
        }
        
        self.use_nnunet = arch_selection.get('nnunet', False)
        
        # Log which architectures will be trained
        enabled_archs = list(self.architectures.keys())
        if self.use_nnunet:
            enabled_archs.append('nnunet')
        print(f"Enabled architectures: {enabled_archs}")
        
    def setup_logging(self, output_dir):
        """Setup logging"""
        log_file = output_dir / f'multi_unet_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def create_loss_function(self, loss_type):
        """Create loss function based on type"""
        if loss_type == 'dice':
            return DiceLoss()
        elif loss_type == 'focal':
            return FocalLoss(gamma=2.0)
        elif loss_type == 'combined':
            return CombinedLoss(dice_weight=0.5, focal_weight=0.5)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def train_single_architecture(self, arch_name, arch_config, dataset, output_dir, logger):
        """Train a single architecture with 5-fold cross-validation"""
        logger.info(f"Training {arch_name} architecture")
        
        # Create architecture-specific output directory
        arch_output_dir = output_dir / arch_name
        arch_output_dir.mkdir(exist_ok=True)
        
        # Save architecture config
        with open(arch_output_dir / 'config.yaml', 'w') as f:
            yaml.dump(arch_config, f)
        
        # Setup 5-fold cross-validation
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            logger.info(f"  Training fold {fold + 1}/5")
            
            # Create fold directory
            fold_dir = arch_output_dir / f'fold_{fold}'
            fold_dir.mkdir(exist_ok=True)
            
            # Train this fold
            fold_result = self._train_fold(
                arch_name, arch_config, dataset, train_idx, val_idx, 
                fold_dir, fold, logger
            )
            
            fold_results.append(fold_result)
        
        # Compute overall statistics
        avg_dice = np.mean([r['val_dice'] for r in fold_results])
        std_dice = np.std([r['val_dice'] for r in fold_results])
        
        logger.info(f"{arch_name} - Average Dice: {avg_dice:.4f} ± {std_dice:.4f}")
        
        # Log to wandb
        if self.use_wandb:
            wandb.log({
                f'{arch_name}_avg_dice': avg_dice,
                f'{arch_name}_std_dice': std_dice,
                f'{arch_name}_completed_folds': len(fold_results)
            })
        
        # Save overall results
        results = {
            'architecture': arch_name,
            'fold_results': fold_results,
            'average_dice': float(avg_dice),
            'std_dice': float(std_dice)
        }
        
        with open(arch_output_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def _train_single_architecture_on_gpu(self, arch_name, arch_config, dataset, output_dir, logger, gpu_id):
        """Train a single architecture on a specific GPU"""
        # Set device for this worker
        device = torch.device(f'cuda:{gpu_id}')
        
        logger.info(f"Training {arch_name} on GPU {gpu_id}")
        
        # Create architecture-specific output directory
        arch_output_dir = output_dir / arch_name
        arch_output_dir.mkdir(exist_ok=True)
        
        # Save architecture config
        with open(arch_output_dir / 'config.yaml', 'w') as f:
            yaml.dump(arch_config, f)
        
        # Setup 5-fold cross-validation
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            logger.info(f"  Training {arch_name} fold {fold + 1}/5 on GPU {gpu_id}")
            
            # Create fold directory
            fold_dir = arch_output_dir / f'fold_{fold}'
            fold_dir.mkdir(exist_ok=True)
            
            # Train this fold with specific device
            fold_result = self._train_fold_on_device(
                arch_name, arch_config, dataset, train_idx, val_idx, 
                fold_dir, fold, logger, device
            )
            
            fold_results.append(fold_result)
        
        # Compute overall statistics
        avg_dice = np.mean([r['val_dice'] for r in fold_results])
        std_dice = np.std([r['val_dice'] for r in fold_results])
        
        logger.info(f"{arch_name} (GPU {gpu_id}) - Average Dice: {avg_dice:.4f} ± {std_dice:.4f}")
        
        # Log to wandb
        if self.use_wandb:
            wandb.log({
                f'{arch_name}_avg_dice': avg_dice,
                f'{arch_name}_std_dice': std_dice,
                f'{arch_name}_completed_folds': len(fold_results),
                f'{arch_name}_gpu_id': gpu_id
            })
        
        # Save overall results
        results = {
            'architecture': arch_name,
            'fold_results': fold_results,
            'average_dice': float(avg_dice),
            'std_dice': float(std_dice),
            'gpu_id': gpu_id
        }
        
        with open(arch_output_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Return GPU to queue
        if hasattr(self, 'gpu_queue'):
            self.gpu_queue.put(gpu_id)
        
        return results
    
    def _train_fold(self, arch_name, arch_config, dataset, train_idx, val_idx, 
                   fold_dir, fold, logger):
        """Train a single fold"""
        
        # Initialize wandb for this fold
        if self.use_wandb:
            fold_run = wandb.init(
                project="pulmonary-artery-seg",
                name=f"{arch_name}_fold_{fold}",
                config={
                    'architecture': arch_name,
                    'fold': fold,
                    **arch_config['config'],
                    **arch_config['training']
                },
                reinit=True
            )
        
        # Create data loaders
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(
            dataset,
            batch_size=arch_config['training']['batch_size'],
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            dataset,
            batch_size=1,  # Full volume validation
            sampler=val_sampler,
            num_workers=4,
            pin_memory=True
        )
        
        # Create model
        model = arch_config['model_class'](**arch_config['config'])
        model = model.to(self.device)
        
        # Log model info
        if self.use_wandb:
            wandb.watch(model, log="all", log_freq=10)
            wandb.log({"model_parameters": model.get_num_parameters()})
        
        # Create loss and optimizer
        criterion = self.create_loss_function(arch_config['training']['loss'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        
        # Training loop with progress bar
        best_val_dice = 0
        patience = 15
        patience_counter = 0
        max_epochs = 100
        
        # Main progress bar for epochs
        epoch_pbar = tqdm(range(max_epochs), desc=f"{arch_name} Fold {fold}", leave=False)
        
        for epoch in epoch_pbar:
            # Train
            model.train()
            train_loss = 0
            
            # Training progress bar
            train_pbar = tqdm(train_loader, desc="Training", leave=False)
            for batch in train_pbar:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                
                batch_loss = loss.item()
                train_loss += batch_loss
                train_pbar.set_postfix({'loss': f'{batch_loss:.4f}'})
            
            train_loss /= len(train_loader)
            
            # Update epoch progress bar
            epoch_pbar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'best_dice': f'{best_val_dice:.4f}',
                'patience': f'{patience_counter}/{patience}'
            })
            
            # Validate every 2 epochs
            if (epoch + 1) % 2 == 0:
                val_loss, val_dice = self._validate_fold(model, val_loader, criterion)
                
                logger.info(f"    Epoch {epoch+1}: Train Loss: {train_loss:.4f}, "
                           f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
                
                # Log to wandb
                if self.use_wandb:
                    wandb.log({
                        'epoch': epoch + 1,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'val_dice': val_dice,
                        'learning_rate': optimizer.param_groups[0]['lr'],
                        'patience_counter': patience_counter
                    })
                
                # Save best model
                if val_dice > best_val_dice:
                    best_val_dice = val_dice
                    patience_counter = 0
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_dice': val_dice,
                        'config': arch_config
                    }, fold_dir / 'best_model.pth')
                    
                    # Log best model to wandb
                    if self.use_wandb:
                        wandb.log({'best_val_dice': best_val_dice})
                else:
                    patience_counter += 1
                
                # Update progress bar
                epoch_pbar.set_postfix({
                    'train_loss': f'{train_loss:.4f}',
                    'val_dice': f'{val_dice:.4f}',
                    'best_dice': f'{best_val_dice:.4f}',
                    'patience': f'{patience_counter}/{patience}'
                })
                
                # Early stopping
                if patience_counter >= patience:
                    logger.info(f"    Early stopping at epoch {epoch+1}")
                    if self.use_wandb:
                        wandb.log({'early_stopped': True, 'final_epoch': epoch + 1})
                    break
            
            scheduler.step()
        
        epoch_pbar.close()
        
        # Finish wandb run for this fold
        if self.use_wandb:
            wandb.log({'final_best_dice': best_val_dice})
            wandb.finish()
        
        return {
            'fold': fold,
            'best_epoch': epoch + 1 - patience_counter,
            'val_dice': best_val_dice,
            'model_path': str(fold_dir / 'best_model.pth')
        }
    
    def _train_fold_on_device(self, arch_name, arch_config, dataset, train_idx, val_idx, 
                             fold_dir, fold, logger, device):
        """Train a single fold on a specific device"""
        
        # Initialize wandb for this fold
        if self.use_wandb:
            fold_run = wandb.init(
                project="pulmonary-artery-seg",
                name=f"{arch_name}_fold_{fold}_gpu_{device.index}",
                config={
                    'architecture': arch_name,
                    'fold': fold,
                    'gpu_id': device.index,
                    **arch_config['config'],
                    **arch_config['training']
                },
                reinit=True
            )
        
        # Create data loaders
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(
            dataset,
            batch_size=arch_config['training']['batch_size'],
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            dataset,
            batch_size=1,  # Full volume validation
            sampler=val_sampler,
            num_workers=4,
            pin_memory=True
        )
        
        # Create model
        model = arch_config['model_class'](**arch_config['config'])
        model = model.to(device)
        
        # Log model info
        if self.use_wandb:
            wandb.watch(model, log="all", log_freq=10)
            wandb.log({"model_parameters": model.get_num_parameters()})
        
        # Create loss and optimizer
        criterion = self.create_loss_function(arch_config['training']['loss'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        
        # Training loop with progress bar
        best_val_dice = 0
        patience = 15
        patience_counter = 0
        max_epochs = 100
        
        # Main progress bar for epochs
        epoch_pbar = tqdm(range(max_epochs), desc=f"{arch_name} Fold {fold} GPU {device.index}", leave=False)
        
        for epoch in epoch_pbar:
            # Train
            model.train()
            train_loss = 0
            
            # Training progress bar
            train_pbar = tqdm(train_loader, desc="Training", leave=False)
            for batch in train_pbar:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                
                batch_loss = loss.item()
                train_loss += batch_loss
                train_pbar.set_postfix({'loss': f'{batch_loss:.4f}'})
            
            train_loss /= len(train_loader)
            
            # Update epoch progress bar
            epoch_pbar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'best_dice': f'{best_val_dice:.4f}',
                'patience': f'{patience_counter}/{patience}'
            })
            
            # Validate every 2 epochs
            if (epoch + 1) % 2 == 0:
                val_loss, val_dice = self._validate_fold_on_device(model, val_loader, criterion, device)
                
                logger.info(f"    {arch_name} GPU {device.index} Epoch {epoch+1}: Train Loss: {train_loss:.4f}, "
                           f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
                
                # Log to wandb
                if self.use_wandb:
                    wandb.log({
                        'epoch': epoch + 1,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'val_dice': val_dice,
                        'learning_rate': optimizer.param_groups[0]['lr'],
                        'patience_counter': patience_counter
                    })
                
                # Save best model
                if val_dice > best_val_dice:
                    best_val_dice = val_dice
                    patience_counter = 0
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_dice': val_dice,
                        'config': arch_config
                    }, fold_dir / 'best_model.pth')
                    
                    # Log best model to wandb
                    if self.use_wandb:
                        wandb.log({'best_val_dice': best_val_dice})
                else:
                    patience_counter += 1
                
                # Update progress bar
                epoch_pbar.set_postfix({
                    'train_loss': f'{train_loss:.4f}',
                    'val_dice': f'{val_dice:.4f}',
                    'best_dice': f'{best_val_dice:.4f}',
                    'patience': f'{patience_counter}/{patience}'
                })
                
                # Early stopping
                if patience_counter >= patience:
                    logger.info(f"    {arch_name} GPU {device.index} Early stopping at epoch {epoch+1}")
                    if self.use_wandb:
                        wandb.log({'early_stopped': True, 'final_epoch': epoch + 1})
                    break
            
            scheduler.step()
        
        epoch_pbar.close()
        
        # Finish wandb run for this fold
        if self.use_wandb:
            wandb.log({'final_best_dice': best_val_dice})
            wandb.finish()
        
        return {
            'fold': fold,
            'best_epoch': epoch + 1 - patience_counter,
            'val_dice': best_val_dice,
            'model_path': str(fold_dir / 'best_model.pth'),
            'gpu_id': device.index
        }
    
    def _validate_fold(self, model, val_loader, criterion):
        """Validate a single fold"""
        return self._validate_fold_on_device(model, val_loader, criterion, self.device)
    
    def _validate_fold_on_device(self, model, val_loader, criterion, device):
        """Validate a single fold on a specific device"""
        model.eval()
        val_loss = 0
        dice_scores = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation", leave=False)
            for batch in val_pbar:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                # Compute metrics
                predictions = torch.sigmoid(outputs) > 0.5
                metrics = compute_metrics(predictions.cpu(), masks.cpu())
                dice_scores.append(metrics['dice'])
                
                val_pbar.set_postfix({'dice': f'{metrics["dice"]:.4f}'})
        
        val_loss /= len(val_loader)
        avg_dice = np.mean(dice_scores)
        
        return val_loss, avg_dice
    
    def _train_architectures_parallel(self, dataset, output_dir, logger):
        """Train multiple architectures in parallel on different GPUs"""
        results = {}
        
        # Create a list of architectures to train
        arch_items = list(self.architectures.items())
        
        # Use ThreadPoolExecutor to train architectures in parallel
        with ThreadPoolExecutor(max_workers=len(self.gpu_ids)) as executor:
            # Submit training tasks
            future_to_arch = {}
            
            for i, (arch_name, arch_config) in enumerate(arch_items):
                # Get GPU for this architecture (round-robin)
                gpu_id = self.gpu_ids[i % len(self.gpu_ids)]
                
                logger.info(f"Submitting {arch_name} to GPU {gpu_id}")
                future = executor.submit(
                    self._train_single_architecture_on_gpu,
                    arch_name, arch_config, dataset, output_dir, logger, gpu_id
                )
                future_to_arch[future] = arch_name
            
            # Collect results as they complete
            completed_count = 0
            total_count = len(future_to_arch)
            
            for future in as_completed(future_to_arch):
                arch_name = future_to_arch[future]
                completed_count += 1
                
                try:
                    result = future.result()
                    results[arch_name] = result
                    
                    if result and 'average_dice' in result:
                        logger.info(f"✅ {arch_name} completed ({completed_count}/{total_count}): "
                                   f"Dice = {result['average_dice']:.4f} ± {result['std_dice']:.4f}")
                    else:
                        logger.info(f"✅ {arch_name} completed ({completed_count}/{total_count})")
                        
                except Exception as e:
                    logger.error(f"❌ {arch_name} failed ({completed_count}/{total_count}): {str(e)}")
                    results[arch_name] = None
        
        return results
    
    def train_nnunet(self, dataset_path, output_dir, logger):
        """Train nnU-Net if requested"""
        if not self.use_nnunet:
            return None
            
        if not check_nnunet_installation():
            logger.warning("nnU-Net not available, skipping")
            return None
        
        logger.info("Training nnU-Net")
        
        try:
            from .train_nnunet_standalone import nnUNetTrainingManager
            
            # Create nnU-Net training manager
            nnunet_output_dir = output_dir / 'nnunet'
            manager = nnUNetTrainingManager(task_id=501, output_dir=nnunet_output_dir)
            
            # First, prepare dataset if not already done
            dataset_preparation_success = self._prepare_nnunet_dataset(dataset_path, logger)
            if not dataset_preparation_success:
                return {
                    'architecture': 'nnunet',
                    'status': 'failed',
                    'error': 'Dataset preparation failed'
                }
            
            # Run planning and preprocessing
            if not manager.run_plan_and_preprocess():
                return {
                    'architecture': 'nnunet',
                    'status': 'failed',
                    'error': 'Planning and preprocessing failed'
                }
            
            # Train all folds for 3d_fullres
            results = manager.train_all_folds(network='3d_fullres', trainer='nnUNetTrainerV2')
            
            successful_folds = sum(results)
            logger.info(f"nnU-Net training: {successful_folds}/5 folds completed successfully")
            
            if successful_folds >= 3:  # Consider successful if at least 3 folds work
                return {
                    'architecture': 'nnunet',
                    'status': 'completed',
                    'successful_folds': successful_folds,
                    'model_path': str(nnunet_output_dir),
                    'results_folder': os.environ.get('RESULTS_FOLDER')
                }
            else:
                return {
                    'architecture': 'nnunet',
                    'status': 'partial_failure',
                    'successful_folds': successful_folds,
                    'error': f'Only {successful_folds}/5 folds completed successfully'
                }
            
        except Exception as e:
            logger.error(f"nnU-Net training failed: {str(e)}")
            return {
                'architecture': 'nnunet',
                'status': 'failed',
                'error': str(e)
            }
    
    def _prepare_nnunet_dataset(self, dataset_path, logger):
        """Prepare nnU-Net dataset"""
        try:
            # Check if dataset is already prepared
            raw_data_base = Path(os.environ.get('nnUNet_raw_data_base', './nnUNet_raw_data'))
            task_path = raw_data_base / "Task501_PulmonaryArtery"
            
            if (task_path / "dataset.json").exists():
                logger.info("nnU-Net dataset already prepared, skipping preparation")
                return True
            
            logger.info("Preparing nnU-Net dataset...")
            
            # Import and run dataset preparation
            import subprocess
            import sys
            
            cmd = [
                sys.executable, 
                "scripts/prepare_nnunet_dataset.py",
                "--dataset", str(dataset_path),
                "--task-id", "501",
                "--task-name", "PulmonaryArtery"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            
            if result.returncode == 0:
                logger.info("nnU-Net dataset preparation completed")
                return True
            else:
                logger.error(f"Dataset preparation failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Dataset preparation failed: {e}")
            return False
    
    def generate_cv_predictions(self, results, dataset, output_dir, logger):
        """Generate cross-validation predictions for graph correction training"""
        logger.info("Generating cross-validation predictions for graph correction")
        
        predictions_dir = output_dir / 'cv_predictions'
        predictions_dir.mkdir(exist_ok=True)
        
        # For each architecture that was successfully trained
        for arch_name, arch_results in results.items():
            if arch_results is None or arch_results.get('status') == 'failed':
                continue
                
            if arch_name == 'nnunet':
                continue  # Handle nnU-Net separately if needed
            
            logger.info(f"Generating predictions for {arch_name}")
            
            arch_pred_dir = predictions_dir / arch_name
            arch_pred_dir.mkdir(exist_ok=True)
            
            # Load architecture config
            arch_config = self.architectures[arch_name]
            
            # Setup cross-validation again
            kfold = KFold(n_splits=5, shuffle=True, random_state=42)
            
            for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
                # Load trained model for this fold
                model_path = Path(arch_results['fold_results'][fold]['model_path'])
                checkpoint = torch.load(model_path, map_location=self.device)
                
                model = arch_config['model_class'](**arch_config['config'])
                model.load_state_dict(checkpoint['model_state_dict'])
                model = model.to(self.device)
                model.eval()
                
                # Generate predictions for validation set
                val_sampler = SubsetRandomSampler(val_idx)
                val_loader = DataLoader(dataset, batch_size=1, sampler=val_sampler)
                
                with torch.no_grad():
                    for batch_idx, batch in enumerate(val_loader):
                        images = batch['image'].to(self.device)
                        outputs = model(images)
                        
                        # Convert to probability and binary prediction
                        prob = torch.sigmoid(outputs).cpu().numpy()
                        pred = (prob > 0.5).astype(np.uint8)
                        
                        # Save predictions
                        patient_id = batch['patient_id'][0]
                        
                        # Save binary prediction and probability map
                        np.save(arch_pred_dir / f'{patient_id}_pred.npy', pred[0, 0])
                        np.save(arch_pred_dir / f'{patient_id}_prob.npy', prob[0, 0])
        
        logger.info("Cross-validation predictions generated successfully")
    
    def run(self, dataset_path, output_dir):
        """Run the complete multi-architecture training"""
        # Setup
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger = self.setup_logging(output_dir)
        logger.info("Starting multi-architecture U-Net training")
        
        # Check if any architectures are enabled
        if not self.architectures and not self.use_nnunet:
            logger.error("No architectures enabled for training!")
            logger.error("Please enable at least one architecture in config or command line")
            return
        
        # Create dataset
        dataset = PulmonaryArteryDataset(
            data_dir=dataset_path,
            mode='train',
            patch_size=self.config['data']['patch_size'],
            cache_num=self.config['data']['cache_num']
        )
        
        logger.info(f"Dataset loaded: {len(dataset)} samples")
        
        # Train each enabled architecture
        results = {}
        
        # Overall progress bar for architectures
        arch_names = list(self.architectures.keys())
        if self.use_nnunet:
            arch_names.append('nnunet')
        
        if self.parallel_training and len(self.gpu_ids) > 1 and len(self.architectures) > 1:
            logger.info(f"Training {len(self.architectures)} architectures in parallel on {len(self.gpu_ids)} GPUs")
            results = self._train_architectures_parallel(dataset, output_dir, logger)
        else:
            logger.info("Training architectures sequentially")
            arch_pbar = tqdm(self.architectures.items(), desc="Training Architectures", total=len(arch_names))
            
            for arch_name, arch_config in arch_pbar:
                try:
                    arch_pbar.set_description(f"Training {arch_name}")
                    logger.info(f"Training {arch_name}...")
                    result = self.train_single_architecture(
                        arch_name, arch_config, dataset, output_dir, logger
                    )
                    results[arch_name] = result
                    
                    # Update progress bar with results
                    if result and 'average_dice' in result:
                        arch_pbar.set_postfix({'last_dice': f'{result["average_dice"]:.4f}'})
                    
                except Exception as e:
                    logger.error(f"Failed to train {arch_name}: {str(e)}")
                    results[arch_name] = None
                    arch_pbar.set_postfix({'status': 'FAILED'})
        
        # Train nnU-Net if requested
        if self.use_nnunet:
            try:
                logger.info("Training nnU-Net...")
                nnunet_result = self.train_nnunet(dataset_path, output_dir, logger)
                results['nnunet'] = nnunet_result
            except Exception as e:
                logger.error(f"nnU-Net training failed: {str(e)}")
                results['nnunet'] = {'status': 'failed', 'error': str(e)}
        
        # Generate cross-validation predictions
        self.generate_cv_predictions(results, dataset, output_dir, logger)
        
        # Save overall results
        with open(output_dir / 'all_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info("Multi-architecture training completed!")
        
        # Print summary
        logger.info("=== TRAINING SUMMARY ===")
        for arch_name, result in results.items():
            if result and 'average_dice' in result:
                logger.info(f"{arch_name}: {result['average_dice']:.4f} ± {result['std_dice']:.4f}")
            elif result and result.get('status') == 'completed':
                logger.info(f"{arch_name}: Training completed")
            else:
                logger.info(f"{arch_name}: Training failed")


def main():
    parser = argparse.ArgumentParser(description='Train multiple U-Net architectures')
    parser.add_argument('--config', type=str, default='configs/unet_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--dataset', type=str, default='DATASET/Parse_dataset',
                        help='Path to dataset directory')
    parser.add_argument('--output-dir', type=str, default='experiments/multi_unet',
                        help='Output directory for all results')
    
    # Architecture selection arguments
    parser.add_argument('--architectures', nargs='+', 
                        choices=['unet3d', 'attention_unet3d', 'unet2d', 'nnunet'],
                        help='Specific architectures to train (overrides config)')
    parser.add_argument('--use-nnunet', action='store_true',
                        help='Enable nnU-Net training (legacy flag)')
    
    # Individual architecture flags
    parser.add_argument('--enable-unet3d', action='store_true',
                        help='Enable UNet3D training')
    parser.add_argument('--enable-attention-unet3d', action='store_true',
                        help='Enable Attention UNet3D training')
    parser.add_argument('--enable-unet2d', action='store_true',
                        help='Enable UNet2D training')
    parser.add_argument('--enable-nnunet', action='store_true',
                        help='Enable nnU-Net training')
    
    # Disable flags (to override config)
    parser.add_argument('--disable-unet3d', action='store_true',
                        help='Disable UNet3D training')
    parser.add_argument('--disable-attention-unet3d', action='store_true',
                        help='Disable Attention UNet3D training')
    parser.add_argument('--disable-unet2d', action='store_true',
                        help='Disable UNet2D training')
    parser.add_argument('--disable-nnunet', action='store_true',
                        help='Disable nnU-Net training')
    
    # Experiment tracking arguments
    parser.add_argument('--use-wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='pulmonary-artery-seg',
                        help='WandB project name')
    parser.add_argument('--wandb-entity', type=str,
                        help='WandB entity (username or team)')
    
    # Multi-GPU parallel training arguments
    parser.add_argument('--parallel', action='store_true',
                        help='Enable parallel training on multiple GPUs')
    parser.add_argument('--gpu-ids', nargs='+', type=int,
                        help='Specific GPU IDs to use (e.g., --gpu-ids 0 1)')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Determine which architectures to enable
    enable_architectures = None
    
    if args.architectures:
        # Use explicit architecture list
        enable_architectures = args.architectures
        print(f"Using command line architecture selection: {enable_architectures}")
    else:
        # Build from individual flags
        enabled = []
        
        # Get base configuration
        multi_unet_config = config.get('multi_unet', {})
        
        # Apply enable flags
        if args.enable_unet3d or (multi_unet_config.get('train_unet3d', True) and not args.disable_unet3d):
            enabled.append('unet3d')
        if args.enable_attention_unet3d or (multi_unet_config.get('train_attention_unet3d', True) and not args.disable_attention_unet3d):
            enabled.append('attention_unet3d')
        if args.enable_unet2d or (multi_unet_config.get('train_unet2d', True) and not args.disable_unet2d):
            enabled.append('unet2d')
        if args.enable_nnunet or args.use_nnunet or (multi_unet_config.get('train_nnunet', False) and not args.disable_nnunet):
            enabled.append('nnunet')
        
        if enabled:
            enable_architectures = enabled
            print(f"Using processed architecture selection: {enable_architectures}")
    
    # Initialize wandb if requested
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.login()
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"multi_unet_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                'architectures': enable_architectures,
                'dataset': args.dataset,
                **config
            }
        )
        print("✅ WandB initialized successfully")
    elif args.use_wandb:
        print("❌ WandB requested but not available. Install with: pip install wandb")
    
    # Create trainer and run
    trainer = MultiUNetTrainer(
        config, 
        enable_architectures, 
        args.use_wandb, 
        args.wandb_project,
        parallel_training=args.parallel,
        gpu_ids=args.gpu_ids
    )
    trainer.run(args.dataset, args.output_dir)
    
    # Finish main wandb run
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == '__main__':
    main()