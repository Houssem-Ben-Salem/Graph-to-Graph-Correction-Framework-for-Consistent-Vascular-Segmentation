#!/usr/bin/env python
"""Train U-Net model for pulmonary artery segmentation"""

import os
import sys
import argparse
import yaml
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.unet import UNet3D, AttentionUNet3D
from src.data.loaders.dataset import PulmonaryArteryDataset
from src.data.augmentation.transforms import get_train_transforms, get_val_transforms
from src.training.losses import CombinedLoss, DiceLoss, FocalLoss
from src.utils.metrics import compute_metrics
from src.utils.visualization import save_prediction_visualization


def setup_logging(output_dir):
    """Setup logging configuration"""
    log_file = output_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
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


def create_model(config):
    """Create model based on configuration"""
    model_config = config['model']
    
    if model_config['architecture'] == 'UNet3D':
        model = UNet3D(
            in_channels=model_config['in_channels'],
            out_channels=model_config['out_channels'],
            features=model_config['features'],
            dropout=model_config['dropout']
        )
    elif model_config['architecture'] == 'AttentionUNet3D':
        model = AttentionUNet3D(
            in_channels=model_config['in_channels'],
            out_channels=model_config['out_channels'],
            features=model_config['features'],
            dropout=model_config['dropout']
        )
    else:
        raise ValueError(f"Unknown architecture: {model_config['architecture']}")
    
    return model


def create_loss_function(config):
    """Create loss function based on configuration"""
    loss_config = config['loss']
    
    if loss_config['type'] == 'dice':
        return DiceLoss()
    elif loss_config['type'] == 'focal':
        return FocalLoss(gamma=loss_config['focal_gamma'])
    elif loss_config['type'] == 'combined':
        return CombinedLoss(
            dice_weight=loss_config['dice_weight'],
            focal_weight=loss_config['focal_weight'],
            focal_gamma=loss_config['focal_gamma']
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_config['type']}")


def train_epoch(model, dataloader, criterion, optimizer, scaler, device, logger):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0
    epoch_metrics = {'dice': 0, 'sensitivity': 0, 'precision': 0}
    
    progress_bar = tqdm(dataloader, desc='Training')
    
    for batch_idx, batch in enumerate(progress_bar):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Compute metrics
        with torch.no_grad():
            predictions = torch.sigmoid(outputs) > 0.5
            metrics = compute_metrics(predictions, masks)
        
        epoch_loss += loss.item()
        for key in epoch_metrics:
            epoch_metrics[key] += metrics[key]
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': loss.item(),
            'dice': metrics['dice']
        })
    
    # Average metrics
    n_batches = len(dataloader)
    epoch_loss /= n_batches
    for key in epoch_metrics:
        epoch_metrics[key] /= n_batches
    
    return epoch_loss, epoch_metrics


def validate(model, dataloader, criterion, device, logger):
    """Validate model"""
    model.eval()
    val_loss = 0
    val_metrics = {'dice': 0, 'sensitivity': 0, 'precision': 0}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            predictions = torch.sigmoid(outputs) > 0.5
            metrics = compute_metrics(predictions, masks)
            
            val_loss += loss.item()
            for key in val_metrics:
                val_metrics[key] += metrics[key]
    
    # Average metrics
    n_batches = len(dataloader)
    val_loss /= n_batches
    for key in val_metrics:
        val_metrics[key] /= n_batches
    
    return val_loss, val_metrics


def main():
    parser = argparse.ArgumentParser(description='Train U-Net for pulmonary artery segmentation')
    parser.add_argument('--config', type=str, default='configs/unet_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='experiments/unet',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup output directory
    output_dir = Path(args.output_dir) / datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info(f"Starting training with config: {args.config}")
    
    # Set device
    device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create datasets
    train_dataset = PulmonaryArteryDataset(
        data_dir=config['data']['dataset_path'],
        mode='train',
        transform=get_train_transforms(config['augmentation']),
        patch_size=config['data']['patch_size'],
        patch_overlap=config['data']['patch_overlap'],
        cache_num=config['data']['cache_num']
    )
    
    val_dataset = PulmonaryArteryDataset(
        data_dir=config['data']['dataset_path'],
        mode='val',
        transform=get_val_transforms(),
        cache_num=config['data']['cache_num']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Full volume for validation
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    logger.info(f"Model parameters: {model.get_num_parameters():,}")
    
    # Create loss function and optimizer
    criterion = create_loss_function(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create learning rate scheduler
    if config['training']['scheduler']['type'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs']
        )
    else:
        scheduler = None
    
    # Mixed precision scaler
    scaler = GradScaler() if config['hardware']['mixed_precision'] else None
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_metric = 0
    
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_metric = checkpoint.get('best_val_metric', 0)
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    for epoch in range(start_epoch, config['training']['epochs']):
        logger.info(f"Epoch {epoch + 1}/{config['training']['epochs']}")
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, logger
        )
        
        logger.info(f"Train Loss: {train_loss:.4f}, Dice: {train_metrics['dice']:.4f}, "
                   f"Sensitivity: {train_metrics['sensitivity']:.4f}, "
                   f"Precision: {train_metrics['precision']:.4f}")
        
        # Validate
        if (epoch + 1) % config['validation']['val_interval'] == 0:
            val_loss, val_metrics = validate(
                model, val_loader, criterion, device, logger
            )
            
            logger.info(f"Val Loss: {val_loss:.4f}, Dice: {val_metrics['dice']:.4f}, "
                       f"Sensitivity: {val_metrics['sensitivity']:.4f}, "
                       f"Precision: {val_metrics['precision']:.4f}")
            
            # Save best model
            if val_metrics['dice'] > best_val_metric:
                best_val_metric = val_metrics['dice']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_metric': best_val_metric,
                    'config': config
                }, output_dir / 'best_model.pth')
                logger.info(f"Saved best model with dice: {best_val_metric:.4f}")
        
        # Update learning rate
        if scheduler:
            scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % config['checkpoint']['save_interval'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config
            }, output_dir / f'checkpoint_epoch_{epoch + 1}.pth')
    
    logger.info("Training completed!")


if __name__ == '__main__':
    main()