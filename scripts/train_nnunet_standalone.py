#!/usr/bin/env python
"""Standalone nnU-Net training script with monitoring and management"""

import os
import sys
import argparse
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
import logging


class nnUNetTrainingManager:
    """Manage nnU-Net training with monitoring and logging"""
    
    def __init__(self, task_id, output_dir=None):
        self.task_id = task_id
        self.task_name = f"Task{task_id:03d}"
        
        # Setup output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(f"experiments/nnunet_{self.task_name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Check environment
        self.check_environment()
        
        # Available networks
        self.networks = {
            '2d': '2d',
            '3d_lowres': '3d_lowres', 
            '3d_fullres': '3d_fullres',
            '3d_cascade': '3d_cascade_fullres'
        }
        
    def setup_logging(self):
        """Setup logging for training monitoring"""
        log_file = self.output_dir / f'nnunet_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"nnU-Net Training Manager initialized for {self.task_name}")
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def check_environment(self):
        """Check nnU-Net environment setup"""
        required_vars = ['nnUNet_raw_data_base', 'nnUNet_preprocessed', 'RESULTS_FOLDER']
        missing_vars = []
        
        for var in required_vars:
            if not os.environ.get(var):
                missing_vars.append(var)
        
        if missing_vars:
            self.logger.error(f"Missing environment variables: {missing_vars}")
            self.logger.error("Please set up nnU-Net environment variables")
            sys.exit(1)
        
        # Log environment info
        for var in required_vars:
            self.logger.info(f"{var}: {os.environ[var]}")
    
    def run_plan_and_preprocess(self, verify_dataset=True):
        """Run nnU-Net planning and preprocessing"""
        self.logger.info("Starting nnU-Net planning and preprocessing...")
        
        cmd = ["nnUNet_plan_and_preprocess", "-t", str(self.task_id)]
        
        if verify_dataset:
            cmd.append("--verify_dataset_integrity")
        
        try:
            # Run command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            # Log output
            if result.stdout:
                self.logger.info("Planning output:")
                for line in result.stdout.split('\n'):
                    if line.strip():
                        self.logger.info(f"  {line}")
            
            if result.stderr:
                self.logger.warning("Planning stderr:")
                for line in result.stderr.split('\n'):
                    if line.strip():
                        self.logger.warning(f"  {line}")
            
            if result.returncode == 0:
                self.logger.info("Planning and preprocessing completed successfully!")
                
                # Save planning info
                self.save_planning_info()
                
                return True
            else:
                self.logger.error(f"Planning failed with return code {result.returncode}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("Planning timed out after 1 hour")
            return False
        except Exception as e:
            self.logger.error(f"Planning failed with exception: {e}")
            return False
    
    def save_planning_info(self):
        """Save nnU-Net planning information"""
        try:
            # Find plans file
            preprocessed_dir = Path(os.environ['nnUNet_preprocessed']) / self.task_name
            plans_file = preprocessed_dir / "nnUNetPlansv2.1.pkl"
            
            if plans_file.exists():
                self.logger.info(f"Plans file created: {plans_file}")
                
                # Try to extract some basic info
                try:
                    import pickle
                    with open(plans_file, 'rb') as f:
                        plans = pickle.load(f)
                    
                    info = {
                        'task_name': self.task_name,
                        'plans_file': str(plans_file),
                        'num_stages': len(plans['plans_per_stage']),
                        'modalities': plans.get('modalities', {}),
                        'original_spacing': plans.get('original_spacing', None),
                        'planning_timestamp': datetime.now().isoformat()
                    }
                    
                    # Save info
                    with open(self.output_dir / 'planning_info.json', 'w') as f:
                        json.dump(info, f, indent=2)
                    
                    self.logger.info(f"Planning info saved: {self.output_dir / 'planning_info.json'}")
                    
                except Exception as e:
                    self.logger.warning(f"Could not extract planning details: {e}")
            
        except Exception as e:
            self.logger.warning(f"Could not save planning info: {e}")
    
    def train_network(self, network='3d_fullres', fold=0, trainer='nnUNetTrainerV2', 
                     continue_training=False, validation_only=False):
        """Train a specific network configuration"""
        
        if network not in self.networks:
            self.logger.error(f"Unknown network: {network}")
            self.logger.error(f"Available networks: {list(self.networks.keys())}")
            return False
        
        network_name = self.networks[network]
        
        self.logger.info(f"Training {network_name} (fold {fold}) with {trainer}")
        
        # Build command
        cmd = [
            "nnUNet_train",
            network_name,
            trainer,
            str(self.task_id),
            str(fold)
        ]
        
        if continue_training:
            cmd.append("-c")
        
        if validation_only:
            cmd.append("--validation_only")
        
        try:
            # Create fold-specific log file
            fold_log = self.output_dir / f'training_{network}_{trainer}_fold{fold}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            
            self.logger.info(f"Starting training... (detailed log: {fold_log})")
            self.logger.info(f"Command: {' '.join(cmd)}")
            
            # Run training
            start_time = time.time()
            
            with open(fold_log, 'w') as log_file:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # Monitor training progress
                for line in iter(process.stdout.readline, ''):
                    if line:
                        log_file.write(line)
                        log_file.flush()
                        
                        # Log important lines to main logger
                        if any(keyword in line.lower() for keyword in ['epoch', 'loss', 'dice', 'error', 'finished']):
                            self.logger.info(f"  {line.strip()}")
                
                process.wait()
                return_code = process.returncode
            
            training_time = time.time() - start_time
            
            if return_code == 0:
                self.logger.info(f"Training completed successfully in {training_time/3600:.2f} hours!")
                
                # Save training info
                self.save_training_info(network, fold, trainer, training_time, True)
                
                return True
            else:
                self.logger.error(f"Training failed with return code {return_code}")
                self.save_training_info(network, fold, trainer, training_time, False)
                return False
                
        except Exception as e:
            self.logger.error(f"Training failed with exception: {e}")
            return False
    
    def save_training_info(self, network, fold, trainer, training_time, success):
        """Save training information"""
        info = {
            'task_id': self.task_id,
            'network': network,
            'fold': fold,
            'trainer': trainer,
            'training_time_hours': training_time / 3600,
            'success': success,
            'timestamp': datetime.now().isoformat()
        }
        
        # Load existing info if available
        info_file = self.output_dir / 'training_info.json'
        if info_file.exists():
            with open(info_file, 'r') as f:
                all_info = json.load(f)
        else:
            all_info = {'trainings': []}
        
        all_info['trainings'].append(info)
        
        # Save updated info
        with open(info_file, 'w') as f:
            json.dump(all_info, f, indent=2)
    
    def train_all_folds(self, network='3d_fullres', trainer='nnUNetTrainerV2'):
        """Train all 5 folds for a network"""
        self.logger.info(f"Training all folds for {network}")
        
        results = []
        for fold in range(5):
            self.logger.info(f"=== FOLD {fold} ===")
            success = self.train_network(network, fold, trainer)
            results.append(success)
            
            if not success:
                self.logger.warning(f"Fold {fold} failed, continuing with next fold...")
        
        successful_folds = sum(results)
        self.logger.info(f"Training summary: {successful_folds}/5 folds completed successfully")
        
        return results
    
    def find_best_configuration(self):
        """Find best network configuration automatically"""
        self.logger.info("Finding best configuration...")
        
        cmd = [
            "nnUNet_find_best_configuration",
            "-m", "2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres",
            "-t", str(self.task_id)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.stdout:
                self.logger.info("Best configuration output:")
                for line in result.stdout.split('\n'):
                    if line.strip():
                        self.logger.info(f"  {line}")
            
            if result.returncode == 0:
                self.logger.info("Best configuration analysis completed!")
                return True
            else:
                self.logger.error("Best configuration analysis failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Best configuration analysis failed: {e}")
            return False
    
    def predict(self, input_folder, output_folder, model_config=None):
        """Run prediction with trained model"""
        if model_config is None:
            model_config = {
                'network': '3d_fullres',
                'trainer': 'nnUNetTrainerV2',
                'folds': [0, 1, 2, 3, 4]
            }
        
        self.logger.info(f"Running prediction on {input_folder}")
        
        cmd = [
            "nnUNet_predict",
            "-i", str(input_folder),
            "-o", str(output_folder),
            "-t", str(self.task_id),
            "-m", model_config['network'],
            "-tr", model_config['trainer']
        ]
        
        # Add folds
        if model_config['folds']:
            cmd.extend(["-f"] + [str(f) for f in model_config['folds']])
        
        try:
            self.logger.info(f"Prediction command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.stdout:
                self.logger.info("Prediction output:")
                for line in result.stdout.split('\n'):
                    if line.strip():
                        self.logger.info(f"  {line}")
            
            if result.returncode == 0:
                self.logger.info(f"Prediction completed! Results saved to: {output_folder}")
                return True
            else:
                self.logger.error("Prediction failed")
                if result.stderr:
                    self.logger.error(result.stderr)
                return False
                
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description='nnU-Net training manager')
    parser.add_argument('--task-id', type=int, default=501,
                        help='nnU-Net task ID')
    parser.add_argument('--output-dir', type=str,
                        help='Output directory for logs and info')
    parser.add_argument('--network', type=str, default='3d_fullres',
                        choices=['2d', '3d_lowres', '3d_fullres', '3d_cascade'],
                        help='Network configuration to train')
    parser.add_argument('--fold', type=int, 
                        help='Specific fold to train (default: train all folds)')
    parser.add_argument('--trainer', type=str, default='nnUNetTrainerV2',
                        help='nnU-Net trainer class')
    parser.add_argument('--skip-preprocessing', action='store_true',
                        help='Skip planning and preprocessing')
    parser.add_argument('--find-best-config', action='store_true',
                        help='Find best configuration after training')
    
    args = parser.parse_args()
    
    # Create training manager
    manager = nnUNetTrainingManager(args.task_id, args.output_dir)
    
    # Run planning and preprocessing
    if not args.skip_preprocessing:
        if not manager.run_plan_and_preprocess():
            manager.logger.error("Preprocessing failed, stopping")
            sys.exit(1)
    
    # Train network
    if args.fold is not None:
        # Train specific fold
        success = manager.train_network(args.network, args.fold, args.trainer)
        if not success:
            sys.exit(1)
    else:
        # Train all folds
        results = manager.train_all_folds(args.network, args.trainer)
        if not any(results):
            manager.logger.error("All folds failed")
            sys.exit(1)
    
    # Find best configuration
    if args.find_best_config:
        manager.find_best_configuration()
    
    manager.logger.info("nnU-Net training completed!")


if __name__ == '__main__':
    main()