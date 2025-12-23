#!/usr/bin/env python
"""Generate predictions from trained nnU-Net models"""

import os
import sys
import argparse
import subprocess
import shutil
import tempfile
from pathlib import Path
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


class nnUNetPredictor:
    """Generate predictions from trained nnU-Net models"""
    
    def __init__(self, task_id=501, results_folder=None):
        self.task_id = task_id
        self.task_name = f"Task{task_id:03d}_PulmonaryArtery"
        
        # Get results folder
        if results_folder:
            self.results_folder = Path(results_folder)
        else:
            self.results_folder = Path(os.environ.get('RESULTS_FOLDER', './nnUNet_results'))
        
        print(f"nnU-Net Predictor for {self.task_name}")
        print(f"Results folder: {self.results_folder}")
        
        # Find available models
        self.available_models = self._find_available_models()
        
    def _find_available_models(self):
        """Find available trained models"""
        models = {}
        
        # Look for different network configurations
        networks = ['2d', '3d_lowres', '3d_fullres', '3d_cascade_fullres']
        trainer = 'nnUNetTrainerV2'
        
        for network in networks:
            model_dir = self.results_folder / network / self.task_name / f"{trainer}__nnUNetPlansv2.1"
            
            if model_dir.exists():
                # Find available folds
                folds = []
                for fold_dir in model_dir.iterdir():
                    if fold_dir.is_dir() and fold_dir.name.startswith('fold_'):
                        fold_num = fold_dir.name.split('_')[1]
                        # Check if model file exists
                        model_file = fold_dir / 'model_final_checkpoint.model'
                        if model_file.exists():
                            folds.append(int(fold_num))
                
                if folds:
                    models[network] = {
                        'path': model_dir,
                        'folds': sorted(folds),
                        'trainer': trainer
                    }
                    print(f"Found {network} model with folds: {folds}")
        
        if not models:
            print("Warning: No trained nnU-Net models found!")
            
        return models
    
    def predict_dataset(self, dataset_path, output_dir, network='3d_fullres', 
                       use_all_folds=True, save_probabilities=True):
        """Generate predictions for entire dataset"""
        
        if network not in self.available_models:
            print(f"Error: Network {network} not available")
            print(f"Available networks: {list(self.available_models.keys())}")
            return False
        
        dataset_path = Path(dataset_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get patient directories
        patient_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
        print(f"Found {len(patient_dirs)} patients to process")
        
        # Create temporary directories for nnU-Net format
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            temp_input = temp_path / 'input'
            temp_output = temp_path / 'output'
            temp_input.mkdir()
            temp_output.mkdir()
            
            # Copy and rename files to nnU-Net format
            patient_mapping = {}
            for i, patient_dir in enumerate(patient_dirs):
                patient_id = patient_dir.name
                src_image = patient_dir / 'image' / f'{patient_id}.nii.gz'
                
                if not src_image.exists():
                    print(f"Warning: Image not found for {patient_id}")
                    continue
                
                # nnU-Net naming convention
                nnunet_name = f"{self.task_name}_{i:03d}_0000.nii.gz"
                dst_image = temp_input / nnunet_name
                
                shutil.copy2(src_image, dst_image)
                patient_mapping[nnunet_name] = patient_id
            
            print(f"Prepared {len(patient_mapping)} cases for prediction")
            
            # Run nnU-Net prediction
            success = self._run_nnunet_predict(
                temp_input, temp_output, network, use_all_folds, save_probabilities
            )
            
            if not success:
                print("nnU-Net prediction failed")
                return False
            
            # Copy results back with original patient IDs
            self._copy_results_back(temp_output, output_dir, patient_mapping, save_probabilities)
        
        print(f"Predictions completed! Results saved to: {output_dir}")
        return True
    
    def _run_nnunet_predict(self, input_dir, output_dir, network, use_all_folds, save_probabilities):
        """Run nnU-Net prediction command"""
        
        model_info = self.available_models[network]
        
        cmd = [
            "nnUNet_predict",
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-t", str(self.task_id),
            "-m", network,
            "-tr", model_info['trainer']
        ]
        
        # Add folds
        if use_all_folds:
            cmd.extend(["-f"] + [str(f) for f in model_info['folds']])
        else:
            cmd.extend(["-f", str(model_info['folds'][0])])  # Use first available fold
        
        # Save probabilities if requested
        if save_probabilities:
            cmd.append("--save_npz")
        
        # Disable test time augmentation for faster prediction
        cmd.append("--disable_tta")
        
        print(f"Running prediction command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )
            
            if result.returncode == 0:
                print("nnU-Net prediction completed successfully")
                return True
            else:
                print(f"nnU-Net prediction failed with return code {result.returncode}")
                if result.stderr:
                    print(f"Error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("nnU-Net prediction timed out")
            return False
        except Exception as e:
            print(f"nnU-Net prediction failed: {e}")
            return False
    
    def _copy_results_back(self, temp_output, final_output, patient_mapping, save_probabilities):
        """Copy results back with original patient IDs"""
        
        for nnunet_name, patient_id in patient_mapping.items():
            # Base name without extension
            base_name = nnunet_name.replace('_0000.nii.gz', '')
            
            # Copy binary prediction
            pred_file = temp_output / f"{base_name}.nii.gz"
            if pred_file.exists():
                final_pred = final_output / f"{patient_id}_pred.nii.gz"
                shutil.copy2(pred_file, final_pred)
            
            # Copy probability map if available
            if save_probabilities:
                prob_file = temp_output / f"{base_name}.npz"
                if prob_file.exists():
                    # Convert npz to nii.gz for consistency
                    self._convert_npz_to_nifti(
                        prob_file, 
                        final_output / f"{patient_id}_prob.nii.gz",
                        patient_mapping[nnunet_name]
                    )
    
    def _convert_npz_to_nifti(self, npz_file, output_file, patient_id):
        """Convert nnU-Net npz probability file to NIfTI"""
        try:
            # Load npz file
            data = np.load(npz_file)
            probabilities = data['probabilities']
            
            # nnU-Net saves probabilities as [classes, z, y, x]
            # We want the probability of class 1 (pulmonary artery)
            if probabilities.shape[0] > 1:
                prob_map = probabilities[1]  # Class 1 probabilities
            else:
                prob_map = probabilities[0]
            
            # Create NIfTI image
            prob_image = sitk.GetImageFromArray(prob_map.astype(np.float32))
            
            # Try to copy spacing and origin from original image if possible
            try:
                # Look for original image to copy metadata
                dataset_root = Path("DATASET/Parse_dataset")  # Assumes standard location
                original_image_path = dataset_root / patient_id / 'image' / f'{patient_id}.nii.gz'
                
                if original_image_path.exists():
                    original_image = sitk.ReadImage(str(original_image_path))
                    prob_image.CopyInformation(original_image)
            except:
                pass  # Use default spacing if original not found
            
            # Save probability map
            sitk.WriteImage(prob_image, str(output_file))
            
        except Exception as e:
            print(f"Warning: Could not convert probability map for {patient_id}: {e}")
    
    def predict_single_case(self, image_path, output_dir, network='3d_fullres', 
                           case_id=None, save_probabilities=True):
        """Predict single case"""
        
        if case_id is None:
            case_id = Path(image_path).stem.replace('.nii', '')
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            temp_input = temp_path / 'input'
            temp_output = temp_path / 'output'
            temp_input.mkdir()
            temp_output.mkdir()
            
            # Copy to nnU-Net format
            nnunet_name = f"{self.task_name}_000_0000.nii.gz"
            shutil.copy2(image_path, temp_input / nnunet_name)
            
            # Run prediction
            success = self._run_nnunet_predict(
                temp_input, temp_output, network, True, save_probabilities
            )
            
            if success:
                # Copy results
                pred_file = temp_output / f"{self.task_name}_000.nii.gz"
                if pred_file.exists():
                    shutil.copy2(pred_file, output_dir / f"{case_id}_pred.nii.gz")
                
                if save_probabilities:
                    prob_file = temp_output / f"{self.task_name}_000.npz"
                    if prob_file.exists():
                        self._convert_npz_to_nifti(
                            prob_file,
                            output_dir / f"{case_id}_prob.nii.gz",
                            case_id
                        )
                
                print(f"Prediction completed for {case_id}")
                return True
            
        return False
    
    def generate_cv_predictions_for_graph_training(self, dataset_path, output_dir):
        """Generate cross-validation style predictions for graph correction training"""
        
        print("Generating nnU-Net predictions for graph correction training...")
        
        # Use the best available network
        best_network = self._find_best_network()
        if not best_network:
            print("No trained nnU-Net models available")
            return False
        
        # Generate predictions
        success = self.predict_dataset(
            dataset_path, 
            output_dir,
            network=best_network,
            use_all_folds=True,
            save_probabilities=True
        )
        
        if success:
            # Create summary info
            summary = {
                'model_type': 'nnunet',
                'network': best_network,
                'task_id': self.task_id,
                'task_name': self.task_name,
                'available_folds': self.available_models[best_network]['folds'],
                'prediction_timestamp': str(Path(output_dir).stat().st_mtime)
            }
            
            with open(Path(output_dir) / 'nnunet_prediction_info.json', 'w') as f:
                json.dump(summary, f, indent=2)
        
        return success
    
    def _find_best_network(self):
        """Find best available network (prefer 3d_fullres)"""
        preference_order = ['3d_fullres', '3d_cascade_fullres', '3d_lowres', '2d']
        
        for network in preference_order:
            if network in self.available_models:
                return network
        
        return None


def main():
    parser = argparse.ArgumentParser(description='Generate predictions from trained nnU-Net')
    parser.add_argument('--dataset', type=str, default='DATASET/Parse_dataset',
                        help='Path to dataset directory')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for predictions')
    parser.add_argument('--task-id', type=int, default=501,
                        help='nnU-Net task ID')
    parser.add_argument('--network', type=str, default='3d_fullres',
                        choices=['2d', '3d_lowres', '3d_fullres', '3d_cascade_fullres'],
                        help='Network configuration to use')
    parser.add_argument('--single-case', type=str,
                        help='Predict single case (provide image path)')
    parser.add_argument('--no-probabilities', action='store_true',
                        help='Do not save probability maps')
    parser.add_argument('--results-folder', type=str,
                        help='Path to nnU-Net results folder (default: from env)')
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = nnUNetPredictor(args.task_id, args.results_folder)
    
    if args.single_case:
        # Predict single case
        success = predictor.predict_single_case(
            args.single_case,
            args.output_dir,
            network=args.network,
            save_probabilities=not args.no_probabilities
        )
    else:
        # Predict entire dataset
        success = predictor.predict_dataset(
            args.dataset,
            args.output_dir,
            network=args.network,
            save_probabilities=not args.no_probabilities
        )
    
    if success:
        print("Prediction completed successfully!")
    else:
        print("Prediction failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()