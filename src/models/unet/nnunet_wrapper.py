"""Wrapper for nnU-Net integration"""

import os
import subprocess
import shutil
from pathlib import Path
import SimpleITK as sitk
import numpy as np
from typing import Dict, Optional


class nnUNetWrapper:
    """
    Wrapper for nnU-Net framework
    
    Note: nnU-Net must be installed separately:
    pip install nnunet
    """
    
    def __init__(self, task_name: str = "Task501_PulmonaryArtery", 
                 nnunet_raw_data: Optional[str] = None,
                 nnunet_preprocessed: Optional[str] = None,
                 nnunet_results: Optional[str] = None):
        """
        Initialize nnU-Net wrapper
        
        Args:
            task_name: Name for the nnU-Net task
            nnunet_raw_data: Path to nnUNet_raw_data_base (uses env var if None)
            nnunet_preprocessed: Path to nnUNet_preprocessed (uses env var if None)
            nnunet_results: Path to RESULTS_FOLDER (uses env var if None)
        """
        self.task_name = task_name
        self.task_id = int(task_name.split('_')[0].replace('Task', ''))
        
        # Set up nnU-Net paths
        self.raw_data_base = Path(nnunet_raw_data or os.environ.get('nnUNet_raw_data_base', './nnUNet_raw_data'))
        self.preprocessed = Path(nnunet_preprocessed or os.environ.get('nnUNet_preprocessed', './nnUNet_preprocessed'))
        self.results = Path(nnunet_results or os.environ.get('RESULTS_FOLDER', './nnUNet_results'))
        
        # Create directories if they don't exist
        self.raw_data_base.mkdir(parents=True, exist_ok=True)
        self.preprocessed.mkdir(parents=True, exist_ok=True)
        self.results.mkdir(parents=True, exist_ok=True)

        # nnU-Net expects tasks inside nnUNet_raw_data subdirectory
        (self.raw_data_base / "nnUNet_raw_data").mkdir(parents=True, exist_ok=True)
        self.task_path = self.raw_data_base / "nnUNet_raw_data" / self.task_name
        
    def prepare_dataset(self, dataset_path: str, train_test_split: float = 0.8):
        """
        Prepare dataset in nnU-Net format
        
        Args:
            dataset_path: Path to Parse_dataset directory
            train_test_split: Train/test split ratio
        """
        print(f"Preparing dataset for nnU-Net task: {self.task_name}")
        
        # Create task directory structure
        self.task_path.mkdir(exist_ok=True)
        (self.task_path / "imagesTr").mkdir(exist_ok=True)
        (self.task_path / "imagesTs").mkdir(exist_ok=True)
        (self.task_path / "labelsTr").mkdir(exist_ok=True)
        
        # Get all patient directories
        dataset_path = Path(dataset_path)
        patient_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
        
        # Split into train and test
        n_train = int(len(patient_dirs) * train_test_split)
        train_patients = patient_dirs[:n_train]
        test_patients = patient_dirs[n_train:]
        
        # Copy training data
        for i, patient_dir in enumerate(train_patients):
            # Copy image
            src_image = patient_dir / 'image' / f'{patient_dir.name}.nii.gz'
            dst_image = self.task_path / "imagesTr" / f"{self.task_name}_{i:03d}_0000.nii.gz"
            shutil.copy2(src_image, dst_image)
            
            # Copy label
            src_label = patient_dir / 'label' / f'{patient_dir.name}.nii.gz'
            dst_label = self.task_path / "labelsTr" / f"{self.task_name}_{i:03d}.nii.gz"
            shutil.copy2(src_label, dst_label)
        
        # Copy test data
        for i, patient_dir in enumerate(test_patients):
            src_image = patient_dir / 'image' / f'{patient_dir.name}.nii.gz'
            dst_image = self.task_path / "imagesTs" / f"{self.task_name}_{i:03d}_0000.nii.gz"
            shutil.copy2(src_image, dst_image)
        
        # Create dataset.json
        self._create_dataset_json(len(train_patients))
        
        print(f"Dataset prepared: {len(train_patients)} training, {len(test_patients)} test cases")
        
    def _create_dataset_json(self, num_training: int):
        """Create dataset.json for nnU-Net"""
        import json
        
        dataset_dict = {
            "name": "Pulmonary Artery Segmentation",
            "description": "Pulmonary artery segmentation from CT scans",
            "tensorImageSize": "3D",
            "reference": "Internal dataset",
            "licence": "Internal use",
            "release": "0.0",
            "modality": {
                "0": "CT"
            },
            "labels": {
                "0": "background",
                "1": "pulmonary_artery"
            },
            "numTraining": num_training,
            "numTest": 0,  # Will be updated when we run inference
            "training": [],
            "test": []
        }
        
        # Add training files
        for i in range(num_training):
            dataset_dict["training"].append({
                "image": f"./imagesTr/{self.task_name}_{i:03d}.nii.gz",
                "label": f"./labelsTr/{self.task_name}_{i:03d}.nii.gz"
            })
        
        # Save dataset.json
        with open(self.task_path / "dataset.json", 'w') as f:
            json.dump(dataset_dict, f, indent=4)
    
    def plan_and_preprocess(self):
        """Run nnU-Net planning and preprocessing"""
        print("Running nnU-Net planning and preprocessing...")
        
        cmd = [
            "nnUNet_plan_and_preprocess",
            "-t", str(self.task_id),
            "--verify_dataset_integrity"
        ]
        
        subprocess.run(cmd, check=True)
        print("Planning and preprocessing completed!")
    
    def train(self, network: str = "3d_fullres", fold: int = 0):
        """
        Train nnU-Net model
        
        Args:
            network: Network configuration (2d, 3d_lowres, 3d_fullres, 3d_cascade_fullres)
            fold: Which fold to train (0-4 for 5-fold CV, or 'all')
        """
        print(f"Training nnU-Net {network} on fold {fold}...")
        
        cmd = [
            "nnUNet_train",
            network,
            "nnUNetTrainerV2",
            str(self.task_id),
            str(fold)
        ]
        
        subprocess.run(cmd, check=True)
        print("Training completed!")
    
    def predict(self, input_folder: str, output_folder: str, 
                network: str = "3d_fullres", fold: int = 0,
                save_npz: bool = True, disable_tta: bool = False):
        """
        Run nnU-Net prediction
        
        Args:
            input_folder: Folder containing test images
            output_folder: Where to save predictions
            network: Network configuration used for training
            fold: Which fold model to use
            save_npz: Save softmax outputs
            disable_tta: Disable test time augmentation
        """
        print(f"Running nnU-Net prediction...")
        
        model_folder = self.results / network / self.task_name / f"nnUNetTrainerV2__nnUNetPlansv2.1/fold_{fold}"
        
        cmd = [
            "nnUNet_predict",
            "-i", input_folder,
            "-o", output_folder,
            "-t", str(self.task_id),
            "-m", network,
            "-f", str(fold),
            "--save_npz" if save_npz else "",
            "--disable_tta" if disable_tta else ""
        ]
        
        # Remove empty strings from command
        cmd = [c for c in cmd if c]
        
        subprocess.run(cmd, check=True)
        print(f"Predictions saved to {output_folder}")
    
    def predict_simple(self, image_path: str, output_path: str,
                      network: str = "3d_fullres", fold: int = 0):
        """
        Simple prediction interface for single images
        
        Args:
            image_path: Path to input image
            output_path: Path to save prediction
            network: Network configuration
            fold: Which fold model to use
        """
        # Create temporary directories
        temp_input = Path("./temp_nnunet_input")
        temp_output = Path("./temp_nnunet_output")
        temp_input.mkdir(exist_ok=True)
        temp_output.mkdir(exist_ok=True)
        
        # Copy input to temp directory with correct naming
        temp_image = temp_input / f"{self.task_name}_999_0000.nii.gz"
        shutil.copy2(image_path, temp_image)
        
        # Run prediction
        self.predict(str(temp_input), str(temp_output), network, fold, 
                    save_npz=True, disable_tta=True)
        
        # Copy result to output path
        pred_file = temp_output / f"{self.task_name}_999.nii.gz"
        shutil.copy2(pred_file, output_path)
        
        # Also copy probability map if available
        prob_file = temp_output / f"{self.task_name}_999.npz"
        if prob_file.exists():
            prob_output = Path(output_path).parent / f"{Path(output_path).stem}_prob.npz"
            shutil.copy2(prob_file, prob_output)
        
        # Clean up
        shutil.rmtree(temp_input)
        shutil.rmtree(temp_output)


def check_nnunet_installation():
    """Check if nnU-Net is properly installed"""
    try:
        import nnunet
        print("nnU-Net is installed")
        
        # Check for required environment variables
        required_vars = ['nnUNet_raw_data_base', 'nnUNet_preprocessed', 'RESULTS_FOLDER']
        missing_vars = [var for var in required_vars if not os.environ.get(var)]
        
        if missing_vars:
            print(f"Warning: Missing environment variables: {missing_vars}")
            print("You can set them or pass paths directly to nnUNetWrapper")
        
        return True
    except ImportError:
        print("nnU-Net is not installed. Please install it with: pip install nnunet")
        return False