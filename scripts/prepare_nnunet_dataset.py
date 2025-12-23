#!/usr/bin/env python
"""Prepare dataset for nnU-Net training"""

import os
import sys
import argparse
import json
import shutil
from pathlib import Path
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


class nnUNetDatasetPreparator:
    """Prepare pulmonary artery dataset for nnU-Net"""
    
    def __init__(self, task_id=501, task_name="PulmonaryArtery"):
        self.task_id = task_id
        self.task_name = task_name
        self.full_task_name = f"Task{task_id:03d}_{task_name}"
        
        # Get nnU-Net paths from environment
        self.raw_data_base = Path(os.environ.get('nnUNet_raw_data_base', './nnUNet_raw_data'))
        self.task_path = self.raw_data_base / self.full_task_name
        
        print(f"nnU-Net task: {self.full_task_name}")
        print(f"Task path: {self.task_path}")
    
    def prepare_dataset(self, source_dataset_path, train_ratio=0.8, val_ratio=0.1):
        """
        Prepare dataset in nnU-Net format
        
        Args:
            source_dataset_path: Path to Parse_dataset directory
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation (rest goes to test)
        """
        source_path = Path(source_dataset_path)
        
        # Create directory structure
        self._create_directory_structure()
        
        # Get all patient directories
        patient_dirs = sorted([d for d in source_path.iterdir() if d.is_dir()])
        print(f"Found {len(patient_dirs)} patients")
        
        # Split data
        n_total = len(patient_dirs)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_patients = patient_dirs[:n_train]
        val_patients = patient_dirs[n_train:n_train + n_val] 
        test_patients = patient_dirs[n_train + n_val:]
        
        print(f"Split: {len(train_patients)} train, {len(val_patients)} val, {len(test_patients)} test")
        
        # Process training data
        self._process_patients(train_patients, "imagesTr", "labelsTr", "Training")
        
        # Process test data (no labels for nnU-Net test set)
        test_data_info = self._process_patients(test_patients, "imagesTs", None, "Test")
        val_data_info = self._process_patients(val_patients, "imagesTs", None, "Validation")
        
        # Create dataset.json
        training_data_info = self._process_patients(train_patients, None, None, None, info_only=True)
        self._create_dataset_json(training_data_info, len(train_patients))
        
        # Save data split information
        split_info = {
            'train': [p.name for p in train_patients],
            'val': [p.name for p in val_patients], 
            'test': [p.name for p in test_patients],
            'task_name': self.full_task_name,
            'task_id': self.task_id
        }
        
        with open(self.task_path / 'data_split.json', 'w') as f:
            json.dump(split_info, f, indent=2)
        
        print(f"\nDataset preparation completed!")
        print(f"Task directory: {self.task_path}")
        print(f"Next steps:")
        print(f"1. Run: nnUNet_plan_and_preprocess -t {self.task_id}")
        print(f"2. Run: nnUNet_train 3d_fullres nnUNetTrainerV2 {self.task_id} FOLD")
    
    def _create_directory_structure(self):
        """Create nnU-Net directory structure"""
        directories = [
            self.task_path,
            self.task_path / "imagesTr",
            self.task_path / "labelsTr", 
            self.task_path / "imagesTs"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {directory}")
    
    def _process_patients(self, patient_dirs, image_subdir, label_subdir, description, info_only=False):
        """Process patient data"""
        if description and not info_only:
            print(f"\nProcessing {description.lower()} data...")
        
        data_info = []
        
        for i, patient_dir in enumerate(tqdm(patient_dirs, desc=description or "Processing")):
            patient_id = patient_dir.name
            
            # Source files
            src_image = patient_dir / 'image' / f'{patient_id}.nii.gz'
            src_label = patient_dir / 'label' / f'{patient_id}.nii.gz'
            
            if not src_image.exists():
                print(f"Warning: Image not found for {patient_id}")
                continue
            
            if info_only:
                # Just collect information for dataset.json
                data_info.append({
                    'image': f"./imagesTr/{self.full_task_name}_{i:03d}_0000.nii.gz",
                    'label': f"./labelsTr/{self.full_task_name}_{i:03d}.nii.gz"
                })
                continue
            
            # Destination files (nnU-Net naming convention)
            if image_subdir:
                dst_image = self.task_path / image_subdir / f"{self.full_task_name}_{i:03d}_0000.nii.gz"
                
                # Process and copy image
                self._process_and_copy_image(src_image, dst_image)
            
            if label_subdir and src_label.exists():
                dst_label = self.task_path / label_subdir / f"{self.full_task_name}_{i:03d}.nii.gz"
                
                # Process and copy label
                self._process_and_copy_label(src_label, dst_label)
            
            if image_subdir:  # Only add to info if we're actually processing files
                data_info.append({
                    'patient_id': patient_id,
                    'case_id': f"{self.full_task_name}_{i:03d}"
                })
        
        return data_info
    
    def _process_and_copy_image(self, src_path, dst_path):
        """Process and copy image with validation"""
        try:
            # Load image
            image = sitk.ReadImage(str(src_path))
            
            # Validate image
            array = sitk.GetArrayFromImage(image)
            
            # Basic validation
            if array.size == 0:
                raise ValueError("Empty image")
            
            if not np.isfinite(array).all():
                print(f"Warning: Non-finite values in {src_path}")
                array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
                image = sitk.GetImageFromArray(array)
                image.CopyInformation(sitk.ReadImage(str(src_path)))
            
            # Ensure proper data type (float32 for images)
            if image.GetPixelID() != sitk.sitkFloat32:
                image = sitk.Cast(image, sitk.sitkFloat32)
            
            # Write processed image
            sitk.WriteImage(image, str(dst_path))
            
        except Exception as e:
            print(f"Error processing image {src_path}: {e}")
            # Fallback: just copy the file
            shutil.copy2(src_path, dst_path)
    
    def _process_and_copy_label(self, src_path, dst_path):
        """Process and copy label with validation"""
        try:
            # Load label
            label = sitk.ReadImage(str(src_path))
            
            # Validate label
            array = sitk.GetArrayFromImage(label)
            
            # Ensure binary values
            unique_values = np.unique(array)
            if not all(v in [0, 1] for v in unique_values):
                print(f"Warning: Non-binary values in label {src_path}: {unique_values}")
                array = (array > 0.5).astype(np.uint8)
                label = sitk.GetImageFromArray(array)
                label.CopyInformation(sitk.ReadImage(str(src_path)))
            
            # Ensure proper data type (uint8 or uint16 for labels)
            if label.GetPixelID() not in [sitk.sitkUInt8, sitk.sitkUInt16]:
                label = sitk.Cast(label, sitk.sitkUInt8)
            
            # Write processed label
            sitk.WriteImage(label, str(dst_path))
            
        except Exception as e:
            print(f"Error processing label {src_path}: {e}")
            # Fallback: just copy the file
            shutil.copy2(src_path, dst_path)
    
    def _create_dataset_json(self, training_data, num_training):
        """Create dataset.json file required by nnU-Net"""
        
        dataset_dict = {
            "name": "Pulmonary Artery Segmentation",
            "description": "3D segmentation of pulmonary arteries from chest CT scans",
            "tensorImageSize": "3D",
            "reference": "Internal medical imaging dataset",
            "licence": "Internal use",
            "release": "1.0",
            "modality": {
                "0": "CT"
            },
            "labels": {
                "0": "background", 
                "1": "pulmonary_artery"
            },
            "numTraining": num_training,
            "numTest": 0,
            "training": training_data,
            "test": []
        }
        
        # Write dataset.json
        with open(self.task_path / "dataset.json", 'w') as f:
            json.dump(dataset_dict, f, indent=2)
        
        print(f"Created dataset.json with {num_training} training cases")
    
    def verify_dataset(self):
        """Verify the prepared dataset"""
        print(f"\nVerifying dataset: {self.full_task_name}")
        
        # Check directory structure
        required_dirs = ["imagesTr", "labelsTr", "imagesTs"]
        for dirname in required_dirs:
            dir_path = self.task_path / dirname
            if dir_path.exists():
                n_files = len(list(dir_path.glob("*.nii.gz")))
                print(f"✓ {dirname}: {n_files} files")
            else:
                print(f"✗ {dirname}: Missing")
        
        # Check dataset.json
        dataset_json = self.task_path / "dataset.json"
        if dataset_json.exists():
            with open(dataset_json, 'r') as f:
                data = json.load(f)
            print(f"✓ dataset.json: {data['numTraining']} training cases")
        else:
            print("✗ dataset.json: Missing")
        
        # Verify image-label pairs
        images_tr = sorted(list((self.task_path / "imagesTr").glob("*.nii.gz")))
        labels_tr = sorted(list((self.task_path / "labelsTr").glob("*.nii.gz")))
        
        print(f"Image-label pairs: {len(images_tr)} images, {len(labels_tr)} labels")
        
        if len(images_tr) != len(labels_tr):
            print("⚠ Warning: Mismatch between number of images and labels")
        
        # Sample verification
        if images_tr and labels_tr:
            try:
                sample_image = sitk.ReadImage(str(images_tr[0]))
                sample_label = sitk.ReadImage(str(labels_tr[0]))
                
                img_size = sample_image.GetSize()
                lbl_size = sample_label.GetSize()
                
                print(f"Sample image size: {img_size}")
                print(f"Sample label size: {lbl_size}")
                
                if img_size == lbl_size:
                    print("✓ Image and label dimensions match")
                else:
                    print("✗ Image and label dimensions don't match")
                    
            except Exception as e:
                print(f"Error reading sample files: {e}")


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset for nnU-Net')
    parser.add_argument('--dataset', type=str, default='DATASET/Parse_dataset',
                        help='Path to Parse_dataset directory')
    parser.add_argument('--task-id', type=int, default=501,
                        help='nnU-Net task ID (default: 501)')
    parser.add_argument('--task-name', type=str, default='PulmonaryArtery',
                        help='nnU-Net task name (default: PulmonaryArtery)')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Training data ratio (default: 0.8)')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='Validation data ratio (default: 0.1)')
    parser.add_argument('--verify-only', action='store_true',
                        help='Only verify existing dataset')
    
    args = parser.parse_args()
    
    # Check if nnU-Net environment is set up
    if not os.environ.get('nnUNet_raw_data_base'):
        print("Error: nnUNet_raw_data_base environment variable not set!")
        print("Please run: export nnUNet_raw_data_base=/path/to/nnunet/raw/data")
        sys.exit(1)
    
    # Create preparator
    preparator = nnUNetDatasetPreparator(args.task_id, args.task_name)
    
    if args.verify_only:
        preparator.verify_dataset()
    else:
        # Prepare dataset
        preparator.prepare_dataset(
            args.dataset,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio
        )
        
        # Verify after preparation
        preparator.verify_dataset()
        
        print("\n" + "="*50)
        print("NEXT STEPS:")
        print("="*50)
        print(f"1. Plan and preprocess:")
        print(f"   nnUNet_plan_and_preprocess -t {args.task_id}")
        print()
        print(f"2. Train (example for fold 0):")
        print(f"   nnUNet_train 3d_fullres nnUNetTrainerV2 {args.task_id} 0")
        print()
        print(f"3. Train all folds:")
        print(f"   for fold in {{0,1,2,3,4}}; do")
        print(f"       nnUNet_train 3d_fullres nnUNetTrainerV2 {args.task_id} $fold")
        print(f"   done")


if __name__ == '__main__':
    main()