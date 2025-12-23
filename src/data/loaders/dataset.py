"""Dataset classes for pulmonary artery segmentation"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
from typing import Dict, List, Tuple, Optional
import glob
from pathlib import Path


class PulmonaryArteryDataset(Dataset):
    """Dataset for pulmonary artery segmentation"""
    
    def __init__(self, 
                 data_dir: str,
                 mode: str = 'train',
                 transform=None,
                 patch_size: Tuple[int, int, int] = (128, 128, 128),
                 patch_overlap: float = 0.5,
                 cache_num: int = 0):
        """
        Args:
            data_dir: Path to Parse_dataset directory
            mode: 'train', 'val', or 'test'
            transform: Optional transforms to apply
            patch_size: Size of patches to extract for training
            patch_overlap: Overlap between patches (0-1)
            cache_num: Number of volumes to cache in memory
        """
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.transform = transform
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.cache_num = cache_num
        
        # Get list of patient directories and validate them
        all_patient_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        self.patient_dirs = self._validate_patient_dirs(all_patient_dirs)
        
        # Split data based on mode (simple 80/20 split for now)
        n_total = len(self.patient_dirs)
        n_train = int(0.8 * n_total)
        
        if mode == 'train':
            self.patient_dirs = self.patient_dirs[:n_train]
        else:  # val/test
            self.patient_dirs = self.patient_dirs[n_train:]
        
        # Cache for loaded volumes
        self.cache = {}
        
        # If using patches, pre-compute patch locations
        if mode == 'train':
            self.patches = self._compute_patch_locations()
        
    def __len__(self):
        if self.mode == 'train' and hasattr(self, 'patches'):
            return len(self.patches)
        return len(self.patient_dirs)
    
    def __getitem__(self, idx):
        if self.mode == 'train' and hasattr(self, 'patches'):
            return self._get_patch(idx)
        else:
            return self._get_volume(idx)
    
    def _get_volume(self, idx):
        """Load full volume and mask"""
        patient_dir = self.patient_dirs[idx]
        
        # Check cache
        if patient_dir in self.cache:
            image, mask = self.cache[patient_dir]
        else:
            # Load image and mask
            image_path = patient_dir / 'image' / f'{patient_dir.name}.nii.gz'
            mask_path = patient_dir / 'label' / f'{patient_dir.name}.nii.gz'
            
            image = sitk.ReadImage(str(image_path))
            mask = sitk.ReadImage(str(mask_path))
            
            # Convert to numpy arrays
            image_array = sitk.GetArrayFromImage(image).astype(np.float32)
            mask_array = sitk.GetArrayFromImage(mask).astype(np.float32)
            
            # Normalize image
            image_array = self._normalize(image_array)
            
            # Cache if enabled
            if len(self.cache) < self.cache_num:
                self.cache[patient_dir] = (image_array, mask_array)
            
            image, mask = image_array, mask_array
        
        # Convert to torch tensors
        image = torch.from_numpy(image).unsqueeze(0)  # Add channel dimension
        mask = torch.from_numpy(mask).unsqueeze(0)
        
        # Apply transforms if any
        if self.transform:
            image, mask = self.transform(image, mask)
        
        return {
            'image': image,
            'mask': mask,
            'patient_id': patient_dir.name
        }
    
    def _get_patch(self, idx):
        """Get a patch for training"""
        patch_info = self.patches[idx]
        volume_data = self._get_volume(patch_info['volume_idx'])
        
        # Extract patch
        z, y, x = patch_info['position']
        pz, py, px = self.patch_size
        
        image_patch = volume_data['image'][
            :, z:z+pz, y:y+py, x:x+px
        ]
        mask_patch = volume_data['mask'][
            :, z:z+pz, y:y+py, x:x+px
        ]
        
        return {
            'image': image_patch,
            'mask': mask_patch,
            'patient_id': volume_data['patient_id'],
            'patch_position': patch_info['position']
        }
    
    def _compute_patch_locations(self):
        """Pre-compute valid patch locations for all volumes"""
        patches = []
        
        for vol_idx, patient_dir in enumerate(self.patient_dirs):
            # Load mask to find valid regions
            mask_path = patient_dir / 'label' / f'{patient_dir.name}.nii.gz'
            mask = sitk.ReadImage(str(mask_path))
            mask_array = sitk.GetArrayFromImage(mask)
            
            # Get volume shape
            shape = mask_array.shape
            
            # Compute stride
            stride = [int(s * (1 - self.patch_overlap)) for s in self.patch_size]
            
            # Generate patch positions
            for z in range(0, shape[0] - self.patch_size[0] + 1, stride[0]):
                for y in range(0, shape[1] - self.patch_size[1] + 1, stride[1]):
                    for x in range(0, shape[2] - self.patch_size[2] + 1, stride[2]):
                        # Check if patch contains any positive labels
                        patch = mask_array[
                            z:z+self.patch_size[0],
                            y:y+self.patch_size[1],
                            x:x+self.patch_size[2]
                        ]
                        
                        if patch.sum() > 0:  # Contains vessel
                            patches.append({
                                'volume_idx': vol_idx,
                                'position': (z, y, x)
                            })
        
        return patches
    
    def _validate_patient_dirs(self, all_patient_dirs):
        """Validate patient directories and return only complete cases"""
        valid_dirs = []
        
        for patient_dir in all_patient_dirs:
            patient_id = patient_dir.name
            image_path = patient_dir / 'image' / f'{patient_id}.nii.gz'
            label_path = patient_dir / 'label' / f'{patient_id}.nii.gz'
            
            # Check if both files exist and are not empty
            if (image_path.exists() and label_path.exists() and 
                image_path.stat().st_size > 1000 and 
                label_path.stat().st_size > 1000):
                valid_dirs.append(patient_dir)
        
        print(f"Dataset validation: {len(valid_dirs)}/{len(all_patient_dirs)} complete cases")
        return valid_dirs
    
    def _normalize(self, image):
        """Normalize image intensities"""
        # Simple percentile normalization
        p1, p99 = np.percentile(image, [1, 99])
        image = np.clip(image, p1, p99)
        image = (image - p1) / (p99 - p1 + 1e-8)
        return image


class GraphCorrectionDataset(Dataset):
    """Dataset for graph correction training"""
    
    def __init__(self,
                 data_dir: str,
                 unet_predictions_dir: str,
                 mode: str = 'train',
                 graph_extractor=None):
        """
        Args:
            data_dir: Path to Parse_dataset directory with ground truth
            unet_predictions_dir: Path to U-Net prediction masks
            mode: 'train', 'val', or 'test'
            graph_extractor: GraphExtractor instance for converting masks to graphs
        """
        self.data_dir = Path(data_dir)
        self.predictions_dir = Path(unet_predictions_dir)
        self.mode = mode
        self.graph_extractor = graph_extractor
        
        # Get list of available predictions
        self.patient_ids = self._get_available_patients()
        
        # Split data
        n_total = len(self.patient_ids)
        n_train = int(0.8 * n_total)
        
        if mode == 'train':
            self.patient_ids = self.patient_ids[:n_train]
        else:
            self.patient_ids = self.patient_ids[n_train:]
    
    def __len__(self):
        return len(self.patient_ids)
    
    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        
        # Load ground truth mask
        gt_mask_path = self.data_dir / patient_id / 'label' / f'{patient_id}.nii.gz'
        gt_mask = sitk.ReadImage(str(gt_mask_path))
        gt_mask_array = sitk.GetArrayFromImage(gt_mask).astype(np.uint8)
        spacing = gt_mask.GetSpacing()[::-1]  # Convert to ZYX
        
        # Load predicted mask
        pred_mask_path = self.predictions_dir / f'{patient_id}_pred.nii.gz'
        pred_mask = sitk.ReadImage(str(pred_mask_path))
        pred_mask_array = sitk.GetArrayFromImage(pred_mask).astype(np.uint8)
        
        # Load confidence map if available
        conf_path = self.predictions_dir / f'{patient_id}_conf.nii.gz'
        if conf_path.exists():
            conf_map = sitk.ReadImage(str(conf_path))
            conf_array = sitk.GetArrayFromImage(conf_map).astype(np.float32)
        else:
            conf_array = None
        
        # Extract graphs
        gt_graph = self.graph_extractor.extract_graph(gt_mask_array, spacing)
        pred_graph = self.graph_extractor.extract_graph(
            pred_mask_array, spacing, conf_array
        )
        
        # Convert to PyTorch Geometric format
        gt_data = gt_graph.to_pytorch_geometric()
        pred_data = pred_graph.to_pytorch_geometric()
        
        return {
            'gt_graph': gt_data,
            'pred_graph': pred_data,
            'patient_id': patient_id,
            'gt_mask': gt_mask_array,
            'pred_mask': pred_mask_array,
            'spacing': spacing
        }
    
    def _get_available_patients(self):
        """Get list of patients with both GT and predictions"""
        gt_patients = set(d.name for d in self.data_dir.iterdir() if d.is_dir())
        pred_files = list(self.predictions_dir.glob('*_pred.nii.gz'))
        pred_patients = set(f.stem.replace('_pred', '') for f in pred_files)
        
        # Return intersection
        return sorted(list(gt_patients & pred_patients))