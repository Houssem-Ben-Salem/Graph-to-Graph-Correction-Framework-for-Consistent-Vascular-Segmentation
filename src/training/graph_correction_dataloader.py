"""
Graph Correction Data Loader
Enhanced data loader for training the graph correction model with curriculum learning
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Dict, List, Tuple, Optional, Iterator
import pickle
import gzip
from pathlib import Path
import numpy as np
import logging
from dataclasses import dataclass
import random

from src.models.graph_extraction.vascular_graph import VascularGraph
from src.utils.graph_correspondence import CorrespondenceResult


@dataclass
class DataLoaderConfig:
    """Configuration for graph correction data loader"""
    batch_size: int = 4
    num_workers: int = 2
    shuffle: bool = True
    pin_memory: bool = True
    drop_last: bool = True
    
    # Curriculum learning
    curriculum_enabled: bool = True
    adaptive_sampling: bool = True
    level_weights: Optional[Dict[str, float]] = None
    
    # Data augmentation
    online_augmentation: bool = False
    augmentation_prob: float = 0.3
    
    # Memory optimization
    preload_correspondences: bool = True
    cache_graphs: bool = False
    max_cache_size: int = 100


class GraphCorrectionDataset(Dataset):
    """
    Dataset for graph correction training with curriculum learning support
    """
    
    def __init__(self, 
                 dataset_path: Path,
                 curriculum_level: Optional[str] = None,
                 config: Optional[DataLoaderConfig] = None):
        """
        Initialize dataset
        
        Args:
            dataset_path: Path to generated training dataset
            curriculum_level: Specific curriculum level to load
            config: Data loader configuration
        """
        self.dataset_path = Path(dataset_path)
        self.curriculum_level = curriculum_level
        self.config = config or DataLoaderConfig()
        self.logger = logging.getLogger(__name__)
        
        # Load dataset
        self.samples = self._load_dataset()
        
        # Filter by curriculum level if specified
        if curriculum_level:
            self.samples = [s for s in self.samples if s['curriculum_level'] == curriculum_level]
        
        self.logger.info(f"Loaded {len(self.samples)} samples")
        if curriculum_level:
            self.logger.info(f"Filtered to {curriculum_level} level")
        
        # Initialize caching if enabled
        self.graph_cache = {} if self.config.cache_graphs else None
        
        # Precompute curriculum weights
        self.curriculum_weights = self._compute_curriculum_weights()
        
    def _load_dataset(self) -> List[Dict]:
        """Load dataset from file"""
        
        if str(self.dataset_path).endswith('.gz'):
            with gzip.open(self.dataset_path, 'rb') as f:
                samples = pickle.load(f)
        else:
            with open(self.dataset_path, 'rb') as f:
                samples = pickle.load(f)
        
        return samples
    
    def _compute_curriculum_weights(self) -> Dict[str, float]:
        """Compute sampling weights for curriculum levels"""
        
        if not self.config.curriculum_enabled:
            return {}
        
        # Count samples per level
        level_counts = {}
        for sample in self.samples:
            level = sample['curriculum_level']
            level_counts[level] = level_counts.get(level, 0) + 1
        
        # Compute weights (inverse frequency for balancing)
        total_samples = len(self.samples)
        weights = {}
        
        for level, count in level_counts.items():
            if self.config.level_weights and level in self.config.level_weights:
                weights[level] = self.config.level_weights[level]
            else:
                # Inverse frequency weighting
                weights[level] = total_samples / (len(level_counts) * count)
        
        return weights
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single training sample"""
        
        sample = self.samples[idx]
        
        try:
            # Get graphs (with caching if enabled)
            gt_graph = self._get_graph(sample, 'gt_graph')
            degraded_graph = self._get_graph(sample, 'degraded_graph')
            
            # Get correspondences
            correspondences = sample['correspondences']
            
            # Prepare training data
            training_sample = {
                'sample_id': sample['sample_id'],
                'patient_id': sample['patient_id'],
                'curriculum_level': sample['curriculum_level'],
                'gt_graph': gt_graph,
                'degraded_graph': degraded_graph,
                'correspondences': correspondences,
                'degradation_metadata': sample['degradation_metadata'],
                'curriculum_weight': self.curriculum_weights.get(sample['curriculum_level'], 1.0)
            }
            
            # Apply online augmentation if enabled
            if self.config.online_augmentation and np.random.random() < self.config.augmentation_prob:
                training_sample = self._apply_online_augmentation(training_sample)
            
            return training_sample
            
        except Exception as e:
            self.logger.error(f"Failed to load sample {idx}: {e}")
            # Return a dummy sample to avoid breaking the batch
            return self._get_dummy_sample()
    
    def _get_graph(self, sample: Dict, graph_key: str) -> VascularGraph:
        """Get graph with optional caching"""
        
        if self.graph_cache is not None:
            cache_key = f"{sample['sample_id']}_{graph_key}"
            if cache_key in self.graph_cache:
                return self.graph_cache[cache_key]
            
            graph = sample[graph_key]
            
            # Add to cache if not full
            if len(self.graph_cache) < self.config.max_cache_size:
                self.graph_cache[cache_key] = graph
            
            return graph
        else:
            return sample[graph_key]
    
    def _apply_online_augmentation(self, sample: Dict) -> Dict:
        """Apply online data augmentation"""
        
        # Simple augmentation: add small noise to positions
        degraded_graph = sample['degraded_graph']
        
        # Create a copy to avoid modifying original
        augmented_nodes = []
        for node in degraded_graph.nodes:
            aug_node = node.copy()
            
            if 'position' in node and np.random.random() < 0.5:
                position = np.array(aug_node['position'])
                noise = np.random.normal(0, 0.1, 3)  # Small noise
                aug_node['position'] = (position + noise).tolist()
            
            augmented_nodes.append(aug_node)
        
        # Create augmented graph
        augmented_graph = VascularGraph(
            nodes=augmented_nodes,
            edges=degraded_graph.edges.copy(),
            global_properties=degraded_graph.global_properties.copy(),
            metadata={**degraded_graph.metadata, 'online_augmented': True}
        )
        
        sample['degraded_graph'] = augmented_graph
        return sample
    
    def _get_dummy_sample(self) -> Dict:
        """Create dummy sample for error handling"""
        
        dummy_nodes = [{'id': 0, 'position': [0, 0, 0], 'type': 'regular', 'radius_voxels': 1.0}]
        dummy_edges = []
        
        dummy_graph = VascularGraph(
            nodes=dummy_nodes,
            edges=dummy_edges,
            global_properties={'num_nodes': 1, 'num_edges': 0},
            metadata={'dummy': True}
        )
        
        from src.utils.graph_correspondence import CorrespondenceResult
        dummy_correspondences = CorrespondenceResult(
            node_correspondences={0: 0},
            node_confidences={0: 1.0},
            unmatched_pred_nodes=set(),
            unmatched_gt_nodes=set(),
            edge_correspondences={},
            edge_confidences={},
            unmatched_pred_edges=set(),
            unmatched_gt_edges=set(),
            topology_differences={},
            alignment_transform={},
            correspondence_quality={'overall_quality': 1.0},
            metadata={}
        )
        
        return {
            'sample_id': 'dummy',
            'patient_id': 'dummy',
            'curriculum_level': 'easy',
            'gt_graph': dummy_graph,
            'degraded_graph': dummy_graph,
            'correspondences': dummy_correspondences,
            'degradation_metadata': {},
            'curriculum_weight': 1.0
        }
    
    def get_level_distribution(self) -> Dict[str, int]:
        """Get distribution of samples across curriculum levels"""
        
        distribution = {}
        for sample in self.samples:
            level = sample['curriculum_level']
            distribution[level] = distribution.get(level, 0) + 1
        
        return distribution
    
    def get_weighted_sampler(self) -> WeightedRandomSampler:
        """Get weighted random sampler for curriculum learning"""
        
        if not self.config.curriculum_enabled:
            return None
        
        # Compute sample weights
        sample_weights = []
        for sample in self.samples:
            level = sample['curriculum_level']
            weight = self.curriculum_weights.get(level, 1.0)
            sample_weights.append(weight)
        
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )


def collate_graph_correction_batch(batch: List[Dict]) -> Dict:
    """
    Custom collate function for graph correction training
    
    Args:
        batch: List of training samples
        
    Returns:
        Batched data ready for model training
    """
    # Separate components
    sample_ids = [item['sample_id'] for item in batch]
    patient_ids = [item['patient_id'] for item in batch]
    curriculum_levels = [item['curriculum_level'] for item in batch]
    curriculum_weights = torch.tensor([item['curriculum_weight'] for item in batch])
    
    gt_graphs = [item['gt_graph'] for item in batch]
    degraded_graphs = [item['degraded_graph'] for item in batch]
    correspondences = [item['correspondences'] for item in batch]
    degradation_metadata = [item['degradation_metadata'] for item in batch]
    
    # Batch statistics
    batch_stats = {
        'batch_size': len(batch),
        'sample_ids': sample_ids,
        'patient_ids': patient_ids,
        'curriculum_levels': curriculum_levels,
        'curriculum_weights': curriculum_weights,
        'level_distribution': {
            level: curriculum_levels.count(level) 
            for level in set(curriculum_levels)
        }
    }
    
    return {
        'gt_graphs': gt_graphs,
        'degraded_graphs': degraded_graphs,
        'correspondences': correspondences,
        'degradation_metadata': degradation_metadata,
        'batch_stats': batch_stats
    }


class CurriculumDataManager:
    """
    Manages curriculum learning progression and data loading
    """
    
    def __init__(self, 
                 train_dataset_dir: Path,
                 val_dataset_dir: Path,
                 config: Optional[DataLoaderConfig] = None):
        """
        Initialize curriculum manager
        
        Args:
            train_dataset_dir: Directory containing training datasets
            val_dataset_dir: Directory containing validation datasets
            config: Data loader configuration
        """
        self.train_dataset_dir = Path(train_dataset_dir)
        self.val_dataset_dir = Path(val_dataset_dir)
        self.config = config or DataLoaderConfig()
        self.logger = logging.getLogger(__name__)
        
        # Curriculum progression
        self.curriculum_stages = ['easy', 'medium', 'hard', 'expert']
        self.current_stage = 0
        
        # Track curriculum progress
        self.stage_epochs = {}
        self.stage_performance = {}
        
    def get_current_stage(self) -> str:
        """Get current curriculum stage"""
        return self.curriculum_stages[min(self.current_stage, len(self.curriculum_stages) - 1)]
    
    def get_dataloaders_for_stage(self, stage: str) -> Tuple[DataLoader, DataLoader]:
        """
        Get train and validation data loaders for specific curriculum stage
        
        Args:
            stage: Curriculum stage ('easy', 'medium', 'hard', 'expert')
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Load datasets for stage - check for .gz versions first
        train_dataset_path = self.train_dataset_dir / f"train_dataset.pkl.gz"
        val_dataset_path = self.val_dataset_dir / f"val_dataset.pkl.gz"
        
        # Fallback to uncompressed versions if .gz don't exist
        if not train_dataset_path.exists():
            train_dataset_path = self.train_dataset_dir / f"train_dataset.pkl"
        if not val_dataset_path.exists():
            val_dataset_path = self.val_dataset_dir / f"val_dataset.pkl"
        
        # Check if stage-specific datasets exist (with .gz priority)
        stage_train_path_gz = self.train_dataset_dir / f"train_{stage}_dataset.pkl.gz"
        stage_val_path_gz = self.val_dataset_dir / f"val_{stage}_dataset.pkl.gz"
        stage_train_path = self.train_dataset_dir / f"train_{stage}_dataset.pkl"
        stage_val_path = self.val_dataset_dir / f"val_{stage}_dataset.pkl"
        
        if stage_train_path_gz.exists():
            train_dataset_path = stage_train_path_gz
        elif stage_train_path.exists():
            train_dataset_path = stage_train_path
            
        if stage_val_path_gz.exists():
            val_dataset_path = stage_val_path_gz
        elif stage_val_path.exists():
            val_dataset_path = stage_val_path
        
        # Create datasets with mixed real+synthetic support
        train_dataset = self._create_mixed_dataset(
            stage=stage,
            split="train",
            synthetic_path=train_dataset_path
        )
        
        val_dataset = self._create_mixed_dataset(
            stage=stage,
            split="val", 
            synthetic_path=val_dataset_path
        )
        
        # Create samplers
        train_sampler = train_dataset.get_weighted_sampler() if self.config.curriculum_enabled else None
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=(train_sampler is None) and self.config.shuffle,
            sampler=train_sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=self.config.drop_last,
            collate_fn=collate_graph_correction_batch
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=False,
            collate_fn=collate_graph_correction_batch
        )
        
        self.logger.info(f"Created data loaders for {stage} stage:")
        self.logger.info(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
        self.logger.info(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
        
        # Log level distribution
        train_dist = train_dataset.get_level_distribution()
        val_dist = val_dataset.get_level_distribution()
        self.logger.info(f"  Train distribution: {train_dist}")
        self.logger.info(f"  Val distribution: {val_dist}")
        
        return train_loader, val_loader
    
    def _create_mixed_dataset(self, stage: str, split: str, synthetic_path: Path):
        """Create mixed real+synthetic dataset based on curriculum stage"""
        
        # Define mixing ratios for each stage
        mixing_ratios = {
            "easy": {"real": 0.0, "synthetic": 1.0},      # 100% synthetic
            "medium": {"real": 0.5, "synthetic": 0.5},    # 50% real, 50% synthetic  
            "hard": {"real": 0.7, "synthetic": 0.3},      # 70% real, 30% synthetic
            "expert": {"real": 0.8, "synthetic": 0.2},    # 80% real, 20% synthetic
            "real": {"real": 1.0, "synthetic": 0.0}       # 100% real
        }
        
        # Get mixing ratio for current stage
        ratio = mixing_ratios.get(stage, {"real": 0.0, "synthetic": 1.0})
        
        # If no real data needed, use standard synthetic dataset
        if ratio["real"] == 0.0:
            return GraphCorrectionDataset(
                dataset_path=synthetic_path,
                curriculum_level=stage,
                config=self.config
            )
        
        # Load synthetic data
        synthetic_samples = []
        if ratio["synthetic"] > 0.0 and synthetic_path.exists():
            synthetic_dataset = GraphCorrectionDataset(
                dataset_path=synthetic_path,
                curriculum_level=stage, 
                config=self.config
            )
            synthetic_samples = synthetic_dataset.samples
        
        # Load real data
        real_samples = []
        real_path = self.train_dataset_dir / f"{split}_real_dataset.pkl"
        if not real_path.exists():
            real_path = self.train_dataset_dir / f"{split}_real_dataset.pkl.gz"
            
        if ratio["real"] > 0.0 and real_path.exists():
            real_dataset = GraphCorrectionDataset(
                dataset_path=real_path,
                curriculum_level="real",
                config=self.config
            )
            real_samples = real_dataset.samples
        
        # Mix samples according to ratios
        mixed_samples = []
        
        # Calculate target total based on available samples
        max_synthetic = len(synthetic_samples) if synthetic_samples else 0
        max_real = len(real_samples) if real_samples else 0
        
        if ratio["real"] > 0 and ratio["synthetic"] > 0:
            # Calculate how many samples we can use while maintaining ratio
            # If we want 70% real, 30% synthetic:
            # total = real/0.7 or total = synthetic/0.3, whichever is smaller
            total_from_real = int(max_real / ratio["real"]) if ratio["real"] > 0 else float('inf')
            total_from_synthetic = int(max_synthetic / ratio["synthetic"]) if ratio["synthetic"] > 0 else float('inf')
            target_total = min(total_from_real, total_from_synthetic)
            
            n_real = int(target_total * ratio["real"])
            n_synthetic = int(target_total * ratio["synthetic"])
        else:
            # If only one type is needed
            n_real = max_real if ratio["real"] > 0 else 0
            n_synthetic = max_synthetic if ratio["synthetic"] > 0 else 0
        
        # Add samples
        if real_samples and n_real > 0:
            mixed_samples.extend(real_samples[:n_real])
        if synthetic_samples and n_synthetic > 0:
            mixed_samples.extend(synthetic_samples[:n_synthetic])
        
        # Create mixed dataset
        mixed_dataset = GraphCorrectionDataset.__new__(GraphCorrectionDataset)
        mixed_dataset.dataset_path = synthetic_path
        mixed_dataset.curriculum_level = stage
        mixed_dataset.config = self.config
        mixed_dataset.logger = logging.getLogger(__name__)
        mixed_dataset.samples = mixed_samples
        mixed_dataset.graph_cache = {} if self.config.cache_graphs else None
        mixed_dataset.curriculum_weights = mixed_dataset._compute_curriculum_weights()
        
        self.logger.info(f"Created mixed {split} dataset for {stage}: {len(mixed_samples)} samples "
                        f"({len([s for s in mixed_samples if s.get('curriculum_level') == 'real'])} real, "
                        f"{len([s for s in mixed_samples if s.get('curriculum_level') != 'real'])} synthetic)")
        
        return mixed_dataset
    
    def should_advance_curriculum(self, 
                                epoch: int, 
                                performance_metric: float,
                                min_epochs: int = 10,
                                performance_threshold: float = 0.8) -> bool:
        """
        Determine if curriculum should advance to next stage
        
        Args:
            epoch: Current epoch number
            performance_metric: Current stage performance (e.g., validation accuracy)
            min_epochs: Minimum epochs before advancing
            performance_threshold: Performance threshold for advancement
            
        Returns:
            Whether to advance to next stage
        """
        current_stage = self.get_current_stage()
        
        # Track epochs for current stage
        if current_stage not in self.stage_epochs:
            self.stage_epochs[current_stage] = 0
        self.stage_epochs[current_stage] += 1
        
        # Track performance
        if current_stage not in self.stage_performance:
            self.stage_performance[current_stage] = []
        self.stage_performance[current_stage].append(performance_metric)
        
        # Check advancement criteria
        epochs_in_stage = self.stage_epochs[current_stage]
        
        if epochs_in_stage < min_epochs:
            return False
        
        # Check if performance is stable and above threshold
        if epochs_in_stage >= 5:  # Need some history
            recent_performance = self.stage_performance[current_stage][-5:]
            avg_recent = np.mean(recent_performance)
            
            if avg_recent >= performance_threshold:
                return True
        
        return False
    
    def advance_curriculum(self) -> bool:
        """
        Advance to next curriculum stage
        
        Returns:
            Whether advancement was successful
        """
        if self.current_stage < len(self.curriculum_stages) - 1:
            old_stage = self.get_current_stage()
            self.current_stage += 1
            new_stage = self.get_current_stage()
            
            self.logger.info(f"Curriculum advanced: {old_stage} → {new_stage}")
            return True
        else:
            self.logger.info("Curriculum at final stage (expert)")
            return False
    
    def get_curriculum_summary(self) -> Dict:
        """Get summary of curriculum progression"""
        
        return {
            'current_stage': self.get_current_stage(),
            'stage_index': self.current_stage,
            'total_stages': len(self.curriculum_stages),
            'stage_epochs': self.stage_epochs,
            'stage_performance': {
                stage: {
                    'avg': np.mean(perf) if perf else 0,
                    'best': np.max(perf) if perf else 0,
                    'latest': perf[-1] if perf else 0
                }
                for stage, perf in self.stage_performance.items()
            }
        }


def create_curriculum_data_manager(train_dir: Path, 
                                 val_dir: Path,
                                 config: Optional[Dict] = None) -> CurriculumDataManager:
    """
    Factory function to create curriculum data manager
    
    Args:
        train_dir: Training dataset directory
        val_dir: Validation dataset directory
        config: Configuration dictionary
        
    Returns:
        CurriculumDataManager instance
    """
    if config:
        dataloader_config = DataLoaderConfig(**config)
    else:
        dataloader_config = DataLoaderConfig()
    
    return CurriculumDataManager(train_dir, val_dir, dataloader_config)