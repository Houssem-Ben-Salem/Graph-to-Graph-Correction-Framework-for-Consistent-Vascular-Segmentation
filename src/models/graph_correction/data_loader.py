"""
Data Loader for Graph Correction Training
Handles loading and batching of graph correspondence pairs
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import pickle
from pathlib import Path
import numpy as np
import logging

from src.models.graph_extraction.vascular_graph import VascularGraph
from src.utils.graph_correspondence import CorrespondenceResult


class GraphCorrespondenceDataset(Dataset):
    """
    Dataset for graph correction training using correspondence pairs
    """
    
    def __init__(self, 
                 extracted_graphs_dir: Path,
                 correspondences_dir: Optional[Path] = None,
                 split: str = 'train',
                 synthetic_degradation: bool = True,
                 degradation_levels: List[float] = [0.2, 0.3, 0.4]):
        """
        Initialize dataset
        
        Args:
            extracted_graphs_dir: Directory containing extracted graphs
            correspondences_dir: Directory containing pre-computed correspondences
            split: Dataset split ('train', 'val', 'test')
            synthetic_degradation: Whether to create synthetic predictions
            degradation_levels: Levels of synthetic degradation to apply
        """
        self.extracted_graphs_dir = Path(extracted_graphs_dir)
        self.correspondences_dir = Path(correspondences_dir) if correspondences_dir else None
        self.split = split
        self.synthetic_degradation = synthetic_degradation
        self.degradation_levels = degradation_levels
        
        self.logger = logging.getLogger(__name__)
        
        # Find available samples
        self.samples = self._find_samples()
        
        # Split data
        self.samples = self._split_data(self.samples, split)
        
        self.logger.info(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _find_samples(self) -> List[str]:
        """Find available patient samples"""
        samples = []
        
        for patient_dir in self.extracted_graphs_dir.iterdir():
            if patient_dir.is_dir() and patient_dir.name.startswith('PA'):
                gt_file = patient_dir / f"{patient_dir.name}_GT.pkl"
                if gt_file.exists():
                    samples.append(patient_dir.name)
        
        return sorted(samples)
    
    def _split_data(self, samples: List[str], split: str) -> List[str]:
        """Split data into train/val/test"""
        num_samples = len(samples)
        
        if split == 'train':
            return samples[:int(0.7 * num_samples)]
        elif split == 'val':
            return samples[int(0.7 * num_samples):int(0.85 * num_samples)]
        elif split == 'test':
            return samples[int(0.85 * num_samples):]
        else:
            return samples
    
    def __len__(self) -> int:
        """Dataset length"""
        if self.synthetic_degradation:
            return len(self.samples) * len(self.degradation_levels)
        else:
            return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single training sample"""
        
        if self.synthetic_degradation:
            # Multiple samples per patient (different degradation levels)
            sample_idx = idx // len(self.degradation_levels)
            degradation_idx = idx % len(self.degradation_levels)
            sample_id = self.samples[sample_idx]
            degradation_level = self.degradation_levels[degradation_idx]
        else:
            sample_id = self.samples[idx]
            degradation_level = None
        
        try:
            # Load ground truth graph
            gt_file = self.extracted_graphs_dir / sample_id / f"{sample_id}_GT.pkl"
            gt_graph = VascularGraph.load(gt_file)
            
            # Load or create prediction graph
            if self.synthetic_degradation:
                pred_graph = self._create_synthetic_prediction(gt_graph, degradation_level)
                correspondences = self._compute_correspondences(pred_graph, gt_graph)
            else:
                # Try to load existing prediction
                pred_file = self.extracted_graphs_dir / sample_id / f"{sample_id}_PRED.pkl"
                if pred_file.exists():
                    pred_graph = VascularGraph.load(pred_file)
                else:
                    # Fallback to synthetic
                    pred_graph = self._create_synthetic_prediction(gt_graph, 0.3)
                
                # Load or compute correspondences
                correspondences = self._load_or_compute_correspondences(
                    sample_id, pred_graph, gt_graph
                )
            
            # Convert to training format
            training_data = self._prepare_training_data(
                pred_graph, gt_graph, correspondences, sample_id
            )
            
            return training_data
            
        except Exception as e:
            self.logger.error(f"Error loading sample {sample_id}: {e}")
            # Return dummy data to avoid breaking the batch
            return self._get_dummy_sample()
    
    def _create_synthetic_prediction(self, gt_graph: VascularGraph, degradation_level: float) -> VascularGraph:
        """Create synthetic prediction by degrading ground truth"""
        
        # Copy nodes with degradation
        pred_nodes = []
        for i, node in enumerate(gt_graph.nodes):
            pred_node = node.copy()
            
            # Add position noise
            if 'position' in node:
                position = np.array(node['position'])
                noise = np.random.normal(0, degradation_level, 3)
                pred_node['position'] = (position + noise).tolist()
            
            # Add radius noise
            if 'radius_voxels' in node:
                radius = node['radius_voxels']
                noise_factor = 1 + np.random.normal(0, degradation_level * 0.5)
                pred_node['radius_voxels'] = max(0.1, radius * noise_factor)
            
            # Occasionally change node type
            if np.random.random() < degradation_level * 0.2:
                original_type = node.get('type', 'regular')
                if original_type == 'bifurcation':
                    pred_node['type'] = 'regular'
                elif original_type == 'regular' and np.random.random() < 0.1:
                    pred_node['type'] = 'bifurcation'
            
            pred_nodes.append(pred_node)
        
        # Copy edges with some removal
        pred_edges = []
        for edge in gt_graph.edges:
            if np.random.random() > degradation_level * 0.15:  # Keep most edges
                pred_edge = edge.copy()
                
                # Add length noise
                if 'euclidean_length' in edge:
                    length = edge['euclidean_length']
                    noise_factor = 1 + np.random.normal(0, degradation_level * 0.3)
                    pred_edge['euclidean_length'] = max(0.1, length * noise_factor)
                
                pred_edges.append(pred_edge)
        
        # Create prediction graph
        pred_graph = VascularGraph(
            nodes=pred_nodes,
            edges=pred_edges,
            global_properties=gt_graph.global_properties.copy(),
            metadata={
                **gt_graph.metadata,
                'synthetic_prediction': True,
                'degradation_level': degradation_level
            }
        )
        
        return pred_graph
    
    def _compute_correspondences(self, pred_graph: VascularGraph, gt_graph: VascularGraph) -> CorrespondenceResult:
        """Compute correspondences between prediction and ground truth"""
        from src.utils.graph_correspondence import create_correspondence_matcher
        
        matcher = create_correspondence_matcher()
        correspondences = matcher.find_correspondence(pred_graph, gt_graph)
        
        return correspondences
    
    def _load_or_compute_correspondences(self, sample_id: str, pred_graph: VascularGraph, gt_graph: VascularGraph) -> CorrespondenceResult:
        """Load existing correspondences or compute new ones"""
        
        if self.correspondences_dir:
            correspondence_file = self.correspondences_dir / f"{sample_id}_correspondence.pkl"
            if correspondence_file.exists():
                try:
                    with open(correspondence_file, 'rb') as f:
                        correspondences = pickle.load(f)
                    return correspondences
                except Exception as e:
                    self.logger.warning(f"Failed to load correspondences for {sample_id}: {e}")
        
        # Compute correspondences
        return self._compute_correspondences(pred_graph, gt_graph)
    
    def _prepare_training_data(self, 
                             pred_graph: VascularGraph, 
                             gt_graph: VascularGraph,
                             correspondences: CorrespondenceResult,
                             sample_id: str) -> Dict:
        """Prepare data for training"""
        
        training_data = {
            'sample_id': sample_id,
            'pred_graph': pred_graph,
            'gt_graph': gt_graph,
            'correspondences': correspondences,
            
            # Pre-computed features for efficiency
            'pred_num_nodes': len(pred_graph.nodes),
            'pred_num_edges': len(pred_graph.edges),
            'gt_num_nodes': len(gt_graph.nodes),
            'gt_num_edges': len(gt_graph.edges),
            
            # Correspondence statistics
            'num_node_correspondences': len(correspondences.node_correspondences),
            'num_edge_correspondences': len(correspondences.edge_correspondences),
            'correspondence_quality': correspondences.correspondence_quality.get('overall_quality', 0.5)
        }
        
        return training_data
    
    def _get_dummy_sample(self) -> Dict:
        """Create dummy sample for error handling"""
        # Create minimal dummy graphs
        dummy_nodes = [{'id': 0, 'position': [0, 0, 0], 'type': 'regular', 'radius_voxels': 1.0}]
        dummy_edges = []
        
        dummy_graph = VascularGraph(
            nodes=dummy_nodes,
            edges=dummy_edges,
            global_properties={'num_nodes': 1, 'num_edges': 0},
            metadata={'dummy': True}
        )
        
        # Create dummy correspondences
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
            'pred_graph': dummy_graph,
            'gt_graph': dummy_graph,
            'correspondences': dummy_correspondences,
            'pred_num_nodes': 1,
            'pred_num_edges': 0,
            'gt_num_nodes': 1,
            'gt_num_edges': 0,
            'num_node_correspondences': 1,
            'num_edge_correspondences': 0,
            'correspondence_quality': 1.0
        }


def collate_graph_data(batch: List[Dict]) -> Dict:
    """
    Custom collate function for batching graph data
    
    Args:
        batch: List of training samples
        
    Returns:
        Batched data dictionary
    """
    # Separate different components
    sample_ids = [item['sample_id'] for item in batch]
    pred_graphs = [item['pred_graph'] for item in batch]
    gt_graphs = [item['gt_graph'] for item in batch]
    correspondences = [item['correspondences'] for item in batch]
    
    # Statistics
    stats = {
        'sample_ids': sample_ids,
        'batch_size': len(batch),
        'pred_num_nodes': [item['pred_num_nodes'] for item in batch],
        'pred_num_edges': [item['pred_num_edges'] for item in batch],
        'gt_num_nodes': [item['gt_num_nodes'] for item in batch],
        'gt_num_edges': [item['gt_num_edges'] for item in batch],
        'num_node_correspondences': [item['num_node_correspondences'] for item in batch],
        'correspondence_quality': [item['correspondence_quality'] for item in batch]
    }
    
    return {
        'pred_graphs': pred_graphs,
        'gt_graphs': gt_graphs,
        'correspondences': correspondences,
        'stats': stats
    }


def create_data_loaders(extracted_graphs_dir: Path,
                       correspondences_dir: Optional[Path] = None,
                       batch_size: int = 4,
                       num_workers: int = 2,
                       synthetic_degradation: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders
    
    Args:
        extracted_graphs_dir: Directory with extracted graphs
        correspondences_dir: Directory with correspondences
        batch_size: Batch size
        num_workers: Number of worker processes
        synthetic_degradation: Use synthetic degradation
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    # Create datasets
    train_dataset = GraphCorrespondenceDataset(
        extracted_graphs_dir=extracted_graphs_dir,
        correspondences_dir=correspondences_dir,
        split='train',
        synthetic_degradation=synthetic_degradation
    )
    
    val_dataset = GraphCorrespondenceDataset(
        extracted_graphs_dir=extracted_graphs_dir,
        correspondences_dir=correspondences_dir,
        split='val',
        synthetic_degradation=synthetic_degradation,
        degradation_levels=[0.3]  # Single level for validation
    )
    
    test_dataset = GraphCorrespondenceDataset(
        extracted_graphs_dir=extracted_graphs_dir,
        correspondences_dir=correspondences_dir,
        split='test',
        synthetic_degradation=False  # Use real predictions if available
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_graph_data,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_graph_data,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Single sample for testing
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_graph_data,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader