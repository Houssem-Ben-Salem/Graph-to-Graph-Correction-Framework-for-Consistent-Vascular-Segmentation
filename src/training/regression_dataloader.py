"""
DataLoader for regression-based graph correction model
Converts VascularGraph to PyTorch Geometric Data objects
"""

import torch
from torch_geometric.data import Data, Dataset
import numpy as np
from pathlib import Path
import pickle
import gzip
from typing import Dict, List, Optional
import logging


class RegressionGraphDataset(Dataset):
    """Dataset that converts VascularGraph to PyG Data for regression model"""
    
    def __init__(self, dataset_path: Path):
        super().__init__()
        self.dataset_path = Path(dataset_path)
        self.logger = logging.getLogger(__name__)
        
        # Load samples
        self.samples = self._load_dataset()
        self.logger.info(f"Loaded {len(self.samples)} samples")
        
    def _load_dataset(self) -> List[Dict]:
        """Load dataset from file"""
        if str(self.dataset_path).endswith('.gz'):
            with gzip.open(self.dataset_path, 'rb') as f:
                samples = pickle.load(f)
        else:
            with open(self.dataset_path, 'rb') as f:
                samples = pickle.load(f)
        return samples
    
    def len(self):
        return len(self.samples)
    
    def get(self, idx):
        """Get item and convert to PyG Data"""
        sample = self.samples[idx]
        
        # Convert to PyG Data
        data = self._convert_to_pyg_data(
            sample['degraded_graph'],
            sample['gt_graph'],
            sample['correspondences']
        )
        
        # Add metadata
        data.sample_id = sample['sample_id']
        data.patient_id = sample['patient_id']
        
        return data
    
    def _convert_to_pyg_data(self, pred_graph, gt_graph, correspondences):
        """Convert VascularGraph to PyTorch Geometric Data object"""
        
        # Extract predicted graph features
        pred_nodes = pred_graph.nodes
        num_pred_nodes = len(pred_nodes)
        
        # Node features: position(3) + radius(1) + confidence(1) + placeholder(2)
        pred_node_features = []
        pred_positions = []
        
        for node in pred_nodes:
            pos = np.array(node['position'], dtype=np.float32)
            radius = float(node.get('radius', 1.0))
            confidence = float(correspondences.node_confidences.get(node['id'], 0.5))
            
            # Additional features (can add more graph features later)
            feat1 = 0.0  # Placeholder for node degree
            feat2 = 0.0  # Placeholder for centrality
            
            features = np.concatenate([pos, [radius, confidence, feat1, feat2]])
            pred_node_features.append(features)
            pred_positions.append(pos)
        
        # Convert to tensors
        x = torch.tensor(np.array(pred_node_features), dtype=torch.float32)
        pos = torch.tensor(np.array(pred_positions), dtype=torch.float32)
        
        # Extract edges
        edge_list = []
        for edge in pred_graph.edges:
            # Handle different edge formats
            if isinstance(edge, dict):
                if 'start_node' in edge and 'end_node' in edge:
                    start, end = edge['start_node'], edge['end_node']
                elif 'source' in edge and 'target' in edge:
                    start, end = edge['source'], edge['target']
                else:
                    # Try to get node IDs from the edge keys
                    edge_keys = list(edge.keys())
                    if len(edge_keys) >= 2:
                        start, end = edge[edge_keys[0]], edge[edge_keys[1]]
                    else:
                        continue
            elif hasattr(edge, 'start_node') and hasattr(edge, 'end_node'):
                start, end = edge.start_node, edge.end_node
            else:
                # Skip edges we can't parse
                continue
                
            edge_list.append([start, end])
            edge_list.append([end, start])  # Undirected
        
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        # Create PyG Data object
        data = Data(x=x, edge_index=edge_index, pos=pos)
        
        # Add ground truth information for loss computation
        data.pred_features = {
            'node_positions': pos,
            'node_radii': torch.tensor([n.get('radius', 1.0) for n in pred_nodes], dtype=torch.float32),
            'num_nodes': num_pred_nodes
        }
        
        # Extract GT positions for matched nodes
        gt_positions = []
        gt_radii = []
        gt_node_map = {node['id']: node for node in gt_graph.nodes}
        
        for gt_node in gt_graph.nodes:
            pos = np.array(gt_node['position'], dtype=np.float32)
            radius = float(gt_node.get('radius', 1.0))
            gt_positions.append(pos)
            gt_radii.append(radius)
        
        data.gt_features = {
            'node_positions': torch.tensor(np.array(gt_positions), dtype=torch.float32),
            'node_radii': torch.tensor(np.array(gt_radii), dtype=torch.float32),
            'num_nodes': len(gt_graph.nodes)
        }
        
        # Store correspondences
        data.correspondences = correspondences
        
        return data


def create_regression_dataloader(dataset_path, batch_size=8, shuffle=True, num_workers=4):
    """Create a dataloader for regression training"""
    dataset = RegressionGraphDataset(dataset_path)
    
    from torch_geometric.loader import DataLoader
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    
    return loader