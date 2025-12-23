"""Main graph correction network for enhancing vascular graphs"""

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from typing import Dict, Tuple

from .topology_corrector import TopologyCorrector
from .anatomy_preserver import AnatomyPreserver

# Legacy aliases for backward compatibility
TopologyCorrectionModule = TopologyCorrector
AnatomyCorrectionModule = AnatomyPreserver
ConsistencyEnforcer = TopologyCorrector  # Simplified mapping


class GraphCorrector(nn.Module):
    """Hierarchical graph correction network"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Input dimensions
        self.node_input_dim = 7  # x, y, z, radius, curvature, node_type, confidence
        self.edge_input_dim = 4  # length, avg_radius, curvature, confidence
        self.hidden_dim = config.get('hidden_dim', 256)
        
        # Initial feature projection
        self.node_encoder = nn.Sequential(
            nn.Linear(self.node_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim)
        )
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(self.edge_input_dim, self.hidden_dim // 4),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim // 4)
        )
        
        # Core correction modules
        self.topology_corrector = TopologyCorrector(
            self.hidden_dim, 
            config.get('topology_corrector', {})
        )
        
        self.anatomy_corrector = AnatomyPreserver(
            self.hidden_dim,
            config.get('anatomy_corrector', {})
        )
        
        self.consistency_enforcer = TopologyCorrector(
            self.hidden_dim,
            config.get('consistency_enforcer', {})
        )
        
        # Output heads for different correction types
        self.node_operation_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 3)  # keep, remove, modify
        )
        
        self.node_correction_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 4)  # dx, dy, dz, d_radius
        )
        
        self.edge_operation_head = nn.Sequential(
            nn.Linear(self.hidden_dim // 4, 3)  # keep, remove, add
        )
        
    def forward(self, pred_graph_data, gt_graph_data=None, correspondences=None):
        """
        Forward pass for graph correction
        
        Args:
            pred_graph_data: PyTorch Geometric Data object for predicted graph
            gt_graph_data: Optional ground truth graph for training
            correspondences: Optional node correspondences for training
            
        Returns:
            Dictionary containing correction predictions
        """
        # Encode input features
        node_features = self.node_encoder(pred_graph_data.x)
        edge_features = self.edge_encoder(pred_graph_data.edge_attr)
        
        # Apply topology correction
        node_features, edge_features, topology_outputs = self.topology_corrector(
            node_features, pred_graph_data.edge_index, edge_features,
            pred_graph_data.batch if hasattr(pred_graph_data, 'batch') else None
        )
        
        # Apply anatomy correction
        node_features, anatomy_outputs = self.anatomy_corrector(
            node_features, pred_graph_data.edge_index, edge_features,
            pred_graph_data.batch if hasattr(pred_graph_data, 'batch') else None
        )
        
        # Enforce global consistency
        consistency_outputs = self.consistency_enforcer(
            node_features, pred_graph_data.edge_index,
            pred_graph_data.batch if hasattr(pred_graph_data, 'batch') else None
        )
        
        # Generate correction predictions
        node_operations = self.node_operation_head(node_features)
        node_corrections = self.node_correction_head(node_features)
        edge_operations = self.edge_operation_head(edge_features)
        
        outputs = {
            'node_operations': node_operations,  # Classification logits
            'node_corrections': node_corrections,  # Regression values
            'edge_operations': edge_operations,  # Classification logits
            'topology_outputs': topology_outputs,
            'anatomy_outputs': anatomy_outputs,
            'consistency_outputs': consistency_outputs,
            'corrected_node_features': node_features,
            'corrected_edge_features': edge_features
        }
        
        # If training, compute additional supervision signals
        if gt_graph_data is not None and correspondences is not None:
            outputs['training_signals'] = self._compute_training_signals(
                pred_graph_data, gt_graph_data, correspondences, outputs
            )
        
        return outputs
    
    def _compute_training_signals(self, pred_graph, gt_graph, correspondences, outputs):
        """Compute additional training signals based on ground truth"""
        signals = {}
        
        # Node operation targets
        node_op_targets = torch.zeros(pred_graph.x.size(0), dtype=torch.long)
        for pred_idx, (gt_idx, confidence) in correspondences['node_correspondences'].items():
            if confidence > 0.7:
                node_op_targets[pred_idx] = 0  # keep
            else:
                node_op_targets[pred_idx] = 3  # modify
        
        # Mark nodes for removal if no correspondence
        for i in range(pred_graph.x.size(0)):
            if i not in correspondences['node_correspondences']:
                node_op_targets[i] = 1  # remove
        
        signals['node_op_targets'] = node_op_targets
        
        # Node correction targets (for matched nodes)
        node_correction_targets = torch.zeros(pred_graph.x.size(0), 4)
        for pred_idx, (gt_idx, confidence) in correspondences['node_correspondences'].items():
            if confidence > 0.5:
                # Compute position and radius corrections
                pred_pos = pred_graph.x[pred_idx, :3]
                gt_pos = gt_graph.x[gt_idx, :3]
                pred_radius = pred_graph.x[pred_idx, 3]
                gt_radius = gt_graph.x[gt_idx, 3]
                
                node_correction_targets[pred_idx, :3] = gt_pos - pred_pos
                node_correction_targets[pred_idx, 3] = gt_radius - pred_radius
        
        signals['node_correction_targets'] = node_correction_targets
        
        return signals
    
    def apply_corrections(self, graph_data, corrections):
        """Apply predicted corrections to generate corrected graph"""
        # This would be implemented to actually modify the graph structure
        # based on the predicted operations and corrections
        pass