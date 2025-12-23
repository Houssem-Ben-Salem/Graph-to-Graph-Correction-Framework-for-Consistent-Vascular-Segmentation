"""
Regression-based Graph Correction Model
Directly predicts position corrections instead of classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import numpy as np

class GraphCorrectionRegressionModel(nn.Module):
    """
    Predicts position corrections directly, derives classes from magnitude
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Feature dimensions
        self.node_feat_dim = 7  # position(3) + radius(1) + confidence(1) + feat1(1) + feat2(1)
        
        # Handle both dict and object configs
        if isinstance(config, dict):
            hidden_dim = config['hidden_dim']
            num_heads = config['num_heads']
            num_layers = config['num_layers']
            self.dropout = config['dropout']
        else:
            hidden_dim = config.hidden_dim
            num_heads = config.num_heads
            num_layers = config.num_layers
            self.dropout = config.dropout
        
        # Multi-head GAT layers
        self.gat_layers = nn.ModuleList()
        
        # First layer
        self.gat_layers.append(
            GATConv(self.node_feat_dim, hidden_dim, heads=num_heads, 
                   dropout=self.dropout, concat=True)
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads,
                       dropout=self.dropout, concat=True)
            )
        
        # Final GAT layer
        self.gat_layers.append(
            GATConv(hidden_dim * num_heads, hidden_dim, heads=1,
                   dropout=self.dropout, concat=False)
        )
        
        # Regression head for position corrections
        self.position_correction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim // 2, 3)  # 3D position correction
        )
        
        # Optional: confidence head to predict reliability of correction
        self.correction_confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Single threshold for binary classification
        # Modify < 2.5, Remove >= 2.5 (from statistical analysis)
        self.register_buffer('magnitude_threshold', 
                           torch.tensor(2.5, dtype=torch.float32))
        
    def forward(self, data):
        """
        Forward pass
        Args:
            data: PyG Data object with node features and edge indices
        """
        x = data.x
        edge_index = data.edge_index
        
        # Extract current positions for skip connection
        current_positions = x[:, :3]
        
        # GAT feature extraction
        for i, layer in enumerate(self.gat_layers):
            x = layer(x, edge_index)
            if i < len(self.gat_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Predict position corrections (deltas)
        position_corrections = self.position_correction_head(x)
        
        # Predict correction confidence
        correction_confidence = self.correction_confidence_head(x)
        
        # Calculate correction magnitudes
        correction_magnitudes = torch.norm(position_corrections, dim=1)
        
        # Derive node operations from magnitudes
        node_operations = self.magnitude_to_class(correction_magnitudes)
        
        # Predicted positions = current + corrections
        predicted_positions = current_positions + position_corrections
        
        return {
            'position_corrections': position_corrections,
            'correction_magnitudes': correction_magnitudes,
            'correction_confidence': correction_confidence,
            'predicted_positions': predicted_positions,
            'node_operations': node_operations,  # For compatibility
            'node_features': x
        }
    
    def magnitude_to_class(self, magnitudes):
        """
        Convert correction magnitudes to binary classes
        """
        # Binary classification: 0=Modify, 1=Remove
        classes = (magnitudes >= self.magnitude_threshold).long()
        return classes
    
    def compute_targets(self, pred_features, gt_features, correspondences):
        """
        Compute regression targets from ground truth
        """
        batch_size = pred_features['batch'].max().item() + 1 if 'batch' in pred_features else 1
        device = pred_features['node_positions'].device
        
        num_pred_nodes = pred_features['node_positions'].shape[0]
        
        # Initialize targets
        position_correction_targets = torch.zeros(num_pred_nodes, 3, device=device)
        has_correspondence = torch.zeros(num_pred_nodes, dtype=torch.bool, device=device)
        
        # For each correspondence, compute position correction
        for pred_idx, gt_idx in correspondences.node_correspondences.items():
            pred_pos = pred_features['node_positions'][pred_idx]
            gt_pos = gt_features['node_positions'][gt_idx]
            
            # Position correction = GT position - Predicted position
            position_correction_targets[pred_idx] = gt_pos - pred_pos
            has_correspondence[pred_idx] = True
        
        # For unmatched nodes, target is to remove (large correction)
        # We don't have GT position, so we use a large magnitude in current direction
        unmatched_mask = ~has_correspondence
        if unmatched_mask.any():
            # Set large corrections for removal
            removal_magnitude = 5.0  # > 2.5 threshold
            current_positions = pred_features['node_positions'][unmatched_mask]
            
            # Random direction for removal (or could use distance from center)
            random_directions = torch.randn_like(current_positions)
            random_directions = F.normalize(random_directions, dim=1)
            
            position_correction_targets[unmatched_mask] = random_directions * removal_magnitude
        
        # Compute derived targets
        correction_magnitudes = torch.norm(position_correction_targets, dim=1)
        node_operations = self.magnitude_to_class(correction_magnitudes)
        
        return {
            'position_corrections': position_correction_targets,
            'correction_magnitudes': correction_magnitudes,
            'node_operations': node_operations,
            'has_correspondence': has_correspondence
        }
    
    def compute_loss(self, predictions, targets, loss_config):
        """
        Compute regression loss
        """
        losses = {}
        
        # Main loss: L2 regression loss on position corrections
        position_loss = F.mse_loss(
            predictions['position_corrections'],
            targets['position_corrections']
        )
        losses['position'] = position_loss
        
        # Magnitude loss (auxiliary, helps training)
        magnitude_loss = F.mse_loss(
            predictions['correction_magnitudes'],
            targets['correction_magnitudes']
        )
        losses['magnitude'] = magnitude_loss * 0.5
        
        # Optional: Binary classification loss for compatibility/comparison
        if loss_config.get('use_classification_loss', False):
            # Binary classification: Modify ~80%, Remove ~20%
            class_weights = torch.tensor([1.25, 4.0], device=position_loss.device)  # Balanced weights
            
            classification_loss = F.cross_entropy(
                predictions['node_operations'],
                targets['node_operations'],
                weight=class_weights
            )
            losses['classification'] = classification_loss * 0.1
        
        # Confidence regularization (prefer high confidence)
        confidence_reg = -predictions['correction_confidence'].mean() * 0.01
        losses['confidence_reg'] = confidence_reg
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total'] = total_loss
        
        return losses
    
    def update_thresholds_from_data(self, train_loader):
        """
        Update magnitude thresholds based on training data distribution
        """
        all_magnitudes = []
        all_labels = []
        
        self.eval()
        with torch.no_grad():
            for batch in train_loader:
                # Get predictions
                outputs = self(batch)
                
                # Get true labels from targets
                targets = self.compute_targets(
                    batch.pred_features,
                    batch.gt_features,
                    batch.correspondences
                )
                
                all_magnitudes.append(targets['correction_magnitudes'].cpu())
                all_labels.append(targets['node_operations'].cpu())
        
        # Concatenate all
        all_magnitudes = torch.cat(all_magnitudes)
        all_labels = torch.cat(all_labels)
        
        # Find optimal thresholds
        # Between Keep (0) and Modify (2)
        keep_mags = all_magnitudes[all_labels == 0]
        modify_mags = all_magnitudes[all_labels == 2]
        
        if len(keep_mags) > 0 and len(modify_mags) > 0:
            threshold1 = (keep_mags.max() + modify_mags.min()) / 2
        else:
            threshold1 = 0.5
        
        # Between Modify (2) and Remove (1)
        remove_mags = all_magnitudes[all_labels == 1]
        
        if len(modify_mags) > 0 and len(remove_mags) > 0:
            threshold2 = (modify_mags.max() + remove_mags.min()) / 2
        else:
            threshold2 = 2.5
        
        print(f"Updated thresholds: Keep/Modify={threshold1:.3f}, Modify/Remove={threshold2:.3f}")
        self.magnitude_thresholds = torch.tensor([threshold1, threshold2])