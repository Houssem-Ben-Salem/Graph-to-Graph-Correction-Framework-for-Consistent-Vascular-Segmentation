"""
Enhanced Regression-based Graph Correction Model
Addresses loss plateau and improves accuracy with:
1. Residual connections
2. Attention mechanisms
3. Better loss functions
4. Learning rate scheduling
5. Advanced regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, TransformerConv, BatchNorm, LayerNorm
import numpy as np
import math


class EnhancedGraphCorrectionModel(nn.Module):
    """
    Enhanced model with residual connections, better loss functions,
    and improved architecture to address training plateaus
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Feature dimensions
        self.node_feat_dim = 7  # position(3) + radius(1) + confidence(1) + feat1(1) + feat2(1)
        
        # Handle both dict and object configs
        if isinstance(config, dict):
            hidden_dim = config.get('hidden_dim', 256)  # Increased from 128
            num_heads = config.get('num_heads', 8)
            num_layers = config.get('num_layers', 6)  # Increased from 4
            self.dropout = config.get('dropout', 0.15)  # Slight increase
            self.use_residual = config.get('use_residual', True)
            self.use_layer_norm = config.get('use_layer_norm', True)
            self.use_transformer = config.get('use_transformer', True)
        else:
            hidden_dim = getattr(config, 'hidden_dim', 256)
            num_heads = getattr(config, 'num_heads', 8)
            num_layers = getattr(config, 'num_layers', 6)
            self.dropout = getattr(config, 'dropout', 0.15)
            self.use_residual = getattr(config, 'use_residual', True)
            self.use_layer_norm = getattr(config, 'use_layer_norm', True)
            self.use_transformer = getattr(config, 'use_transformer', True)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection with better initialization
        self.input_projection = nn.Sequential(
            nn.Linear(self.node_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if self.use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Enhanced feature extraction layers
        self.feature_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if self.use_layer_norm else None
        self.residual_projections = nn.ModuleList() if self.use_residual else None
        
        for i in range(num_layers):
            # Mix of different graph convolution types
            if self.use_transformer and i % 2 == 0:
                # TransformerConv for better long-range dependencies
                layer = TransformerConv(
                    hidden_dim, hidden_dim, heads=num_heads, 
                    dropout=self.dropout, concat=False
                )
            else:
                # GAT for local attention
                layer = GATConv(
                    hidden_dim, hidden_dim, heads=num_heads,
                    dropout=self.dropout, concat=False, add_self_loops=True
                )
            
            self.feature_layers.append(layer)
            
            if self.use_layer_norm:
                self.layer_norms.append(LayerNorm(hidden_dim))
            
            if self.use_residual:
                # Residual projection in case dimensions change
                self.residual_projections.append(
                    nn.Linear(hidden_dim, hidden_dim) if i == 0 else nn.Identity()
                )
        
        # Multi-scale feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Concat last two layers
            nn.LayerNorm(hidden_dim) if self.use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Enhanced position correction head with skip connections
        self.position_correction_head = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),  # +3 for input position
            nn.LayerNorm(hidden_dim) if self.use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2) if self.use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 3),  # 3D position correction
            nn.Tanh()  # Bound corrections to [-1, 1] then scale
        )
        
        # Correction magnitude predictor (direct prediction)
        self.magnitude_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4) if self.use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softplus()  # Ensure positive magnitudes
        )
        
        # Enhanced confidence head (without Sigmoid for autocast safety)
        self.correction_confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4) if self.use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.ReLU(),
            nn.Linear(hidden_dim // 8, 1)
            # No Sigmoid - will use logits directly
        )
        
        # Classification head (for auxiliary loss)
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4) if self.use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim // 4, 2)  # Binary: Modify/Remove
        )
        
        # Learnable threshold (instead of fixed)
        self.magnitude_threshold = nn.Parameter(torch.tensor(2.5, dtype=torch.float32))
        
        # Scale factor for corrections (learnable)
        self.correction_scale = nn.Parameter(torch.tensor(5.0, dtype=torch.float32))
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Better weight initialization"""
        if isinstance(module, nn.Linear):
            # Xavier initialization with gain
            gain = nn.init.calculate_gain('relu')
            nn.init.xavier_uniform_(module.weight, gain=gain)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, data):
        """Enhanced forward pass with residual connections"""
        x = data.x
        edge_index = data.edge_index
        
        # Extract current positions for skip connection
        current_positions = x[:, :3]
        
        # Input projection
        x = self.input_projection(x)
        
        # Store features from each layer for multi-scale fusion
        layer_features = []
        
        # Enhanced feature extraction with residuals
        for i, layer in enumerate(self.feature_layers):
            # Store input for residual
            residual = x
            
            # Graph convolution
            x = layer(x, edge_index)
            
            # Layer normalization
            if self.use_layer_norm:
                x = self.layer_norms[i](x)
            
            # Residual connection
            if self.use_residual:
                if i == 0:
                    residual = self.residual_projections[i](residual)
                x = x + residual
            
            # Activation and dropout
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Store features from last two layers
            if i >= self.num_layers - 2:
                layer_features.append(x)
        
        # Multi-scale feature fusion
        if len(layer_features) >= 2:
            fused_features = torch.cat(layer_features[-2:], dim=1)
            x = self.feature_fusion(fused_features)
        
        # Position correction with input skip connection
        position_input = torch.cat([x, current_positions], dim=1)
        position_corrections = self.position_correction_head(position_input)
        
        # Scale corrections
        position_corrections = position_corrections * self.correction_scale
        
        # Direct magnitude prediction
        predicted_magnitudes = self.magnitude_head(x).squeeze(-1)
        
        # Confidence prediction (logits)
        confidence_logits = self.correction_confidence_head(x).squeeze(-1)
        correction_confidence = torch.sigmoid(confidence_logits)  # For output compatibility
        
        # Classification logits
        classification_logits = self.classification_head(x)
        
        # Derive classes from predicted magnitudes
        node_operations = (predicted_magnitudes >= self.magnitude_threshold).long()
        
        # Predicted positions
        predicted_positions = current_positions + position_corrections
        
        # Calculate actual correction magnitudes for comparison
        actual_magnitudes = torch.norm(position_corrections, dim=1)
        
        return {
            'position_corrections': position_corrections,
            'correction_magnitudes': actual_magnitudes,  # From corrections
            'predicted_magnitudes': predicted_magnitudes,  # Direct prediction
            'correction_confidence': correction_confidence,
            'confidence_logits': confidence_logits,  # For safe autocast loss
            'predicted_positions': predicted_positions,
            'node_operations': node_operations,
            'classification_logits': classification_logits,
            'node_features': x,
            'learnable_threshold': self.magnitude_threshold
        }
    
    def compute_targets(self, pred_features, gt_features, correspondences):
        """Enhanced target computation with better handling"""
        batch_size = pred_features['batch'].max().item() + 1 if 'batch' in pred_features else 1
        device = pred_features['node_positions'].device
        
        num_pred_nodes = pred_features['node_positions'].shape[0]
        
        # Initialize targets
        position_correction_targets = torch.zeros(num_pred_nodes, 3, device=device)
        magnitude_targets = torch.zeros(num_pred_nodes, device=device)
        confidence_targets = torch.ones(num_pred_nodes, device=device)  # Default high confidence
        classification_targets = torch.zeros(num_pred_nodes, dtype=torch.long, device=device)  # Default modify
        has_correspondence = torch.zeros(num_pred_nodes, dtype=torch.bool, device=device)
        
        # For matched nodes, compute corrections
        for pred_idx, gt_idx in correspondences.node_correspondences.items():
            pred_pos = pred_features['node_positions'][pred_idx]
            gt_pos = gt_features['node_positions'][gt_idx]
            
            # Position correction = GT position - Predicted position
            correction = gt_pos - pred_pos
            magnitude = torch.norm(correction).item()
            
            position_correction_targets[pred_idx] = correction
            magnitude_targets[pred_idx] = magnitude
            has_correspondence[pred_idx] = True
            
            # Classification target based on magnitude
            if magnitude >= 2.5:
                classification_targets[pred_idx] = 1  # Remove
                confidence_targets[pred_idx] = 0.8  # High confidence for removal
            else:
                classification_targets[pred_idx] = 0  # Modify
                confidence_targets[pred_idx] = max(0.3, 1.0 - magnitude / 5.0)  # Confidence based on magnitude
        
        # For unmatched nodes (should be removed)
        unmatched_mask = ~has_correspondence
        if unmatched_mask.any():
            # Large magnitude for removal
            removal_magnitude = 4.0
            current_positions = pred_features['node_positions'][unmatched_mask]
            
            # Direction away from center or based on current position
            directions = F.normalize(current_positions, dim=1)
            
            position_correction_targets[unmatched_mask] = directions * removal_magnitude
            magnitude_targets[unmatched_mask] = removal_magnitude
            classification_targets[unmatched_mask] = 1  # Remove
            confidence_targets[unmatched_mask] = 0.9  # High confidence for removal
        
        return {
            'position_corrections': position_correction_targets,
            'magnitude_targets': magnitude_targets,
            'confidence_targets': confidence_targets,
            'classification_targets': classification_targets,
            'has_correspondence': has_correspondence
        }
    
    def compute_enhanced_loss(self, predictions, targets, loss_config):
        """Enhanced loss function with multiple components"""
        losses = {}
        
        # 1. Position correction loss (with normalization)
        position_loss = F.smooth_l1_loss(  # More robust than MSE
            predictions['position_corrections'],
            targets['position_corrections']
        )
        losses['position'] = position_loss
        
        # 2. Magnitude prediction loss
        magnitude_loss = F.smooth_l1_loss(
            predictions['predicted_magnitudes'],
            targets['magnitude_targets']
        )
        losses['magnitude'] = magnitude_loss * loss_config.get('magnitude_weight', 0.5)
        
        # 3. Confidence loss (use autocast-safe version)
        confidence_loss = F.binary_cross_entropy_with_logits(
            predictions['confidence_logits'],
            targets['confidence_targets']
        )
        losses['confidence'] = confidence_loss * loss_config.get('confidence_weight', 0.3)
        
        # 4. Classification loss (auxiliary) with strong class weights
        # Heavily penalize modify class errors since model ignores minority class
        class_weights = torch.tensor([4.0, 1.0], device=predictions['classification_logits'].device)  
        classification_loss = F.cross_entropy(
            predictions['classification_logits'],
            targets['classification_targets'],
            weight=class_weights
        )
        losses['classification'] = classification_loss * loss_config.get('classification_weight', 0.2)
        
        # 5. Consistency loss (magnitude from corrections vs direct prediction)
        consistency_loss = F.mse_loss(
            predictions['correction_magnitudes'],
            predictions['predicted_magnitudes']
        )
        losses['consistency'] = consistency_loss * 0.1
        
        # 6. Threshold regularization (keep threshold reasonable)
        threshold_reg = F.relu(predictions['learnable_threshold'] - 5.0) + \
                      F.relu(1.0 - predictions['learnable_threshold'])
        losses['threshold_reg'] = threshold_reg * 0.01
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total'] = total_loss
        
        return losses
    
    def get_learning_rate_schedule(self, optimizer, num_epochs):
        """Create learning rate scheduler"""
        # Cosine annealing with warm restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=1e-6
        )
        return scheduler


def create_enhanced_model(config):
    """Factory function to create enhanced model"""
    return EnhancedGraphCorrectionModel(config)


if __name__ == "__main__":
    # Test model creation
    config = {
        'hidden_dim': 256,
        'num_heads': 8,
        'num_layers': 6,
        'dropout': 0.15,
        'use_residual': True,
        'use_layer_norm': True,
        'use_transformer': True
    }
    
    model = EnhancedGraphCorrectionModel(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    from torch_geometric.data import Data
    
    # Create dummy data
    x = torch.randn(100, 7)  # 100 nodes, 7 features
    edge_index = torch.randint(0, 100, (2, 200))  # 200 edges
    
    data = Data(x=x, edge_index=edge_index)
    
    with torch.no_grad():
        output = model(data)
        print("Forward pass successful!")
        print(f"Output keys: {output.keys()}")
        print(f"Position corrections shape: {output['position_corrections'].shape}")
        print(f"Learnable threshold: {output['learnable_threshold'].item():.3f}")