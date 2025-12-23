"""
Dual Graph Encoder for Graph Correction
Implements separate encoding for prediction and ground truth graphs with cross-graph attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


class MLP(nn.Module):
    """Multi-layer perceptron with configurable architecture"""
    
    def __init__(self, dims, dropout=0.1, activation='relu'):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # Don't add activation/dropout after last layer
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'gelu':
                    layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class MultiHeadCrossGraphAttention(nn.Module):
    """
    Multi-head cross-attention between predicted and ground truth graphs
    Enables the prediction graph to attend to relevant GT structures
    """
    
    def __init__(self, d_model=64, n_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections for queries, keys, values
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, query_features, key_features, value_features, mask=None):
        """
        Args:
            query_features: Features from prediction graph [N_pred, d_model]
            key_features: Features from ground truth graph [N_gt, d_model]
            value_features: Features from ground truth graph [N_gt, d_model]
            mask: Optional attention mask
        """
        batch_size = query_features.size(0)
        seq_len_q = query_features.size(0)
        seq_len_k = key_features.size(0)
        
        # Linear projections
        Q = self.w_q(query_features).view(seq_len_q, self.n_heads, self.d_k).transpose(0, 1)  # [n_heads, N_pred, d_k]
        K = self.w_k(key_features).view(seq_len_k, self.n_heads, self.d_k).transpose(0, 1)    # [n_heads, N_gt, d_k]
        V = self.w_v(value_features).view(seq_len_k, self.n_heads, self.d_k).transpose(0, 1)  # [n_heads, N_gt, d_k]
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # [n_heads, N_pred, N_gt]
        
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, V)  # [n_heads, N_pred, d_k]
        
        # Concatenate heads and project
        attended_values = attended_values.transpose(0, 1).contiguous().view(seq_len_q, self.d_model)
        output = self.w_o(attended_values)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + query_features)
        
        return output, attention_weights


class DualGraphEncoder(nn.Module):
    """
    Dual graph encoder that processes prediction and ground truth graphs separately
    then enables cross-graph attention for learning correspondences
    """
    
    def __init__(self, 
                 node_feature_dim=16, 
                 edge_feature_dim=8,
                 hidden_dim=64,
                 num_attention_heads=8):
        super().__init__()
        
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        
        # Ground truth graph encoders
        self.gt_node_encoder = MLP([node_feature_dim, 32, hidden_dim])
        self.gt_edge_encoder = MLP([edge_feature_dim, 16, 32])
        
        # Prediction graph encoders (with additional confidence features)
        self.pred_node_encoder = MLP([node_feature_dim + 4, 32, hidden_dim])  # +4 for confidence features
        self.pred_edge_encoder = MLP([edge_feature_dim + 2, 16, 32])          # +2 for confidence features
        
        # Cross-graph attention mechanism
        self.cross_attention = MultiHeadCrossGraphAttention(
            d_model=hidden_dim,
            n_heads=num_attention_heads
        )
        
        # Feature fusion layers
        self.feature_fusion = MLP([hidden_dim * 2, hidden_dim, hidden_dim])
        
    def encode_ground_truth_features(self, node_attrs, edge_attrs):
        """
        Encode ground truth graph features
        
        Args:
            node_attrs: [N_gt, node_feature_dim] ground truth node attributes
            edge_attrs: [E_gt, edge_feature_dim] ground truth edge attributes
            
        Returns:
            Encoded features
        """
        gt_node_features = self.gt_node_encoder(node_attrs)
        gt_edge_features = self.gt_edge_encoder(edge_attrs)
        
        return gt_node_features, gt_edge_features
    
    def encode_prediction_features(self, node_attrs, edge_attrs, 
                                 node_confidences, edge_confidences):
        """
        Encode prediction graph features with confidence information
        
        Args:
            node_attrs: [N_pred, node_feature_dim] prediction node attributes
            edge_attrs: [E_pred, edge_feature_dim] prediction edge attributes
            node_confidences: [N_pred, 4] node confidence features
            edge_confidences: [E_pred, 2] edge confidence features
            
        Returns:
            Encoded features
        """
        # Concatenate base features with confidence
        enhanced_node_attrs = torch.cat([node_attrs, node_confidences], dim=-1)
        enhanced_edge_attrs = torch.cat([edge_attrs, edge_confidences], dim=-1)
        
        pred_node_features = self.pred_node_encoder(enhanced_node_attrs)
        pred_edge_features = self.pred_edge_encoder(enhanced_edge_attrs)
        
        return pred_node_features, pred_edge_features
    
    def forward(self, 
                pred_node_attrs, pred_edge_attrs,
                pred_node_confidences, pred_edge_confidences,
                gt_node_attrs=None, gt_edge_attrs=None):
        """
        Forward pass for dual graph encoding
        
        Args:
            pred_node_attrs: [N_pred, node_feature_dim] prediction node attributes
            pred_edge_attrs: [E_pred, edge_feature_dim] prediction edge attributes  
            pred_node_confidences: [N_pred, 4] prediction node confidences
            pred_edge_confidences: [E_pred, 2] prediction edge confidences
            gt_node_attrs: [N_gt, node_feature_dim] optional GT node attributes
            gt_edge_attrs: [E_gt, edge_feature_dim] optional GT edge attributes
            
        Returns:
            Dictionary containing encoded features
        """
        # Encode prediction graph features
        pred_node_features, pred_edge_features = self.encode_prediction_features(
            pred_node_attrs, pred_edge_attrs, 
            pred_node_confidences, pred_edge_confidences
        )
        
        outputs = {
            'pred_node_features': pred_node_features,
            'pred_edge_features': pred_edge_features,
        }
        
        # If ground truth is available, encode and apply cross-attention
        if gt_node_attrs is not None and gt_edge_attrs is not None:
            gt_node_features, gt_edge_features = self.encode_ground_truth_features(
                gt_node_attrs, gt_edge_attrs
            )
            
            # Apply cross-graph attention
            attended_pred_features, attention_weights = self.cross_attention(
                query_features=pred_node_features,    # Prediction attends to GT
                key_features=gt_node_features,
                value_features=gt_node_features
            )
            
            # Fuse original and attended features
            fused_features = self.feature_fusion(
                torch.cat([pred_node_features, attended_pred_features], dim=-1)
            )
            
            outputs.update({
                'gt_node_features': gt_node_features,
                'gt_edge_features': gt_edge_features,
                'attended_pred_features': attended_pred_features,
                'fused_pred_features': fused_features,
                'attention_weights': attention_weights
            })
        else:
            # During inference, use prediction features directly
            outputs['fused_pred_features'] = pred_node_features
        
        return outputs
    
    def extract_node_features(self, graph_data, is_prediction=True):
        """
        Extract standardized node features from VascularGraph data
        
        Args:
            graph_data: VascularGraph instance
            is_prediction: Boolean indicating if this is prediction data
            
        Returns:
            Tensor of node features [N, feature_dim]
        """
        features = []
        
        for node in graph_data.nodes:
            # Base features: [position(3), radius(1), curvature(1), node_type_onehot(3), degree(1)]
            # Handle position - can be array or list
            position = node.get('position', [0, 0, 0])
            if hasattr(position, 'tolist'):
                position = position.tolist()[:3]
            else:
                position = list(position)[:3]
            
            # Radius
            radius = [float(node.get('radius_voxels', 1.0))]
            
            # Curvature
            curvature = [float(node.get('local_curvature', 0.0))]
            
            # One-hot encoding for node type
            node_type = node.get('type', 'regular')
            if node_type == 'bifurcation':
                type_onehot = [1, 0, 0]
            elif node_type == 'endpoint':
                type_onehot = [0, 1, 0]
            else:  # regular
                type_onehot = [0, 0, 1]
            
            # Degree
            degree = [float(node.get('degree', 2))]
            
            # Additional features to reach 16 total
            # [vessel_direction(3), neighbor_distance(1), node_density(1), physical_radius(1)]
            vessel_direction = node.get('vessel_direction', [0, 0, 1])
            if hasattr(vessel_direction, 'tolist'):
                vessel_direction = vessel_direction.tolist()[:3]
            else:
                vessel_direction = list(vessel_direction)[:3]
            
            neighbor_distance = [float(node.get('nearest_neighbor_distance', 1.0))]
            node_density = [float(node.get('local_node_density', 1.0))]
            physical_radius = [float(node.get('radius_mm', node.get('radius_voxels', 1.0)))]
            
            # Combine all features: 3+1+1+3+1+3+1+1+1 = 15, add one padding
            node_features = (position + radius + curvature + type_onehot + degree + 
                           vessel_direction + neighbor_distance + node_density + physical_radius)
            
            # Pad to exactly 16 features if needed
            while len(node_features) < 16:
                node_features.append(0.0)
            
            features.append(node_features[:16])  # Ensure exactly 16 features
        
        return torch.tensor(features, dtype=torch.float32)
    
    def extract_edge_features(self, graph_data):
        """
        Extract standardized edge features from VascularGraph data
        
        Args:
            graph_data: VascularGraph instance
            
        Returns:
            Tensor of edge features [E, feature_dim]
        """
        features = []
        
        for edge in graph_data.edges:
            # Base features: [length(1), direction(3), avg_radius(1), curvature(1), edge_type(1)]
            length = [float(edge.get('euclidean_length', 1.0))]
            
            # Direction vector (normalized)
            direction = edge.get('direction_vector', [0, 0, 1])
            if hasattr(direction, 'tolist'):
                direction = direction.tolist()[:3]
            else:
                direction = list(direction)[:3]
            
            avg_radius = [float(edge.get('average_radius', 1.0))]
            curvature = [float(edge.get('path_curvature', 0.0))]
            
            # Additional features for richer representation
            physical_length = [float(edge.get('physical_length', edge.get('euclidean_length', 1.0)))]
            confidence = [float(edge.get('confidence_score', 1.0))]
            
            # Combine features: 1+3+1+1+1+1 = 8 features
            edge_features = length + direction + avg_radius + curvature + physical_length + confidence
            
            # Pad to exactly 8 features if needed
            while len(edge_features) < 8:
                edge_features.append(0.0)
            
            features.append(edge_features[:8])  # Ensure exactly 8 features
        
        return torch.tensor(features, dtype=torch.float32)
    
    def extract_confidence_features(self, graph_data, correspondences=None):
        """
        Extract confidence features for prediction graph
        
        Args:
            graph_data: VascularGraph instance (prediction)
            correspondences: Optional correspondence result for confidence extraction
            
        Returns:
            node_confidences: [N, 4] node confidence features
            edge_confidences: [E, 2] edge confidence features
        """
        num_nodes = len(graph_data.nodes)
        num_edges = len(graph_data.edges)
        
        # Node confidence features: [confidence, uncertainty, local_confidence_mean, local_confidence_std]
        node_confidences = torch.ones(num_nodes, 4) * 0.5  # Default moderate confidence
        
        # Edge confidence features: [confidence, path_confidence_variance]
        edge_confidences = torch.ones(num_edges, 2) * 0.5  # Default moderate confidence
        
        # If correspondences are available, use actual confidence scores
        if correspondences is not None:
            for i, (node_id, confidence) in enumerate(correspondences.node_confidences.items()):
                if node_id < num_nodes:
                    node_confidences[node_id, 0] = confidence
                    node_confidences[node_id, 1] = 1.0 - confidence  # Uncertainty
            
            for i, ((src, tgt), confidence) in enumerate(correspondences.edge_confidences.items()):
                if i < num_edges:
                    edge_confidences[i, 0] = confidence
                    edge_confidences[i, 1] = 0.1  # Low variance for matched edges
        
        return node_confidences, edge_confidences