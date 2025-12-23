"""
Graph Attention Networks for Graph Correction
Implements multi-head GAT layers specialized for vascular graph processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from typing import Optional, Tuple
import math


class GraphAttentionLayer(nn.Module):
    """
    Single Graph Attention Layer with edge features
    Specialized for vascular graph topology understanding
    """
    
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 num_heads: int = 1,
                 dropout: float = 0.1,
                 alpha: float = 0.2,
                 concat: bool = True,
                 edge_dim: Optional[int] = None):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        self.edge_dim = edge_dim
        
        # Use PyTorch Geometric's GATConv as base
        self.gat_conv = GATConv(
            in_channels=in_features,
            out_channels=out_features,
            heads=num_heads,
            dropout=dropout,
            edge_dim=edge_dim,
            concat=concat
        )
        
        # Additional edge processing if edge features are provided
        if edge_dim is not None:
            self.edge_processor = nn.Sequential(
                nn.Linear(edge_dim, edge_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
        # Layer normalization
        final_dim = out_features * num_heads if concat else out_features
        self.layer_norm = nn.LayerNorm(final_dim)
        
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        Forward pass
        
        Args:
            x: Node features [N, in_features]
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge features [E, edge_dim]
            batch: Batch assignment for nodes
            
        Returns:
            Updated node features and attention weights
        """
        # Process edge attributes if provided
        if edge_attr is not None and self.edge_dim is not None:
            edge_attr = self.edge_processor(edge_attr)
        
        # Apply GAT convolution
        output = self.gat_conv(x, edge_index, edge_attr)
        
        # Layer normalization with residual connection
        if x.size(-1) == output.size(-1):
            output = self.layer_norm(output + x)
        else:
            output = self.layer_norm(output)
        
        return output


class GraphAttentionNetwork(nn.Module):
    """
    Multi-layer Graph Attention Network for vascular graph analysis
    """
    
    def __init__(self,
                 input_dim: int = 64,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 edge_dim: Optional[int] = 32):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.edge_dim = edge_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                # First layer
                in_dim = hidden_dim
                out_dim = hidden_dim // num_heads
                concat = True
            elif i == num_layers - 1:
                # Last layer - don't concat heads
                in_dim = hidden_dim
                out_dim = hidden_dim
                concat = False
            else:
                # Middle layers
                in_dim = hidden_dim
                out_dim = hidden_dim // num_heads
                concat = True
            
            gat_layer = GraphAttentionLayer(
                in_features=in_dim,
                out_features=out_dim,
                num_heads=num_heads,
                dropout=dropout,
                concat=concat,
                edge_dim=edge_dim if i == 0 else None  # Only first layer processes edges
            )
            
            self.gat_layers.append(gat_layer)
        
        # Final projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        Forward pass through GAT layers
        
        Args:
            x: Node features [N, input_dim]
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge features [E, edge_dim]
            batch: Batch assignment
            
        Returns:
            Updated node features [N, hidden_dim]
        """
        # Input projection
        x = self.input_projection(x)
        x = F.relu(x)
        
        # Apply GAT layers
        for i, gat_layer in enumerate(self.gat_layers):
            # Only pass edge attributes to first layer
            current_edge_attr = edge_attr if i == 0 else None
            x = gat_layer(x, edge_index, current_edge_attr, batch)
        
        # Final projection
        x = self.output_projection(x)
        
        return x


class BifurcationAttentionModule(nn.Module):
    """
    Specialized attention module for bifurcation analysis
    Focuses on parent-child relationships at vessel branching points
    """
    
    def __init__(self, feature_dim: int = 128, num_heads: int = 4):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        # Multi-head attention for bifurcation analysis
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Bifurcation-specific processing
        self.bifurcation_processor = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim * 2),  # Parent + 2 children
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        # Murray's law checker
        self.murray_law_checker = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, node_features, edge_index, node_types):
        """
        Process bifurcation nodes with specialized attention
        
        Args:
            node_features: [N, feature_dim] node features
            edge_index: [2, E] edge connectivity
            node_types: [N] node type indicators (0=regular, 1=bifurcation, 2=endpoint)
            
        Returns:
            Updated node features and bifurcation analysis
        """
        device = node_features.device
        num_nodes = node_features.size(0)
        
        # Identify bifurcation nodes
        bifurcation_mask = (node_types == 1)
        bifurcation_indices = torch.where(bifurcation_mask)[0]
        
        if len(bifurcation_indices) == 0:
            return node_features, {}
        
        # For each bifurcation, find parent and children
        bifurcation_features = []
        murray_violations = []
        
        for bifurc_idx in bifurcation_indices:
            # Find connected nodes
            connected_edges = (edge_index[0] == bifurc_idx) | (edge_index[1] == bifurc_idx)
            connected_nodes = edge_index[:, connected_edges].flatten()
            connected_nodes = connected_nodes[connected_nodes != bifurc_idx].unique()
            
            if len(connected_nodes) >= 2:
                # Get features for bifurcation and connected nodes
                bifurc_feature = node_features[bifurc_idx:bifurc_idx+1]  # [1, feature_dim]
                connected_features = node_features[connected_nodes]      # [K, feature_dim]
                
                # Apply attention to understand relationships
                attended_features, attention_weights = self.attention(
                    bifurc_feature,  # Query: bifurcation node
                    connected_features,  # Key: connected nodes
                    connected_features   # Value: connected nodes
                )
                
                # Process bifurcation context (simplified - take first 2 connected nodes)
                if len(connected_nodes) >= 2:
                    context_features = torch.cat([
                        bifurc_feature.squeeze(0),
                        connected_features[0],
                        connected_features[1]
                    ])
                    
                    processed_features = self.bifurcation_processor(context_features)
                    bifurcation_features.append(processed_features)
                    
                    # Check Murray's law compliance
                    murray_score = self.murray_law_checker(processed_features)
                    murray_violations.append(murray_score.item())
                else:
                    bifurcation_features.append(node_features[bifurc_idx])
                    murray_violations.append(0.5)  # Neutral score
            else:
                bifurcation_features.append(node_features[bifurc_idx])
                murray_violations.append(0.5)
        
        # Update bifurcation node features
        updated_features = node_features.clone()
        if bifurcation_features:
            bifurcation_tensor = torch.stack(bifurcation_features)
            updated_features[bifurcation_indices] = bifurcation_tensor
        
        analysis = {
            'bifurcation_indices': bifurcation_indices,
            'murray_violations': murray_violations,
            'num_bifurcations': len(bifurcation_indices)
        }
        
        return updated_features, analysis


class MultiHeadCrossGraphAttention(nn.Module):
    """
    Multi-head cross-attention between two different graphs
    Enables correspondence learning between prediction and ground truth
    """
    
    def __init__(self, d_model: int = 64, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Spatial distance weighting
        self.spatial_weighting = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, 
                query_features, 
                key_features, 
                value_features,
                query_positions=None,
                key_positions=None,
                mask=None):
        """
        Cross-graph attention with spatial awareness
        
        Args:
            query_features: [N_pred, d_model] prediction graph features
            key_features: [N_gt, d_model] GT graph features  
            value_features: [N_gt, d_model] GT graph features
            query_positions: [N_pred, 3] prediction node positions
            key_positions: [N_gt, 3] GT node positions
            mask: Optional attention mask
            
        Returns:
            attended_features: [N_pred, d_model]
            attention_weights: [n_heads, N_pred, N_gt]
        """
        seq_len_q = query_features.size(0)
        seq_len_k = key_features.size(0)
        
        # Linear projections and reshape for multi-head attention
        Q = self.w_q(query_features).view(seq_len_q, self.n_heads, self.d_k).transpose(0, 1)
        K = self.w_k(key_features).view(seq_len_k, self.n_heads, self.d_k).transpose(0, 1)
        V = self.w_v(value_features).view(seq_len_k, self.n_heads, self.d_k).transpose(0, 1)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Add spatial bias if positions are provided
        if query_positions is not None and key_positions is not None:
            # Compute pairwise spatial distances
            distances = torch.cdist(query_positions, key_positions, p=2)  # [N_pred, N_gt]
            
            # Convert distances to spatial weights
            spatial_weights = self.spatial_weighting(distances.unsqueeze(-1)).squeeze(-1)
            
            # Add spatial bias to attention scores (broadcast to all heads)
            spatial_bias = spatial_weights.unsqueeze(0).expand(self.n_heads, -1, -1)
            attention_scores = attention_scores + spatial_bias
        
        # Apply mask if provided
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        
        # Softmax and dropout
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


class HierarchicalGraphAttention(nn.Module):
    """
    Hierarchical attention that operates at multiple scales:
    1. Local: immediate neighbors
    2. Regional: 2-hop neighborhoods  
    3. Global: entire graph context
    """
    
    def __init__(self, feature_dim: int = 128, num_heads: int = 4):
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # Local attention (1-hop neighbors)
        self.local_attention = GraphAttentionLayer(
            in_features=feature_dim,
            out_features=feature_dim // num_heads,
            num_heads=num_heads,
            concat=True
        )
        
        # Regional attention (2-hop neighbors)
        self.regional_attention = GraphAttentionLayer(
            in_features=feature_dim,
            out_features=feature_dim // num_heads,
            num_heads=num_heads,
            concat=True
        )
        
        # Global attention (graph-level pooling + broadcast)
        self.global_pooling = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
    def forward(self, x, edge_index, batch=None):
        """
        Hierarchical attention forward pass
        
        Args:
            x: Node features [N, feature_dim]
            edge_index: Edge connectivity [2, E]
            batch: Batch assignment
            
        Returns:
            Multi-scale attended features [N, feature_dim]
        """
        # Local attention
        local_features = self.local_attention(x, edge_index)
        
        # Regional attention (could use 2-hop edge index, simplified here)
        regional_features = self.regional_attention(x, edge_index)
        
        # Global context
        if batch is not None:
            global_context = global_mean_pool(x, batch)  # [batch_size, feature_dim]
            # Broadcast to all nodes
            global_features = global_context[batch]  # [N, feature_dim]
        else:
            # Single graph case
            global_context = torch.mean(x, dim=0, keepdim=True)  # [1, feature_dim]
            global_features = global_context.expand_as(x)  # [N, feature_dim]
        
        global_features = self.global_pooling(global_features)
        
        # Fuse multi-scale features
        fused_features = self.fusion(torch.cat([
            local_features, regional_features, global_features
        ], dim=-1))
        
        return fused_features