"""
Topology Correction Module
Implements specialized modules for correcting topological errors in vascular graphs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
from typing import Dict, Tuple, List, Optional
import numpy as np

from .graph_attention import GraphAttentionNetwork, BifurcationAttentionModule


class TopologyCorrector(nn.Module):
    """
    Main topology correction module that handles:
    - Node operations (insert/delete/keep/move)
    - Edge operations (add/remove/keep)
    - Bifurcation correction
    - Connectivity preservation
    """
    
    def __init__(self, 
                 feature_dim: int = 128,
                 config: Optional[Dict] = None):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.config = config or {}
        
        # Topology analysis network
        self.topology_analyzer = GraphAttentionNetwork(
            input_dim=feature_dim,
            hidden_dim=feature_dim,
            num_layers=3,
            num_heads=4,
            edge_dim=32
        )
        
        # Node operation predictor
        self.node_operation_predictor = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),  # Node + context features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # keep, remove, modify
        )
        
        # Edge operation predictor
        self.edge_operation_predictor = nn.Sequential(
            nn.Linear(feature_dim * 2 + 32, 128),  # 2 nodes + edge features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # add, remove, keep
        )
        
        # Specialized bifurcation corrector
        self.bifurcation_corrector = BifurcationAttentionModule(
            feature_dim=feature_dim,
            num_heads=4
        )
        
        # Node position corrector (for move operations)
        self.position_corrector = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # dx, dy, dz
        )
        
        # Connectivity validator
        self.connectivity_validator = ConnectivityValidator(feature_dim)
        
        # New node generator (for insert operations)
        self.node_generator = NodeGenerator(feature_dim)
        
    def forward(self, 
                node_features, 
                edge_index, 
                edge_features, 
                batch=None,
                node_positions=None,
                node_types=None):
        """
        Forward pass for topology correction
        
        Args:
            node_features: [N, feature_dim] node features
            edge_index: [2, E] edge connectivity
            edge_features: [E, edge_feature_dim] edge features
            batch: Batch assignment for nodes
            node_positions: [N, 3] node positions
            node_types: [N] node type indicators
            
        Returns:
            Dictionary containing topology corrections
        """
        # Analyze topology with GAT
        analyzed_features = self.topology_analyzer(
            node_features, edge_index, edge_features, batch
        )
        
        # Generate context features for each node
        context_features = self._generate_context_features(
            analyzed_features, edge_index, batch
        )
        
        # Predict node operations
        node_input = torch.cat([analyzed_features, context_features], dim=-1)
        node_operations = self.node_operation_predictor(node_input)
        
        # Predict position corrections for move operations
        position_corrections = self.position_corrector(analyzed_features)
        
        # Predict edge operations
        edge_operations = self._predict_edge_operations(
            analyzed_features, edge_index, edge_features
        )
        
        # Specialized bifurcation correction
        bifurcation_features, bifurcation_analysis = self.bifurcation_corrector(
            analyzed_features, edge_index, node_types
        )
        
        # Validate connectivity
        connectivity_analysis = self.connectivity_validator(
            analyzed_features, edge_index, batch
        )
        
        # Generate new nodes for insert operations
        insertion_candidates = self.node_generator(
            analyzed_features, edge_index, node_operations, batch
        )
        
        outputs = {
            'analyzed_features': analyzed_features,
            'node_operations': node_operations,
            'position_corrections': position_corrections,
            'edge_operations': edge_operations,
            'bifurcation_features': bifurcation_features,
            'bifurcation_analysis': bifurcation_analysis,
            'connectivity_analysis': connectivity_analysis,
            'insertion_candidates': insertion_candidates,
            'topology_quality_score': self._compute_topology_quality(
                node_operations, edge_operations, connectivity_analysis
            )
        }
        
        return analyzed_features, edge_features, outputs
    
    def _generate_context_features(self, node_features, edge_index, batch):
        """Generate contextual features for each node based on neighborhood"""
        num_nodes = node_features.size(0)
        device = node_features.device
        
        # Compute node degrees
        degrees = torch.zeros(num_nodes, device=device)
        for i in range(num_nodes):
            degree = ((edge_index[0] == i) | (edge_index[1] == i)).sum().float()
            degrees[i] = degree
        
        # Local neighborhood aggregation
        neighbor_features = torch.zeros_like(node_features)
        for i in range(num_nodes):
            # Find neighbors
            neighbors = edge_index[1][edge_index[0] == i]
            neighbors = torch.cat([neighbors, edge_index[0][edge_index[1] == i]])
            neighbors = neighbors.unique()
            
            if len(neighbors) > 0:
                neighbor_features[i] = node_features[neighbors].mean(dim=0)
            else:
                neighbor_features[i] = node_features[i]
        
        # Global context (graph-level)
        if batch is not None:
            global_context = global_mean_pool(node_features, batch)
            global_features = global_context[batch]
        else:
            global_context = node_features.mean(dim=0, keepdim=True)
            global_features = global_context.expand_as(node_features)
        
        # Combine local and global context
        context_features = torch.cat([
            neighbor_features,
            global_features,
            degrees.unsqueeze(-1).expand(-1, node_features.size(-1))
        ], dim=-1)
        
        # Project to correct dimension
        context_projection = nn.Linear(
            context_features.size(-1), 
            node_features.size(-1)
        ).to(device)
        
        return context_projection(context_features)
    
    def _predict_edge_operations(self, node_features, edge_index, edge_features):
        """Predict operations for each edge"""
        num_edges = edge_index.size(1)
        
        edge_operations = []
        for i in range(num_edges):
            src_idx, tgt_idx = edge_index[0, i], edge_index[1, i]
            src_features = node_features[src_idx]
            tgt_features = node_features[tgt_idx]
            edge_feature = edge_features[i] if edge_features is not None else torch.zeros(32, device=node_features.device)
            
            # Concatenate source, target, and edge features
            edge_input = torch.cat([src_features, tgt_features, edge_feature])
            edge_operation = self.edge_operation_predictor(edge_input.unsqueeze(0))
            edge_operations.append(edge_operation)
        
        if edge_operations:
            return torch.cat(edge_operations, dim=0)
        else:
            return torch.empty(0, 3, device=node_features.device)
    
    def _compute_topology_quality(self, node_operations, edge_operations, connectivity_analysis):
        """Compute overall topology quality score"""
        # Simple quality metric based on operation confidence
        node_confidence = F.softmax(node_operations, dim=-1).max(dim=-1)[0].mean()
        
        if edge_operations.size(0) > 0:
            edge_confidence = F.softmax(edge_operations, dim=-1).max(dim=-1)[0].mean()
        else:
            edge_confidence = torch.tensor(1.0, device=node_operations.device)
        
        connectivity_score = connectivity_analysis.get('connectivity_score', 0.5)
        if isinstance(connectivity_score, (int, float)):
            connectivity_score = torch.tensor(connectivity_score, device=node_operations.device)
        
        quality_score = (node_confidence + edge_confidence + connectivity_score) / 3
        return quality_score


class ConnectivityValidator(nn.Module):
    """
    Validates and suggests improvements for graph connectivity
    """
    
    def __init__(self, feature_dim: int):
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # Connectivity analysis network
        self.connectivity_analyzer = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, node_features, edge_index, batch=None):
        """
        Analyze connectivity patterns and suggest improvements
        
        Returns:
            Dictionary with connectivity analysis
        """
        num_nodes = node_features.size(0)
        
        # Compute connectivity metrics
        connectivity_metrics = self._compute_connectivity_metrics(
            node_features, edge_index, batch
        )
        
        # Predict connectivity quality for each node
        connectivity_scores = self.connectivity_analyzer(node_features)
        
        # Identify disconnected components
        disconnected_components = self._find_disconnected_components(
            edge_index, num_nodes
        )
        
        analysis = {
            'connectivity_score': connectivity_scores.mean().item(),
            'connectivity_metrics': connectivity_metrics,
            'disconnected_components': disconnected_components,
            'node_connectivity_scores': connectivity_scores
        }
        
        return analysis
    
    def _compute_connectivity_metrics(self, node_features, edge_index, batch):
        """Compute basic connectivity metrics"""
        num_nodes = node_features.size(0)
        num_edges = edge_index.size(1)
        
        # Basic metrics
        density = (2 * num_edges) / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        
        # Degree distribution
        degrees = torch.zeros(num_nodes, device=node_features.device)
        for i in range(num_nodes):
            degree = ((edge_index[0] == i) | (edge_index[1] == i)).sum()
            degrees[i] = degree.float()
        
        avg_degree = degrees.mean().item()
        degree_std = degrees.std().item()
        
        return {
            'density': density,
            'avg_degree': avg_degree,
            'degree_std': degree_std,
            'num_isolated_nodes': (degrees == 0).sum().item()
        }
    
    def _find_disconnected_components(self, edge_index, num_nodes):
        """Find disconnected components using simple traversal"""
        # Convert to adjacency list
        adj_list = [[] for _ in range(num_nodes)]
        for i in range(edge_index.size(1)):
            src, tgt = edge_index[0, i].item(), edge_index[1, i].item()
            adj_list[src].append(tgt)
            adj_list[tgt].append(src)
        
        visited = [False] * num_nodes
        components = []
        
        for i in range(num_nodes):
            if not visited[i]:
                component = []
                stack = [i]
                
                while stack:
                    node = stack.pop()
                    if not visited[node]:
                        visited[node] = True
                        component.append(node)
                        stack.extend(adj_list[node])
                
                components.append(component)
        
        return components


class NodeGenerator(nn.Module):
    """
    Generates new nodes for insertion operations
    """
    
    def __init__(self, feature_dim: int):
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # Node generation network
        self.node_generator = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),  # Context + reference features
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim + 3)  # Features + position
        )
        
        # Insertion probability network
        self.insertion_predictor = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, node_features, edge_index, node_operations, batch=None):
        """
        Generate candidate nodes for insertion
        
        Args:
            node_features: [N, feature_dim] existing node features
            edge_index: [2, E] edge connectivity
            node_operations: [N, 4] predicted node operations
            batch: Batch assignment
            
        Returns:
            Dictionary with insertion candidates
        """
        # Find nodes that need insertion (high insert probability)
        insert_probs = F.softmax(node_operations, dim=-1)[:, 0]  # Insert operation probability
        insertion_threshold = 0.5
        insertion_candidates = insert_probs > insertion_threshold
        
        if not insertion_candidates.any():
            return {
                'insertion_positions': torch.empty(0, 3, device=node_features.device),
                'insertion_features': torch.empty(0, self.feature_dim, device=node_features.device),
                'insertion_probabilities': torch.empty(0, device=node_features.device)
            }
        
        candidate_indices = torch.where(insertion_candidates)[0]
        
        insertion_positions = []
        insertion_features = []
        insertion_probabilities = []
        
        for idx in candidate_indices:
            # Find neighbors for context
            neighbors = self._find_neighbors(idx, edge_index)
            
            if len(neighbors) > 0:
                neighbor_features = node_features[neighbors].mean(dim=0)
                context_features = torch.cat([node_features[idx], neighbor_features])
                
                # Generate new node
                generated = self.node_generator(context_features)
                new_position = generated[:3]
                new_features = generated[3:]
                
                # Compute insertion probability
                insertion_prob = self.insertion_predictor(new_features)
                
                insertion_positions.append(new_position)
                insertion_features.append(new_features)
                insertion_probabilities.append(insertion_prob.item())
        
        return {
            'insertion_positions': torch.stack(insertion_positions) if insertion_positions else torch.empty(0, 3, device=node_features.device),
            'insertion_features': torch.stack(insertion_features) if insertion_features else torch.empty(0, self.feature_dim, device=node_features.device),
            'insertion_probabilities': torch.tensor(insertion_probabilities, device=node_features.device)
        }
    
    def _find_neighbors(self, node_idx, edge_index):
        """Find neighbors of a given node"""
        neighbors = []
        
        # Find outgoing edges
        outgoing = edge_index[1][edge_index[0] == node_idx]
        neighbors.extend(outgoing.tolist())
        
        # Find incoming edges
        incoming = edge_index[0][edge_index[1] == node_idx]
        neighbors.extend(incoming.tolist())
        
        return list(set(neighbors))  # Remove duplicates