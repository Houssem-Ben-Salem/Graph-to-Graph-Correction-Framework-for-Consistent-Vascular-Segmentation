"""
Main Graph Correction Model
Integrates all components for end-to-end graph correction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np

from .dual_graph_encoder import DualGraphEncoder
from .graph_attention import GraphAttentionNetwork, MultiHeadCrossGraphAttention
from .topology_corrector import TopologyCorrector
from .anatomy_preserver import AnatomyPreserver
from .loss_functions import GraphCorrectionLoss


class GraphCorrectionModel(nn.Module):
    """
    Complete Graph-to-Graph Correction Model
    
    Architecture:
    1. Dual Graph Encoder - Processes prediction and GT graphs separately
    2. Cross-Graph Attention - Learns correspondences
    3. Topology Corrector - Fixes topological errors
    4. Anatomy Preserver - Enforces physiological constraints
    5. Output Heads - Generates corrections and operations
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        
        self.config = self._get_default_config()
        if config:
            self.config.update(config)
        
        # Model dimensions
        self.node_feature_dim = self.config['node_feature_dim']
        self.edge_feature_dim = self.config['edge_feature_dim'] 
        self.hidden_dim = self.config['hidden_dim']
        self.num_attention_heads = self.config['num_attention_heads']
        
        # Core components
        self.dual_encoder = DualGraphEncoder(
            node_feature_dim=self.node_feature_dim,
            edge_feature_dim=self.edge_feature_dim,
            hidden_dim=self.hidden_dim,
            num_attention_heads=self.num_attention_heads
        )
        
        self.topology_corrector = TopologyCorrector(
            feature_dim=self.hidden_dim,
            config=self.config.get('topology_corrector', {})
        )
        
        self.anatomy_preserver = AnatomyPreserver(
            feature_dim=self.hidden_dim,
            config=self.config.get('anatomy_preserver', {})
        )
        
        # Additional processing layers
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(0.1)
        )
        
        # Output heads
        self.node_operation_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # keep, remove, modify
        )
        
        self.node_correction_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 7)  # dx, dy, dz, d_radius, d_curvature, type_prob_1, type_prob_2
        )
        
        self.edge_operation_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # add, remove, keep
        )
        
        self.quality_assessment_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Training mode flags
        self.training_stage = 1  # 1: topology, 2: anatomy, 3: joint
        
    def _get_default_config(self):
        """Get default model configuration"""
        return {
            'node_feature_dim': 16,
            'edge_feature_dim': 8,
            'hidden_dim': 128,
            'num_attention_heads': 8,
            'dropout': 0.1,
            'use_cross_attention': True,
            'topology_corrector': {
                'num_gat_layers': 3,
                'num_gat_heads': 4
            },
            'anatomy_preserver': {
                'murray_law_threshold': 0.2,
                'angle_tolerance': 30.0
            }
        }
    
    def forward(self, 
                pred_graph_data,
                gt_graph_data=None,
                correspondences=None,
                training_mode=True):
        """
        Forward pass for graph correction
        
        Args:
            pred_graph_data: Prediction graph data (VascularGraph or processed tensors)
            gt_graph_data: Optional ground truth graph data
            correspondences: Optional correspondence information
            training_mode: Whether in training mode
            
        Returns:
            Dictionary containing all correction outputs
        """
        # Extract features from graph data
        if hasattr(pred_graph_data, 'nodes'):  # VascularGraph format
            pred_features = self._extract_features_from_vascular_graph(pred_graph_data)
            if gt_graph_data is not None:
                gt_features = self._extract_features_from_vascular_graph(gt_graph_data)
            else:
                gt_features = None
        else:  # Pre-processed tensor format
            pred_features = pred_graph_data
            gt_features = gt_graph_data
        
        # Stage 1: Dual Graph Encoding
        encoding_outputs = self._encode_graphs(pred_features, gt_features, correspondences)
        
        # Get primary features for processing
        if 'fused_pred_features' in encoding_outputs:
            primary_features = encoding_outputs['fused_pred_features']
        else:
            primary_features = encoding_outputs['pred_node_features']
        
        # Stage 2: Topology Correction
        if self.training_stage in [1, 3]:  # Topology or joint training
            topology_features, edge_features, topology_outputs = self.topology_corrector(
                primary_features,
                pred_features['edge_index'],
                encoding_outputs.get('pred_edge_features'),
                pred_features.get('batch'),
                pred_features.get('node_positions'),
                pred_features.get('node_types')
            )
        else:
            topology_features = primary_features
            edge_features = encoding_outputs.get('pred_edge_features')
            topology_outputs = {}
        
        # Stage 3: Anatomy Preservation
        if self.training_stage in [2, 3]:  # Anatomy or joint training
            anatomy_features, anatomy_outputs = self.anatomy_preserver(
                topology_features,
                pred_features['edge_index'],
                edge_features,
                pred_features.get('batch'),
                pred_features.get('node_positions'),
                pred_features.get('node_radii'),
                pred_features.get('node_types')
            )
        else:
            anatomy_features = topology_features
            anatomy_outputs = {}
        
        # Stage 4: Feature Fusion
        if topology_features.size(-1) == anatomy_features.size(-1):
            fused_features = self.feature_fusion(
                torch.cat([topology_features, anatomy_features], dim=-1)
            )
        else:
            fused_features = anatomy_features
        
        # Stage 5: Generate Final Outputs
        outputs = self._generate_outputs(
            fused_features, edge_features, pred_features, 
            topology_outputs, anatomy_outputs, encoding_outputs
        )
        
        # Add training-specific outputs
        if training_mode and gt_graph_data is not None:
            outputs['training_signals'] = self._compute_training_signals(
                pred_features, gt_features, correspondences, outputs
            )
        
        return outputs
    
    def _extract_features_from_vascular_graph(self, graph_data):
        """Extract standardized features from VascularGraph"""
        features = {}
        
        # Get device from model parameters
        device = next(self.parameters()).device
        
        # Node features: [position(3), radius(1), curvature(1), node_type_onehot(3), degree(1)]
        node_features = self.dual_encoder.extract_node_features(graph_data, is_prediction=True)
        features['node_features'] = node_features.to(device)
        
        # Edge features and indices
        edge_features = self.dual_encoder.extract_edge_features(graph_data)
        features['edge_features'] = edge_features.to(device)
        
        # Edge index from graph structure
        edge_index = self._build_edge_index(graph_data, device)
        features['edge_index'] = edge_index
        
        # Additional features
        # Convert positions efficiently
        positions = []
        for node in graph_data.nodes:
            pos = node.get('position', [0, 0, 0])
            if hasattr(pos, 'tolist'):
                pos = pos.tolist()[:3]
            else:
                pos = list(pos)[:3]
            positions.append(pos)
        features['node_positions'] = torch.tensor(np.array(positions), dtype=torch.float32, device=device)
        
        features['node_radii'] = torch.tensor([
            node.get('radius_voxels', 1.0) for node in graph_data.nodes
        ], dtype=torch.float32, device=device)
        
        features['node_types'] = torch.tensor([
            1 if node.get('type') == 'bifurcation' else 
            2 if node.get('type') == 'endpoint' else 0
            for node in graph_data.nodes
        ], dtype=torch.long, device=device)
        
        return features
    
    def _build_edge_index(self, graph_data, device=None):
        """Build edge index tensor from VascularGraph"""
        if device is None:
            device = next(self.parameters()).device
            
        edges = []
        for edge in graph_data.edges:
            src = edge['source']
            tgt = edge['target'] 
            edges.append([src, tgt])
            # Note: Not adding reverse edges here - let PyTorch Geometric handle undirected conversion
        
        if edges:
            return torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
        else:
            return torch.empty(2, 0, dtype=torch.long, device=device)
    
    def _encode_graphs(self, pred_features, gt_features, correspondences):
        """Encode prediction and ground truth graphs"""
        # Extract confidence features
        num_nodes = pred_features['node_features'].size(0)
        num_edges = pred_features['edge_features'].size(0)
        device = pred_features['node_features'].device
        
        if correspondences is not None:
            # Extract confidence from correspondences
            node_confidences = torch.ones(num_nodes, 4, device=device) * 0.5  # Default
            edge_confidences = torch.ones(num_edges, 2, device=device) * 0.5  # Default
            
            # Update with actual correspondence confidences
            for pred_id, confidence in correspondences.node_confidences.items():
                if pred_id < num_nodes:
                    node_confidences[pred_id, 0] = confidence
                    node_confidences[pred_id, 1] = 1.0 - confidence  # Uncertainty
                    node_confidences[pred_id, 2] = confidence  # Local confidence mean
                    node_confidences[pred_id, 3] = 0.1  # Local confidence std
            
            # Update edge confidences
            for i, ((src, tgt), confidence) in enumerate(correspondences.edge_confidences.items()):
                if i < num_edges:
                    edge_confidences[i, 0] = confidence
                    edge_confidences[i, 1] = 0.1  # Low variance for matched edges
        else:
            # Default confidence values
            node_confidences = torch.ones(num_nodes, 4, device=device) * 0.5
            edge_confidences = torch.ones(num_edges, 2, device=device) * 0.5
        
        # Encode using dual encoder
        if gt_features is not None:
            encoding_outputs = self.dual_encoder(
                pred_features['node_features'],
                pred_features['edge_features'],
                node_confidences,
                edge_confidences,
                gt_features['node_features'],
                gt_features['edge_features']
            )
        else:
            encoding_outputs = self.dual_encoder(
                pred_features['node_features'],
                pred_features['edge_features'],
                node_confidences,
                edge_confidences
            )
        
        return encoding_outputs
    
    def _generate_outputs(self, fused_features, edge_features, pred_features,
                         topology_outputs, anatomy_outputs, encoding_outputs):
        """Generate final model outputs"""
        outputs = {}
        
        # Node operations and corrections
        node_operations = self.node_operation_head(fused_features)
        node_corrections = self.node_correction_head(fused_features)
        
        outputs['node_operations'] = node_operations
        outputs['node_corrections'] = node_corrections
        
        # Edge operations (if edge features available)
        if edge_features is not None and edge_features.size(0) > 0:
            # Pool edge features for classification
            edge_pooled = torch.mean(edge_features, dim=0, keepdim=True)
            
            # Project edge features to match hidden_dim if needed
            if edge_pooled.size(-1) != self.hidden_dim:
                # Create temporary projection layer
                edge_projection = torch.nn.Linear(edge_pooled.size(-1), self.hidden_dim).to(edge_pooled.device)
                edge_pooled = edge_projection(edge_pooled)
            
            edge_operations = self.edge_operation_head(edge_pooled)
            outputs['edge_operations'] = edge_operations
        
        # Quality assessment
        graph_features = torch.mean(fused_features, dim=0, keepdim=True)
        quality_score = self.quality_assessment_head(graph_features)
        outputs['quality_score'] = quality_score
        
        # Include component outputs
        outputs['topology_outputs'] = topology_outputs
        outputs['anatomy_outputs'] = anatomy_outputs
        outputs['encoding_outputs'] = encoding_outputs
        
        # Feature outputs
        outputs['corrected_node_features'] = fused_features
        outputs['corrected_edge_features'] = edge_features
        
        return outputs
    
    def _compute_training_signals(self, pred_features, gt_features, correspondences, outputs):
        """Compute training targets based on correspondences"""
        signals = {}
        
        if correspondences is None:
            return signals
        
        num_pred_nodes = pred_features['node_features'].size(0)
        device = pred_features['node_features'].device
        
        # Node operation targets
        node_op_targets = torch.zeros(num_pred_nodes, dtype=torch.long, device=device)
        
        # Node correction targets
        node_correction_targets = torch.zeros(num_pred_nodes, 7, device=device)
        
        # Data-driven thresholds based on actual distribution
        # From debug: confidence range 0.506-0.790, spatial error range 0.0-3.8
        high_conf_thresh = 0.77   # ~75th percentile for "keep" 
        low_conf_thresh = 0.52    # ~25th percentile for "remove"
        spatial_error_high = 2.0  # Higher spatial error threshold
        spatial_error_low = 0.5   # Lower spatial error threshold
        
        # Track class assignment for debugging
        class_counts = {'keep': 0, 'remove': 0, 'modify': 0}
        confidence_stats = {'min': float('inf'), 'max': float('-inf'), 'values': []}
        spatial_error_stats = {'min': float('inf'), 'max': float('-inf'), 'values': []}
        
        for pred_idx, gt_idx in correspondences.node_correspondences.items():
            confidence = correspondences.node_confidences.get(pred_idx, 0.5)
            confidence_stats['values'].append(confidence)
            confidence_stats['min'] = min(confidence_stats['min'], confidence)
            confidence_stats['max'] = max(confidence_stats['max'], confidence)
            
            # Check spatial displacement for additional criteria
            spatial_error = 0.0
            if gt_features is not None:
                pred_pos = pred_features['node_positions'][pred_idx]
                gt_pos = gt_features['node_positions'][gt_idx]
                spatial_error = torch.norm(gt_pos - pred_pos).item()
                spatial_error_stats['values'].append(spatial_error)
                spatial_error_stats['min'] = min(spatial_error_stats['min'], spatial_error)
                spatial_error_stats['max'] = max(spatial_error_stats['max'], spatial_error)
            
            # Balanced class assignment using data-driven thresholds
            if confidence > high_conf_thresh and spatial_error < spatial_error_low:
                node_op_targets[pred_idx] = 0  # keep (class 0) - high confidence, low spatial error
                class_counts['keep'] += 1
            elif confidence < low_conf_thresh or spatial_error > spatial_error_high:
                node_op_targets[pred_idx] = 1  # remove (class 1) - low confidence or high spatial error  
                class_counts['remove'] += 1
            else:
                node_op_targets[pred_idx] = 2  # modify (class 2) - medium confidence/error
                class_counts['modify'] += 1
                
                # Compute correction targets for modify class
                if gt_features is not None:
                    pos_correction = gt_pos - pred_pos
                    node_correction_targets[pred_idx, :3] = pos_correction
                    
                    pred_radius = pred_features['node_radii'][pred_idx]
                    gt_radius = gt_features['node_radii'][gt_idx]
                    radius_correction = gt_radius - pred_radius
                    node_correction_targets[pred_idx, 3] = radius_correction
        
        # Debug statistics
        if len(confidence_stats['values']) > 0:
            conf_mean = sum(confidence_stats['values']) / len(confidence_stats['values'])
            conf_std = (sum((x - conf_mean)**2 for x in confidence_stats['values']) / len(confidence_stats['values']))**0.5
            print(f"DEBUG: Confidence - min: {confidence_stats['min']:.3f}, max: {confidence_stats['max']:.3f}, mean: {conf_mean:.3f}, std: {conf_std:.3f}")
            
        if len(spatial_error_stats['values']) > 0:
            se_mean = sum(spatial_error_stats['values']) / len(spatial_error_stats['values'])
            se_std = (sum((x - se_mean)**2 for x in spatial_error_stats['values']) / len(spatial_error_stats['values']))**0.5
            print(f"DEBUG: Spatial Error - min: {spatial_error_stats['min']:.3f}, max: {spatial_error_stats['max']:.3f}, mean: {se_mean:.3f}, std: {se_std:.3f}")
        
        print(f"DEBUG: Class assignment - Keep: {class_counts['keep']}, Remove: {class_counts['remove']}, Modify: {class_counts['modify']}")
        
        # Mark unmatched nodes for removal
        for i in range(num_pred_nodes):
            if i not in correspondences.node_correspondences:
                node_op_targets[i] = 1  # remove (class 1)
        
        signals['node_op_targets'] = node_op_targets
        signals['node_correction_targets'] = node_correction_targets
        
        # Edge operation targets (simplified)
        if 'edge_operations' in outputs:
            num_edges = outputs['edge_operations'].size(0)
            edge_op_targets = torch.zeros(num_edges, dtype=torch.long, device=device)  # keep (class 0) by default
            signals['edge_op_targets'] = edge_op_targets
        
        return signals
    
    def set_training_stage(self, stage: int):
        """Set training stage (1: topology, 2: anatomy, 3: joint)"""
        self.training_stage = stage
        
        if stage == 1:  # Topology focus
            self.anatomy_preserver.requires_grad_(False)
        elif stage == 2:  # Anatomy focus  
            self.anatomy_preserver.requires_grad_(True)
            # Reduce topology learning rate (handled externally)
        elif stage == 3:  # Joint optimization
            self.anatomy_preserver.requires_grad_(True)
            self.topology_corrector.requires_grad_(True)
    
    def apply_corrections(self, graph_data, predictions, threshold=0.5):
        """
        Apply predicted corrections to generate corrected graph
        
        Args:
            graph_data: Original VascularGraph
            predictions: Model predictions
            threshold: Confidence threshold for applying corrections
            
        Returns:
            Corrected VascularGraph
        """
        # This would implement the actual graph modification
        # based on predicted operations and corrections
        
        # Extract operations
        node_operations = F.softmax(predictions['node_operations'], dim=-1)
        node_corrections = predictions['node_corrections']
        
        # Apply corrections based on predicted operations
        corrected_nodes = []
        corrected_edges = []
        
        for i, node in enumerate(graph_data.nodes):
            op_probs = node_operations[i]
            max_op = torch.argmax(op_probs).item()
            confidence = torch.max(op_probs).item()
            
            if confidence > threshold:
                if max_op == 0:  # insert - would need additional logic
                    corrected_nodes.append(node)
                elif max_op == 1:  # delete
                    continue  # Skip this node
                elif max_op == 2:  # keep
                    corrected_nodes.append(node)
                elif max_op == 3:  # move/modify
                    corrected_node = node.copy()
                    corrections = node_corrections[i]
                    
                    # Apply position correction
                    if 'position' in node:
                        device = corrections.device
                        old_pos = torch.tensor(node['position'][:3], device=device)
                        new_pos = old_pos + corrections[:3]
                        corrected_node['position'] = new_pos.tolist()
                    
                    # Apply radius correction
                    if 'radius_voxels' in node:
                        old_radius = node['radius_voxels']
                        new_radius = old_radius + corrections[3].item()
                        corrected_node['radius_voxels'] = max(0.1, new_radius)
                    
                    corrected_nodes.append(corrected_node)
            else:
                corrected_nodes.append(node)  # Keep if uncertain
        
        # For now, keep all edges (edge correction would be more complex)
        corrected_edges = graph_data.edges.copy()
        
        # Create new VascularGraph with corrections
        from src.models.graph_extraction.vascular_graph import VascularGraph
        
        corrected_graph = VascularGraph(
            nodes=corrected_nodes,
            edges=corrected_edges,
            global_properties=graph_data.global_properties.copy(),
            metadata={
                **graph_data.metadata,
                'corrected': True,
                'correction_threshold': threshold
            }
        )
        
        return corrected_graph
    
    def get_model_summary(self):
        """Get summary of model architecture"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        summary = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'config': self.config,
            'training_stage': self.training_stage,
            'components': {
                'dual_encoder': sum(p.numel() for p in self.dual_encoder.parameters()),
                'topology_corrector': sum(p.numel() for p in self.topology_corrector.parameters()),
                'anatomy_preserver': sum(p.numel() for p in self.anatomy_preserver.parameters())
            }
        }
        
        return summary