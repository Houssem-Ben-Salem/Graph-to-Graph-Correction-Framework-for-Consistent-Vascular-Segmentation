"""
Multi-Objective Loss Functions for Graph Correction
Implements comprehensive loss functions for topology and anatomy correction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math
import numpy as np


class GraphCorrectionLoss(nn.Module):
    """
    Multi-objective loss function for graph correction training
    
    Components:
    1. Topology Loss - Node/edge operation accuracy
    2. Anatomy Loss - Murray's law, tapering, angles, continuity
    3. Consistency Loss - Feature consistency and smoothness
    4. Improvement Loss - Overall quality improvement
    5. Regularization Loss - Model complexity control
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        
        self.config = config or {}
        
        # Default loss weights
        self.loss_weights = {
            'topology': self.config.get('topology_weight', 1.0),
            'anatomy': self.config.get('anatomy_weight', 0.8),
            'consistency': self.config.get('consistency_weight', 0.6),
            'improvement': self.config.get('improvement_weight', 1.2),
            'regularization': self.config.get('regularization_weight', 0.1)
        }
        
        # Topology loss components with class weighting for imbalanced data
        use_focal_loss = self.config.get('use_focal_loss', False)
        
        if use_focal_loss:
            from .focal_loss import FocalLoss
            node_class_weights = self.config.get('node_class_weights', [1.0, 1.5, 1.2])
            # Don't set device here - will be handled dynamically in FocalLoss
            alpha = torch.tensor(node_class_weights, dtype=torch.float32)
            gamma = self.config.get('focal_gamma', 2.0)
            self.node_operation_loss = FocalLoss(alpha=alpha, gamma=gamma)
        else:
            # Default weights for [keep, remove, modify] - will be updated based on data
            node_class_weights = self.config.get('node_class_weights', [1.0, 1.5, 1.2])
            # Store weights to move to device later
            self.class_weights = torch.tensor(node_class_weights, dtype=torch.float32)
            self.node_operation_loss = nn.CrossEntropyLoss()  # Will set weight dynamically
        self.edge_operation_loss = nn.CrossEntropyLoss()
        self.position_loss = nn.MSELoss()
        self.radius_loss = nn.MSELoss()
        
        # Anatomy loss components
        self.murray_law_loss = MurrayLawLoss()
        self.tapering_loss = TaperingConsistencyLoss()
        self.angle_loss = BranchingAngleLoss()
        self.continuity_loss = VesselContinuityLoss()
        
        # Consistency loss components
        self.feature_consistency_loss = FeatureConsistencyLoss()
        self.smoothness_loss = SmoothnessLoss()
        
        # Improvement loss
        self.improvement_loss = QualityImprovementLoss()
        
        # Regularization
        self.regularization_loss = RegularizationLoss()
        
    def forward(self, 
                predictions: Dict,
                targets: Dict,
                correspondences: Dict,
                metadata: Optional[Dict] = None) -> Dict:
        """
        Compute multi-objective loss
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            correspondences: Node/edge correspondences
            metadata: Additional metadata
            
        Returns:
            Dictionary with loss components and total loss
        """
        losses = {}
        
        # 1. Topology Loss
        topology_loss = self._compute_topology_loss(predictions, targets, correspondences)
        losses['topology'] = topology_loss
        
        # 2. Anatomy Loss
        anatomy_loss = self._compute_anatomy_loss(predictions, targets, correspondences)
        losses['anatomy'] = anatomy_loss
        
        # 3. Consistency Loss
        consistency_loss = self._compute_consistency_loss(predictions, targets)
        losses['consistency'] = consistency_loss
        
        # 4. Improvement Loss
        improvement_loss = self._compute_improvement_loss(predictions, targets, metadata)
        losses['improvement'] = improvement_loss
        
        # 5. Regularization Loss
        regularization_loss = self._compute_regularization_loss(predictions)
        losses['regularization'] = regularization_loss
        
        # Weighted total loss
        total_loss = sum(self.loss_weights[key] * loss for key, loss in losses.items())
        losses['total'] = total_loss
        
        return losses
    
    def _compute_topology_loss(self, predictions, targets, correspondences):
        """Compute topology correction loss"""
        topology_losses = []
        
        # Node operation loss
        if 'node_operations' in predictions and 'node_op_targets' in targets:
            # Handle device placement for class weights
            if hasattr(self, 'class_weights'):
                # Move class weights to same device as predictions
                device = predictions['node_operations'].device
                weight = self.class_weights.to(device)
                node_op_loss = nn.functional.cross_entropy(
                    predictions['node_operations'], 
                    targets['node_op_targets'],
                    weight=weight
                )
            else:
                # Use focal loss or default CrossEntropyLoss
                node_op_loss = self.node_operation_loss(
                    predictions['node_operations'], 
                    targets['node_op_targets']
                )
            topology_losses.append(node_op_loss)
        
        # Edge operation loss
        if 'edge_operations' in predictions and 'edge_op_targets' in targets:
            edge_op_loss = self.edge_operation_loss(
                predictions['edge_operations'],
                targets['edge_op_targets']
            )
            topology_losses.append(edge_op_loss)
        
        # Position correction loss
        if 'position_corrections' in predictions and 'position_targets' in targets:
            pos_loss = self.position_loss(
                predictions['position_corrections'],
                targets['position_targets']
            )
            topology_losses.append(pos_loss)
        
        # Bifurcation detection loss
        if 'bifurcation_analysis' in predictions:
            bifurc_loss = self._compute_bifurcation_loss(
                predictions['bifurcation_analysis'], targets
            )
            topology_losses.append(bifurc_loss)
        
        return sum(topology_losses) / max(len(topology_losses), 1)
    
    def _compute_anatomy_loss(self, predictions, targets, correspondences):
        """Compute anatomy preservation loss"""
        anatomy_losses = []
        
        # Murray's law loss
        if 'anatomy_outputs' in predictions:
            anatomy_outputs = predictions['anatomy_outputs']
            
            if 'murray_analysis' in anatomy_outputs:
                murray_loss = self.murray_law_loss(anatomy_outputs['murray_analysis'])
                anatomy_losses.append(murray_loss)
            
            # Tapering consistency loss
            if 'tapering_analysis' in anatomy_outputs:
                tapering_loss = self.tapering_loss(anatomy_outputs['tapering_analysis'])
                anatomy_losses.append(tapering_loss)
            
            # Branching angle loss
            if 'angle_analysis' in anatomy_outputs:
                angle_loss = self.angle_loss(anatomy_outputs['angle_analysis'])
                anatomy_losses.append(angle_loss)
            
            # Vessel continuity loss
            if 'continuity_analysis' in anatomy_outputs:
                continuity_loss = self.continuity_loss(anatomy_outputs['continuity_analysis'])
                anatomy_losses.append(continuity_loss)
        
        return sum(anatomy_losses) / max(len(anatomy_losses), 1)
    
    def _compute_consistency_loss(self, predictions, targets):
        """Compute feature consistency loss"""
        consistency_losses = []
        
        # Feature consistency
        if 'corrected_node_features' in predictions:
            feature_loss = self.feature_consistency_loss(
                predictions['corrected_node_features']
            )
            consistency_losses.append(feature_loss)
        
        # Smoothness loss
        if 'corrected_node_features' in predictions and 'edge_index' in targets:
            smoothness_loss = self.smoothness_loss(
                predictions['corrected_node_features'],
                targets['edge_index']
            )
            consistency_losses.append(smoothness_loss)
        
        return sum(consistency_losses) / max(len(consistency_losses), 1)
    
    def _compute_improvement_loss(self, predictions, targets, metadata):
        """Compute quality improvement loss"""
        if 'topology_quality_score' in predictions:
            quality_score = predictions['topology_quality_score']
            improvement_loss = self.improvement_loss(quality_score)
            return improvement_loss
        
        return torch.tensor(0.0, device=predictions['node_operations'].device)
    
    def _compute_regularization_loss(self, predictions):
        """Compute regularization loss"""
        regularization_losses = []
        
        # L2 regularization on corrected features
        if 'corrected_node_features' in predictions:
            l2_loss = torch.norm(predictions['corrected_node_features'], p=2)
            regularization_losses.append(l2_loss)
        
        # Sparsity regularization on operations
        if 'node_operations' in predictions:
            node_probs = F.softmax(predictions['node_operations'], dim=-1)
            sparsity_loss = -torch.sum(node_probs * torch.log(node_probs + 1e-8))
            regularization_losses.append(0.1 * sparsity_loss)
        
        return sum(regularization_losses) / max(len(regularization_losses), 1)
    
    def _compute_bifurcation_loss(self, bifurcation_analysis, targets):
        """Compute bifurcation-specific loss"""
        if isinstance(bifurcation_analysis, dict):
            # Penalize Murray's law violations
            violations = bifurcation_analysis.get('murray_violations', [])
            if violations:
                violation_loss = torch.tensor(np.mean(violations), 
                                            dtype=torch.float32,
                                            device=next(iter(targets.values())).device)
                return violation_loss
        
        return torch.tensor(0.0, device=next(iter(targets.values())).device)


class MurrayLawLoss(nn.Module):
    """Loss function for Murray's law compliance"""
    
    def __init__(self, violation_penalty: float = 2.0):
        super().__init__()
        self.violation_penalty = violation_penalty
    
    def forward(self, murray_analysis: Dict) -> torch.Tensor:
        """
        Compute Murray's law loss
        
        Args:
            murray_analysis: Dictionary with Murray's law analysis
            
        Returns:
            Murray's law violation loss
        """
        if not isinstance(murray_analysis, dict):
            return torch.tensor(0.0)
        
        violations = murray_analysis.get('murray_violations', [])
        compliance_scores = murray_analysis.get('compliance_scores', [])
        
        total_loss = 0.0
        num_components = 0
        
        # Violation penalty
        if violations:
            violation_loss = torch.tensor(np.mean(violations), dtype=torch.float32)
            total_loss += self.violation_penalty * violation_loss
            num_components += 1
        
        # Compliance score loss (encourage high compliance)
        if compliance_scores:
            compliance_loss = 1.0 - torch.tensor(np.mean(compliance_scores), dtype=torch.float32)
            total_loss += compliance_loss
            num_components += 1
        
        return total_loss / max(num_components, 1)


class TaperingConsistencyLoss(nn.Module):
    """Loss function for vessel tapering consistency"""
    
    def forward(self, tapering_analysis: Dict) -> torch.Tensor:
        """Compute tapering consistency loss"""
        if not isinstance(tapering_analysis, dict):
            return torch.tensor(0.0)
        
        avg_consistency = tapering_analysis.get('avg_tapering_consistency', 1.0)
        num_violations = tapering_analysis.get('num_tapering_violations', 0)
        total_nodes = len(tapering_analysis.get('tapering_scores', [1]))
        
        # Inconsistency penalty
        inconsistency_loss = 1.0 - avg_consistency
        
        # Violation ratio penalty
        violation_ratio = num_violations / max(total_nodes, 1)
        violation_loss = violation_ratio
        
        return torch.tensor(inconsistency_loss + violation_loss, dtype=torch.float32)


class BranchingAngleLoss(nn.Module):
    """Loss function for branching angle realism"""
    
    def forward(self, angle_analysis: Dict) -> torch.Tensor:
        """Compute branching angle loss"""
        if not isinstance(angle_analysis, dict):
            return torch.tensor(0.0)
        
        avg_quality = angle_analysis.get('avg_angle_quality', 1.0)
        num_violations = angle_analysis.get('num_angle_violations', 0)
        total_bifurcations = len(angle_analysis.get('angle_scores', [1]))
        
        # Angle quality loss
        quality_loss = 1.0 - avg_quality
        
        # Violation penalty
        violation_ratio = num_violations / max(total_bifurcations, 1)
        violation_loss = violation_ratio
        
        return torch.tensor(quality_loss + violation_loss, dtype=torch.float32)


class VesselContinuityLoss(nn.Module):
    """Loss function for vessel continuity"""
    
    def forward(self, continuity_analysis: Dict) -> torch.Tensor:
        """Compute vessel continuity loss"""
        if not isinstance(continuity_analysis, dict):
            return torch.tensor(0.0)
        
        avg_continuity = continuity_analysis.get('avg_continuity', 1.0)
        num_violations = continuity_analysis.get('num_continuity_violations', 0)
        total_edges = len(continuity_analysis.get('continuity_scores', [1]))
        
        # Continuity loss
        continuity_loss = 1.0 - avg_continuity
        
        # Violation penalty
        violation_ratio = num_violations / max(total_edges, 1)
        violation_loss = violation_ratio
        
        return torch.tensor(continuity_loss + violation_loss, dtype=torch.float32)


class FeatureConsistencyLoss(nn.Module):
    """Loss function for feature consistency"""
    
    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        Compute feature consistency loss
        
        Args:
            node_features: [N, feature_dim] node features
            
        Returns:
            Feature consistency loss
        """
        # Variance regularization - encourage consistent feature magnitudes
        feature_vars = torch.var(node_features, dim=0)
        variance_loss = torch.mean(feature_vars)
        
        # Extreme value penalty
        feature_norms = torch.norm(node_features, dim=-1)
        extreme_penalty = torch.mean(torch.clamp(feature_norms - 10.0, min=0.0))
        
        return variance_loss + extreme_penalty


class SmoothnessLoss(nn.Module):
    """Loss function for feature smoothness across edges"""
    
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute smoothness loss - features should vary smoothly along edges
        
        Args:
            node_features: [N, feature_dim] node features
            edge_index: [2, E] edge connectivity
            
        Returns:
            Smoothness loss
        """
        if edge_index.size(1) == 0:
            return torch.tensor(0.0, device=node_features.device)
        
        # Compute feature differences along edges
        src_features = node_features[edge_index[0]]  # [E, feature_dim]
        tgt_features = node_features[edge_index[1]]  # [E, feature_dim]
        
        feature_diffs = torch.norm(src_features - tgt_features, dim=-1)  # [E]
        smoothness_loss = torch.mean(feature_diffs)
        
        return smoothness_loss


class QualityImprovementLoss(nn.Module):
    """Loss function that encourages overall quality improvement"""
    
    def __init__(self, target_quality: float = 0.8):
        super().__init__()
        self.target_quality = target_quality
    
    def forward(self, quality_score: torch.Tensor) -> torch.Tensor:
        """
        Compute quality improvement loss
        
        Args:
            quality_score: Predicted quality score [0, 1]
            
        Returns:
            Quality improvement loss
        """
        # Encourage high quality scores
        quality_loss = torch.clamp(self.target_quality - quality_score, min=0.0)
        
        return quality_loss


class RegularizationLoss(nn.Module):
    """General regularization loss"""
    
    def __init__(self, l1_weight: float = 0.01, l2_weight: float = 0.01):
        super().__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute regularization loss
        
        Args:
            features: Feature tensor to regularize
            
        Returns:
            Regularization loss
        """
        l1_loss = torch.norm(features, p=1)
        l2_loss = torch.norm(features, p=2)
        
        return self.l1_weight * l1_loss + self.l2_weight * l2_loss


class AdaptiveLossWeighting(nn.Module):
    """
    Adaptive loss weighting that adjusts weights during training
    Based on loss magnitudes and training progress
    """
    
    def __init__(self, initial_weights: Dict[str, float]):
        super().__init__()
        
        self.initial_weights = initial_weights
        self.weight_history = {key: [] for key in initial_weights}
        self.current_weights = initial_weights.copy()
        
    def update_weights(self, loss_dict: Dict[str, torch.Tensor], epoch: int):
        """Update loss weights based on training progress"""
        
        # Store loss magnitudes
        for key, loss in loss_dict.items():
            if key in self.weight_history:
                self.weight_history[key].append(loss.item())
        
        # Adaptive weighting based on loss balance
        if epoch > 10:  # Start adapting after initial epochs
            loss_means = {}
            for key, history in self.weight_history.items():
                if len(history) >= 10:
                    loss_means[key] = np.mean(history[-10:])  # Recent average
            
            if loss_means:
                # Normalize weights based on relative loss magnitudes
                total_loss = sum(loss_means.values())
                for key in self.current_weights:
                    if key in loss_means:
                        # Inverse relationship: higher loss gets lower weight
                        relative_magnitude = loss_means[key] / total_loss
                        adaptation_factor = 1.0 / (1.0 + relative_magnitude)
                        self.current_weights[key] = (
                            0.9 * self.current_weights[key] + 
                            0.1 * self.initial_weights[key] * adaptation_factor
                        )
    
    def get_weights(self) -> Dict[str, float]:
        """Get current adaptive weights"""
        return self.current_weights.copy()


class CurriculumLoss(nn.Module):
    """
    Curriculum learning loss that adjusts difficulty over training
    """
    
    def __init__(self, base_loss_fn: nn.Module):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.current_difficulty = 0.0  # [0, 1] - 0 = easy, 1 = hard
        
    def set_difficulty(self, difficulty: float):
        """Set current curriculum difficulty"""
        self.current_difficulty = np.clip(difficulty, 0.0, 1.0)
    
    def forward(self, predictions, targets, correspondences, metadata=None):
        """
        Compute curriculum-adjusted loss
        """
        # Compute base loss
        base_losses = self.base_loss_fn(predictions, targets, correspondences, metadata)
        
        # Adjust loss weights based on difficulty
        difficulty_weights = {
            'topology': 1.0,  # Always important
            'anatomy': 0.5 + 0.5 * self.current_difficulty,  # Gradually increase
            'consistency': 0.3 + 0.7 * self.current_difficulty,  # More important later
            'improvement': 1.0 + self.current_difficulty,  # Increase emphasis
            'regularization': 0.2 - 0.1 * self.current_difficulty  # Decrease over time
        }
        
        # Apply curriculum weights
        adjusted_losses = {}
        for key, loss in base_losses.items():
            if key in difficulty_weights:
                adjusted_losses[key] = difficulty_weights[key] * loss
            else:
                adjusted_losses[key] = loss
        
        # Recompute total
        adjusted_losses['total'] = sum(
            loss for key, loss in adjusted_losses.items() if key != 'total'
        )
        
        return adjusted_losses