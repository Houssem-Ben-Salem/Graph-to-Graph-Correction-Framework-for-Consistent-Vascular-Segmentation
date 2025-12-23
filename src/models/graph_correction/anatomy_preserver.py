"""
Anatomy Preservation Module
Implements physiological constraint enforcement for vascular graphs including Murray's law
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional
import math
import numpy as np


class AnatomyPreserver(nn.Module):
    """
    Main anatomy preservation module that enforces:
    - Murray's law at bifurcations
    - Vessel tapering consistency
    - Realistic branching angles
    - Vessel continuity
    """
    
    def __init__(self, 
                 feature_dim: int = 128,
                 config: Optional[Dict] = None):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.config = config or {}
        
        # Murray's law enforcer
        self.murray_law_enforcer = MurrayLawModule(feature_dim)
        
        # Vessel tapering consistency module
        self.tapering_consistency_module = TaperingModule(feature_dim)
        
        # Branching angle validator
        self.angle_validator = BranchingAngleModule(feature_dim)
        
        # Vessel continuity enforcer
        self.continuity_enforcer = VesselContinuityModule(feature_dim)
        
        # Radius corrector
        self.radius_corrector = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Radius correction factor
        )
        
        # Anatomical quality assessor
        self.quality_assessor = nn.Sequential(
            nn.Linear(feature_dim * 4, 128),  # Combined anatomy features
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, 
                node_features, 
                edge_index, 
                edge_features, 
                batch=None,
                node_positions=None,
                node_radii=None,
                node_types=None):
        """
        Forward pass for anatomy preservation
        
        Args:
            node_features: [N, feature_dim] node features
            edge_index: [2, E] edge connectivity
            edge_features: [E, edge_feature_dim] edge features
            batch: Batch assignment
            node_positions: [N, 3] node positions
            node_radii: [N] node radii
            node_types: [N] node type indicators
            
        Returns:
            Updated node features and anatomy analysis
        """
        # Murray's law enforcement
        murray_features, murray_analysis = self.murray_law_enforcer(
            node_features, edge_index, node_radii, node_types
        )
        
        # Vessel tapering analysis
        tapering_features, tapering_analysis = self.tapering_consistency_module(
            node_features, edge_index, node_positions, node_radii
        )
        
        # Branching angle validation
        angle_features, angle_analysis = self.angle_validator(
            node_features, edge_index, node_positions, node_types
        )
        
        # Vessel continuity enforcement
        continuity_features, continuity_analysis = self.continuity_enforcer(
            node_features, edge_index, node_positions, node_radii
        )
        
        # Combine anatomy features
        combined_features = torch.cat([
            murray_features,
            tapering_features, 
            angle_features,
            continuity_features
        ], dim=-1)
        
        # Assess anatomical quality
        anatomy_quality = self.quality_assessor(combined_features)
        
        # Generate radius corrections
        radius_corrections = self.radius_corrector(node_features)
        
        # Update node features with anatomy-aware information
        updated_features = node_features + 0.1 * (
            murray_features + tapering_features + 
            angle_features + continuity_features
        ) / 4
        
        anatomy_outputs = {
            'murray_analysis': murray_analysis,
            'tapering_analysis': tapering_analysis,
            'angle_analysis': angle_analysis,
            'continuity_analysis': continuity_analysis,
            'anatomy_quality': anatomy_quality,
            'radius_corrections': radius_corrections,
            'combined_anatomy_features': combined_features
        }
        
        return updated_features, anatomy_outputs


class MurrayLawModule(nn.Module):
    """
    Enforces Murray's law: r_parent³ = Σ r_child³ at bifurcations
    """
    
    def __init__(self, feature_dim: int):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.violation_threshold = 0.2  # 20% violation threshold
        
        # Radius corrector for Murray's law violations
        self.radius_corrector = nn.Sequential(
            nn.Linear(feature_dim * 3, 64),  # Parent + 2 children features
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Radius correction factor
        )
        
        # Murray's law compliance predictor
        self.compliance_predictor = nn.Sequential(
            nn.Linear(feature_dim * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, node_features, edge_index, node_radii, node_types):
        """
        Enforce Murray's law at bifurcation points
        
        Args:
            node_features: [N, feature_dim] node features
            edge_index: [2, E] edge connectivity
            node_radii: [N] node radii (if available)
            node_types: [N] node type indicators
            
        Returns:
            Updated features and Murray's law analysis
        """
        num_nodes = node_features.size(0)
        device = node_features.device
        
        # Initialize outputs
        updated_features = node_features.clone()
        murray_violations = []
        correction_factors = []
        compliance_scores = []
        
        # Find bifurcation nodes
        if node_types is not None:
            bifurcation_mask = (node_types == 1)  # Assuming 1 = bifurcation
            bifurcation_indices = torch.where(bifurcation_mask)[0]
        else:
            # Find nodes with degree > 2 (potential bifurcations)
            degrees = torch.zeros(num_nodes, device=device)
            for i in range(num_nodes):
                degree = ((edge_index[0] == i) | (edge_index[1] == i)).sum()
                degrees[i] = degree.float()
            bifurcation_indices = torch.where(degrees > 2)[0]
        
        for bifurc_idx in bifurcation_indices:
            # Find connected nodes
            connected_edges = (edge_index[0] == bifurc_idx) | (edge_index[1] == bifurc_idx)
            connected_nodes = edge_index[:, connected_edges].flatten()
            connected_nodes = connected_nodes[connected_nodes != bifurc_idx].unique()
            
            if len(connected_nodes) >= 2:
                # Get features for bifurcation analysis
                bifurc_features = node_features[bifurc_idx]
                
                # Take first two connected nodes (simplified)
                child1_features = node_features[connected_nodes[0]]
                child2_features = node_features[connected_nodes[1]] if len(connected_nodes) > 1 else child1_features
                
                # Combine features for analysis
                combined_features = torch.cat([bifurc_features, child1_features, child2_features])
                
                # Predict compliance
                compliance = self.compliance_predictor(combined_features.unsqueeze(0))
                compliance_scores.append(compliance.item())
                
                # If we have actual radii, check Murray's law violation
                if node_radii is not None:
                    parent_radius = node_radii[bifurc_idx]
                    child1_radius = node_radii[connected_nodes[0]]
                    child2_radius = node_radii[connected_nodes[1]] if len(connected_nodes) > 1 else child1_radius
                    
                    # Murray's law: r_parent³ = r_child1³ + r_child2³
                    expected_parent_cubed = child1_radius**3 + child2_radius**3
                    actual_parent_cubed = parent_radius**3
                    
                    violation = abs(expected_parent_cubed - actual_parent_cubed) / actual_parent_cubed
                    murray_violations.append(violation.item())
                    
                    # Generate correction if violation is significant
                    if violation > self.violation_threshold:
                        correction_factor = self.radius_corrector(combined_features.unsqueeze(0))
                        correction_factors.append(correction_factor.item())
                        
                        # Update features with correction information
                        correction_signal = correction_factor * torch.tanh(combined_features)
                        # Ensure correction signal has the right shape [feature_dim]
                        if correction_signal.dim() > 1:
                            correction_signal = correction_signal.squeeze(0)
                        if correction_signal.size(-1) != self.feature_dim:
                            correction_signal = correction_signal[:self.feature_dim]
                        updated_features[bifurc_idx] += 0.1 * correction_signal
                else:
                    # Without actual radii, use feature-based prediction
                    correction_factor = self.radius_corrector(combined_features.unsqueeze(0))
                    correction_factors.append(correction_factor.item())
                    murray_violations.append(0.5)  # Neutral violation score
        
        analysis = {
            'num_bifurcations': len(bifurcation_indices),
            'murray_violations': murray_violations,
            'avg_violation': np.mean(murray_violations) if murray_violations else 0.0,
            'correction_factors': correction_factors,
            'compliance_scores': compliance_scores,
            'avg_compliance': np.mean(compliance_scores) if compliance_scores else 0.5
        }
        
        return updated_features, analysis


class TaperingModule(nn.Module):
    """
    Enforces realistic vessel tapering (gradual radius decrease)
    """
    
    def __init__(self, feature_dim: int):
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # Tapering analysis network
        self.tapering_analyzer = nn.Sequential(
            nn.Linear(feature_dim * 2, 64),  # Current and downstream features
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, node_features, edge_index, node_positions, node_radii):
        """
        Analyze and enforce vessel tapering consistency
        """
        num_nodes = node_features.size(0)
        device = node_features.device
        
        updated_features = node_features.clone()
        tapering_scores = []
        
        for i in range(num_nodes):
            # Find downstream neighbors (simplified: all neighbors)
            neighbors = self._find_neighbors(i, edge_index)
            
            if neighbors:
                current_features = node_features[i]
                neighbor_features = node_features[neighbors].mean(dim=0)
                
                combined_features = torch.cat([current_features, neighbor_features])
                tapering_score = self.tapering_analyzer(combined_features.unsqueeze(0))
                tapering_scores.append(tapering_score.item())
                
                # Update features based on tapering consistency
                tapering_signal = tapering_score * torch.tanh(combined_features)
                # Ensure tapering signal has the right shape [feature_dim]
                if tapering_signal.dim() > 1:
                    tapering_signal = tapering_signal.squeeze(0)
                if tapering_signal.size(-1) != self.feature_dim:
                    tapering_signal = tapering_signal[:self.feature_dim]
                updated_features[i] += 0.05 * tapering_signal
            else:
                tapering_scores.append(1.0)  # Endpoints have perfect tapering
        
        analysis = {
            'tapering_scores': tapering_scores,
            'avg_tapering_consistency': np.mean(tapering_scores),
            'num_tapering_violations': sum(1 for score in tapering_scores if score < 0.7)
        }
        
        return updated_features, analysis
    
    def _find_neighbors(self, node_idx, edge_index):
        """Find neighbors of a node"""
        neighbors = []
        for i in range(edge_index.size(1)):
            if edge_index[0, i] == node_idx:
                neighbors.append(edge_index[1, i].item())
            elif edge_index[1, i] == node_idx:
                neighbors.append(edge_index[0, i].item())
        return list(set(neighbors))


class BranchingAngleModule(nn.Module):
    """
    Validates and corrects branching angles at bifurcations
    """
    
    def __init__(self, feature_dim: int):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.optimal_angle_range = (30, 60)  # Degrees
        
        # Angle analysis network
        self.angle_analyzer = nn.Sequential(
            nn.Linear(feature_dim * 3 + 3, 64),  # 3 nodes + angle info
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, node_features, edge_index, node_positions, node_types):
        """
        Analyze branching angles at bifurcations
        """
        num_nodes = node_features.size(0)
        device = node_features.device
        
        updated_features = node_features.clone()
        angle_scores = []
        angle_violations = []
        
        # Find bifurcation nodes
        if node_types is not None:
            bifurcation_indices = torch.where(node_types == 1)[0]
        else:
            degrees = torch.zeros(num_nodes, device=device)
            for i in range(num_nodes):
                degree = ((edge_index[0] == i) | (edge_index[1] == i)).sum()
                degrees[i] = degree.float()
            bifurcation_indices = torch.where(degrees > 2)[0]
        
        for bifurc_idx in bifurcation_indices:
            if node_positions is not None:
                # Calculate actual branching angles
                angle_score, angle_violation = self._analyze_bifurcation_angles(
                    bifurc_idx, edge_index, node_positions, node_features
                )
                angle_scores.append(angle_score)
                angle_violations.append(angle_violation)
            else:
                # Feature-based angle analysis
                connected_nodes = self._find_connected_nodes(bifurc_idx, edge_index)
                if len(connected_nodes) >= 2:
                    bifurc_features = node_features[bifurc_idx]
                    child1_features = node_features[connected_nodes[0]]
                    child2_features = node_features[connected_nodes[1]]
                    
                    # Dummy angle info
                    angle_info = torch.tensor([45.0], device=device)  # Assume optimal angle
                    
                    combined_input = torch.cat([bifurc_features, child1_features, child2_features, angle_info])
                    angle_score = self.angle_analyzer(combined_input.unsqueeze(0))
                    
                    angle_scores.append(angle_score.item())
                    angle_violations.append(0.0 if angle_score.item() > 0.7 else 1.0)
        
        analysis = {
            'angle_scores': angle_scores,
            'angle_violations': angle_violations,
            'avg_angle_quality': np.mean(angle_scores) if angle_scores else 1.0,
            'num_angle_violations': len([v for v in angle_violations if v > 0.5])
        }
        
        return updated_features, analysis
    
    def _analyze_bifurcation_angles(self, bifurc_idx, edge_index, node_positions, node_features):
        """Analyze angles at a specific bifurcation"""
        connected_nodes = self._find_connected_nodes(bifurc_idx, edge_index)
        
        if len(connected_nodes) < 2 or node_positions is None:
            return 1.0, 0.0
        
        # Get positions
        bifurc_pos = node_positions[bifurc_idx]
        child1_pos = node_positions[connected_nodes[0]]
        child2_pos = node_positions[connected_nodes[1]]
        
        # Calculate vectors
        vec1 = child1_pos - bifurc_pos
        vec2 = child2_pos - bifurc_pos
        
        # Calculate angle
        cos_angle = torch.dot(vec1, vec2) / (torch.norm(vec1) * torch.norm(vec2))
        angle_rad = torch.acos(torch.clamp(cos_angle, -1, 1))
        angle_deg = angle_rad * 180 / math.pi
        
        # Check if angle is in optimal range
        optimal_min, optimal_max = self.optimal_angle_range
        if optimal_min <= angle_deg <= optimal_max:
            angle_score = 1.0
            violation = 0.0
        else:
            # Score based on distance from optimal range
            if angle_deg < optimal_min:
                distance = optimal_min - angle_deg
            else:
                distance = angle_deg - optimal_max
            
            angle_score = max(0.0, 1.0 - distance / 30.0)  # Normalize by 30 degrees
            violation = 1.0 - angle_score
        
        return angle_score, violation
    
    def _find_connected_nodes(self, node_idx, edge_index):
        """Find nodes connected to a given node"""
        connected = []
        for i in range(edge_index.size(1)):
            if edge_index[0, i] == node_idx:
                connected.append(edge_index[1, i].item())
            elif edge_index[1, i] == node_idx:
                connected.append(edge_index[0, i].item())
        return list(set(connected))


class VesselContinuityModule(nn.Module):
    """
    Enforces smooth transitions between vessel segments
    """
    
    def __init__(self, feature_dim: int):
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # Continuity analyzer
        self.continuity_analyzer = nn.Sequential(
            nn.Linear(feature_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, node_features, edge_index, node_positions, node_radii):
        """
        Analyze vessel continuity along edges
        """
        num_edges = edge_index.size(1)
        device = node_features.device
        
        updated_features = node_features.clone()
        continuity_scores = []
        
        for i in range(num_edges):
            src_idx = edge_index[0, i]
            tgt_idx = edge_index[1, i]
            
            src_features = node_features[src_idx]
            tgt_features = node_features[tgt_idx]
            
            # Analyze continuity between connected nodes
            combined_features = torch.cat([src_features, tgt_features])
            continuity_score = self.continuity_analyzer(combined_features.unsqueeze(0))
            continuity_scores.append(continuity_score.item())
            
            # Update features to improve continuity
            if continuity_score.item() < 0.7:  # Poor continuity
                continuity_signal = (1.0 - continuity_score) * torch.tanh(combined_features)
                # Ensure continuity signal has the right shape
                if continuity_signal.dim() > 1:
                    continuity_signal = continuity_signal.squeeze(0)
                # Split signal for source and target features
                src_signal = continuity_signal[:self.feature_dim]
                tgt_signal = continuity_signal[self.feature_dim:self.feature_dim*2] if continuity_signal.size(0) >= self.feature_dim*2 else continuity_signal[:self.feature_dim]
                updated_features[src_idx] += 0.05 * src_signal
                updated_features[tgt_idx] += 0.05 * tgt_signal
        
        analysis = {
            'continuity_scores': continuity_scores,
            'avg_continuity': np.mean(continuity_scores) if continuity_scores else 1.0,
            'num_continuity_violations': sum(1 for score in continuity_scores if score < 0.7)
        }
        
        return updated_features, analysis