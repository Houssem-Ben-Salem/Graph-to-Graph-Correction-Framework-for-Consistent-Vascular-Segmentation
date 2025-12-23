"""
Comprehensive metrics for vascular segmentation evaluation.

This module implements all metrics needed for comparing traditional post-processing,
learning-based methods, and the graph-to-graph correction framework.
"""

import numpy as np
import torch
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff
from skimage import measure, morphology
try:
    from skimage.morphology import skeletonize_3d
except ImportError:
    # In newer versions of scikit-image, it's just called skeletonize
    from skimage.morphology import skeletonize as skeletonize_3d
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union
import warnings
from collections import defaultdict
from tqdm import tqdm

# Import existing basic metrics
from .metrics import (
    compute_dice_score,
    compute_jaccard_score,
    compute_hausdorff_distance,
    compute_sensitivity,
    compute_specificity,
    compute_precision,
    compute_volume_similarity,
    compute_average_symmetric_surface_distance
)


class TopologicalMetrics:
    """Compute topological and structural metrics for vascular segmentation."""
    
    def __init__(self, voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        """
        Initialize topological metrics calculator.
        
        Args:
            voxel_spacing: Physical spacing between voxels (z, y, x)
        """
        self.voxel_spacing = np.array(voxel_spacing)
    
    def compute_connected_components_count(
        self, 
        mask: np.ndarray,
        connectivity: int = 3
    ) -> Dict[str, Union[int, float]]:
        """
        Count connected components in the mask.
        
        Args:
            mask: Binary mask
            connectivity: Connectivity for connected components (1, 2, or 3 for 3D)
            
        Returns:
            Dictionary with component statistics
        """
        labeled_mask = measure.label(mask > 0.5, connectivity=connectivity)
        num_components = labeled_mask.max()
        
        if num_components == 0:
            return {
                'num_components': 0,
                'largest_component_ratio': 0.0,
                'component_size_std': 0.0,
                'total_volume': 0
            }
        
        # Get component sizes
        component_sizes = np.bincount(labeled_mask.ravel())[1:]  # Exclude background
        total_volume = int(mask.sum())
        
        return {
            'num_components': num_components,
            'largest_component_ratio': float(component_sizes.max() / total_volume),
            'component_size_std': float(np.std(component_sizes)),
            'total_volume': total_volume,
            'component_sizes': component_sizes.tolist()
        }
    
    def compute_connectivity_aware_dice(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
        connectivity_weight: float = 0.3
    ) -> float:
        """
        Compute connectivity-aware Dice score (cDice).
        
        Combines standard Dice with a connectivity penalty based on 
        topological differences.
        
        Args:
            pred_mask: Predicted binary mask
            gt_mask: Ground truth binary mask  
            connectivity_weight: Weight for connectivity component (0-1)
            
        Returns:
            Connectivity-aware Dice score
        """
        # Standard Dice score
        dice = compute_dice_score(pred_mask, gt_mask)
        
        if dice == 0:  # No overlap, connectivity doesn't matter
            return 0.0
        
        # Connectivity component
        pred_cc = self.compute_connected_components_count(pred_mask)
        gt_cc = self.compute_connected_components_count(gt_mask)
        
        # Connectivity score based on component count difference
        if gt_cc['num_components'] == 0:
            connectivity_score = 1.0 if pred_cc['num_components'] == 0 else 0.0
        else:
            cc_diff = abs(pred_cc['num_components'] - gt_cc['num_components'])
            connectivity_score = max(0.0, 1.0 - cc_diff / max(gt_cc['num_components'], 1))
        
        # Combine Dice and connectivity
        cdice = (1 - connectivity_weight) * dice + connectivity_weight * connectivity_score
        
        return float(cdice)
    
    def extract_skeleton(self, mask: np.ndarray) -> np.ndarray:
        """Extract 3D skeleton from binary mask."""
        if mask.ndim == 2:
            return morphology.skeletonize(mask > 0.5)
        else:
            return skeletonize_3d(mask > 0.5)
    
    def skeleton_to_graph(self, skeleton: np.ndarray) -> nx.Graph:
        """Convert 3D skeleton to NetworkX graph."""
        # Find skeleton points
        skeleton_points = np.where(skeleton)
        
        if len(skeleton_points[0]) == 0:
            return nx.Graph()
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes
        for i in range(len(skeleton_points[0])):
            node_id = i
            coords = tuple(skeleton_points[j][i] for j in range(len(skeleton_points)))
            G.add_node(node_id, coords=coords)
        
        # Add edges based on adjacency
        coords_to_id = {}
        for i, node in enumerate(G.nodes()):
            coords = G.nodes[node]['coords']
            coords_to_id[coords] = node
        
        # Check 26-connectivity for 3D or 8-connectivity for 2D
        if skeleton.ndim == 3:
            offsets = [(dx, dy, dz) for dx in [-1, 0, 1] for dy in [-1, 0, 1] 
                      for dz in [-1, 0, 1] if not (dx == dy == dz == 0)]
        else:
            offsets = [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] 
                      if not (dx == dy == 0)]
        
        for node in G.nodes():
            coords = G.nodes[node]['coords']
            for offset in offsets:
                neighbor_coords = tuple(coords[i] + offset[i] for i in range(len(coords)))
                if neighbor_coords in coords_to_id:
                    neighbor_node = coords_to_id[neighbor_coords]
                    if not G.has_edge(node, neighbor_node):
                        # Compute edge length
                        dist = np.linalg.norm(
                            np.array(neighbor_coords) - np.array(coords)
                        )
                        G.add_edge(node, neighbor_node, length=dist)
        
        return G
    
    def compute_vessel_tree_isomorphism(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute vessel tree isomorphism score.
        
        This measures structural similarity between predicted and ground truth
        vessel trees by comparing their graph representations.
        """
        try:
            # Extract skeletons
            pred_skeleton = self.extract_skeleton(pred_mask)
            gt_skeleton = self.extract_skeleton(gt_mask)
            
            # Convert to graphs
            pred_graph = self.skeleton_to_graph(pred_skeleton)
            gt_graph = self.skeleton_to_graph(gt_skeleton)
            
            if len(pred_graph.nodes) == 0 and len(gt_graph.nodes) == 0:
                return {'tree_isomorphism': 1.0, 'node_count_ratio': 1.0, 'edge_count_ratio': 1.0}
            
            if len(pred_graph.nodes) == 0 or len(gt_graph.nodes) == 0:
                return {'tree_isomorphism': 0.0, 'node_count_ratio': 0.0, 'edge_count_ratio': 0.0}
            
            # Basic structural similarity
            node_ratio = min(len(pred_graph.nodes), len(gt_graph.nodes)) / max(len(pred_graph.nodes), len(gt_graph.nodes))
            edge_ratio = min(len(pred_graph.edges), len(gt_graph.edges)) / max(len(pred_graph.edges), len(gt_graph.edges))
            
            # Degree distribution similarity
            pred_degrees = [d for n, d in pred_graph.degree()]
            gt_degrees = [d for n, d in gt_graph.degree()]
            
            # Simple isomorphism score based on structural properties
            degree_similarity = self._compute_degree_distribution_similarity(pred_degrees, gt_degrees)
            
            tree_iso_score = (node_ratio + edge_ratio + degree_similarity) / 3.0
            
            return {
                'tree_isomorphism': float(tree_iso_score),
                'node_count_ratio': float(node_ratio),
                'edge_count_ratio': float(edge_ratio),
                'degree_similarity': float(degree_similarity)
            }
            
        except Exception as e:
            warnings.warn(f"Error computing vessel tree isomorphism: {e}")
            return {'tree_isomorphism': 0.0, 'node_count_ratio': 0.0, 'edge_count_ratio': 0.0}
    
    def _compute_degree_distribution_similarity(self, degrees1: List[int], degrees2: List[int]) -> float:
        """Compute similarity between degree distributions."""
        if not degrees1 and not degrees2:
            return 1.0
        if not degrees1 or not degrees2:
            return 0.0
        
        # Convert to histograms
        max_degree = max(max(degrees1), max(degrees2))
        hist1 = np.histogram(degrees1, bins=range(max_degree + 2), density=True)[0]
        hist2 = np.histogram(degrees2, bins=range(max_degree + 2), density=True)[0]
        
        # Compute intersection over union
        intersection = np.minimum(hist1, hist2).sum()
        union = np.maximum(hist1, hist2).sum()
        
        return intersection / union if union > 0 else 0.0
    
    def compute_bifurcation_detection_accuracy(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
        tolerance: float = 3.0
    ) -> Dict[str, float]:
        """
        Compute bifurcation detection accuracy.
        
        Args:
            pred_mask: Predicted binary mask
            gt_mask: Ground truth binary mask
            tolerance: Maximum distance to consider a bifurcation as correctly detected
            
        Returns:
            Dictionary with bifurcation detection metrics
        """
        try:
            # Extract skeletons
            pred_skeleton = self.extract_skeleton(pred_mask)
            gt_skeleton = self.extract_skeleton(gt_mask)
            
            # Convert to graphs
            pred_graph = self.skeleton_to_graph(pred_skeleton)
            gt_graph = self.skeleton_to_graph(gt_skeleton)
            
            # Find bifurcations (nodes with degree > 2)
            pred_bifurcations = [n for n, d in pred_graph.degree() if d > 2]
            gt_bifurcations = [n for n, d in gt_graph.degree() if d > 2]
            
            if len(gt_bifurcations) == 0:
                # No bifurcations in ground truth
                precision = 1.0 if len(pred_bifurcations) == 0 else 0.0
                recall = 1.0
                f1 = 1.0
            else:
                # Get coordinates of bifurcations
                pred_coords = [pred_graph.nodes[n]['coords'] for n in pred_bifurcations]
                gt_coords = [gt_graph.nodes[n]['coords'] for n in gt_bifurcations]
                
                if len(pred_bifurcations) == 0:
                    precision = 0.0
                    recall = 0.0
                    f1 = 0.0
                else:
                    # Match bifurcations based on distance
                    matches = self._match_points(pred_coords, gt_coords, tolerance)
                    
                    true_positives = len(matches)
                    precision = true_positives / len(pred_bifurcations)
                    recall = true_positives / len(gt_bifurcations)
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            return {
                'bifurcation_precision': float(precision),
                'bifurcation_recall': float(recall),
                'bifurcation_f1': float(f1),
                'pred_bifurcations': len(pred_bifurcations),
                'gt_bifurcations': len(gt_bifurcations)
            }
            
        except Exception as e:
            warnings.warn(f"Error computing bifurcation detection accuracy: {e}")
            return {
                'bifurcation_precision': 0.0,
                'bifurcation_recall': 0.0,
                'bifurcation_f1': 0.0,
                'pred_bifurcations': 0,
                'gt_bifurcations': 0
            }
    
    def _match_points(self, points1: List, points2: List, tolerance: float) -> List[Tuple]:
        """Match points between two sets based on distance threshold."""
        matches = []
        used_indices = set()
        
        for i, p1 in enumerate(points1):
            best_match = None
            best_distance = float('inf')
            
            for j, p2 in enumerate(points2):
                if j in used_indices:
                    continue
                
                distance = np.linalg.norm(np.array(p1) - np.array(p2))
                if distance < tolerance and distance < best_distance:
                    best_distance = distance
                    best_match = j
            
            if best_match is not None:
                matches.append((i, best_match))
                used_indices.add(best_match)
        
        return matches


class AnatomicalMetrics:
    """Compute anatomical consistency metrics for vascular segmentation."""
    
    def __init__(self, voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        """
        Initialize anatomical metrics calculator.
        
        Args:
            voxel_spacing: Physical spacing between voxels (z, y, x)
        """
        self.voxel_spacing = np.array(voxel_spacing)
    
    def compute_murrays_law_compliance(
        self,
        mask: np.ndarray,
        skeleton: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute Murray's law compliance score.
        
        Murray's law states that at bifurcations: r_parent^3 ≈ r_child1^3 + r_child2^3
        where r is the vessel radius.
        
        Args:
            mask: Binary mask
            skeleton: Optional pre-computed skeleton
            
        Returns:
            Dictionary with Murray's law compliance metrics
        """
        try:
            if skeleton is None:
                print("      [1/4] Skeletonizing mask...", flush=True)
                if mask.ndim == 2:
                    skeleton = morphology.skeletonize(mask > 0.5)
                else:
                    skeleton = skeletonize_3d(mask > 0.5)

            # Compute distance transform for radius estimation
            print("      [2/4] Computing distance transform...", flush=True)
            distance_transform = ndimage.distance_transform_edt(mask > 0.5)

            # Extract skeleton points and their radii
            skeleton_points = np.where(skeleton)
            if len(skeleton_points[0]) == 0:
                return {'murrays_law_score': 1.0, 'num_bifurcations': 0, 'compliance_violations': 0}

            # Create graph from skeleton
            print("      [3/4] Building graph from skeleton...", flush=True)
            topo_metrics = TopologicalMetrics(self.voxel_spacing)
            graph = topo_metrics.skeleton_to_graph(skeleton)

            # Find bifurcations (degree > 2)
            bifurcations = [n for n, d in graph.degree() if d > 2]

            if len(bifurcations) == 0:
                return {'murrays_law_score': 1.0, 'num_bifurcations': 0, 'compliance_violations': 0}

            compliance_scores = []

            print(f"      [4/4] Analyzing Murray's law at {len(bifurcations)} bifurcations...", flush=True)
            for bifurcation in tqdm(bifurcations, desc="      Murray's Law", leave=False):
                coords = graph.nodes[bifurcation]['coords']
                radius = distance_transform[coords]
                
                # Get neighboring nodes
                neighbors = list(graph.neighbors(bifurcation))
                neighbor_radii = []
                
                for neighbor in neighbors:
                    neighbor_coords = graph.nodes[neighbor]['coords']
                    neighbor_radius = distance_transform[neighbor_coords]
                    neighbor_radii.append(neighbor_radius)
                
                if len(neighbor_radii) >= 2:
                    # Apply Murray's law
                    # Assume the largest radius is the parent vessel
                    neighbor_radii.sort(reverse=True)
                    parent_radius = neighbor_radii[0]
                    child_radii = neighbor_radii[1:]
                    
                    # Murray's law: r_parent^3 = sum(r_child^3)
                    expected_parent_cubed = sum(r**3 for r in child_radii)
                    actual_parent_cubed = parent_radius**3
                    
                    if expected_parent_cubed > 0:
                        compliance = 1.0 - abs(actual_parent_cubed - expected_parent_cubed) / expected_parent_cubed
                        compliance_scores.append(max(0.0, compliance))
            
            if compliance_scores:
                mean_compliance = np.mean(compliance_scores)
                violations = sum(1 for score in compliance_scores if score < 0.8)
            else:
                mean_compliance = 1.0
                violations = 0
            
            return {
                'murrays_law_score': float(mean_compliance),
                'num_bifurcations': len(bifurcations),
                'compliance_violations': violations
            }
            
        except Exception as e:
            warnings.warn(f"Error computing Murray's law compliance: {e}")
            return {'murrays_law_score': 0.0, 'num_bifurcations': 0, 'compliance_violations': 0}
    
    def compute_vessel_tapering_consistency(
        self,
        mask: np.ndarray,
        skeleton: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute vessel tapering consistency score.
        
        Vessels should generally taper (get narrower) as they move away from the root.
        
        Args:
            mask: Binary mask
            skeleton: Optional pre-computed skeleton
            
        Returns:
            Dictionary with tapering consistency metrics
        """
        try:
            if skeleton is None:
                print("      [1/3] Skeletonizing for tapering...", flush=True)
                if mask.ndim == 2:
                    skeleton = morphology.skeletonize(mask > 0.5)
                else:
                    skeleton = skeletonize_3d(mask > 0.5)

            # Compute distance transform for radius estimation
            print("      [2/3] Computing distance transform...", flush=True)
            distance_transform = ndimage.distance_transform_edt(mask > 0.5)

            # Create graph from skeleton
            print("      [3/3] Building graph...", flush=True)
            topo_metrics = TopologicalMetrics(self.voxel_spacing)
            graph = topo_metrics.skeleton_to_graph(skeleton)

            if len(graph.nodes) < 3:
                return {'tapering_score': 1.0, 'monotonic_paths': 0, 'total_paths': 0}

            # Find paths from high-degree nodes to endpoints
            high_degree_nodes = [n for n, d in graph.degree() if d > 1]
            endpoints = [n for n, d in graph.degree() if d == 1]

            tapering_scores = []
            total_paths = 0

            # OPTIMIZATION: Sample paths instead of checking all combinations
            # For large graphs, checking all paths is prohibitively expensive (can be millions)
            # Sampling 1000 random paths provides statistically valid results
            import random

            max_paths_to_check = 1000  # Sample size - sufficient for statistical validity
            total_possible_paths = len(high_degree_nodes) * len(endpoints)

            if total_possible_paths > max_paths_to_check:
                print(f"      OPTIMIZATION: Sampling {max_paths_to_check} random paths (out of {total_possible_paths:,} possible)...", flush=True)
                # Create all possible pairs and sample randomly
                all_pairs = [(s, e) for s in high_degree_nodes for e in endpoints]
                sampled_pairs = random.sample(all_pairs, max_paths_to_check)
            else:
                print(f"      Analyzing all {total_possible_paths:,} paths...", flush=True)
                sampled_pairs = [(s, e) for s in high_degree_nodes for e in endpoints]

            for start_node, end_node in tqdm(sampled_pairs, desc="      Tapering", leave=False):
                try:
                    path = nx.shortest_path(graph, start_node, end_node)
                    if len(path) > 2:  # Need at least 3 points for tapering
                        total_paths += 1

                        # Get radii along the path
                        radii = []
                        for node in path:
                            coords = graph.nodes[node]['coords']
                            radius = distance_transform[coords]
                            radii.append(radius)

                        # Check if tapering is monotonic or nearly monotonic
                        tapering_violations = 0
                        for i in range(len(radii) - 1):
                            if radii[i] < radii[i + 1]:  # Radius increased
                                tapering_violations += 1

                        # Tapering score for this path
                        path_score = 1.0 - tapering_violations / max(len(radii) - 1, 1)
                        tapering_scores.append(path_score)

                except nx.NetworkXNoPath:
                    continue
            
            if tapering_scores:
                mean_tapering = np.mean(tapering_scores)
                monotonic_paths = sum(1 for score in tapering_scores if score > 0.8)
            else:
                mean_tapering = 1.0
                monotonic_paths = 0
            
            return {
                'tapering_score': float(mean_tapering),
                'monotonic_paths': monotonic_paths,
                'total_paths': total_paths
            }
            
        except Exception as e:
            warnings.warn(f"Error computing vessel tapering consistency: {e}")
            return {'tapering_score': 0.0, 'monotonic_paths': 0, 'total_paths': 0}
    
    def compute_branching_angle_distribution(
        self,
        mask: np.ndarray,
        skeleton: Optional[np.ndarray] = None
    ) -> Dict[str, Union[float, List[float]]]:
        """
        Compute branching angle distribution metrics.
        
        Args:
            mask: Binary mask
            skeleton: Optional pre-computed skeleton
            
        Returns:
            Dictionary with branching angle statistics
        """
        try:
            if skeleton is None:
                print("      [1/2] Skeletonizing for branching angles...", flush=True)
                if mask.ndim == 2:
                    skeleton = morphology.skeletonize(mask > 0.5)
                else:
                    skeleton = skeletonize_3d(mask > 0.5)

            # Create graph from skeleton
            print("      [2/2] Building graph...", flush=True)
            topo_metrics = TopologicalMetrics(self.voxel_spacing)
            graph = topo_metrics.skeleton_to_graph(skeleton)

            # Find bifurcations
            bifurcations = [n for n, d in graph.degree() if d > 2]

            if len(bifurcations) == 0:
                return {
                    'mean_branching_angle': 0.0,
                    'std_branching_angle': 0.0,
                    'physiological_angle_ratio': 1.0,
                    'branching_angles': []
                }

            all_angles = []

            print(f"      Computing angles at {len(bifurcations)} bifurcations...", flush=True)
            for bifurcation in tqdm(bifurcations, desc="      Angles", leave=False):
                bifurcation_coords = np.array(graph.nodes[bifurcation]['coords'])
                neighbors = list(graph.neighbors(bifurcation))
                
                # Get vectors to all neighbors
                vectors = []
                for neighbor in neighbors:
                    neighbor_coords = np.array(graph.nodes[neighbor]['coords'])
                    vector = neighbor_coords - bifurcation_coords
                    # Normalize vector
                    norm = np.linalg.norm(vector)
                    if norm > 0:
                        vectors.append(vector / norm)
                
                # Compute angles between all pairs of vectors
                for i in range(len(vectors)):
                    for j in range(i + 1, len(vectors)):
                        dot_product = np.clip(np.dot(vectors[i], vectors[j]), -1.0, 1.0)
                        angle = np.arccos(dot_product) * 180.0 / np.pi
                        all_angles.append(angle)
            
            if all_angles:
                mean_angle = np.mean(all_angles)
                std_angle = np.std(all_angles)
                
                # Physiological angle range is typically 30-120 degrees
                physiological_angles = [a for a in all_angles if 30 <= a <= 120]
                physiological_ratio = len(physiological_angles) / len(all_angles)
            else:
                mean_angle = 0.0
                std_angle = 0.0
                physiological_ratio = 1.0
            
            return {
                'mean_branching_angle': float(mean_angle),
                'std_branching_angle': float(std_angle),
                'physiological_angle_ratio': float(physiological_ratio),
                'branching_angles': all_angles
            }
            
        except Exception as e:
            warnings.warn(f"Error computing branching angle distribution: {e}")
            return {
                'mean_branching_angle': 0.0,
                'std_branching_angle': 0.0,
                'physiological_angle_ratio': 0.0,
                'branching_angles': []
            }


def compute_comprehensive_metrics(
    pred_mask: Union[np.ndarray, torch.Tensor],
    gt_mask: Union[np.ndarray, torch.Tensor],
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    include_anatomical: bool = True,
    include_topological: bool = True
) -> Dict[str, Union[float, int, List]]:
    """
    Compute all comprehensive metrics for vascular segmentation evaluation.
    
    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
        voxel_spacing: Physical spacing between voxels
        include_anatomical: Whether to compute anatomical metrics
        include_topological: Whether to compute topological metrics
        
    Returns:
        Dictionary containing all computed metrics
    """
    # Convert to numpy if needed
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.detach().cpu().numpy()
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.detach().cpu().numpy()
    
    # Ensure binary
    pred_mask = (pred_mask > 0.5).astype(np.uint8)
    gt_mask = (gt_mask > 0.5).astype(np.uint8)
    
    results = {}
    
    # 1. Standard Volumetric Segmentation Metrics
    results.update({
        'dice': compute_dice_score(pred_mask, gt_mask),
        'jaccard': compute_jaccard_score(pred_mask, gt_mask),
        'hausdorff_95': compute_hausdorff_distance(pred_mask, gt_mask),
        'assd': compute_average_symmetric_surface_distance(pred_mask, gt_mask),
        'sensitivity': compute_sensitivity(pred_mask, gt_mask),
        'specificity': compute_specificity(pred_mask, gt_mask),
        'precision': compute_precision(pred_mask, gt_mask),
        'volume_similarity': compute_volume_similarity(pred_mask, gt_mask)
    })
    
    if include_topological:
        # 2. Topological & Structural Metrics
        topo_metrics = TopologicalMetrics(voxel_spacing)
        
        # Connected components
        cc_stats = topo_metrics.compute_connected_components_count(pred_mask)
        gt_cc_stats = topo_metrics.compute_connected_components_count(gt_mask)
        results.update({
            'num_components': cc_stats['num_components'],
            'gt_num_components': gt_cc_stats['num_components'],
            'component_count_error': abs(cc_stats['num_components'] - gt_cc_stats['num_components'])
        })
        
        # Connectivity-aware Dice
        results['connectivity_aware_dice'] = topo_metrics.compute_connectivity_aware_dice(pred_mask, gt_mask)
        
        # Vessel tree isomorphism
        tree_iso_results = topo_metrics.compute_vessel_tree_isomorphism(pred_mask, gt_mask)
        results.update({f'tree_{k}': v for k, v in tree_iso_results.items()})
        
        # Bifurcation detection accuracy
        bda_results = topo_metrics.compute_bifurcation_detection_accuracy(pred_mask, gt_mask)
        results.update(bda_results)
    
    if include_anatomical:
        # 3. Anatomical Consistency Metrics
        print("    Computing anatomy metrics...", flush=True)
        anat_metrics = AnatomicalMetrics(voxel_spacing)

        # Murray's law compliance
        print("    [ANATOMY 1/3] Murray's Law Compliance...", flush=True)
        murray_results = anat_metrics.compute_murrays_law_compliance(pred_mask)
        results.update({f'murray_{k}' if k != 'murrays_law_score' else k: v for k, v in murray_results.items()})

        # Vessel tapering consistency
        print("    [ANATOMY 2/3] Vessel Tapering Consistency...", flush=True)
        tapering_results = anat_metrics.compute_vessel_tapering_consistency(pred_mask)
        results.update({f'taper_{k}' if k != 'tapering_score' else k: v for k, v in tapering_results.items()})

        # Branching angle distribution
        print("    [ANATOMY 3/3] Branching Angle Distribution...", flush=True)
        angle_results = anat_metrics.compute_branching_angle_distribution(pred_mask)
        results.update({f'angle_{k}': v for k, v in angle_results.items() if k != 'branching_angles'})
        # Store angles separately if needed
        results['branching_angles'] = angle_results['branching_angles']
    
    return results