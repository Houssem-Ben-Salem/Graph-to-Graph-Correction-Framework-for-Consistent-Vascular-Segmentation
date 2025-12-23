"""
Main Graph Extraction Pipeline
Orchestrates the complete graph extraction process from segmentation masks
"""

import numpy as np
from typing import Dict, Tuple, Optional
import logging
from pathlib import Path

from .vascular_graph import VascularGraph
from .centerline_extraction import CenterlineExtractor
from .node_placement import NodePlacer
from .node_attributes import NodeAttributeExtractor
from .edge_creation import EdgeCreator
from ...data.preprocessing.mask_preprocessing import MaskPreprocessor
from ...data.preprocessing.quality_assessment import QualityAssessment


class GraphExtractor:
    """
    Main graph extraction pipeline for converting segmentation masks to vascular graphs
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize graph extractor with configuration
        
        Args:
            config: Configuration dictionary for extraction parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize pipeline components
        self.preprocessor = MaskPreprocessor(
            min_component_size=self.config.get('min_component_size', 27),
            gaussian_sigma=self.config.get('gaussian_sigma', 0.5),
            fill_holes=self.config.get('fill_holes', True)
        )
        
        self.quality_assessor = QualityAssessment()
        
        self.centerline_extractor = CenterlineExtractor(
            spur_length_threshold=self.config.get('spur_length_threshold', 3),
            smoothing_sigma=self.config.get('centerline_smoothing_sigma', 0.5),
            min_branch_length=self.config.get('min_branch_length', 2)
        )
        
        self.node_placer = NodePlacer(
            base_density=self.config.get('base_node_density', 2.0),
            curvature_sensitivity=self.config.get('curvature_sensitivity', 7.0),
            min_distance=self.config.get('min_node_distance', 0.5),
            max_distance=self.config.get('max_node_distance', 3.0),
            bifurcation_buffer=self.config.get('bifurcation_buffer', 2),
            max_nodes=self.config.get('max_nodes', 500)  # Conservative limit for large volumes
        )
        
        self.attribute_extractor = NodeAttributeExtractor(
            neighborhood_radius=self.config.get('neighborhood_radius', 3.0)
        )
        
        self.edge_creator = EdgeCreator(
            max_edge_length=self.config.get('max_edge_length', 10.0),
            connectivity_tolerance=self.config.get('connectivity_tolerance', 2.0)
        )
    
    def extract_graph(self,
                     mask: np.ndarray,
                     voxel_spacing: Optional[Tuple[float, float, float]] = None,
                     is_prediction: bool = True,
                     prediction_probabilities: Optional[np.ndarray] = None,
                     confidence_map: Optional[np.ndarray] = None,
                     patient_id: Optional[str] = None) -> VascularGraph:
        """
        Extract vascular graph from segmentation mask
        
        Args:
            mask: Binary segmentation mask (0s and 1s)
            voxel_spacing: Voxel spacing in mm (z, y, x)
            is_prediction: Whether this is a predicted mask (vs ground truth)
            prediction_probabilities: Raw prediction probabilities (optional)
            confidence_map: Pre-computed confidence map (optional)
            patient_id: Patient identifier for metadata
            
        Returns:
            VascularGraph object containing nodes, edges, and attributes
        """
        self.logger.info(f"Starting graph extraction for {'prediction' if is_prediction else 'ground truth'} mask")
        
        # Validate input
        if mask.dtype not in [bool, np.uint8]:
            raise ValueError(f"Mask must be binary (bool or uint8), got {mask.dtype}")
        
        original_mask_shape = mask.shape
        mask_bool = mask.astype(bool)
        
        if np.sum(mask_bool) == 0:
            self.logger.warning("Empty mask provided, returning empty graph")
            return self._create_empty_graph(original_mask_shape, voxel_spacing, patient_id)
        
        try:
            # Step 1: Preprocessing
            self.logger.info("Step 1: Preprocessing mask")
            cleaned_mask, preprocessing_info = self.preprocessor.preprocess_mask(
                mask, is_prediction, voxel_spacing
            )
            
            # Generate confidence map if not provided
            if is_prediction and confidence_map is None and prediction_probabilities is not None:
                confidence_map = self.preprocessor.generate_confidence_map(
                    cleaned_mask, prediction_probabilities
                )
            
            # Step 2: Quality Assessment
            self.logger.info("Step 2: Quality assessment")
            quality_metrics = self.quality_assessor.assess_mask_quality(
                cleaned_mask, voxel_spacing
            )
            
            # Step 3: Centerline Extraction
            self.logger.info("Step 3: Centerline extraction")
            centerline_data = self.centerline_extractor.extract_centerline(
                cleaned_mask, voxel_spacing
            )
            
            if centerline_data['metrics']['total_length_voxels'] == 0:
                self.logger.warning("No centerline extracted, returning empty graph")
                return self._create_empty_graph(original_mask_shape, voxel_spacing, patient_id)
            
            # Step 4: Strategic Node Placement
            self.logger.info("Step 4: Strategic node placement")
            nodes = self.node_placer.place_nodes(centerline_data, voxel_spacing)
            
            if len(nodes['positions']) == 0:
                self.logger.warning("No nodes placed, returning empty graph")
                return self._create_empty_graph(original_mask_shape, voxel_spacing, patient_id)
            
            # Step 5: Node Attribute Extraction
            self.logger.info("Step 5: Node attribute extraction")
            node_attributes = self.attribute_extractor.extract_all_attributes(
                nodes,
                centerline_data['distance_map'],
                centerline_data['skeleton'],
                cleaned_mask,
                prediction_probabilities,
                confidence_map,
                voxel_spacing
            )
            
            # Step 6: Edge Creation
            self.logger.info("Step 6: Edge creation")
            edges = self.edge_creator.create_edges(
                nodes,
                node_attributes,
                centerline_data,
                centerline_data['distance_map'],
                voxel_spacing
            )
            
            # Step 7: Create VascularGraph
            self.logger.info("Step 7: Assembling final graph")
            vascular_graph = VascularGraph.from_extraction_results(
                nodes=nodes,
                node_attributes=edges['updated_node_attributes'],  # Use updated attributes with degrees
                edges=edges,
                centerline_data=centerline_data,
                preprocessing_info=preprocessing_info,
                quality_metrics=quality_metrics,
                voxel_spacing=voxel_spacing,
                original_mask_shape=original_mask_shape
            )
            
            # Add additional metadata
            vascular_graph.metadata.update({
                'patient_id': patient_id,
                'is_prediction': is_prediction,
                'extraction_config': self.config,
                'extraction_success': True
            })
            
            self.logger.info(f"Graph extraction completed successfully: {vascular_graph}")
            self.logger.info(f"Quality score: {quality_metrics.get('overall_quality_score', 0):.3f}")
            
            return vascular_graph
            
        except Exception as e:
            self.logger.error(f"Graph extraction failed: {str(e)}")
            # Return empty graph with error information
            empty_graph = self._create_empty_graph(original_mask_shape, voxel_spacing, patient_id)
            empty_graph.metadata.update({
                'extraction_success': False,
                'extraction_error': str(e)
            })
            return empty_graph
    
    def extract_graph_pair(self,
                          predicted_mask: np.ndarray,
                          ground_truth_mask: np.ndarray,
                          voxel_spacing: Optional[Tuple[float, float, float]] = None,
                          prediction_probabilities: Optional[np.ndarray] = None,
                          patient_id: Optional[str] = None) -> Tuple[VascularGraph, VascularGraph]:
        """
        Extract graphs from both predicted and ground truth masks
        
        Args:
            predicted_mask: Predicted segmentation mask
            ground_truth_mask: Ground truth segmentation mask
            voxel_spacing: Voxel spacing in mm
            prediction_probabilities: Raw prediction probabilities
            patient_id: Patient identifier
            
        Returns:
            Tuple of (predicted_graph, ground_truth_graph)
        """
        self.logger.info(f"Extracting graph pair for patient {patient_id}")
        
        # Extract predicted graph
        predicted_graph = self.extract_graph(
            predicted_mask,
            voxel_spacing=voxel_spacing,
            is_prediction=True,
            prediction_probabilities=prediction_probabilities,
            patient_id=patient_id
        )
        
        # Extract ground truth graph
        gt_graph = self.extract_graph(
            ground_truth_mask,
            voxel_spacing=voxel_spacing,
            is_prediction=False,
            prediction_probabilities=None,
            patient_id=patient_id
        )
        
        return predicted_graph, gt_graph
    
    def _create_empty_graph(self,
                           original_shape: Tuple[int, int, int],
                           voxel_spacing: Optional[Tuple[float, float, float]],
                           patient_id: Optional[str]) -> VascularGraph:
        """Create an empty graph for cases where extraction fails"""
        empty_graph = VascularGraph(
            nodes=[],
            edges=[],
            global_properties={
                'num_nodes': 0,
                'num_edges': 0,
                'node_type_counts': {'bifurcations': 0, 'endpoints': 0, 'regular': 0, 'buffer': 0},
                'total_length_voxels': 0,
                'total_length_mm': 0,
                'average_radius_voxels': 0,
                'complexity_score': 0,
                'is_connected': False,
                'density': 0
            },
            metadata={
                'original_mask_shape': original_shape,
                'voxel_spacing': voxel_spacing,
                'patient_id': patient_id,
                'extraction_parameters': {},
                'extraction_success': False
            }
        )
        
        return empty_graph
    
    def extract_from_file(self,
                         mask_path: Path,
                         voxel_spacing: Optional[Tuple[float, float, float]] = None,
                         is_prediction: bool = True,
                         prob_path: Optional[Path] = None) -> VascularGraph:
        """
        Extract graph from mask file
        
        Args:
            mask_path: Path to binary mask file (NIfTI format)
            voxel_spacing: Voxel spacing in mm
            is_prediction: Whether this is a predicted mask
            prob_path: Path to probability map file (optional)
            
        Returns:
            VascularGraph object
        """
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError("nibabel required for file I/O. Install with: pip install nibabel")
        
        # Load mask
        mask_img = nib.load(str(mask_path))
        mask = mask_img.get_fdata().astype(np.uint8)
        
        # Get voxel spacing from header if not provided
        if voxel_spacing is None:
            voxel_spacing = tuple(mask_img.header.get_zooms()[:3])
        
        # Load probabilities if provided
        prediction_probabilities = None
        if prob_path and prob_path.exists():
            prob_img = nib.load(str(prob_path))
            prediction_probabilities = prob_img.get_fdata().astype(np.float32)
        
        # Extract patient ID from filename
        patient_id = mask_path.stem
        
        return self.extract_graph(
            mask=mask,
            voxel_spacing=voxel_spacing,
            is_prediction=is_prediction,
            prediction_probabilities=prediction_probabilities,
            patient_id=patient_id
        )
    
    def get_extraction_summary(self) -> str:
        """Get summary of extraction parameters"""
        summary = ["=== Graph Extraction Configuration ==="]
        summary.append(f"Min component size: {self.config.get('min_component_size', 27)} voxels")
        summary.append(f"Gaussian sigma: {self.config.get('gaussian_sigma', 0.5)}")
        summary.append(f"Base node density: {self.config.get('base_node_density', 2.0)}")
        summary.append(f"Curvature sensitivity: {self.config.get('curvature_sensitivity', 7.0)}")
        summary.append(f"Max edge length: {self.config.get('max_edge_length', 10.0)}")
        summary.append(f"Neighborhood radius: {self.config.get('neighborhood_radius', 3.0)}")
        
        return "\n".join(summary)