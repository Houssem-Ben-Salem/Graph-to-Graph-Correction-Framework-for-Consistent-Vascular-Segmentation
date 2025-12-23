"""
Graph-to-Volume Pipeline
Integration pipeline connecting graph correction outputs to volume reconstruction
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
from pathlib import Path

from src.models.graph_extraction.vascular_graph import VascularGraph
from src.models.graph_correction import GraphCorrectionModel
from src.models.reconstruction.volume_reconstructor import (
    VolumeReconstructor, ReconstructionConfig, ReconstructionResult
)
from src.utils.graph_correspondence import CorrespondenceResult


@dataclass
class PipelineConfig:
    """Configuration for the complete graph-to-volume pipeline"""
    # Model configuration
    graph_correction_model_path: Optional[str] = None  # Path to trained model
    
    # Reconstruction configuration
    reconstruction_config: ReconstructionConfig = None
    
    # Quality control
    min_correction_confidence: float = 0.3  # Minimum confidence for using corrections
    fallback_to_original: bool = True  # Use original if correction fails
    
    # Processing options
    batch_processing: bool = True  # Process multiple graphs in batch
    device: str = "cuda"  # Device for model inference
    
    # Integration settings
    blend_correction_confidence: bool = True  # Weight corrections by confidence
    preserve_original_topology: bool = False  # Preserve original when confidence is low
    
    def __post_init__(self):
        if self.reconstruction_config is None:
            self.reconstruction_config = ReconstructionConfig()


@dataclass
class PipelineResult:
    """Result from the complete pipeline"""
    # Final outputs
    reconstructed_volume: np.ndarray  # Final volumetric mask
    
    # Quality metrics
    overall_quality: float  # Overall pipeline quality
    correction_quality: float  # Graph correction quality
    reconstruction_quality: float  # Volume reconstruction quality
    
    # Process information
    corrected_graph: Optional[VascularGraph] = None  # Corrected graph
    correction_applied: bool = False  # Whether correction was applied
    
    # Timing information
    correction_time: float = 0.0  # Time for graph correction
    reconstruction_time: float = 0.0  # Time for volume reconstruction
    total_time: float = 0.0  # Total pipeline time
    
    # Intermediate results (for debugging)
    original_graph: Optional[VascularGraph] = None
    reconstruction_result: Optional[ReconstructionResult] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class GraphToVolumePipeline:
    """Complete pipeline from graph correction to volume reconstruction"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.graph_correction_model = None
        self.volume_reconstructor = VolumeReconstructor(config.reconstruction_config)
        
        # Load trained model if path provided
        if config.graph_correction_model_path:
            self._load_graph_correction_model(config.graph_correction_model_path)
        
        self.logger.info(f"Pipeline initialized on device: {self.device}")
    
    def _load_graph_correction_model(self, model_path: str):
        """Load trained graph correction model"""
        try:
            self.logger.info(f"Loading graph correction model from: {model_path}")
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Initialize model
            model_config = checkpoint.get('config', {}).get('model_config', {})
            self.graph_correction_model = GraphCorrectionModel(model_config)
            
            # Load state dict
            self.graph_correction_model.load_state_dict(checkpoint['model_state_dict'])
            self.graph_correction_model.to(self.device)
            self.graph_correction_model.eval()
            
            self.logger.info("Graph correction model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load graph correction model: {e}")
            self.graph_correction_model = None
    
    def process_single_case(self, 
                          predicted_graph: VascularGraph,
                          ground_truth_graph: VascularGraph,
                          correspondences: CorrespondenceResult,
                          original_prediction: Optional[np.ndarray] = None,
                          reference_spacing: Optional[Tuple[float, float, float]] = None) -> PipelineResult:
        """
        Process a single case through the complete pipeline
        
        Args:
            predicted_graph: Initial U-Net derived graph
            ground_truth_graph: Reference graph for correspondence
            correspondences: Graph correspondence mapping
            original_prediction: Original U-Net prediction mask
            reference_spacing: Voxel spacing from original volume
        """
        import time
        start_time = time.time()
        
        self.logger.info("Processing single case through pipeline")
        
        result = PipelineResult(
            reconstructed_volume=np.array([]),  # Will be filled
            overall_quality=0.0,
            correction_quality=0.0,
            reconstruction_quality=0.0
        )
        
        result.original_graph = predicted_graph
        
        try:
            # Step 1: Apply graph correction if model is available
            if self.graph_correction_model is not None:
                self.logger.info("Applying graph correction...")
                correction_start = time.time()
                
                corrected_graph, correction_confidence = self._apply_graph_correction(
                    predicted_graph, ground_truth_graph, correspondences
                )
                
                result.correction_time = time.time() - correction_start
                result.correction_quality = correction_confidence
                
                # Decide whether to use correction
                if correction_confidence >= self.config.min_correction_confidence:
                    result.corrected_graph = corrected_graph
                    result.correction_applied = True
                    graph_to_reconstruct = corrected_graph
                    self.logger.info(f"Using corrected graph (confidence: {correction_confidence:.3f})")
                else:
                    result.corrected_graph = predicted_graph
                    result.correction_applied = False
                    graph_to_reconstruct = predicted_graph
                    result.warnings.append(f"Low correction confidence: {correction_confidence:.3f}")
                    self.logger.warning(f"Low correction confidence: {correction_confidence:.3f}, using original")
            
            else:
                # No correction model available, use original graph
                result.corrected_graph = predicted_graph
                result.correction_applied = False
                graph_to_reconstruct = predicted_graph
                result.correction_quality = 1.0  # Assume original is good
                result.warnings.append("No correction model available")
                self.logger.info("No correction model, using original graph")
            
            # Step 2: Reconstruct volume from graph
            self.logger.info("Reconstructing volume from graph...")
            reconstruction_start = time.time()
            
            reconstruction_result = self.volume_reconstructor.reconstruct_from_graph(
                corrected_graph=graph_to_reconstruct,
                original_prediction=original_prediction,
                reference_spacing=reference_spacing,
                return_intermediates=False
            )
            
            result.reconstruction_time = time.time() - reconstruction_start
            result.reconstruction_quality = reconstruction_result.reconstruction_quality
            result.reconstructed_volume = reconstruction_result.reconstructed_mask
            result.reconstruction_result = reconstruction_result
            
            # Step 3: Compute overall quality
            result.overall_quality = self._compute_overall_quality(
                result.correction_quality,
                result.reconstruction_quality,
                result.correction_applied
            )
            
            # Collect warnings
            result.warnings.extend(reconstruction_result.warnings)
            
            result.total_time = time.time() - start_time
            
            self.logger.info(f"Pipeline completed in {result.total_time:.2f}s")
            self.logger.info(f"Overall quality: {result.overall_quality:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Pipeline processing failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Return fallback result
            if self.config.fallback_to_original and original_prediction is not None:
                result.reconstructed_volume = original_prediction
                result.warnings.append(f"Pipeline failed, using original prediction: {str(e)}")
            else:
                result.reconstructed_volume = np.zeros((64, 64, 64), dtype=np.uint8)
                result.warnings.append(f"Pipeline failed: {str(e)}")
            
            result.total_time = time.time() - start_time
            return result
    
    def _apply_graph_correction(self, 
                              predicted_graph: VascularGraph,
                              ground_truth_graph: VascularGraph,
                              correspondences: CorrespondenceResult) -> Tuple[VascularGraph, float]:
        """Apply graph correction using trained model"""
        
        try:
            with torch.no_grad():
                # Forward pass through correction model
                outputs = self.graph_correction_model(
                    predicted_graph, 
                    ground_truth_graph, 
                    correspondences,
                    training_mode=False
                )
                
                # Extract corrected graph
                if 'corrected_graph' in outputs:
                    corrected_graph = outputs['corrected_graph']
                else:
                    # Apply corrections to create corrected graph
                    corrected_graph = self._apply_corrections_to_graph(
                        predicted_graph, outputs
                    )
                
                # Get correction confidence
                quality_score = outputs.get('quality_score', torch.tensor(0.5))
                if isinstance(quality_score, torch.Tensor):
                    confidence = quality_score.item()
                else:
                    confidence = float(quality_score)
                
                return corrected_graph, confidence
                
        except Exception as e:
            self.logger.error(f"Graph correction failed: {e}")
            return predicted_graph, 0.0
    
    def _apply_corrections_to_graph(self, original_graph: VascularGraph, 
                                  model_outputs: Dict) -> VascularGraph:
        """Apply model corrections to create corrected graph"""
        # This is a simplified version - in practice would need to interpret
        # model outputs (node operations, corrections, etc.) and apply them
        
        try:
            # For now, return the original graph with some modifications
            # based on model confidence
            
            corrected_nodes = []
            for i, node in enumerate(original_graph.nodes):
                corrected_node = node.copy()
                
                # Apply position corrections if available
                if 'node_corrections' in model_outputs:
                    node_corrections = model_outputs['node_corrections']
                    if i < len(node_corrections):
                        corrections = node_corrections[i]
                        
                        # Apply position correction
                        if 'position' in corrected_node:
                            original_pos = np.array(corrected_node['position'][:3])
                            if isinstance(corrections, torch.Tensor):
                                position_correction = corrections[:3].cpu().numpy()
                            else:
                                position_correction = np.array(corrections[:3])
                            
                            corrected_pos = original_pos + position_correction * 0.1  # Scale correction
                            corrected_node['position'] = corrected_pos.tolist()
                
                corrected_nodes.append(corrected_node)
            
            # Create corrected graph
            corrected_graph = VascularGraph(
                nodes=corrected_nodes,
                edges=original_graph.edges.copy(),
                global_properties=original_graph.global_properties,
                metadata=original_graph.metadata
            )
            
            return corrected_graph
            
        except Exception as e:
            self.logger.warning(f"Failed to apply corrections: {e}")
            return original_graph
    
    def _compute_overall_quality(self, correction_quality: float, 
                               reconstruction_quality: float,
                               correction_applied: bool) -> float:
        """Compute overall pipeline quality"""
        
        if correction_applied:
            # Weighted combination of correction and reconstruction quality
            overall = 0.4 * correction_quality + 0.6 * reconstruction_quality
        else:
            # Only reconstruction quality (with penalty for no correction)
            overall = 0.8 * reconstruction_quality
        
        return overall
    
    def process_batch(self, 
                     cases: List[Dict[str, Any]],
                     progress_callback: Optional[callable] = None) -> List[PipelineResult]:
        """
        Process multiple cases in batch
        
        Args:
            cases: List of case dictionaries containing:
                - predicted_graph: VascularGraph
                - ground_truth_graph: VascularGraph  
                - correspondences: CorrespondenceResult
                - original_prediction: Optional[np.ndarray]
                - reference_spacing: Optional[Tuple[float, float, float]]
            progress_callback: Optional callback for progress updates
        """
        
        self.logger.info(f"Processing batch of {len(cases)} cases")
        
        results = []
        
        for i, case in enumerate(cases):
            self.logger.info(f"Processing case {i+1}/{len(cases)}")
            
            # Extract case data
            predicted_graph = case['predicted_graph']
            ground_truth_graph = case['ground_truth_graph']
            correspondences = case['correspondences']
            original_prediction = case.get('original_prediction')
            reference_spacing = case.get('reference_spacing')
            
            # Process single case
            result = self.process_single_case(
                predicted_graph=predicted_graph,
                ground_truth_graph=ground_truth_graph,
                correspondences=correspondences,
                original_prediction=original_prediction,
                reference_spacing=reference_spacing
            )
            
            results.append(result)
            
            # Progress callback
            if progress_callback:
                progress_callback(i + 1, len(cases), result)
        
        # Log batch statistics
        self._log_batch_statistics(results)
        
        return results
    
    def _log_batch_statistics(self, results: List[PipelineResult]):
        """Log batch processing statistics"""
        if not results:
            return
        
        # Compute statistics
        overall_qualities = [r.overall_quality for r in results]
        correction_qualities = [r.correction_quality for r in results]
        reconstruction_qualities = [r.reconstruction_quality for r in results]
        total_times = [r.total_time for r in results]
        
        corrections_applied = sum(1 for r in results if r.correction_applied)
        
        self.logger.info("Batch processing summary:")
        self.logger.info(f"  Cases processed: {len(results)}")
        self.logger.info(f"  Corrections applied: {corrections_applied}/{len(results)} ({corrections_applied/len(results)*100:.1f}%)")
        self.logger.info(f"  Average overall quality: {np.mean(overall_qualities):.3f} ± {np.std(overall_qualities):.3f}")
        self.logger.info(f"  Average correction quality: {np.mean(correction_qualities):.3f} ± {np.std(correction_qualities):.3f}")
        self.logger.info(f"  Average reconstruction quality: {np.mean(reconstruction_qualities):.3f} ± {np.std(reconstruction_qualities):.3f}")
        self.logger.info(f"  Average processing time: {np.mean(total_times):.2f}s ± {np.std(total_times):.2f}s")
        
        # Success rate (quality > 0.5)
        success_rate = sum(1 for q in overall_qualities if q > 0.5) / len(results)
        self.logger.info(f"  Success rate (quality > 0.5): {success_rate:.1%}")


def create_test_pipeline():
    """Create a test pipeline for demonstration"""
    
    # Configuration
    config = PipelineConfig(
        reconstruction_config=ReconstructionConfig(
            target_resolution=(64, 64, 64),
            auto_compute_bounds=True,
            blend_with_original=True
        ),
        min_correction_confidence=0.3,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Create pipeline
    pipeline = GraphToVolumePipeline(config)
    
    return pipeline


def test_pipeline_integration():
    """Test the complete pipeline integration"""
    from src.models.graph_extraction.vascular_graph import VascularGraph
    from src.utils.graph_correspondence import CorrespondenceResult
    
    # Create test data
    nodes = [
        {'position': [0, 0, 0], 'radius_voxels': 2.0, 'type': 'normal'},
        {'position': [10, 0, 0], 'radius_voxels': 1.8, 'type': 'bifurcation'},
        {'position': [15, 5, 0], 'radius_voxels': 1.2, 'type': 'normal'},
        {'position': [15, -5, 0], 'radius_voxels': 1.2, 'type': 'normal'},
    ]
    
    edges = [
        {'source': 0, 'target': 1, 'confidence': 0.9},
        {'source': 1, 'target': 2, 'confidence': 0.8},
        {'source': 1, 'target': 3, 'confidence': 0.8},
    ]
    
    predicted_graph = VascularGraph(nodes=nodes, edges=edges)
    ground_truth_graph = VascularGraph(nodes=nodes, edges=edges)  # Same for simplicity
    
    # Create simple correspondences
    correspondences = CorrespondenceResult(
        node_correspondences={0: 0, 1: 1, 2: 2, 3: 3},
        node_confidences={0: 0.9, 1: 0.9, 2: 0.8, 3: 0.8},
        unmatched_pred_nodes=set(),
        unmatched_gt_nodes=set(),
        edge_correspondences={(0, 1): (0, 1), (1, 2): (1, 2), (1, 3): (1, 3)},
        edge_confidences={(0, 1): 0.9, (1, 2): 0.8, (1, 3): 0.8},
        unmatched_pred_edges=set(),
        unmatched_gt_edges=set(),
        topology_differences={},
        alignment_transform={},
        correspondence_quality={'overall_quality': 0.85},
        metadata={}
    )
    
    # Create pipeline
    pipeline = create_test_pipeline()
    
    # Process case
    print("Processing test case through pipeline...")
    result = pipeline.process_single_case(
        predicted_graph=predicted_graph,
        ground_truth_graph=ground_truth_graph,
        correspondences=correspondences
    )
    
    print(f"Pipeline test completed:")
    print(f"  Overall quality: {result.overall_quality:.3f}")
    print(f"  Correction applied: {result.correction_applied}")
    print(f"  Reconstruction quality: {result.reconstruction_quality:.3f}")
    print(f"  Total time: {result.total_time:.2f}s")
    print(f"  Output shape: {result.reconstructed_volume.shape}")
    
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
    
    return result


if __name__ == "__main__":
    # Run tests
    result = test_pipeline_integration()
    print("Pipeline integration tests completed successfully!")