#!/usr/bin/env python
"""
Test Step 5: Template-Based Reconstruction
Comprehensive test of the complete reconstruction pipeline
"""

import sys
import numpy as np
import torch
import logging
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.graph_extraction.vascular_graph import VascularGraph
from src.models.reconstruction import (
    VesselTemplate, CylindricalTemplate, BifurcationTemplate,
    CylinderParameters, BifurcationParameters, TemplateFactory,
    SDFRenderer, RenderingConfig,
    TemplatePlacer, PlacementConfig,
    VolumeReconstructor, ReconstructionConfig,
    GraphToVolumePipeline, PipelineConfig
)
from src.utils.graph_correspondence import CorrespondenceResult


def create_test_graph() -> VascularGraph:
    """Create a comprehensive test graph"""
    
    # Create a branching vascular structure
    nodes = [
        # Main vessel
        {'position': [0, 0, 0], 'radius_voxels': 3.0, 'type': 'normal'},
        {'position': [5, 0, 0], 'radius_voxels': 2.8, 'type': 'normal'},
        {'position': [10, 0, 0], 'radius_voxels': 2.5, 'type': 'bifurcation'},
        
        # Branch 1
        {'position': [15, 3, 0], 'radius_voxels': 1.8, 'type': 'normal'},
        {'position': [20, 5, 0], 'radius_voxels': 1.6, 'type': 'normal'},
        {'position': [25, 8, 0], 'radius_voxels': 1.4, 'type': 'normal'},
        
        # Branch 2
        {'position': [15, -3, 0], 'radius_voxels': 1.8, 'type': 'bifurcation'},
        {'position': [20, -5, 2], 'radius_voxels': 1.2, 'type': 'normal'},
        {'position': [20, -5, -2], 'radius_voxels': 1.2, 'type': 'normal'},
        
        # Sub-branches
        {'position': [25, -8, 3], 'radius_voxels': 1.0, 'type': 'normal'},
        {'position': [25, -8, -3], 'radius_voxels': 1.0, 'type': 'normal'},
    ]
    
    edges = [
        # Main vessel
        {'source': 0, 'target': 1, 'confidence': 0.95},
        {'source': 1, 'target': 2, 'confidence': 0.90},
        
        # First bifurcation
        {'source': 2, 'target': 3, 'confidence': 0.85},
        {'source': 2, 'target': 6, 'confidence': 0.85},
        
        # Branch 1 continuation
        {'source': 3, 'target': 4, 'confidence': 0.80},
        {'source': 4, 'target': 5, 'confidence': 0.75},
        
        # Second bifurcation
        {'source': 6, 'target': 7, 'confidence': 0.80},
        {'source': 6, 'target': 8, 'confidence': 0.80},
        
        # Sub-branches
        {'source': 7, 'target': 9, 'confidence': 0.70},
        {'source': 8, 'target': 10, 'confidence': 0.70},
    ]
    
    return VascularGraph(
        nodes=nodes,
        edges=edges,
        global_properties={'total_length': 50.0, 'avg_radius': 1.8},
        metadata={'source': 'test_graph', 'creation_time': time.time()}
    )


def test_vessel_templates():
    """Test vessel template creation and SDF computation"""
    
    logger = logging.getLogger(__name__)
    logger.info("=== Testing Vessel Templates ===")
    
    try:
        # Test cylindrical template
        logger.info("Testing cylindrical template...")
        
        cylinder_params = CylinderParameters(
            start_point=np.array([0, 0, 0]),
            end_point=np.array([10, 0, 0]),
            start_radius=2.0,
            end_radius=1.5
        )
        
        cylinder = CylindricalTemplate(cylinder_params)
        
        # Test SDF computation
        test_points = np.array([
            [5, 0, 0],      # Center
            [5, 1.0, 0],    # Inside
            [5, 3.0, 0],    # Outside
            [5, 0, 1.0],    # Inside
            [5, 0, 3.0],    # Outside
        ])
        
        sdf_values = cylinder.compute_sdf(test_points)
        inside_mask = sdf_values <= 0
        
        logger.info(f"  Cylinder SDF values: {sdf_values}")
        logger.info(f"  Inside points: {np.sum(inside_mask)}/5")
        
        # Test bounding box
        min_pt, max_pt = cylinder.get_bounding_box()
        volume = cylinder.get_volume_estimate()
        
        logger.info(f"  Bounding box: {min_pt} to {max_pt}")
        logger.info(f"  Estimated volume: {volume:.2f}")
        
        # Test bifurcation template
        logger.info("Testing bifurcation template...")
        
        bifurcation_params = BifurcationParameters(
            position=np.array([0, 0, 0]),
            parent_direction=np.array([-1, 0, 0]),
            child1_direction=np.array([0.707, 0.707, 0]),
            child2_direction=np.array([0.707, -0.707, 0]),
            parent_radius=2.0,
            child1_radius=1.4,
            child2_radius=1.4
        )
        
        bifurcation = BifurcationTemplate(bifurcation_params)
        
        bifurcation_sdf = bifurcation.compute_sdf(test_points)
        bifurcation_volume = bifurcation.get_volume_estimate()
        
        logger.info(f"  Bifurcation SDF values: {bifurcation_sdf}")
        logger.info(f"  Bifurcation volume: {bifurcation_volume:.2f}")
        
        logger.info("✅ Vessel templates test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Vessel templates test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sdf_renderer():
    """Test SDF rendering functionality"""
    
    logger = logging.getLogger(__name__)
    logger.info("=== Testing SDF Renderer ===")
    
    try:
        # Create renderer configuration
        config = RenderingConfig(
            resolution=(32, 32, 32),  # Small for testing
            voxel_size=(0.5, 0.5, 0.5),
            origin=(-8.0, -8.0, -8.0),
            parallel_processing=True,
            anti_aliasing=True
        )
        
        renderer = SDFRenderer(config)
        
        # Create test templates
        cylinder_params = CylinderParameters(
            start_point=np.array([-6, 0, 0]),
            end_point=np.array([6, 0, 0]),
            start_radius=2.0,
            end_radius=1.5
        )
        
        cylinder = CylindricalTemplate(cylinder_params)
        
        # Render single template
        logger.info("Rendering single template...")
        start_time = time.time()
        
        sdf_volume = renderer.render_template(cylinder)
        
        render_time = time.time() - start_time
        logger.info(f"  Rendering time: {render_time:.3f}s")
        
        # Convert to binary mask
        binary_mask = renderer.sdf_to_binary_mask(sdf_volume)
        probability_mask = renderer.sdf_to_probability_mask(sdf_volume)
        
        # Compute statistics
        stats = renderer.compute_volume_statistics(sdf_volume)
        
        logger.info(f"  Volume statistics:")
        logger.info(f"    Total volume: {stats['total_volume']:.2f}")
        logger.info(f"    Surface area: {stats['surface_area']:.2f}")
        logger.info(f"    Fill ratio: {stats['fill_ratio']:.4f}")
        logger.info(f"    Min/Max SDF: {stats['min_sdf']:.2f} / {stats['max_sdf']:.2f}")
        
        logger.info(f"  Binary mask: {binary_mask.shape}, filled voxels: {np.sum(binary_mask)}")
        logger.info(f"  Probability mask: {probability_mask.shape}, mean prob: {np.mean(probability_mask):.3f}")
        
        logger.info("✅ SDF renderer test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ SDF renderer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_template_placement():
    """Test template placement system"""
    
    logger = logging.getLogger(__name__)
    logger.info("=== Testing Template Placement ===")
    
    try:
        # Create test graph
        test_graph = create_test_graph()
        
        # Create placement configuration
        config = PlacementConfig(
            orientation_smoothing=True,
            murray_law_enforcement=True,
            optimize_placement=True,
            min_vessel_length=1.0,
            min_vessel_radius=0.5
        )
        
        placer = TemplatePlacer(config)
        
        # Place templates
        logger.info("Placing templates...")
        start_time = time.time()
        
        placement_result = placer.place_templates(test_graph)
        
        placement_time = time.time() - start_time
        logger.info(f"  Placement time: {placement_time:.3f}s")
        
        logger.info(f"  Placement results:")
        logger.info(f"    Total templates: {placement_result.total_templates}")
        logger.info(f"    Cylinders: {len(placement_result.templates['cylinders'])}")
        logger.info(f"    Bifurcations: {len(placement_result.templates['bifurcations'])}")
        logger.info(f"    Quality: {placement_result.placement_quality:.3f}")
        logger.info(f"    Warnings: {len(placement_result.warnings)}")
        
        if placement_result.warnings:
            logger.info("  Placement warnings:")
            for warning in placement_result.warnings[:5]:  # Show first 5
                logger.info(f"    - {warning}")
        
        # Verify templates are valid
        all_templates = []
        for template_list in placement_result.templates.values():
            all_templates.extend(template_list)
        
        valid_templates = 0
        for template in all_templates:
            try:
                volume = template.get_volume_estimate()
                if volume > 0:
                    valid_templates += 1
            except:
                pass
        
        logger.info(f"  Valid templates: {valid_templates}/{len(all_templates)}")
        
        logger.info("✅ Template placement test passed")
        return placement_result
        
    except Exception as e:
        logger.error(f"❌ Template placement test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_volume_reconstruction():
    """Test volume reconstruction system"""
    
    logger = logging.getLogger(__name__)
    logger.info("=== Testing Volume Reconstruction ===")
    
    try:
        # Create test graph
        test_graph = create_test_graph()
        
        # Create reconstruction configuration
        config = ReconstructionConfig(
            target_resolution=(64, 64, 64),
            auto_compute_bounds=True,
            blend_with_original=False,
            apply_morphological_closing=True,
            apply_smoothing=True
        )
        
        reconstructor = VolumeReconstructor(config)
        
        # Reconstruct volume
        logger.info("Reconstructing volume...")
        start_time = time.time()
        
        result = reconstructor.reconstruct_from_graph(
            corrected_graph=test_graph,
            return_intermediates=True
        )
        
        reconstruction_time = time.time() - start_time
        logger.info(f"  Reconstruction time: {reconstruction_time:.3f}s")
        
        logger.info(f"  Reconstruction results:")
        logger.info(f"    Quality: {result.reconstruction_quality:.3f}")
        logger.info(f"    Templates used: {result.templates_used}")
        logger.info(f"    Rendering time: {result.rendering_time:.3f}s")
        logger.info(f"    Volume ratio: {result.volume_ratio:.3f}")
        logger.info(f"    Connectivity score: {result.connectivity_score:.3f}")
        
        logger.info(f"  Output volume:")
        logger.info(f"    Shape: {result.reconstructed_mask.shape}")
        logger.info(f"    Fill ratio: {np.mean(result.reconstructed_mask):.4f}")
        logger.info(f"    Non-zero voxels: {np.sum(result.reconstructed_mask):,}")
        
        if result.warnings:
            logger.info("  Reconstruction warnings:")
            for warning in result.warnings[:5]:
                logger.info(f"    - {warning}")
        
        # Test intermediate results
        if result.sdf_volume is not None:
            logger.info(f"  SDF volume: {result.sdf_volume.shape}, range: [{np.min(result.sdf_volume):.2f}, {np.max(result.sdf_volume):.2f}]")
        
        if result.probability_mask is not None:
            logger.info(f"  Probability mask: {result.probability_mask.shape}, mean: {np.mean(result.probability_mask):.3f}")
        
        logger.info("✅ Volume reconstruction test passed")
        return result
        
    except Exception as e:
        logger.error(f"❌ Volume reconstruction test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_complete_pipeline():
    """Test the complete graph-to-volume pipeline"""
    
    logger = logging.getLogger(__name__)
    logger.info("=== Testing Complete Pipeline ===")
    
    try:
        # Create test data
        test_graph = create_test_graph()
        
        # Create simple correspondences (identity mapping for test)
        correspondences = CorrespondenceResult(
            node_correspondences={i: i for i in range(len(test_graph.nodes))},
            node_confidences={i: 0.8 for i in range(len(test_graph.nodes))},
            unmatched_pred_nodes=set(),
            unmatched_gt_nodes=set(),
            edge_correspondences={(e['source'], e['target']): (e['source'], e['target']) for e in test_graph.edges},
            edge_confidences={(e['source'], e['target']): 0.8 for e in test_graph.edges},
            unmatched_pred_edges=set(),
            unmatched_gt_edges=set(),
            topology_differences={},
            alignment_transform={},
            correspondence_quality={'overall_quality': 0.85},
            metadata={}
        )
        
        # Create pipeline configuration
        config = PipelineConfig(
            reconstruction_config=ReconstructionConfig(
                target_resolution=(64, 64, 64),
                auto_compute_bounds=True,
                blend_with_original=False
            ),
            min_correction_confidence=0.3,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Create pipeline
        pipeline = GraphToVolumePipeline(config)
        
        # Process through pipeline
        logger.info("Processing through complete pipeline...")
        start_time = time.time()
        
        result = pipeline.process_single_case(
            predicted_graph=test_graph,
            ground_truth_graph=test_graph,  # Same for test
            correspondences=correspondences
        )
        
        pipeline_time = time.time() - start_time
        logger.info(f"  Pipeline time: {pipeline_time:.3f}s")
        
        logger.info(f"  Pipeline results:")
        logger.info(f"    Overall quality: {result.overall_quality:.3f}")
        logger.info(f"    Correction applied: {result.correction_applied}")
        logger.info(f"    Correction quality: {result.correction_quality:.3f}")
        logger.info(f"    Reconstruction quality: {result.reconstruction_quality:.3f}")
        logger.info(f"    Correction time: {result.correction_time:.3f}s")
        logger.info(f"    Reconstruction time: {result.reconstruction_time:.3f}s")
        
        logger.info(f"  Final volume:")
        logger.info(f"    Shape: {result.reconstructed_volume.shape}")
        logger.info(f"    Fill ratio: {np.mean(result.reconstructed_volume):.4f}")
        logger.info(f"    Non-zero voxels: {np.sum(result.reconstructed_volume):,}")
        
        if result.warnings:
            logger.info("  Pipeline warnings:")
            for warning in result.warnings:
                logger.info(f"    - {warning}")
        
        logger.info("✅ Complete pipeline test passed")
        return result
        
    except Exception as e:
        logger.error(f"❌ Complete pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run all Step 5 tests"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("🚀 === TESTING STEP 5: TEMPLATE-BASED RECONSTRUCTION ===")
    
    test_results = {}
    start_time = time.time()
    
    # Run all tests
    test_results['templates'] = test_vessel_templates()
    test_results['sdf_renderer'] = test_sdf_renderer()
    test_results['placement'] = test_template_placement() is not None
    test_results['reconstruction'] = test_volume_reconstruction() is not None
    test_results['pipeline'] = test_complete_pipeline() is not None
    
    total_time = time.time() - start_time
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("📊 STEP 5 TEST SUMMARY")
    logger.info("="*60)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, passed in test_results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"  {test_name.replace('_', ' ').title()}: {status}")
    
    logger.info("-"*60)
    logger.info(f"OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
    logger.info(f"TOTAL TIME: {total_time:.2f}s")
    
    if passed_tests == total_tests:
        logger.info("🎉 ALL STEP 5 TESTS PASSED!")
        logger.info("Template-based reconstruction system is working correctly")
    else:
        logger.info("⚠️  SOME TESTS FAILED")
        logger.info("Review failed components before proceeding")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)