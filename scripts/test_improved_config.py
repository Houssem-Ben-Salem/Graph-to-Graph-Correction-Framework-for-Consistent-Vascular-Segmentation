#!/usr/bin/env python
"""
Test improved graph extraction configuration for better node representation
"""

import sys
import numpy as np
import logging
from pathlib import Path
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.graph_extraction import GraphExtractor

def test_improved_config():
    """Test improved configuration for better node representation"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("=== Testing Improved Graph Extraction Configuration ===")
    
    # Use same crop as before
    dataset_path = Path("DATASET/Parse_dataset")
    sample_dirs = [d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith('PA')]
    
    if not sample_dirs:
        logger.error("No dataset samples found!")
        return False
    
    sample_dir = sample_dirs[0]
    patient_id = sample_dir.name
    
    # Load and crop data (same as before)
    label_files = list((sample_dir / 'label').glob('*.nii.gz'))
    label_file = label_files[0]
    
    try:
        import nibabel as nib
        img = nib.load(str(label_file))
        mask_data = img.get_fdata()
        voxel_spacing = tuple(img.header.get_zooms()[:3])
        mask_data = (mask_data > 0).astype(np.uint8)
        
        # Extract same crop
        coords = np.argwhere(mask_data)
        center = np.mean(coords, axis=0).astype(int)
        crop_size = (128, 128, 128)
        
        z_start = max(0, center[0] - crop_size[0]//2)
        y_start = max(0, center[1] - crop_size[1]//2)
        x_start = max(0, center[2] - crop_size[2]//2)
        
        z_end = min(mask_data.shape[0], z_start + crop_size[0])
        y_end = min(mask_data.shape[1], y_start + crop_size[1])
        x_end = min(mask_data.shape[2], x_start + crop_size[2])
        
        crop_mask = mask_data[z_start:z_end, y_start:y_end, x_start:x_end]
        
        logger.info(f"Using crop: shape={crop_mask.shape}, volume={np.sum(crop_mask)} voxels")
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return False
    
    # Test improved configurations
    configs = [
        {
            'name': 'Dense_Regular',
            'min_component_size': 27,
            'base_node_density': 2.5,  # Denser to get more regular nodes
            'min_node_distance': 1.0,  # Closer spacing
            'max_nodes': 600,  # More nodes allowed
            'max_edge_length': 12.0,  # Longer edges
            'bifurcation_buffer': 1,  # Less buffer nodes
        },
        {
            'name': 'Optimal',
            'min_component_size': 27,
            'base_node_density': 3.0,  # Even denser
            'min_node_distance': 0.8,  # Even closer
            'max_nodes': 800,  # More nodes
            'max_edge_length': 15.0,  # Longer edges for better connectivity
            'bifurcation_buffer': 1,
        }
    ]
    
    results = []
    
    for config_dict in configs:
        logger.info(f"\n=== Testing {config_dict['name']} Configuration ===")
        
        config = config_dict.copy()
        config_name = config.pop('name')
        
        # Add common parameters
        config.update({
            'gaussian_sigma': 0.5,
            'curvature_sensitivity': 5.0,
            'neighborhood_radius': 3.0,
        })
        
        extractor = GraphExtractor(config)
        
        try:
            start_time = time.time()
            
            graph = extractor.extract_graph(
                mask=crop_mask,
                voxel_spacing=voxel_spacing,
                is_prediction=False,
                patient_id=f"{patient_id}_crop_{config_name}"
            )
            
            extraction_time = time.time() - start_time
            
            if graph.metadata.get('extraction_success', False):
                stats = graph.global_properties
                node_types = stats.get('node_type_counts', {})
                
                result = {
                    'config': config_name,
                    'time': extraction_time,
                    'nodes': stats.get('num_nodes', 0),
                    'edges': stats.get('num_edges', 0),
                    'bifurcations': node_types.get('bifurcations', 0),
                    'endpoints': node_types.get('endpoints', 0),
                    'regular': node_types.get('regular', 0),
                    'buffer': node_types.get('buffer', 0),
                    'total_length_mm': stats.get('total_length_mm', 0),
                    'quality_score': graph.metadata.get('extraction_parameters', {}).get('quality_metrics', {}).get('overall_quality_score', 0),
                    'connected': stats.get('is_connected', False)
                }
                
                results.append(result)
                
                logger.info(f"✅ Success! Time: {extraction_time:.2f}s")
                logger.info(f"Nodes: {result['nodes']} (bifur: {result['bifurcations']}, end: {result['endpoints']}, regular: {result['regular']}, buffer: {result['buffer']})")
                logger.info(f"Edges: {result['edges']}, Length: {result['total_length_mm']:.1f}mm")
                logger.info(f"Quality: {result['quality_score']:.3f}, Connected: {result['connected']}")
                
                # Save graph
                output_dir = Path("test_output/improved_configs")
                output_dir.mkdir(parents=True, exist_ok=True)
                graph.save(output_dir / f"{patient_id}_{config_name}.pkl")
                
            else:
                logger.error(f"❌ Extraction failed: {graph.metadata.get('extraction_error', 'Unknown')}")
                
        except Exception as e:
            logger.error(f"❌ Exception: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary comparison
    if results:
        logger.info(f"\n=== CONFIGURATION COMPARISON ===")
        logger.info(f"{'Config':<15} {'Nodes':<8} {'Regular':<8} {'Edges':<8} {'Length(mm)':<12} {'Quality':<8} {'Time(s)':<8}")
        logger.info("-" * 80)
        
        for r in results:
            logger.info(f"{r['config']:<15} {r['nodes']:<8} {r['regular']:<8} {r['edges']:<8} {r['total_length_mm']:<12.0f} {r['quality_score']:<8.3f} {r['time']:<8.2f}")
        
        # Recommendation
        best_regular = max(results, key=lambda x: x['regular'])
        logger.info(f"\n💡 Best for regular nodes: {best_regular['config']} with {best_regular['regular']} regular nodes")
        
        if best_regular['regular'] > 0:
            logger.info("✅ Successfully generated regular nodes along vessel segments!")
        else:
            logger.warning("⚠️  Still no regular nodes - need to investigate node placement algorithm")
    
    return len(results) > 0

if __name__ == "__main__":
    success = test_improved_config()
    sys.exit(0 if success else 1)