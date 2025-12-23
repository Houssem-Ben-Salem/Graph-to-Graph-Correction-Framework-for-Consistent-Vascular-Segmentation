#!/usr/bin/env python
"""
Test graph extraction on a single sample with detailed monitoring
"""

import sys
import numpy as np
import logging
from pathlib import Path
import time
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.graph_extraction import GraphExtractor

def test_single_sample():
    """Test graph extraction on one sample with monitoring"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("=== Testing Single Sample Extraction ===")
    
    # Find one sample
    dataset_path = Path("DATASET/Parse_dataset")
    sample_dirs = [d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith('PA')]
    
    if not sample_dirs:
        logger.error("No dataset samples found!")
        return False
    
    # Use first sample
    sample_dir = sample_dirs[0]
    patient_id = sample_dir.name
    
    logger.info(f"Using sample: {patient_id}")
    
    # Find label file
    label_files = list((sample_dir / 'label').glob('*.nii.gz'))
    if not label_files:
        logger.error("No label files found!")
        return False
    
    label_file = label_files[0]
    
    # Load data
    try:
        import nibabel as nib
        img = nib.load(str(label_file))
        mask_data = img.get_fdata()
        voxel_spacing = tuple(img.header.get_zooms()[:3])
        
        # Convert to binary
        mask_data = (mask_data > 0).astype(np.uint8)
        
        logger.info(f"Loaded mask: shape={mask_data.shape}, volume={np.sum(mask_data)} voxels")
        logger.info(f"Voxel spacing: {voxel_spacing} mm")
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return False
    
    # Create a smaller crop for testing
    logger.info("\nExtracting a crop for testing...")
    
    # Find center of mass
    coords = np.argwhere(mask_data)
    if len(coords) == 0:
        logger.error("Empty mask!")
        return False
    
    center = np.mean(coords, axis=0).astype(int)
    crop_size = (128, 128, 128)
    
    # Extract crop
    z_start = max(0, center[0] - crop_size[0]//2)
    y_start = max(0, center[1] - crop_size[1]//2)
    x_start = max(0, center[2] - crop_size[2]//2)
    
    z_end = min(mask_data.shape[0], z_start + crop_size[0])
    y_end = min(mask_data.shape[1], y_start + crop_size[1])
    x_end = min(mask_data.shape[2], x_start + crop_size[2])
    
    crop_mask = mask_data[z_start:z_end, y_start:y_end, x_start:x_end]
    crop_volume = np.sum(crop_mask)
    
    logger.info(f"Crop: shape={crop_mask.shape}, volume={crop_volume} voxels")
    
    # Test with different configurations
    configs = [
        {
            'name': 'Conservative',
            'min_component_size': 27,
            'base_node_density': 5.0,  # Very sparse
            'max_nodes': 200,
            'max_edge_length': 8.0,
        },
        {
            'name': 'Balanced',
            'min_component_size': 27,
            'base_node_density': 3.0,  # Moderate
            'max_nodes': 300,
            'max_edge_length': 10.0,
        }
    ]
    
    for config_dict in configs:
        logger.info(f"\n=== Testing {config_dict['name']} Configuration ===")
        
        config = config_dict.copy()
        config_name = config.pop('name')
        
        # Add common parameters
        config.update({
            'gaussian_sigma': 0.5,
            'curvature_sensitivity': 5.0,
            'neighborhood_radius': 3.0,
            'min_node_distance': 1.5
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
                logger.info(f"✅ Success! Time: {extraction_time:.2f}s")
                logger.info(graph.get_summary())
                
                # Save graph
                output_dir = Path("test_output/single_sample")
                output_dir.mkdir(parents=True, exist_ok=True)
                graph.save(output_dir / f"{patient_id}_{config_name}.pkl")
                
                # Visualize if small enough
                if graph.global_properties['num_nodes'] < 100:
                    try:
                        nx_graph = graph.to_networkx()
                        
                        # Simple visualization
                        import networkx as nx
                        plt.figure(figsize=(10, 8))
                        pos = nx.spring_layout(nx_graph, dim=3)
                        nx.draw(nx_graph, pos, node_size=50, with_labels=False, 
                               edge_color='gray', node_color='red')
                        plt.title(f"{config_name} Graph ({graph.global_properties['num_nodes']} nodes)")
                        plt.savefig(output_dir / f"{patient_id}_{config_name}_graph.png")
                        plt.close()
                        logger.info(f"Graph visualization saved")
                    except:
                        pass
                
            else:
                logger.error(f"❌ Extraction failed: {graph.metadata.get('extraction_error', 'Unknown')}")
                
        except Exception as e:
            logger.error(f"❌ Exception: {e}")
            import traceback
            traceback.print_exc()
    
    return True

if __name__ == "__main__":
    success = test_single_sample()
    sys.exit(0 if success else 1)