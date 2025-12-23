#!/usr/bin/env python
"""
Quick test of optimized graph extraction
"""

import sys
import numpy as np
import logging
from pathlib import Path
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.graph_extraction import GraphExtractor

def test_optimized_extraction():
    """Test optimized graph extraction with conservative settings"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("=== Testing Optimized Graph Extraction ===")
    
    # Create a larger test mask
    mask = np.zeros((128, 128, 128), dtype=np.uint8)
    
    # Create a more complex vessel structure
    # Main trunk
    mask[50:80, 50:80, 20:100] = 1
    
    # Branches
    mask[60:90, 60:90, 40:80] = 1
    mask[40:70, 40:70, 60:100] = 1
    
    # Add some thickness variation
    from scipy.ndimage import binary_dilation, binary_erosion
    from skimage.morphology import ball
    
    structure = ball(3)
    mask = binary_dilation(mask, structure)
    mask = binary_erosion(mask, ball(2))
    
    logger.info(f"Test mask shape: {mask.shape}, volume: {np.sum(mask)} voxels")
    
    # Conservative configuration for testing
    config = {
        'min_component_size': 27,
        'gaussian_sigma': 0.5,
        'base_node_density': 3.0,  # Higher density to test scaling
        'curvature_sensitivity': 5.0,
        'max_edge_length': 8.0,
        'max_nodes': 200,  # Conservative limit
    }
    
    extractor = GraphExtractor(config)
    logger.info("Graph extractor configuration:")
    logger.info(extractor.get_extraction_summary())
    
    try:
        start_time = time.time()
        
        graph = extractor.extract_graph(
            mask=mask,
            voxel_spacing=(1.0, 1.0, 1.0),
            is_prediction=False,
            patient_id="test_optimized"
        )
        
        extraction_time = time.time() - start_time
        
        if graph.metadata.get('extraction_success', False):
            logger.info(f"✅ Success! Extraction time: {extraction_time:.2f}s")
            logger.info(graph.get_summary())
            
            # Check if node count is reasonable
            num_nodes = graph.global_properties.get('num_nodes', 0)
            if num_nodes <= config['max_nodes']:
                logger.info(f"✅ Node count within limits: {num_nodes}/{config['max_nodes']}")
            else:
                logger.warning(f"⚠️ Node count exceeded limit: {num_nodes}/{config['max_nodes']}")
            
            return True
        else:
            logger.error(f"❌ Extraction failed: {graph.metadata.get('extraction_error', 'Unknown')}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_optimized_extraction()
    sys.exit(0 if success else 1)