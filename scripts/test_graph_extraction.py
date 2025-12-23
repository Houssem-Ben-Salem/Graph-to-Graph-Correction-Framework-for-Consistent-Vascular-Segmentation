#!/usr/bin/env python
"""
Test script for the graph extraction pipeline
"""

import sys
import numpy as np
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.graph_extraction import GraphExtractor

def create_test_mask():
    """Create a simple test mask for validation"""
    # Create a simple vessel-like structure
    mask = np.zeros((50, 50, 50), dtype=np.uint8)
    
    # Main vessel (straight tube)
    mask[20:30, 20:30, 10:40] = 1
    
    # Branch
    mask[25:35, 25:35, 25:35] = 1
    
    # Make it more vessel-like with morphological operations
    from scipy.ndimage import binary_dilation, binary_erosion
    from skimage.morphology import ball
    
    # Add some thickness variation
    structure = ball(2)
    mask = binary_dilation(mask, structure)
    mask = binary_erosion(mask, ball(1))
    
    return mask.astype(np.uint8)

def test_graph_extraction():
    """Test the complete graph extraction pipeline"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("=== Testing Graph Extraction Pipeline ===")
    
    # Create test mask
    logger.info("Creating test mask...")
    test_mask = create_test_mask()
    logger.info(f"Test mask shape: {test_mask.shape}, volume: {np.sum(test_mask)} voxels")
    
    # Define test parameters
    voxel_spacing = (1.0, 1.0, 1.0)  # 1mm isotropic
    
    # Create graph extractor with default config
    config = {
        'min_component_size': 10,
        'gaussian_sigma': 0.3,
        'base_node_density': 1.5,
        'curvature_sensitivity': 5.0,
        'max_edge_length': 8.0,
    }
    
    extractor = GraphExtractor(config)
    logger.info("Graph extractor configuration:")
    logger.info(extractor.get_extraction_summary())
    
    try:
        # Extract graph
        logger.info("Extracting graph...")
        graph = extractor.extract_graph(
            mask=test_mask,
            voxel_spacing=voxel_spacing,
            is_prediction=False,  # Treat as ground truth for testing
            patient_id="test_case"
        )
        
        # Print results
        logger.info("Graph extraction completed successfully!")
        logger.info(graph.get_summary())
        
        # Test different output formats
        logger.info("\n=== Testing Output Formats ===")
        
        # NetworkX
        nx_graph = graph.to_networkx()
        logger.info(f"NetworkX graph: {nx_graph.number_of_nodes()} nodes, {nx_graph.number_of_edges()} edges")
        
        # Adjacency matrix
        adj_matrix = graph.to_adjacency_matrix()
        logger.info(f"Adjacency matrix shape: {adj_matrix.shape}")
        
        # PyTorch Geometric (if available)
        try:
            pyg_data = graph.to_pytorch_geometric()
            logger.info(f"PyTorch Geometric data: x.shape={pyg_data.x.shape}, edge_index.shape={pyg_data.edge_index.shape}")
        except ImportError:
            logger.info("PyTorch Geometric not available, skipping")
        
        # Test serialization
        logger.info("\n=== Testing Serialization ===")
        
        # Save as pickle
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        
        pickle_path = output_dir / "test_graph.pkl"
        graph.save(pickle_path, format='pickle')
        logger.info(f"Graph saved to {pickle_path}")
        
        # Load back
        loaded_graph = graph.load(pickle_path, format='pickle')
        logger.info(f"Graph loaded successfully: {loaded_graph}")
        
        # Save as JSON
        json_path = output_dir / "test_graph.json"
        graph.save(json_path, format='json')
        logger.info(f"Graph saved to {json_path}")
        
        # Test with prediction probabilities
        logger.info("\n=== Testing with Prediction Data ===")
        
        # Create mock prediction probabilities
        pred_probs = np.random.random(test_mask.shape).astype(np.float32) * 0.3 + 0.7
        pred_probs[test_mask == 0] *= 0.1  # Lower confidence outside vessel
        
        # Create slightly corrupted prediction mask
        pred_mask = test_mask.copy()
        # Add some noise
        noise_indices = np.random.choice(np.prod(pred_mask.shape), size=50, replace=False)
        flat_mask = pred_mask.flatten()
        flat_mask[noise_indices] = 1 - flat_mask[noise_indices]
        pred_mask = flat_mask.reshape(pred_mask.shape)
        
        # Extract graph from prediction
        pred_graph = extractor.extract_graph(
            mask=pred_mask,
            voxel_spacing=voxel_spacing,
            is_prediction=True,
            prediction_probabilities=pred_probs,
            patient_id="test_prediction"
        )
        
        logger.info("Prediction graph extraction completed!")
        logger.info(pred_graph.get_summary())
        
        logger.info("\n=== All Tests Passed Successfully! ===")
        
        return True
        
    except Exception as e:
        logger.error(f"Graph extraction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_graph_extraction()
    sys.exit(0 if success else 1)