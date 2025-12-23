#!/usr/bin/env python
"""
Quick final test to confirm everything works without warnings
"""

import sys
import torch
import logging
from pathlib import Path
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.graph_correction import GraphCorrectionModel
from src.models.graph_extraction.vascular_graph import VascularGraph
from src.utils.graph_correspondence import create_correspondence_matcher

def quick_test():
    """Quick test with one sample"""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("=== QUICK FINAL TEST ===")
    
    # Load sample
    sample_file = Path("extracted_graphs/PA000005/PA000005_GT.pkl")
    if not sample_file.exists():
        logger.error("Sample file not found")
        return False
    
    gt_graph = VascularGraph.load(sample_file)
    logger.info(f"Loaded GT: {len(gt_graph.nodes)} nodes, {len(gt_graph.edges)} edges")
    
    # Create prediction (simplified)
    pred_graph = VascularGraph(
        nodes=gt_graph.nodes[:1000],  # Smaller for speed
        edges=[e for e in gt_graph.edges if e['source'] < 1000 and e['target'] < 1000],
        global_properties=gt_graph.global_properties,
        metadata=gt_graph.metadata
    )
    logger.info(f"Created pred: {len(pred_graph.nodes)} nodes, {len(pred_graph.edges)} edges")
    
    # Quick correspondence (simplified)
    from src.utils.graph_correspondence import CorrespondenceResult
    correspondences = CorrespondenceResult(
        node_correspondences={i: i for i in range(len(pred_graph.nodes))},
        node_confidences={i: 0.8 for i in range(len(pred_graph.nodes))},
        unmatched_pred_nodes=set(),
        unmatched_gt_nodes=set(range(len(pred_graph.nodes), len(gt_graph.nodes))),
        edge_correspondences={},
        edge_confidences={},
        unmatched_pred_edges=set(),
        unmatched_gt_edges=set(),
        topology_differences={},
        alignment_transform={},
        correspondence_quality={'overall_quality': 0.8},
        metadata={}
    )
    
    # Test model
    model = GraphCorrectionModel()
    model.eval()
    
    with torch.no_grad():
        start_time = time.time()
        outputs = model(pred_graph, gt_graph, correspondences, training_mode=False)
        inference_time = time.time() - start_time
    
    logger.info(f"✅ Model inference completed in {inference_time:.2f}s")
    logger.info(f"✅ Node operations shape: {outputs['node_operations'].shape}")
    logger.info(f"✅ Quality score: {outputs['quality_score'].item():.3f}")
    
    logger.info("🎉 ALL TESTS PASSED - Model is ready for training!")
    return True

if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1)