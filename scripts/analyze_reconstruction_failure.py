#!/usr/bin/env python3
"""
Analyze why reconstruction is failing
"""

import sys
sys.path.append('.')

import numpy as np
from pathlib import Path
import pickle
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def analyze_reconstruction_failure():
    """Analyze the reconstruction failure"""
    
    # Load the results
    results_path = Path('experiments/lightweight_results_PA000005.pkl')
    if results_path.exists():
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        logger.info(f"Loaded results: {results['metrics']}")
    
    # Load the graphs
    pred_graph_path = Path('extracted_graphs/PA000005/PA000005_PRED.pkl')
    with open(pred_graph_path, 'rb') as f:
        pred_graph = pickle.load(f)
    
    logger.info("\n=== ANALYSIS ===")
    logger.info(f"Original graph: {len(pred_graph.nodes)} nodes, {len(pred_graph.edges)} edges")
    logger.info(f"After correction: 813 nodes (removed 519 = 39%)")
    logger.info(f"After correction: 6273 edges (removed 4809 = 43%)")
    
    # The problem is clear:
    # 1. We're removing 39% of nodes - this is too aggressive
    # 2. The remaining skeleton is too sparse to reconstruct the full vessel
    # 3. We need a different approach
    
    logger.info("\n=== ROOT CAUSE ===")
    logger.info("1. Graph correction is too aggressive - removing 39% of nodes")
    logger.info("2. Remaining skeleton (813 nodes) is too sparse for full vessel reconstruction")
    logger.info("3. Simple sphere-based reconstruction can't fill the gaps")
    
    logger.info("\n=== SOLUTION ===")
    logger.info("Instead of reconstructing from scratch, we should:")
    logger.info("1. Keep the original U-Net prediction as base")
    logger.info("2. Use graph correction to REFINE specific regions")
    logger.info("3. Remove only the most confident false positives")
    logger.info("4. Add missing connections identified by the graph")


if __name__ == '__main__':
    analyze_reconstruction_failure()