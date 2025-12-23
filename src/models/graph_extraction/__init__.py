"""
Graph Extraction Module for Vascular Segmentation

This module implements the complete pipeline for converting binary segmentation masks
into graph representations suitable for graph neural network processing.

Key Components:
- GraphExtractor: Main pipeline orchestrator
- VascularGraph: Graph data structure
- CenterlineExtractor: 3D skeletonization and centerline extraction
- NodePlacer: Strategic node placement along vessels
- NodeAttributeExtractor: Comprehensive node attribute extraction
- EdgeCreator: Edge creation and attribute computation

The pipeline follows the strategy outlined in graph_correction_strategy.md:
1. Preprocessing: Mask cleaning and quality assessment
2. Centerline Extraction: 3D skeletonization with refinement
3. Node Placement: Adaptive placement based on critical points and curvature
4. Attribute Extraction: Geometric, structural, and context attributes
5. Edge Creation: Connectivity following vessel structure
6. Graph Assembly: Final VascularGraph with comprehensive metadata
"""

from .graph_extractor import GraphExtractor
from .vascular_graph import VascularGraph
from .centerline_extraction import CenterlineExtractor
from .node_placement import NodePlacer
from .node_attributes import NodeAttributeExtractor
from .edge_creation import EdgeCreator

__all__ = [
    'GraphExtractor',
    'VascularGraph',
    'CenterlineExtractor',
    'NodePlacer',
    'NodeAttributeExtractor',
    'EdgeCreator'
]