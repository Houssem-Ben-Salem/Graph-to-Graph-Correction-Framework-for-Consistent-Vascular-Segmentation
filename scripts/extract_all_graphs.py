#!/usr/bin/env python
"""
Extract dense graphs from all available dataset samples
"""

import sys
import numpy as np
import logging
from pathlib import Path
import time
import json
from typing import List, Dict, Tuple
from tqdm import tqdm
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.graph_extraction import GraphExtractor

def get_dense_config() -> Dict:
    """Get optimized dense configuration for better information richness"""
    return {
        # Preprocessing
        'min_component_size': 27,
        'gaussian_sigma': 0.5,
        'fill_holes': True,
        
        # Node placement - DENSER CONFIGURATION
        'base_node_density': 1.8,      # Denser (was 3.0)
        'min_node_distance': 0.4,      # Closer nodes (was 0.8) 
        'max_node_distance': 2.5,      # Shorter max distance
        'max_nodes': 2500,             # More nodes allowed (was 800)
        'bifurcation_buffer': 1,       # Keep minimal buffer
        
        # Curvature and connectivity
        'curvature_sensitivity': 6.0,  # More sensitive to curves
        'max_edge_length': 12.0,       # Moderate edge length
        'connectivity_tolerance': 1.5,  # Tighter connectivity
        
        # Context
        'neighborhood_radius': 2.5,
        
        # Centerline
        'centerline_smoothing_sigma': 0.3,  # Less smoothing to preserve detail
        'spur_length_threshold': 2,         # Keep more small branches
    }

def find_all_samples(dataset_path: Path) -> List[Path]:
    """Find all available patient samples"""
    samples = []
    
    if not dataset_path.exists():
        return samples
    
    # Look for patient directories
    for patient_dir in sorted(dataset_path.iterdir()):
        if patient_dir.is_dir() and patient_dir.name.startswith('PA'):
            image_dir = patient_dir / 'image'
            label_dir = patient_dir / 'label'
            
            if image_dir.exists() and label_dir.exists():
                # Check for NIfTI files
                image_files = list(image_dir.glob('*.nii.gz'))
                label_files = list(label_dir.glob('*.nii.gz'))
                
                if image_files and label_files:
                    samples.append(patient_dir)
    
    return samples

def load_and_process_volume(file_path: Path, crop_strategy: str = 'smart') -> Tuple[np.ndarray, Tuple[float, float, float], bool]:
    """Load and potentially crop volume for efficient processing"""
    try:
        import nibabel as nib
        img = nib.load(str(file_path))
        data = img.get_fdata()
        spacing = tuple(img.header.get_zooms()[:3])
        
        # Convert to binary
        if data.max() > 1:
            data = (data > 0).astype(np.uint8)
        else:
            data = data.astype(np.uint8)
        
        original_volume = np.sum(data)
        
        if original_volume == 0:
            return None, spacing, False
        
        # Crop strategy for large volumes
        if crop_strategy == 'smart' and data.shape[0] * data.shape[1] * data.shape[2] > 200**3:
            # Find vessel region and extract manageable crop
            coords = np.argwhere(data)
            if len(coords) == 0:
                return None, spacing, False
            
            # Find center and extent
            center = np.mean(coords, axis=0).astype(int)
            min_coords = np.min(coords, axis=0)
            max_coords = np.max(coords, axis=0)
            
            # Determine crop size (aim for ~150^3 max)
            vessel_span = max_coords - min_coords
            max_crop_size = 180
            
            # Calculate crop bounds
            crop_size = np.minimum(vessel_span + 40, max_crop_size)  # Add padding
            crop_size = np.maximum(crop_size, 100)  # Minimum size
            
            # Center crop around vessel region
            start = np.maximum(0, center - crop_size // 2)
            end = np.minimum(data.shape, start + crop_size)
            start = np.maximum(0, end - crop_size)  # Adjust if near boundary
            
            cropped_data = data[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
            
            # Check if crop still has reasonable volume
            cropped_volume = np.sum(cropped_data)
            if cropped_volume < original_volume * 0.3:  # Lost too much
                # Try larger crop
                crop_size = np.minimum(vessel_span + 80, 220)
                start = np.maximum(0, center - crop_size // 2)
                end = np.minimum(data.shape, start + crop_size)
                start = np.maximum(0, end - crop_size)
                cropped_data = data[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
            
            return cropped_data, spacing, True
        
        return data, spacing, True
        
    except Exception as e:
        logging.error(f"Failed to load {file_path}: {e}")
        return None, None, False

def extract_sample_graphs(sample_dir: Path, extractor: GraphExtractor, output_dir: Path, logger: logging.Logger) -> Dict:
    """Extract graphs for a single sample"""
    patient_id = sample_dir.name
    
    sample_result = {
        'patient_id': patient_id,
        'gt_extraction': {'success': False, 'time': 0, 'stats': {}},
        'pred_extraction': {'success': False, 'time': 0, 'stats': {}},
        'quality_comparison': {}
    }
    
    # Create patient output directory
    patient_output_dir = output_dir / patient_id
    patient_output_dir.mkdir(exist_ok=True)
    
    # Find files
    label_files = list((sample_dir / 'label').glob('*.nii.gz'))
    image_files = list((sample_dir / 'image').glob('*.nii.gz'))
    
    if not label_files:
        sample_result['error'] = 'No label files found'
        return sample_result
    
    label_file = label_files[0]
    
    logger.info(f"Processing {patient_id}...")
    
    # Extract ground truth graph
    try:
        logger.info(f"  Loading ground truth: {label_file.name}")
        gt_data, spacing, success = load_and_process_volume(label_file, crop_strategy='smart')
        
        if success and gt_data is not None:
            logger.info(f"  GT volume: {gt_data.shape}, vessel voxels: {np.sum(gt_data)}")
            
            start_time = time.time()
            
            gt_graph = extractor.extract_graph(
                mask=gt_data,
                voxel_spacing=spacing,
                is_prediction=False,
                patient_id=f"{patient_id}_GT"
            )
            
            extraction_time = time.time() - start_time
            
            if gt_graph.metadata.get('extraction_success', False):
                sample_result['gt_extraction']['success'] = True
                sample_result['gt_extraction']['time'] = extraction_time
                sample_result['gt_extraction']['stats'] = {
                    'num_nodes': gt_graph.global_properties.get('num_nodes', 0),
                    'num_edges': gt_graph.global_properties.get('num_edges', 0),
                    'total_length_mm': gt_graph.global_properties.get('total_length_mm', 0),
                    'quality_score': gt_graph.metadata.get('extraction_parameters', {}).get('quality_metrics', {}).get('overall_quality_score', 0),
                    'node_type_counts': gt_graph.global_properties.get('node_type_counts', {}),
                    'complexity_score': gt_graph.global_properties.get('complexity_score', 0)
                }
                
                # Save graph
                gt_graph.save(patient_output_dir / f"{patient_id}_GT.pkl")
                logger.info(f"  ✅ GT extracted: {sample_result['gt_extraction']['stats']['num_nodes']} nodes, {extraction_time:.1f}s")
            else:
                sample_result['gt_extraction']['error'] = gt_graph.metadata.get('extraction_error', 'Unknown')
                logger.error(f"  ❌ GT extraction failed: {sample_result['gt_extraction']['error']}")
        else:
            sample_result['gt_extraction']['error'] = 'Failed to load GT data'
            logger.error(f"  ❌ Failed to load GT data")
    
    except Exception as e:
        sample_result['gt_extraction']['error'] = str(e)
        logger.error(f"  ❌ GT extraction exception: {e}")
    
    # TODO: Add predicted mask extraction when U-Net predictions are available
    # For now, we'll focus on ground truth graphs
    
    return sample_result

def extract_all_graphs():
    """Extract graphs from all available samples"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('graph_extraction.log')
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("=== Dense Graph Extraction for All Samples ===")
    
    # Find all samples
    dataset_path = Path("DATASET/Parse_dataset")
    samples = find_all_samples(dataset_path)
    
    if not samples:
        logger.error("No dataset samples found!")
        return False
    
    logger.info(f"Found {len(samples)} samples to process")
    logger.info(f"Samples: {[s.name for s in samples[:10]]}{'...' if len(samples) > 10 else ''}")
    
    # Create dense configuration
    config = get_dense_config()
    logger.info("Dense extraction configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Create extractor
    extractor = GraphExtractor(config)
    
    # Create output directory
    output_dir = Path("extracted_graphs")
    output_dir.mkdir(exist_ok=True)
    
    # Save configuration
    with open(output_dir / 'extraction_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Process all samples
    results = []
    successful_extractions = 0
    
    # Use tqdm for progress tracking
    for sample_dir in tqdm(samples, desc="Extracting graphs"):
        try:
            result = extract_sample_graphs(sample_dir, extractor, output_dir, logger)
            results.append(result)
            
            if result['gt_extraction']['success']:
                successful_extractions += 1
                
        except Exception as e:
            logger.error(f"Failed to process {sample_dir.name}: {e}")
            results.append({
                'patient_id': sample_dir.name,
                'error': str(e),
                'gt_extraction': {'success': False}
            })
    
    # Calculate summary statistics
    success_rate = successful_extractions / len(samples) if samples else 0
    
    # Analyze results
    if successful_extractions > 0:
        successful_results = [r for r in results if r['gt_extraction']['success']]
        
        # Aggregate statistics
        node_counts = [r['gt_extraction']['stats']['num_nodes'] for r in successful_results]
        edge_counts = [r['gt_extraction']['stats']['num_edges'] for r in successful_results]
        quality_scores = [r['gt_extraction']['stats']['quality_score'] for r in successful_results]
        extraction_times = [r['gt_extraction']['time'] for r in successful_results]
        
        summary_stats = {
            'total_samples': len(samples),
            'successful_extractions': successful_extractions,
            'success_rate': success_rate,
            'avg_nodes': float(np.mean(node_counts)),
            'avg_edges': float(np.mean(edge_counts)),
            'avg_quality_score': float(np.mean(quality_scores)),
            'avg_extraction_time': float(np.mean(extraction_times)),
            'node_range': [int(np.min(node_counts)), int(np.max(node_counts))],
            'edge_range': [int(np.min(edge_counts)), int(np.max(edge_counts))],
        }
        
        # Node type analysis
        all_node_types = {}
        for r in successful_results:
            node_types = r['gt_extraction']['stats']['node_type_counts']
            for node_type, count in node_types.items():
                if node_type not in all_node_types:
                    all_node_types[node_type] = []
                all_node_types[node_type].append(count)
        
        avg_node_types = {k: float(np.mean(v)) for k, v in all_node_types.items()}
        summary_stats['avg_node_type_counts'] = avg_node_types
        
    else:
        summary_stats = {
            'total_samples': len(samples),
            'successful_extractions': 0,
            'success_rate': 0.0
        }
    
    # Save results
    results_data = {
        'extraction_config': config,
        'summary_stats': summary_stats,
        'sample_results': results
    }
    
    # Convert numpy types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    results_clean = convert_numpy_types(results_data)
    
    with open(output_dir / 'extraction_results.json', 'w') as f:
        json.dump(results_clean, f, indent=2)
    
    # Print final summary
    logger.info(f"\n=== EXTRACTION COMPLETE ===")
    logger.info(f"Samples processed: {len(samples)}")
    logger.info(f"Successful extractions: {successful_extractions}")
    logger.info(f"Success rate: {success_rate:.1%}")
    
    if successful_extractions > 0:
        logger.info(f"Average nodes: {summary_stats['avg_nodes']:.1f} (range: {summary_stats['node_range']})")
        logger.info(f"Average edges: {summary_stats['avg_edges']:.1f} (range: {summary_stats['edge_range']})")
        logger.info(f"Average quality: {summary_stats['avg_quality_score']:.3f}")
        logger.info(f"Average time: {summary_stats['avg_extraction_time']:.1f}s")
        
        avg_types = summary_stats['avg_node_type_counts']
        logger.info(f"Average node types:")
        for node_type, count in avg_types.items():
            logger.info(f"  {node_type}: {count:.1f}")
    
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Individual graphs saved in: {output_dir}/")
    
    return success_rate > 0.5

if __name__ == "__main__":
    success = extract_all_graphs()
    sys.exit(0 if success else 1)