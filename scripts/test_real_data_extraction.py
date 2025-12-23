#!/usr/bin/env python
"""
Test graph extraction on real pulmonary artery dataset
"""

import sys
import numpy as np
import logging
from pathlib import Path
import time
import json
from typing import List, Dict

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.graph_extraction import GraphExtractor

def find_dataset_samples(dataset_path: Path, max_samples: int = 5) -> List[Path]:
    """Find available dataset samples"""
    samples = []
    
    if not dataset_path.exists():
        logging.warning(f"Dataset path {dataset_path} does not exist")
        return samples
    
    # Look for patient directories
    for patient_dir in dataset_path.iterdir():
        if patient_dir.is_dir() and patient_dir.name.startswith('PA'):
            image_dir = patient_dir / 'image'
            label_dir = patient_dir / 'label'
            
            if image_dir.exists() and label_dir.exists():
                # Look for NIfTI files
                image_files = list(image_dir.glob('*.nii.gz'))
                label_files = list(label_dir.glob('*.nii.gz'))
                
                if image_files and label_files:
                    samples.append(patient_dir)
                    if len(samples) >= max_samples:
                        break
    
    return samples

def load_nifti_safely(file_path: Path):
    """Safely load NIfTI file"""
    try:
        import nibabel as nib
        img = nib.load(str(file_path))
        data = img.get_fdata()
        spacing = tuple(img.header.get_zooms()[:3])
        return data, spacing, True
    except Exception as e:
        logging.error(f"Failed to load {file_path}: {e}")
        return None, None, False

def test_real_data_extraction():
    """Test graph extraction on real dataset"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("=== Testing Graph Extraction on Real Data ===")
    
    # Find dataset
    dataset_path = Path("DATASET/Parse_dataset")
    samples = find_dataset_samples(dataset_path, max_samples=3)
    
    if not samples:
        logger.error("No dataset samples found!")
        logger.error("Please ensure DATASET/Parse_dataset exists with PA* directories")
        return False
    
    logger.info(f"Found {len(samples)} dataset samples: {[s.name for s in samples]}")
    
    # Create graph extractor with optimized settings
    config = {
        'min_component_size': 27,  # As per strategy document
        'gaussian_sigma': 0.5,
        'base_node_density': 4.0,  # Higher base density to space nodes more
        'curvature_sensitivity': 5.0,  # Moderate sensitivity
        'max_edge_length': 10.0,
        'neighborhood_radius': 3.0,
        'max_nodes': 500,  # Limit nodes for large volumes
        'min_node_distance': 1.5  # Increase minimum distance between nodes
    }
    
    extractor = GraphExtractor(config)
    logger.info("Graph extractor configuration:")
    logger.info(extractor.get_extraction_summary())
    
    # Results storage
    results = {
        'config': config,
        'samples': [],
        'summary_stats': {},
        'success_rate': 0.0
    }
    
    successful_extractions = 0
    
    # Test each sample
    for sample_dir in samples:
        patient_id = sample_dir.name
        logger.info(f"\n=== Processing {patient_id} ===")
        
        sample_result = {
            'patient_id': patient_id,
            'extraction_time': 0,
            'success': False,
            'graph_stats': {},
            'quality_metrics': {},
            'error': None
        }
        
        try:
            # Find image and label files
            image_files = list((sample_dir / 'image').glob('*.nii.gz'))
            label_files = list((sample_dir / 'label').glob('*.nii.gz'))
            
            if not image_files or not label_files:
                sample_result['error'] = 'Missing image or label files'
                results['samples'].append(sample_result)
                continue
            
            image_file = image_files[0]  # Use first available
            label_file = label_files[0]  # Use first available
            
            logger.info(f"Loading: {label_file.name}")
            
            # Load ground truth mask
            mask_data, voxel_spacing, load_success = load_nifti_safely(label_file)
            
            if not load_success:
                sample_result['error'] = 'Failed to load mask'
                results['samples'].append(sample_result)
                continue
            
            # Convert to binary if needed
            if mask_data.max() > 1:
                mask_data = (mask_data > 0).astype(np.uint8)
            else:
                mask_data = mask_data.astype(np.uint8)
            
            logger.info(f"Mask shape: {mask_data.shape}, volume: {np.sum(mask_data)} voxels")
            logger.info(f"Voxel spacing: {voxel_spacing} mm")
            
            # Skip if mask is empty
            if np.sum(mask_data) == 0:
                sample_result['error'] = 'Empty mask'
                results['samples'].append(sample_result)
                continue
            
            # Extract graph
            start_time = time.time()
            
            graph = extractor.extract_graph(
                mask=mask_data,
                voxel_spacing=voxel_spacing,
                is_prediction=False,  # Ground truth
                patient_id=patient_id
            )
            
            extraction_time = time.time() - start_time
            sample_result['extraction_time'] = extraction_time
            
            # Check if extraction was successful
            if graph.metadata.get('extraction_success', False):
                sample_result['success'] = True
                successful_extractions += 1
                
                # Collect statistics
                sample_result['graph_stats'] = {
                    'num_nodes': graph.global_properties.get('num_nodes', 0),
                    'num_edges': graph.global_properties.get('num_edges', 0),
                    'total_length_mm': graph.global_properties.get('total_length_mm', 0),
                    'average_radius_voxels': graph.global_properties.get('average_radius_voxels', 0),
                    'complexity_score': graph.global_properties.get('complexity_score', 0),
                    'is_connected': graph.global_properties.get('is_connected', False),
                    'node_type_counts': graph.global_properties.get('node_type_counts', {})
                }
                
                sample_result['quality_metrics'] = graph.metadata.get('extraction_parameters', {}).get('quality_metrics', {})
                
                logger.info(f"✅ Success! Extraction time: {extraction_time:.2f}s")
                logger.info(graph.get_summary())
                
                # Save graph
                output_dir = Path("test_output/real_data")
                output_dir.mkdir(parents=True, exist_ok=True)
                graph.save(output_dir / f"{patient_id}_graph.pkl")
                
            else:
                sample_result['error'] = graph.metadata.get('extraction_error', 'Unknown error')
                logger.error(f"❌ Extraction failed: {sample_result['error']}")
        
        except Exception as e:
            sample_result['error'] = str(e)
            logger.error(f"❌ Exception during processing: {e}")
            import traceback
            traceback.print_exc()
        
        results['samples'].append(sample_result)
    
    # Calculate summary statistics
    results['success_rate'] = successful_extractions / len(samples)
    
    if successful_extractions > 0:
        successful_samples = [s for s in results['samples'] if s['success']]
        
        # Aggregate statistics
        total_nodes = [s['graph_stats']['num_nodes'] for s in successful_samples]
        total_edges = [s['graph_stats']['num_edges'] for s in successful_samples]
        total_lengths = [s['graph_stats']['total_length_mm'] for s in successful_samples if s['graph_stats']['total_length_mm']]
        extraction_times = [s['extraction_time'] for s in successful_samples]
        quality_scores = [s['quality_metrics'].get('overall_quality_score', 0) for s in successful_samples]
        
        results['summary_stats'] = {
            'avg_nodes': np.mean(total_nodes) if total_nodes else 0,
            'avg_edges': np.mean(total_edges) if total_edges else 0,
            'avg_length_mm': np.mean(total_lengths) if total_lengths else 0,
            'avg_extraction_time': np.mean(extraction_times) if extraction_times else 0,
            'avg_quality_score': np.mean(quality_scores) if quality_scores else 0,
            'nodes_range': [int(np.min(total_nodes)), int(np.max(total_nodes))] if total_nodes else [0, 0],
            'edges_range': [int(np.min(total_edges)), int(np.max(total_edges))] if total_edges else [0, 0]
        }
    
    # Save results (convert numpy types to native Python types)
    results_file = Path("test_output/real_data_results.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Custom JSON encoder for numpy types
    import numpy as np
    
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
    
    results_clean = convert_numpy_types(results)
    
    with open(results_file, 'w') as f:
        json.dump(results_clean, f, indent=2)
    
    # Print summary
    logger.info(f"\n=== REAL DATA EXTRACTION SUMMARY ===")
    logger.info(f"Samples processed: {len(samples)}")
    logger.info(f"Successful extractions: {successful_extractions}")
    logger.info(f"Success rate: {results['success_rate']:.1%}")
    
    if successful_extractions > 0:
        stats = results['summary_stats']
        logger.info(f"Average nodes: {stats['avg_nodes']:.1f} (range: {stats['nodes_range']})")
        logger.info(f"Average edges: {stats['avg_edges']:.1f} (range: {stats['edges_range']})")
        logger.info(f"Average length: {stats['avg_length_mm']:.1f} mm")
        logger.info(f"Average extraction time: {stats['avg_extraction_time']:.2f}s")
        logger.info(f"Average quality score: {stats['avg_quality_score']:.3f}")
    
    logger.info(f"Results saved to: {results_file}")
    
    return results['success_rate'] > 0.5  # Consider successful if >50% success rate

if __name__ == "__main__":
    success = test_real_data_extraction()
    sys.exit(0 if success else 1)