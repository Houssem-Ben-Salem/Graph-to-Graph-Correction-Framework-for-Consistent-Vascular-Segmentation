#!/usr/bin/env python
"""
Extract graphs from U-Net predicted masks using the same configuration as GT
"""

import sys
import numpy as np
import logging
from pathlib import Path
import time
import json
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.graph_extraction import GraphExtractor
from scripts.extract_all_graphs import get_dense_config, load_and_process_volume

def find_prediction_files(predictions_dir: Path, extracted_graphs_dir: Path):
    """Find all U-Net prediction files and match with existing GT graphs"""
    
    # Get list of patients with GT graphs
    gt_patients = []
    for patient_dir in extracted_graphs_dir.iterdir():
        if patient_dir.is_dir() and patient_dir.name.startswith('PA'):
            gt_file = patient_dir / f"{patient_dir.name}_GT.pkl"
            if gt_file.exists():
                gt_patients.append(patient_dir.name)
    
    # Find corresponding prediction files
    prediction_pairs = []
    
    # Check different possible structures
    for patient_id in gt_patients:
        # Option 1: predictions_dir/PA000070/prediction.nii.gz
        pred_file1 = predictions_dir / patient_id / "prediction.nii.gz"
        
        # Option 2: predictions_dir/PA000070_pred.nii.gz
        pred_file2 = predictions_dir / f"{patient_id}_pred.nii.gz"
        
        # Option 3: predictions_dir/PA000070.nii.gz
        pred_file3 = predictions_dir / f"{patient_id}.nii.gz"
        
        if pred_file1.exists():
            prediction_pairs.append((patient_id, pred_file1))
        elif pred_file2.exists():
            prediction_pairs.append((patient_id, pred_file2))
        elif pred_file3.exists():
            prediction_pairs.append((patient_id, pred_file3))
    
    return prediction_pairs, gt_patients

def extract_predicted_graphs(predictions_dir: Path, extracted_graphs_dir: Path, 
                           threshold: float = 0.5, use_confidence: bool = True):
    """Extract graphs from U-Net predictions"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('graph_extraction_predictions.log')
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("=== Graph Extraction from U-Net Predictions ===")
    
    # Find prediction files
    prediction_pairs, gt_patients = find_prediction_files(predictions_dir, extracted_graphs_dir)
    
    if not prediction_pairs:
        logger.error(f"No prediction files found in {predictions_dir}")
        logger.info(f"Expected patients with GT graphs: {gt_patients[:5]}...")
        return False
    
    logger.info(f"Found {len(prediction_pairs)} prediction files matching GT graphs")
    
    # Use same configuration as GT extraction
    config = get_dense_config()
    logger.info("Using same dense configuration as GT extraction")
    
    # Create extractor
    extractor = GraphExtractor(config)
    
    # Process predictions
    results = []
    successful_extractions = 0
    
    for patient_id, pred_file in tqdm(prediction_pairs, desc="Extracting predicted graphs"):
        try:
            logger.info(f"\nProcessing {patient_id} prediction...")
            
            # Load prediction
            pred_data, spacing, success = load_and_process_volume(pred_file, crop_strategy='smart')
            
            if not success or pred_data is None:
                logger.error(f"Failed to load prediction for {patient_id}")
                continue
            
            # Handle probability maps vs binary masks
            if pred_data.dtype == np.float32 or pred_data.dtype == np.float64:
                # It's a probability map
                if use_confidence:
                    confidence_map = pred_data.copy()
                else:
                    confidence_map = None
                
                # Threshold to create binary mask
                pred_mask = (pred_data >= threshold).astype(np.uint8)
                logger.info(f"  Thresholded probability map at {threshold}")
            else:
                # Already binary
                pred_mask = pred_data
                confidence_map = None
                logger.info(f"  Using binary mask directly")
            
            logger.info(f"  Prediction shape: {pred_mask.shape}, vessel voxels: {np.sum(pred_mask)}")
            
            # Extract graph
            start_time = time.time()
            
            pred_graph = extractor.extract_graph(
                mask=pred_mask,
                voxel_spacing=spacing,
                is_prediction=True,  # Important: marks this as prediction
                patient_id=f"{patient_id}_PRED",
                confidence_map=confidence_map  # Pass confidence if available
            )
            
            extraction_time = time.time() - start_time
            
            if pred_graph.metadata.get('extraction_success', False):
                # Save in same directory as GT
                output_path = extracted_graphs_dir / patient_id / f"{patient_id}_PRED.pkl"
                pred_graph.save(output_path)
                
                successful_extractions += 1
                logger.info(f"  ✅ Success! {pred_graph.global_properties.get('num_nodes', 0)} nodes, {extraction_time:.1f}s")
                
                results.append({
                    'patient_id': patient_id,
                    'success': True,
                    'num_nodes': pred_graph.global_properties.get('num_nodes', 0),
                    'num_edges': pred_graph.global_properties.get('num_edges', 0),
                    'quality_score': pred_graph.metadata.get('extraction_parameters', {}).get('quality_metrics', {}).get('overall_quality_score', 0),
                    'extraction_time': extraction_time
                })
            else:
                logger.error(f"  ❌ Extraction failed: {pred_graph.metadata.get('extraction_error', 'Unknown')}")
                results.append({
                    'patient_id': patient_id,
                    'success': False,
                    'error': pred_graph.metadata.get('extraction_error', 'Unknown')
                })
                
        except Exception as e:
            logger.error(f"  ❌ Exception processing {patient_id}: {e}")
            results.append({
                'patient_id': patient_id,
                'success': False,
                'error': str(e)
            })
    
    # Summary
    logger.info(f"\n=== EXTRACTION COMPLETE ===")
    logger.info(f"Total predictions processed: {len(prediction_pairs)}")
    logger.info(f"Successful extractions: {successful_extractions}")
    logger.info(f"Success rate: {successful_extractions/len(prediction_pairs):.1%}")
    
    # Save results summary
    results_data = {
        'total_predictions': len(prediction_pairs),
        'successful_extractions': successful_extractions,
        'extraction_config': config,
        'threshold': threshold,
        'results': results
    }
    
    with open(extracted_graphs_dir / 'prediction_extraction_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    return successful_extractions > 0

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract graphs from U-Net predictions')
    parser.add_argument('--predictions-dir', type=str, required=True,
                       help='Directory containing U-Net prediction files')
    parser.add_argument('--graphs-dir', type=str, default='extracted_graphs',
                       help='Directory with extracted GT graphs (default: extracted_graphs)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for probability maps (default: 0.5)')
    parser.add_argument('--no-confidence', action='store_true',
                       help='Do not use confidence maps even if available')
    
    args = parser.parse_args()
    
    predictions_dir = Path(args.predictions_dir)
    graphs_dir = Path(args.graphs_dir)
    
    if not predictions_dir.exists():
        print(f"Error: Predictions directory not found: {predictions_dir}")
        sys.exit(1)
    
    if not graphs_dir.exists():
        print(f"Error: Graphs directory not found: {graphs_dir}")
        sys.exit(1)
    
    success = extract_predicted_graphs(
        predictions_dir=predictions_dir,
        extracted_graphs_dir=graphs_dir,
        threshold=args.threshold,
        use_confidence=not args.no_confidence
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()