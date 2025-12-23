#!/usr/bin/env python3
"""
Full Pipeline Evaluation: U-Net → Graph Extraction → Graph Correction → Reconstruction
Test the complete graph-to-graph correction framework on real data
"""

import os
import sys
sys.path.append('.')

import torch
import numpy as np
from pathlib import Path
import nibabel as nib
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pickle
import time

# Import your modules
from src.models.graph_correction.regression_model import GraphCorrectionRegressionModel
from src.models.reconstruction.volume_reconstructor import VolumeReconstructor
from src.models.graph_extraction.graph_extractor import GraphExtractor
from src.utils.metrics import calculate_dice_score, calculate_hausdorff_distance
from src.training.regression_dataloader import RegressionGraphDataset

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FullPipelineEvaluator:
    """Evaluate the complete graph correction pipeline"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load trained model
        self.graph_correction_model = self._load_trained_model()
        
        # Initialize components
        self.graph_extractor = GraphExtractor()
        self.volume_reconstructor = VolumeReconstructor()
        
        # Results storage
        self.results = defaultdict(list)
        
    def _load_trained_model(self):
        """Load the trained regression model"""
        model_path = Path('experiments/regression_model/best_model.pth')
        
        if not model_path.exists():
            raise FileNotFoundError(f"Trained model not found at {model_path}")
        
        logger.info(f"Loading trained model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Initialize model with config from checkpoint
        model_config = checkpoint['config']['model']
        model = GraphCorrectionRegressionModel(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        logger.info(f"Model loaded successfully (epoch {checkpoint['epoch']})")
        return model
    
    def load_test_data(self, test_data_dir):
        """Load test data (images, predictions, ground truth)"""
        test_dir = Path(test_data_dir)
        
        test_cases = []
        for patient_dir in test_dir.glob('PA*'):
            if not patient_dir.is_dir():
                continue
                
            # Look for required files
            image_path = patient_dir / 'image' / 'image.nii.gz'
            gt_path = patient_dir / 'label' / 'label.nii.gz'
            
            # Look for U-Net prediction (might be in different locations)
            pred_candidates = [
                patient_dir / 'prediction' / 'prediction.nii.gz',
                patient_dir / 'unet_prediction.nii.gz',
                patient_dir / 'pred.nii.gz'
            ]
            
            pred_path = None
            for candidate in pred_candidates:
                if candidate.exists():
                    pred_path = candidate
                    break
            
            if image_path.exists() and gt_path.exists() and pred_path:
                test_cases.append({
                    'patient_id': patient_dir.name,
                    'image_path': image_path,
                    'gt_path': gt_path,
                    'pred_path': pred_path
                })
            else:
                logger.warning(f"Incomplete data for {patient_dir.name}")
        
        logger.info(f"Found {len(test_cases)} test cases")
        return test_cases
    
    def extract_graph_from_mask(self, mask, patient_id):
        """Extract graph representation from segmentation mask"""
        try:
            # Use your graph extraction pipeline
            graph = self.graph_extractor.extract_graph_from_mask(
                mask, 
                spacing=[1.0, 1.0, 1.0],  # Adjust based on your data
                patient_id=patient_id
            )
            return graph
        except Exception as e:
            logger.error(f"Graph extraction failed for {patient_id}: {str(e)}")
            return None
    
    def correct_graph(self, graph, patient_id):
        """Apply graph correction using trained model"""
        try:
            # Convert graph to format expected by model
            # This is a simplified version - you might need to adapt based on your exact format
            
            # Create node features
            nodes = graph.nodes
            node_features = []
            positions = []
            
            for node in nodes:
                pos = np.array(node['position'], dtype=np.float32)
                radius = float(node.get('radius', 1.0))
                confidence = 0.5  # Default confidence for inference
                feat1, feat2 = 0.0, 0.0  # Placeholder features
                
                features = np.concatenate([pos, [radius, confidence, feat1, feat2]])
                node_features.append(features)
                positions.append(pos)
            
            # Convert to tensors
            x = torch.tensor(np.array(node_features), dtype=torch.float32).to(self.device)
            pos = torch.tensor(np.array(positions), dtype=torch.float32).to(self.device)
            
            # Create edge index (simplified - adapt to your graph format)
            edge_list = []
            for edge in graph.edges:
                # Handle different edge formats
                if hasattr(edge, 'start_node') and hasattr(edge, 'end_node'):
                    start, end = edge.start_node, edge.end_node
                elif 'start_node' in edge and 'end_node' in edge:
                    start, end = edge['start_node'], edge['end_node']
                else:
                    continue
                    
                edge_list.append([start, end])
                edge_list.append([end, start])  # Undirected
            
            if edge_list:
                edge_index = torch.tensor(edge_list, dtype=torch.long).t().to(self.device)
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long).to(self.device)
            
            # Create PyG data object
            from torch_geometric.data import Data
            data = Data(x=x, edge_index=edge_index, pos=pos)
            
            # Apply correction
            with torch.no_grad():
                predictions = self.graph_correction_model(data)
            
            # Extract corrected positions
            corrected_positions = predictions['predicted_positions'].cpu().numpy()
            correction_magnitudes = predictions['correction_magnitudes'].cpu().numpy()
            node_operations = predictions['node_operations'].cpu().numpy()
            
            # Create corrected graph
            corrected_graph = self._create_corrected_graph(
                graph, corrected_positions, correction_magnitudes, node_operations
            )
            
            return corrected_graph, {
                'corrections_applied': len(corrected_positions),
                'mean_correction_magnitude': float(np.mean(correction_magnitudes)),
                'operations': {
                    'modify': int(np.sum(node_operations == 0)),
                    'remove': int(np.sum(node_operations == 1))
                }
            }
            
        except Exception as e:
            logger.error(f"Graph correction failed for {patient_id}: {str(e)}")
            return graph, {'error': str(e)}
    
    def _create_corrected_graph(self, original_graph, corrected_positions, magnitudes, operations):
        """Create corrected graph with updated positions"""
        # This is a simplified version - adapt based on your VascularGraph format
        corrected_nodes = []
        
        for i, node in enumerate(original_graph.nodes):
            if i < len(corrected_positions):
                # Update position
                corrected_node = node.copy()
                corrected_node['position'] = corrected_positions[i].tolist()
                corrected_node['correction_magnitude'] = float(magnitudes[i])
                corrected_node['operation'] = int(operations[i])
                
                # Only keep nodes that aren't marked for removal
                if operations[i] == 0:  # Modify operation - keep the node
                    corrected_nodes.append(corrected_node)
                # Remove operation (operations[i] == 1) - skip this node
            else:
                corrected_nodes.append(node)
        
        # Create new graph with corrected nodes
        from src.models.graph_extraction.vascular_graph import VascularGraph
        
        corrected_graph = VascularGraph(
            nodes=corrected_nodes,
            edges=original_graph.edges,  # Keep original edges for now
            global_properties=original_graph.global_properties.copy(),
            metadata={**original_graph.metadata, 'corrected': True}
        )
        
        return corrected_graph
    
    def reconstruct_volume(self, graph, original_shape, patient_id):
        """Reconstruct volume from corrected graph"""
        try:
            reconstructed_mask = self.volume_reconstructor.reconstruct_volume(
                graph, 
                target_shape=original_shape,
                spacing=[1.0, 1.0, 1.0]
            )
            return reconstructed_mask
        except Exception as e:
            logger.error(f"Volume reconstruction failed for {patient_id}: {str(e)}")
            return np.zeros(original_shape, dtype=np.uint8)
    
    def evaluate_case(self, case):
        """Evaluate a single test case through the full pipeline"""
        patient_id = case['patient_id']
        logger.info(f"Evaluating {patient_id}")
        
        start_time = time.time()
        
        try:
            # Load data
            image = nib.load(case['image_path']).get_fdata()
            gt_mask = nib.load(case['gt_path']).get_fdata()
            pred_mask = nib.load(case['pred_path']).get_fdata()
            
            # Ensure binary masks
            gt_mask = (gt_mask > 0).astype(np.uint8)
            pred_mask = (pred_mask > 0).astype(np.uint8)
            
            logger.info(f"{patient_id}: Loaded data - Image: {image.shape}, GT: {gt_mask.shape}, Pred: {pred_mask.shape}")
            
            # Step 1: Extract graph from U-Net prediction
            pred_graph = self.extract_graph_from_mask(pred_mask, patient_id)
            if pred_graph is None:
                return None
            
            logger.info(f"{patient_id}: Extracted graph with {len(pred_graph.nodes)} nodes")
            
            # Step 2: Apply graph correction
            corrected_graph, correction_stats = self.correct_graph(pred_graph, patient_id)
            
            logger.info(f"{patient_id}: Applied corrections - {correction_stats}")
            
            # Step 3: Reconstruct volume from corrected graph
            corrected_mask = self.reconstruct_volume(corrected_graph, pred_mask.shape, patient_id)
            
            # Step 4: Compute metrics
            # Original U-Net vs Ground Truth
            original_dice = calculate_dice_score(pred_mask, gt_mask)
            original_hd = calculate_hausdorff_distance(pred_mask, gt_mask)
            
            # Corrected vs Ground Truth
            corrected_dice = calculate_dice_score(corrected_mask, gt_mask)
            corrected_hd = calculate_hausdorff_distance(corrected_mask, gt_mask)
            
            # Compute improvement
            dice_improvement = corrected_dice - original_dice
            hd_improvement = original_hd - corrected_hd  # Lower HD is better
            
            processing_time = time.time() - start_time
            
            results = {
                'patient_id': patient_id,
                'original_dice': original_dice,
                'corrected_dice': corrected_dice,
                'dice_improvement': dice_improvement,
                'original_hd': original_hd,
                'corrected_hd': corrected_hd,
                'hd_improvement': hd_improvement,
                'processing_time': processing_time,
                'graph_stats': {
                    'original_nodes': len(pred_graph.nodes),
                    'corrected_nodes': len(corrected_graph.nodes),
                    'nodes_removed': len(pred_graph.nodes) - len(corrected_graph.nodes)
                },
                'correction_stats': correction_stats
            }
            
            logger.info(f"{patient_id}: Dice {original_dice:.3f} → {corrected_dice:.3f} "
                       f"(Δ{dice_improvement:+.3f}), HD {original_hd:.1f} → {corrected_hd:.1f} "
                       f"(Δ{hd_improvement:+.1f})")
            
            return results
            
        except Exception as e:
            logger.error(f"Evaluation failed for {patient_id}: {str(e)}")
            return {
                'patient_id': patient_id,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def evaluate_dataset(self, test_data_dir, output_dir):
        """Evaluate the full pipeline on a test dataset"""
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load test cases
        test_cases = self.load_test_data(test_data_dir)
        
        if not test_cases:
            logger.error("No test cases found!")
            return
        
        # Evaluate each case
        all_results = []
        
        for case in tqdm(test_cases, desc="Evaluating pipeline"):
            result = self.evaluate_case(case)
            if result:
                all_results.append(result)
        
        # Save detailed results
        with open(output_dir / 'detailed_results.pkl', 'wb') as f:
            pickle.dump(all_results, f)
        
        # Compute summary statistics
        self.generate_summary_report(all_results, output_dir)
        
        logger.info(f"Evaluation completed! Results saved to {output_dir}")
        
        return all_results
    
    def generate_summary_report(self, results, output_dir):
        """Generate comprehensive summary report"""
        
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            logger.error("No valid results to summarize!")
            return
        
        # Extract metrics
        dice_improvements = [r['dice_improvement'] for r in valid_results]
        hd_improvements = [r['hd_improvement'] for r in valid_results]
        original_dice = [r['original_dice'] for r in valid_results]
        corrected_dice = [r['corrected_dice'] for r in valid_results]
        
        # Summary statistics
        summary = {
            'total_cases': len(results),
            'successful_cases': len(valid_results),
            'failed_cases': len(results) - len(valid_results),
            'dice_metrics': {
                'original_mean': np.mean(original_dice),
                'original_std': np.std(original_dice),
                'corrected_mean': np.mean(corrected_dice),
                'corrected_std': np.std(corrected_dice),
                'improvement_mean': np.mean(dice_improvements),
                'improvement_std': np.std(dice_improvements),
                'cases_improved': sum(1 for d in dice_improvements if d > 0),
                'cases_degraded': sum(1 for d in dice_improvements if d < 0)
            },
            'hausdorff_metrics': {
                'improvement_mean': np.mean(hd_improvements),
                'improvement_std': np.std(hd_improvements),
                'cases_improved': sum(1 for h in hd_improvements if h > 0),
                'cases_degraded': sum(1 for h in hd_improvements if h < 0)
            }
        }
        
        # Save summary
        with open(output_dir / 'summary_report.pkl', 'wb') as f:
            pickle.dump(summary, f)
        
        # Generate visualizations
        self.create_visualizations(valid_results, output_dir)
        
        # Print summary
        logger.info("=" * 80)
        logger.info("FULL PIPELINE EVALUATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Cases evaluated: {summary['successful_cases']}/{summary['total_cases']}")
        logger.info(f"")
        logger.info(f"DICE SCORE RESULTS:")
        logger.info(f"  Original U-Net:     {summary['dice_metrics']['original_mean']:.3f} ± {summary['dice_metrics']['original_std']:.3f}")
        logger.info(f"  After Correction:   {summary['dice_metrics']['corrected_mean']:.3f} ± {summary['dice_metrics']['corrected_std']:.3f}")
        logger.info(f"  Mean Improvement:   {summary['dice_metrics']['improvement_mean']:+.3f} ± {summary['dice_metrics']['improvement_std']:.3f}")
        logger.info(f"  Cases Improved:     {summary['dice_metrics']['cases_improved']}/{len(valid_results)} ({100*summary['dice_metrics']['cases_improved']/len(valid_results):.1f}%)")
        logger.info(f"")
        logger.info(f"HAUSDORFF DISTANCE RESULTS:")
        logger.info(f"  Mean Improvement:   {summary['hausdorff_metrics']['improvement_mean']:+.1f} ± {summary['hausdorff_metrics']['improvement_std']:.1f}")
        logger.info(f"  Cases Improved:     {summary['hausdorff_metrics']['cases_improved']}/{len(valid_results)} ({100*summary['hausdorff_metrics']['cases_improved']/len(valid_results):.1f}%)")
        logger.info("=" * 80)
        
        return summary
    
    def create_visualizations(self, results, output_dir):
        """Create visualization plots"""
        
        # Extract data
        dice_improvements = [r['dice_improvement'] for r in results]
        original_dice = [r['original_dice'] for r in results]
        corrected_dice = [r['corrected_dice'] for r in results]
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Dice improvement histogram
        axes[0, 0].hist(dice_improvements, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.7)
        axes[0, 0].set_xlabel('Dice Improvement')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Distribution of Dice Score Improvements')
        
        # 2. Before vs After scatter
        axes[0, 1].scatter(original_dice, corrected_dice, alpha=0.6)
        axes[0, 1].plot([0, 1], [0, 1], 'r--', alpha=0.7)
        axes[0, 1].set_xlabel('Original Dice')
        axes[0, 1].set_ylabel('Corrected Dice')
        axes[0, 1].set_title('Original vs Corrected Dice Scores')
        
        # 3. Improvement vs Original quality
        axes[1, 0].scatter(original_dice, dice_improvements, alpha=0.6)
        axes[1, 0].axhline(0, color='red', linestyle='--', alpha=0.7)
        axes[1, 0].set_xlabel('Original Dice')
        axes[1, 0].set_ylabel('Dice Improvement')
        axes[1, 0].set_title('Improvement vs Original Quality')
        
        # 4. Success rate by original quality
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        success_rates = []
        
        for i in range(len(bins) - 1):
            mask = (np.array(original_dice) >= bins[i]) & (np.array(original_dice) < bins[i+1])
            if np.sum(mask) > 0:
                improvements = np.array(dice_improvements)[mask]
                success_rate = np.mean(improvements > 0)
                success_rates.append(success_rate)
            else:
                success_rates.append(0)
        
        axes[1, 1].bar(bin_centers, success_rates, width=0.08, alpha=0.7)
        axes[1, 1].set_xlabel('Original Dice Score Bin')
        axes[1, 1].set_ylabel('Success Rate (% Improved)')
        axes[1, 1].set_title('Success Rate by Original Quality')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'evaluation_summary.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {output_dir / 'evaluation_summary.png'}")


def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate full graph correction pipeline')
    parser.add_argument('--test-data', type=str, required=True,
                       help='Directory containing test data')
    parser.add_argument('--output-dir', type=str, 
                       default='experiments/full_pipeline_evaluation',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Simple config (you can make this more sophisticated)
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Run evaluation
    evaluator = FullPipelineEvaluator(config)
    results = evaluator.evaluate_dataset(args.test_data, args.output_dir)
    
    logger.info("Full pipeline evaluation completed!")


if __name__ == '__main__':
    main()