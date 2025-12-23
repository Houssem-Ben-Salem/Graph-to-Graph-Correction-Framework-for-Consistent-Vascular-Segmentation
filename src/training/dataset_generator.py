"""
Training Dataset Generator with Caching
Creates comprehensive training datasets for graph correction using synthetic degradation
Features advanced caching system for improved performance
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Iterator
import logging
from pathlib import Path
import json
import pickle
from tqdm import tqdm
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

from src.models.graph_extraction.vascular_graph import VascularGraph
from src.utils.graph_correspondence import create_correspondence_matcher
from src.training.synthetic_degradation import SyntheticDegradationPipeline, DegradationConfig
from src.training.training_cache import TrainingDataCache, CacheConfig


@dataclass
class CurriculumLevel:
    """Configuration for a curriculum learning level"""
    name: str
    degradation_level: str
    num_samples_per_graph: int
    sample_weight: float
    description: str


class TrainingDatasetGenerator:
    """
    Comprehensive training dataset generator for graph correction
    
    Features:
    - Curriculum learning with multiple difficulty levels
    - Balanced sampling across difficulty levels
    - Quality filtering and validation
    - Parallel processing for efficiency
    - Caching for faster re-use
    """
    
    def __init__(self, 
                 extracted_graphs_dir: Path,
                 output_dir: Path,
                 config: Optional[Dict] = None):
        """
        Initialize dataset generator
        
        Args:
            extracted_graphs_dir: Directory containing extracted GT graphs
            output_dir: Directory to save generated training data
            config: Configuration parameters
        """
        self.extracted_graphs_dir = Path(extracted_graphs_dir)
        self.output_dir = Path(output_dir)
        self.config = self._get_default_config()
        if config:
            self.config.update(config)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.degradation_pipeline = SyntheticDegradationPipeline()
        self.correspondence_matcher = create_correspondence_matcher()
        
        # Initialize caching system
        if self.config['cache_enabled']:
            cache_config = CacheConfig(
                cache_dir=self.output_dir / "cache",
                max_cache_size_gb=self.config['cache_max_size_gb'],
                compress_data=self.config['cache_compress'],
                preload_cache=self.config['cache_preload'],
                auto_cleanup=self.config['cache_auto_cleanup']
            )
            self.cache = TrainingDataCache(cache_config)
            self.logger.info("Training data cache initialized")
        else:
            self.cache = None
            self.logger.info("Training data cache disabled")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define curriculum levels
        self.curriculum_levels = self._define_curriculum_levels()
        
        # Track generation statistics
        self.generation_stats = {
            'total_samples_generated': 0,
            'samples_per_level': {},
            'quality_filtered': 0,
            'generation_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            # Generation parameters
            'samples_per_graph': 10,
            'max_graphs_to_process': None,  # None = all available
            'parallel_workers': 4,
            
            # Quality filtering
            'min_correspondence_quality': 0.3,
            'min_node_coverage': 0.5,
            'max_degradation_rate': 0.7,
            
            # Curriculum learning
            'curriculum_enabled': True,
            'adaptive_sampling': True,
            
            # Caching
            'cache_enabled': True,
            'cache_max_size_gb': 10.0,
            'cache_compress': True,
            'cache_preload': False,
            'cache_auto_cleanup': True,
            
            # Output
            'save_metadata': True,
            'compress_output': True,
            
            # Validation
            'validate_generated_data': True,
            'validation_sample_rate': 0.1
        }
    
    def _define_curriculum_levels(self) -> List[CurriculumLevel]:
        """Define curriculum learning levels"""
        return [
            CurriculumLevel(
                name="easy",
                degradation_level="easy",
                num_samples_per_graph=3,
                sample_weight=0.2,
                description="Minimal degradation - spatial noise only"
            ),
            CurriculumLevel(
                name="medium", 
                degradation_level="medium",
                num_samples_per_graph=4,
                sample_weight=0.4,
                description="Moderate degradation - spatial + topology errors"
            ),
            CurriculumLevel(
                name="hard",
                degradation_level="hard", 
                num_samples_per_graph=2,
                sample_weight=0.3,
                description="Severe degradation - multiple error types"
            ),
            CurriculumLevel(
                name="expert",
                degradation_level="expert",
                num_samples_per_graph=1, 
                sample_weight=0.1,
                description="Extreme degradation - stress testing"
            ),
            CurriculumLevel(
                name="real",
                degradation_level="real",
                num_samples_per_graph=1,
                sample_weight=1.0,
                description="Real U-Net prediction vs GT pairs"
            )
        ]
    
    def generate_training_dataset(self, 
                                split: str = "train",
                                curriculum_stage: Optional[str] = None) -> Dict:
        """
        Generate comprehensive training dataset
        
        Args:
            split: Dataset split ("train", "val", "test")
            curriculum_stage: Specific curriculum stage to generate
            
        Returns:
            Generation summary statistics
        """
        self.logger.info(f"=== Generating {split} dataset ===")
        start_time = time.time()
        
        # Find available GT graphs
        gt_graphs = self._find_gt_graphs()
        
        if not gt_graphs:
            raise ValueError("No GT graphs found!")
        
        # Split graphs based on split type
        graphs_for_split = self._split_graphs(gt_graphs, split)
        self.logger.info(f"Processing {len(graphs_for_split)} graphs for {split} split")
        
        # Limit number of graphs if specified
        if self.config['max_graphs_to_process'] is not None:
            graphs_for_split = graphs_for_split[:self.config['max_graphs_to_process']]
        
        # Generate samples for each curriculum level
        all_samples = []
        
        levels_to_process = (
            [level for level in self.curriculum_levels if level.name == curriculum_stage]
            if curriculum_stage else self.curriculum_levels
        )
        
        for level in levels_to_process:
            self.logger.info(f"Generating {level.name} level samples...")
            level_samples = self._generate_level_samples(
                graphs_for_split, level, split
            )
            all_samples.extend(level_samples)
            
            self.generation_stats['samples_per_level'][level.name] = len(level_samples)
        
        # Save dataset
        dataset_path = self.output_dir / f"{split}_dataset.pkl"
        self._save_dataset(all_samples, dataset_path)
        
        # Generate summary
        generation_time = time.time() - start_time
        self.generation_stats['generation_time'] = generation_time
        self.generation_stats['total_samples_generated'] = len(all_samples)
        
        # Add cache information to stats if available
        if self.cache is not None:
            cache_info = self.cache.get_cache_info()
            self.generation_stats['cache_info'] = cache_info
        
        summary = {
            'split': split,
            'curriculum_stage': curriculum_stage,
            'total_samples': len(all_samples),
            'generation_time': generation_time,
            'stats': self.generation_stats,
            'dataset_path': str(dataset_path)
        }
        
        # Save summary
        summary_path = self.output_dir / f"{split}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Enhanced logging with cache statistics
        self.logger.info(f"Dataset generation completed in {generation_time:.2f}s")
        self.logger.info(f"Generated {len(all_samples)} samples")
        
        if self.cache is not None:
            cache_hits = self.generation_stats['cache_hits']
            cache_misses = self.generation_stats['cache_misses']
            total_requests = cache_hits + cache_misses
            hit_rate = cache_hits / max(1, total_requests) * 100
            
            self.logger.info(f"Cache performance: {cache_hits}/{total_requests} hits ({hit_rate:.1f}%)")
            
            if cache_info:
                time_saved_hours = cache_info.get('generation_time_saved_hours', 0)
                cache_size_mb = cache_info.get('total_size_mb', 0)
                self.logger.info(f"Time saved: {time_saved_hours:.2f} hours, Cache size: {cache_size_mb:.1f} MB")
        
        self.logger.info(f"Saved to: {dataset_path}")
        
        return summary
    
    def _find_gt_graphs(self) -> List[Path]:
        """Find all available GT graph files"""
        gt_graphs = []
        
        for patient_dir in self.extracted_graphs_dir.iterdir():
            if patient_dir.is_dir() and patient_dir.name.startswith('PA'):
                gt_file = patient_dir / f"{patient_dir.name}_GT.pkl"
                if gt_file.exists():
                    gt_graphs.append(gt_file)
        
        return sorted(gt_graphs)
    
    def _split_graphs(self, gt_graphs: List[Path], split: str) -> List[Path]:
        """Split graphs into train/val/test"""
        # Deterministic split based on patient ID
        np.random.seed(42)  # Fixed seed for consistent splits
        
        total_graphs = len(gt_graphs)
        
        if split == "train":
            indices = list(range(int(0.7 * total_graphs)))
        elif split == "val":
            indices = list(range(int(0.7 * total_graphs), int(0.85 * total_graphs)))
        elif split == "test":
            indices = list(range(int(0.85 * total_graphs), total_graphs))
        else:
            indices = list(range(total_graphs))
        
        return [gt_graphs[i] for i in indices]
    
    def _generate_level_samples(self, 
                              gt_graphs: List[Path], 
                              level: CurriculumLevel,
                              split: str) -> List[Dict]:
        """Generate samples for a specific curriculum level"""
        
        samples = []
        
        # Handle real GT/PRED pairs differently
        if level.name == "real":
            return self._generate_real_data_samples(gt_graphs, level, split)
        
        if self.config['parallel_workers'] > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.config['parallel_workers']) as executor:
                futures = []
                
                for gt_graph_path in gt_graphs:
                    future = executor.submit(
                        self._process_single_graph,
                        gt_graph_path, level, split
                    )
                    futures.append(future)
                
                # Collect results with progress bar
                for future in tqdm(as_completed(futures), 
                                 total=len(futures),
                                 desc=f"Processing {level.name} level"):
                    try:
                        graph_samples = future.result()
                        samples.extend(graph_samples)
                    except Exception as e:
                        self.logger.error(f"Failed to process graph: {e}")
        else:
            # Sequential processing
            for gt_graph_path in tqdm(gt_graphs, desc=f"Processing {level.name} level"):
                try:
                    graph_samples = self._process_single_graph(gt_graph_path, level, split)
                    samples.extend(graph_samples)
                except Exception as e:
                    self.logger.error(f"Failed to process {gt_graph_path}: {e}")
        
        return samples
    
    def _process_single_graph(self, 
                            gt_graph_path: Path, 
                            level: CurriculumLevel,
                            split: str) -> List[Dict]:
        """Process a single GT graph to generate training samples with caching"""
        
        try:
            # Load GT graph
            gt_graph = VascularGraph.load(gt_graph_path)
            patient_id = gt_graph_path.parent.name
            
            samples = []
            
            # Generate multiple degraded versions
            for sample_idx in range(level.num_samples_per_graph):
                seed = hash(f"{patient_id}_{level.name}_{sample_idx}_{split}") % (2**32)
                
                # Check cache first if enabled
                cached_sample = None
                if self.cache is not None:
                    cache_key = self._generate_sample_cache_key(
                        gt_graph_path, level, split, sample_idx, seed
                    )
                    cached_sample = self.cache.get_cached_data(cache_key)
                    
                    if cached_sample is not None:
                        self.generation_stats['cache_hits'] += 1
                        samples.append(cached_sample)
                        continue
                    else:
                        self.generation_stats['cache_misses'] += 1
                
                try:
                    generation_start_time = time.time()
                    
                    # Generate degraded graph
                    degraded_graph, degradation_metadata = self.degradation_pipeline.degrade_graph(
                        gt_graph, 
                        degradation_level=level.degradation_level,
                        seed=seed
                    )
                    
                    # Compute correspondences
                    correspondences = self.correspondence_matcher.find_correspondence(
                        degraded_graph, gt_graph
                    )
                    
                    generation_time = time.time() - generation_start_time
                    
                    # Quality filtering
                    if self._passes_quality_filter(correspondences, degradation_metadata):
                        sample = {
                            'patient_id': patient_id,
                            'sample_id': f"{patient_id}_{level.name}_{sample_idx}",
                            'curriculum_level': level.name,
                            'split': split,
                            'gt_graph': gt_graph,
                            'degraded_graph': degraded_graph,
                            'correspondences': correspondences,
                            'degradation_metadata': degradation_metadata,
                            'level_config': level,
                            'generation_seed': seed,
                            'generation_time': generation_time
                        }
                        
                        samples.append(sample)
                        
                        # Cache the generated sample if caching is enabled
                        if self.cache is not None:
                            generation_params = {
                                'gt_graph_path': str(gt_graph_path),
                                'level_name': level.name,
                                'degradation_level': level.degradation_level,
                                'split': split,
                                'sample_idx': sample_idx,
                                'seed': seed,
                                'degradation_config': getattr(self.degradation_pipeline, 'config_hash', 'default'),
                                'correspondence_config': getattr(self.correspondence_matcher, 'config_hash', 'default')
                            }
                            
                            self.cache.cache_data(
                                cache_key=cache_key,
                                data=sample,
                                generation_params=generation_params,
                                generation_time=generation_time
                            )
                    else:
                        self.generation_stats['quality_filtered'] += 1
                
                except Exception as e:
                    self.logger.warning(f"Failed to generate sample {sample_idx} for {patient_id}: {e}")
            
            return samples
            
        except Exception as e:
            self.logger.error(f"Failed to process graph {gt_graph_path}: {e}")
            return []
    
    def _passes_quality_filter(self, correspondences, degradation_metadata) -> bool:
        """Check if generated sample passes quality filters"""
        
        # Check correspondence quality
        quality = correspondences.correspondence_quality.get('overall_quality', 0.0)
        if quality < self.config['min_correspondence_quality']:
            return False
        
        # Check node coverage
        node_coverage = correspondences.correspondence_quality.get('node_coverage', 0.0)
        if node_coverage < self.config['min_node_coverage']:
            return False
        
        # Check degradation rate is not too extreme
        degradation_summary = degradation_metadata.get('degradation_summary', {})
        node_loss_rate = degradation_summary.get('node_loss_rate', 0.0)
        edge_loss_rate = degradation_summary.get('edge_loss_rate', 0.0)
        
        if max(node_loss_rate, edge_loss_rate) > self.config['max_degradation_rate']:
            return False
        
        return True
    
    def _passes_real_data_quality_filter(self, correspondences, pred_graph, gt_graph) -> bool:
        """Check if real GT/PRED pair passes quality filters"""
        
        # More lenient filters for real data
        quality = correspondences.correspondence_quality.get('overall_quality', 0.0)
        if quality < 0.1:  # Very low threshold - real data might have poor correspondence
            return False
        
        # Check if we have some nodes and edges
        if len(pred_graph.nodes) == 0 or len(gt_graph.nodes) == 0:
            return False
        
        # Check if graphs aren't completely empty
        if len(pred_graph.edges) == 0 and len(gt_graph.edges) == 0:
            return False
            
        return True
    
    def _create_spatial_correspondence(self, pred_graph, gt_graph):
        """Create correspondence using spatial matching for real data"""
        from src.utils.graph_correspondence import CorrespondenceResult
        from scipy.spatial.distance import cdist
        from scipy.optimize import linear_sum_assignment
        
        # Extract positions from nodes
        pred_positions = np.array([node['position'] for node in pred_graph.nodes])
        gt_positions = np.array([node['position'] for node in gt_graph.nodes])
        
        # Compute distance matrix
        distance_matrix = cdist(pred_positions, gt_positions)
        
        # Use Hungarian algorithm for optimal assignment
        pred_indices, gt_indices = linear_sum_assignment(distance_matrix)
        
        # Filter matches by distance threshold (5mm)
        distance_threshold = 5.0
        valid_matches = distance_matrix[pred_indices, gt_indices] < distance_threshold
        
        # Create correspondences
        node_correspondences = {}
        node_confidences = {}
        matched_pred = set()
        matched_gt = set()
        
        valid_distances = []
        for i, (pred_idx, gt_idx) in enumerate(zip(pred_indices, gt_indices)):
            if valid_matches[i]:
                distance = distance_matrix[pred_idx, gt_idx]
                confidence = max(0.1, 1.0 - distance / distance_threshold)
                valid_distances.append(distance)
                
                node_correspondences[pred_idx] = gt_idx
                node_confidences[pred_idx] = confidence
                matched_pred.add(pred_idx)
                matched_gt.add(gt_idx)
        
        # Debug spatial correspondence statistics
        if valid_distances:
            min_dist, max_dist = min(valid_distances), max(valid_distances)
            mean_dist = sum(valid_distances) / len(valid_distances)
            confidences = list(node_confidences.values())
            min_conf, max_conf = min(confidences), max(confidences)
            mean_conf = sum(confidences) / len(confidences)
            print(f"DEBUG: Real data - distances min: {min_dist:.3f}, max: {max_dist:.3f}, mean: {mean_dist:.3f}")
            print(f"DEBUG: Real data - confidences min: {min_conf:.3f}, max: {max_conf:.3f}, mean: {mean_conf:.3f}")
            print(f"DEBUG: Real data - {len(node_correspondences)} matches out of {len(pred_graph.nodes)} pred nodes")
        
        # Unmatched nodes
        unmatched_pred_nodes = set(range(len(pred_graph.nodes))) - matched_pred
        unmatched_gt_nodes = set(range(len(gt_graph.nodes))) - matched_gt
        
        # Calculate quality metrics
        node_match_rate = len(node_correspondences) / max(len(pred_graph.nodes), 1)
        overall_quality = node_match_rate * np.mean(list(node_confidences.values())) if node_confidences else 0.1
        
        correspondences = CorrespondenceResult(
            node_correspondences=node_correspondences,
            node_confidences=node_confidences,
            unmatched_pred_nodes=unmatched_pred_nodes,
            unmatched_gt_nodes=unmatched_gt_nodes,
            
            edge_correspondences={},  # Will be derived from node correspondences
            edge_confidences={},
            unmatched_pred_edges=set(range(len(pred_graph.edges))),
            unmatched_gt_edges=set(range(len(gt_graph.edges))),
            
            topology_differences={
                'node_count_diff': len(gt_graph.nodes) - len(pred_graph.nodes),
                'edge_count_diff': len(gt_graph.edges) - len(pred_graph.edges),
                'matched_nodes': len(node_correspondences)
            },
            
            alignment_transform={},
            
            correspondence_quality={
                'overall_quality': overall_quality,
                'node_match_rate': node_match_rate,
                'edge_match_rate': 0.0,
                'avg_distance': np.mean([distance_matrix[p, g] for p, g in node_correspondences.items()]) if node_correspondences else 999.0
            },
            
            metadata={
                'source': 'spatial_real_data',
                'matching_method': 'hungarian_spatial',
                'distance_threshold': distance_threshold,
                'total_matches': len(node_correspondences)
            }
        )
        
        return correspondences
    
    def _save_dataset(self, samples: List[Dict], output_path: Path):
        """Save generated dataset to file"""
        
        if self.config['compress_output']:
            # Save compressed version
            import gzip
            output_path_gz = Path(str(output_path) + '.gz')
            with gzip.open(output_path_gz, 'wb') as f:
                pickle.dump(samples, f)
            self.logger.info(f"Saved {len(samples)} samples to {output_path_gz}")
            
            # Also save uncompressed version for compatibility
            with open(output_path, 'wb') as f:
                pickle.dump(samples, f)
            self.logger.info(f"Saved {len(samples)} samples to {output_path} (uncompressed)")
        else:
            # Save normally
            with open(output_path, 'wb') as f:
                pickle.dump(samples, f)
            self.logger.info(f"Saved {len(samples)} samples to {output_path}")
    
    def _generate_curriculum_level_samples(self, split: str, curriculum_stage: str) -> List[Dict]:
        """Generate samples for a specific curriculum level and split"""
        
        # Find available GT graphs
        gt_graphs = self._find_gt_graphs()
        
        if not gt_graphs:
            raise ValueError("No GT graphs found!")
        
        # Split graphs based on split type
        graphs_for_split = self._split_graphs(gt_graphs, split)
        
        # Limit number of graphs if specified
        if self.config['max_graphs_to_process'] is not None:
            graphs_for_split = graphs_for_split[:self.config['max_graphs_to_process']]
        
        # Find the curriculum level
        level = None
        for l in self.curriculum_levels:
            if l.name == curriculum_stage:
                level = l
                break
        
        if level is None:
            raise ValueError(f"Unknown curriculum stage: {curriculum_stage}")
        
        # Generate samples for this level
        level_samples = self._generate_level_samples(graphs_for_split, level, split)
        
        return level_samples
    
    def _generate_real_data_samples(self, 
                                  gt_graphs: List[Path], 
                                  level: CurriculumLevel,
                                  split: str) -> List[Dict]:
        """Generate training samples from real GT/PRED pairs"""
        
        samples = []
        
        self.logger.info(f"Generating real data samples for {split} split...")
        
        for gt_graph_path in tqdm(gt_graphs, desc="Processing real GT/PRED pairs"):
            try:
                # Load GT graph
                with open(gt_graph_path, 'rb') as f:
                    gt_graph = pickle.load(f)
                
                # Find corresponding PRED graph
                patient_dir = gt_graph_path.parent
                patient_id = patient_dir.name
                pred_graph_path = patient_dir / f"{patient_id}_PRED.pkl"
                
                if not pred_graph_path.exists():
                    self.logger.warning(f"No PRED graph found for {patient_id}")
                    continue
                
                # Load PRED graph
                with open(pred_graph_path, 'rb') as f:
                    pred_graph = pickle.load(f)
                
                # Create correspondence between PRED and GT using spatial matching
                correspondences = self._create_spatial_correspondence(pred_graph, gt_graph)
                
                # Check quality filters for real data
                if not self._passes_real_data_quality_filter(correspondences, pred_graph, gt_graph):
                    continue
                
                # Create training sample
                sample = {
                    'patient_id': patient_id,
                    'sample_id': f"{patient_id}_real",
                    'curriculum_level': 'real',
                    'split': split,
                    'gt_graph': gt_graph,
                    'degraded_graph': pred_graph,  # PRED is the "degraded" input
                    'correspondences': correspondences,
                    'degradation_metadata': {
                        'source': 'real_unet_prediction',
                        'degradation_level': 'real',
                        'applied_degradations': ['unet_prediction_errors']  
                    },
                    'level_config': level,
                    'generation_seed': 0,
                    'generation_time': 0.0
                }
                
                samples.append(sample)
                
            except Exception as e:
                self.logger.error(f"Error processing {gt_graph_path}: {e}")
                continue
        
        self.logger.info(f"Generated {len(samples)} real data samples")
        return samples
    
    def generate_curriculum_datasets(self) -> Dict:
        """Generate datasets for all curriculum stages"""
        
        self.logger.info("=== Generating Curriculum Datasets ===")
        
        results = {}
        
        # Collect all samples for each split across all curriculum levels
        all_train_samples = []
        all_val_samples = []
        
        # Generate training data for each curriculum level
        for level in self.curriculum_levels:
            self.logger.info(f"\nGenerating {level.name} curriculum stage...")
            
            # Generate samples for this level
            train_samples = self._generate_curriculum_level_samples("train", level.name)
            val_samples = self._generate_curriculum_level_samples("val", level.name)
            
            # Add to combined datasets
            all_train_samples.extend(train_samples)
            all_val_samples.extend(val_samples)
            
            # Track individual level results
            train_summary = {
                'split': 'train',
                'curriculum_stage': level.name,
                'total_samples': len(train_samples),
                'generation_time': 0.0,  # Will be updated below
                'stats': self.generation_stats.copy()
            }
            
            val_summary = {
                'split': 'val', 
                'curriculum_stage': level.name,
                'total_samples': len(val_samples),
                'generation_time': 0.0,  # Will be updated below
                'stats': self.generation_stats.copy()
            }
            
            results[level.name] = {
                'train': train_summary,
                'val': val_summary,
                'level_config': level
            }
        
        # Save combined datasets
        self.logger.info(f"\nSaving combined curriculum datasets...")
        
        # Save train dataset with all curriculum levels
        train_dataset_path = self.output_dir / "train_dataset.pkl"
        self._save_dataset(all_train_samples, train_dataset_path)
        self.logger.info(f"Saved combined train dataset: {len(all_train_samples)} samples")
        
        # Save val dataset with all curriculum levels  
        val_dataset_path = self.output_dir / "val_dataset.pkl"
        self._save_dataset(all_val_samples, val_dataset_path)
        self.logger.info(f"Saved combined val dataset: {len(all_val_samples)} samples")
        
        # Generate test dataset (mixed difficulty)
        test_summary = self.generate_training_dataset(split="test")
        results['test'] = test_summary
        
        # Save overall summary
        overall_summary = {
            'curriculum_results': results,
            'generation_config': self.config,
            'total_generation_time': sum(
                r['train']['generation_time'] + r['val']['generation_time'] 
                for r in results.values() 
                if isinstance(r, dict) and 'train' in r
            )
        }
        
        summary_path = self.output_dir / "curriculum_generation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(overall_summary, f, indent=2, default=str)
        
        self.logger.info(f"\nCurriculum dataset generation completed!")
        self.logger.info(f"Summary saved to: {summary_path}")
        
        return overall_summary
    
    def _generate_sample_cache_key(self, gt_graph_path: Path, level: CurriculumLevel, 
                                 split: str, sample_idx: int, seed: int) -> str:
        """Generate cache key for a training sample"""
        cache_params = {
            'gt_graph_path': str(gt_graph_path),
            'level_name': level.name,
            'degradation_level': level.degradation_level,
            'split': split,
            'sample_idx': sample_idx,
            'seed': seed,
            'degradation_version': getattr(self.degradation_pipeline, 'version', '1.0'),
            'correspondence_version': getattr(self.correspondence_matcher, 'version', '1.0')
        }
        
        if self.cache is not None:
            return self.cache.generate_cache_key(cache_params)
        else:
            return str(hash(str(cache_params)))
    
    def get_cache_info(self) -> Optional[Dict]:
        """Get cache information and statistics"""
        if self.cache is not None:
            return self.cache.get_cache_info()
        return None
    
    def warm_cache(self, cache_keys: List[str] = None) -> int:
        """Warm cache by preloading data"""
        if self.cache is not None:
            if cache_keys is None:
                # Get all available cache keys
                cache_keys = list(self.cache.cache_entries.keys())
            return self.cache.warm_cache(cache_keys)
        return 0
    
    def cleanup_cache(self, force: bool = False) -> Dict:
        """Clean up cache"""
        if self.cache is not None:
            return self.cache.cleanup_cache(force=force)
        return {}
    
    def clear_cache(self) -> bool:
        """Clear all cache data"""
        if self.cache is not None:
            return self.cache.clear_cache()
        return True

    def validate_generated_dataset(self, dataset_path: Path) -> Dict:
        """Validate a generated dataset"""
        
        self.logger.info(f"Validating dataset: {dataset_path}")
        
        # Load dataset
        if str(dataset_path).endswith('.gz'):
            import gzip
            with gzip.open(dataset_path, 'rb') as f:
                samples = pickle.load(f)
        else:
            with open(dataset_path, 'rb') as f:
                samples = pickle.load(f)
        
        validation_results = {
            'total_samples': len(samples),
            'valid_samples': 0,
            'invalid_samples': 0,
            'validation_errors': [],
            'quality_stats': {}
        }
        
        # Sample subset for validation if dataset is large
        if len(samples) > 100:
            validation_samples = random.sample(samples, min(100, len(samples)))
        else:
            validation_samples = samples
        
        for sample in tqdm(validation_samples, desc="Validating samples"):
            try:
                # Check sample structure
                required_keys = ['gt_graph', 'degraded_graph', 'correspondences']
                if all(key in sample for key in required_keys):
                    validation_results['valid_samples'] += 1
                else:
                    validation_results['invalid_samples'] += 1
                    validation_results['validation_errors'].append("Missing required keys")
                
            except Exception as e:
                validation_results['invalid_samples'] += 1
                validation_results['validation_errors'].append(str(e))
        
        # Compute quality statistics
        if validation_results['valid_samples'] > 0:
            qualities = []
            coverages = []
            for sample in validation_samples[:50]:  # Sample subset
                if 'correspondences' in sample:
                    quality = sample['correspondences'].correspondence_quality.get('overall_quality', 0)
                    coverage = sample['correspondences'].correspondence_quality.get('node_coverage', 0)
                    qualities.append(quality)
                    coverages.append(coverage)
            
            if qualities:
                validation_results['quality_stats'] = {
                    'avg_quality': np.mean(qualities),
                    'min_quality': np.min(qualities),
                    'max_quality': np.max(qualities),
                    'avg_coverage': np.mean(coverages),
                    'min_coverage': np.min(coverages),
                    'max_coverage': np.max(coverages)
                }
        
        validation_results['validation_success'] = (
            validation_results['valid_samples'] > validation_results['invalid_samples']
        )
        
        self.logger.info(f"Validation completed: {validation_results['valid_samples']}/{len(validation_samples)} valid")
        
        return validation_results


def create_training_dataset(extracted_graphs_dir: Path,
                          output_dir: Path,
                          config: Optional[Dict] = None) -> Dict:
    """
    Convenience function to create training dataset
    
    Args:
        extracted_graphs_dir: Directory with extracted GT graphs
        output_dir: Output directory for training data
        config: Configuration parameters
        
    Returns:
        Generation summary
    """
    generator = TrainingDatasetGenerator(
        extracted_graphs_dir=extracted_graphs_dir,
        output_dir=output_dir,
        config=config
    )
    
    return generator.generate_curriculum_datasets()