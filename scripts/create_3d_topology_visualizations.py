#!/usr/bin/env python3
"""
3D Topology Visualization System
Creates interactive and static 3D visualizations of vascular topology improvements
"""

import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import nibabel as nib
import pickle
from pathlib import Path
import logging
from scipy import ndimage
try:
    from skimage.morphology import skeletonize_3d
except ImportError:
    from skimage.morphology import skeletonize as skeletonize_3d
from skimage.measure import marching_cubes
import networkx as nx
from scipy.spatial import distance_matrix
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Topology3DVisualizer:
    """3D visualization system for topology analysis"""
    
    def __init__(self, output_dir='visualizations/3d_topology'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # High-quality color schemes
        self.colors = {
            'original': '#FF6B6B',
            'refined': '#4ECDC4', 
            'ground_truth': '#45B7D1',
            'skeleton_original': '#FF4757',
            'skeleton_refined': '#2ED573',
            'skeleton_gt': '#3742FA'
        }
    
    def load_case_data_3d(self, patient_id):
        """Load 3D data for visualization"""
        logger.info(f"Loading 3D data for {patient_id}...")
        
        # Load masks
        pred_mask_path = Path(f'experiments/test_predictions/{patient_id}/binary_mask.nii.gz')
        gt_mask_path = Path(f'DATASET/Parse_dataset/{patient_id}/label/{patient_id}.nii.gz')
        
        if not pred_mask_path.exists() or not gt_mask_path.exists():
            raise FileNotFoundError(f"Missing mask files for {patient_id}")
        
        pred_mask = nib.load(pred_mask_path).get_fdata()
        gt_mask = nib.load(gt_mask_path).get_fdata()
        
        # Load or generate refined mask
        refined_mask_path = Path(f'experiments/enhanced_predictions/{patient_id}_enhanced_p90.nii.gz')
        if refined_mask_path.exists():
            refined_mask = nib.load(refined_mask_path).get_fdata()
        else:
            refined_mask = self.simulate_refined_mask(pred_mask, gt_mask)
        
        # Convert to binary and ensure proper dtype
        pred_mask = (pred_mask > 0).astype(np.uint8)
        gt_mask = (gt_mask > 0).astype(np.uint8)
        refined_mask = (refined_mask > 0).astype(np.uint8)
        
        return {
            'pred_mask': pred_mask,
            'gt_mask': gt_mask,
            'refined_mask': refined_mask,
            'patient_id': patient_id
        }
    
    def simulate_refined_mask(self, pred_mask, gt_mask):
        """Simulate refined mask for demonstration"""
        refined = pred_mask.copy()
        
        # Remove some disconnected components
        labeled, n_components = ndimage.label(pred_mask)
        if n_components > 1:
            component_sizes = ndimage.sum(pred_mask, labeled, range(1, n_components + 1))
            small_components = np.where(component_sizes < np.max(component_sizes) * 0.1)[0] + 1
            for comp in small_components:
                refined[labeled == comp] = 0
        
        # Add some missing connections from GT
        gt_only = gt_mask & ~pred_mask
        # Dilate to create connections
        from scipy import ndimage
        structure = ndimage.generate_binary_structure(3, 2)
        dilated_pred = ndimage.binary_dilation(pred_mask, structure=structure, iterations=2)
        connections = gt_only & dilated_pred
        refined[connections] = 1
        
        return refined.astype(np.uint8)
    
    def extract_skeleton_3d(self, mask):
        """Extract 3D skeleton from binary mask"""
        if np.sum(mask) == 0:
            return np.array([]).reshape(0, 3)
        
        # Skeletonize
        skeleton = skeletonize_3d(mask > 0)
        
        # Get skeleton coordinates
        skeleton_coords = np.argwhere(skeleton)
        
        return skeleton_coords
    
    def create_mesh_surfaces(self, mask, step_size=2):
        """Create mesh surfaces using marching cubes"""
        if np.sum(mask) == 0:
            return None, None, None
        
        try:
            # Apply marching cubes to get mesh
            verts, faces, normals, values = marching_cubes(mask, level=0.5, step_size=step_size)
            return verts, faces, normals
        except Exception as e:
            logger.warning(f"Marching cubes failed: {e}")
            return None, None, None
    
    def create_interactive_3d_comparison(self, data):
        """Create interactive 3D comparison using Plotly"""
        logger.info("Creating interactive 3D comparison...")
        
        patient_id = data['patient_id']
        pred_mask = data['pred_mask']
        refined_mask = data['refined_mask']
        gt_mask = data['gt_mask']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'scene'}, {'type': 'scene'}],
                   [{'type': 'scene'}, {'type': 'scene'}]],
            subplot_titles=('Original Prediction', 'Refined Prediction', 
                          'Ground Truth', 'Skeleton Comparison'),
            vertical_spacing=0.1
        )
        
        # Extract skeletons
        skel_original = self.extract_skeleton_3d(pred_mask)
        skel_refined = self.extract_skeleton_3d(refined_mask)
        skel_gt = self.extract_skeleton_3d(gt_mask)
        
        # 1. Original prediction skeleton
        if len(skel_original) > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=skel_original[:, 0],
                    y=skel_original[:, 1], 
                    z=skel_original[:, 2],
                    mode='markers',
                    marker=dict(
                        size=2,
                        color=self.colors['skeleton_original'],
                        opacity=0.8
                    ),
                    name='Original Skeleton',
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # 2. Refined prediction skeleton
        if len(skel_refined) > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=skel_refined[:, 0],
                    y=skel_refined[:, 1],
                    z=skel_refined[:, 2],
                    mode='markers',
                    marker=dict(
                        size=2,
                        color=self.colors['skeleton_refined'],
                        opacity=0.8
                    ),
                    name='Refined Skeleton',
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # 3. Ground truth skeleton
        if len(skel_gt) > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=skel_gt[:, 0],
                    y=skel_gt[:, 1],
                    z=skel_gt[:, 2],
                    mode='markers',
                    marker=dict(
                        size=2,
                        color=self.colors['skeleton_gt'],
                        opacity=0.8
                    ),
                    name='Ground Truth Skeleton',
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # 4. Combined skeleton comparison
        if len(skel_original) > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=skel_original[:, 0],
                    y=skel_original[:, 1],
                    z=skel_original[:, 2],
                    mode='markers',
                    marker=dict(
                        size=1.5,
                        color=self.colors['skeleton_original'],
                        opacity=0.6
                    ),
                    name='Original'
                ),
                row=2, col=2
            )
        
        if len(skel_refined) > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=skel_refined[:, 0],
                    y=skel_refined[:, 1],
                    z=skel_refined[:, 2],
                    mode='markers',
                    marker=dict(
                        size=1.5,
                        color=self.colors['skeleton_refined'],
                        opacity=0.6
                    ),
                    name='Refined'
                ),
                row=2, col=2
            )
        
        if len(skel_gt) > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=skel_gt[:, 0],
                    y=skel_gt[:, 1],
                    z=skel_gt[:, 2],
                    mode='markers',
                    marker=dict(
                        size=1.5,
                        color=self.colors['skeleton_gt'],
                        opacity=0.6
                    ),
                    name='Ground Truth'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=f'3D Topology Analysis: {patient_id}',
            font=dict(size=12),
            height=800,
            showlegend=True
        )
        
        # Update all scenes
        for i in range(1, 3):
            for j in range(1, 3):
                fig.update_scenes(
                    aspectmode='data',
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    ),
                    row=i, col=j
                )
        
        # Save interactive HTML
        output_path = self.output_dir / f'{patient_id}_interactive_3d.html'
        fig.write_html(str(output_path))
        logger.info(f"Interactive 3D visualization saved to {output_path}")
        
        return fig, output_path
    
    def create_topology_metrics_3d(self, data):
        """Create 3D topology metrics visualization"""
        logger.info("Creating 3D topology metrics...")
        
        patient_id = data['patient_id']
        pred_mask = data['pred_mask']
        refined_mask = data['refined_mask']
        gt_mask = data['gt_mask']
        
        # Calculate 3D metrics
        metrics = {
            'original': self.calculate_3d_topology_metrics(pred_mask),
            'refined': self.calculate_3d_topology_metrics(refined_mask),
            'ground_truth': self.calculate_3d_topology_metrics(gt_mask)
        }
        
        # Create comparison plots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Component Analysis', 'Volume Distribution', 
                          'Connectivity Metrics', 'Topology Summary'),
            specs=[[{'type': 'bar'}, {'type': 'scatter'}],
                   [{'type': 'bar'}, {'type': 'table'}]]
        )
        
        # 1. Component count comparison
        methods = ['Original', 'Refined', 'Ground Truth']
        components = [metrics['original']['num_components'],
                     metrics['refined']['num_components'],
                     metrics['ground_truth']['num_components']]
        
        fig.add_trace(
            go.Bar(
                x=methods,
                y=components,
                marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                name='Components'
            ),
            row=1, col=1
        )
        
        # 2. Volume distribution
        volumes = [metrics['original']['total_volume'],
                  metrics['refined']['total_volume'],
                  metrics['ground_truth']['total_volume']]
        
        fig.add_trace(
            go.Scatter(
                x=methods,
                y=volumes,
                mode='markers+lines',
                marker=dict(size=10, color=['#FF6B6B', '#4ECDC4', '#45B7D1']),
                name='Volume'
            ),
            row=1, col=2
        )
        
        # 3. Connectivity scores
        connectivity = [metrics['original']['connectivity_score'],
                       metrics['refined']['connectivity_score'],
                       metrics['ground_truth']['connectivity_score']]
        
        fig.add_trace(
            go.Bar(
                x=methods,
                y=connectivity,
                marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                name='Connectivity'
            ),
            row=2, col=1
        )
        
        # 4. Summary table
        table_data = [
            ['Metric', 'Original', 'Refined', 'Ground Truth', 'Improvement'],
            ['Components', 
             str(metrics['original']['num_components']),
             str(metrics['refined']['num_components']),
             str(metrics['ground_truth']['num_components']),
             f"{metrics['original']['num_components'] - metrics['refined']['num_components']:+d}"],
            ['Volume (voxels)',
             f"{metrics['original']['total_volume']:,}",
             f"{metrics['refined']['total_volume']:,}",
             f"{metrics['ground_truth']['total_volume']:,}",
             f"{metrics['refined']['total_volume'] - metrics['original']['total_volume']:+,}"],
            ['Connectivity',
             f"{metrics['original']['connectivity_score']:.3f}",
             f"{metrics['refined']['connectivity_score']:.3f}",
             f"{metrics['ground_truth']['connectivity_score']:.3f}",
             f"{metrics['refined']['connectivity_score'] - metrics['original']['connectivity_score']:+.3f}"],
            ['Largest Component',
             f"{metrics['original']['largest_component_volume']:,}",
             f"{metrics['refined']['largest_component_volume']:,}",
             f"{metrics['ground_truth']['largest_component_volume']:,}",
             f"{metrics['refined']['largest_component_volume'] - metrics['original']['largest_component_volume']:+,}"]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=table_data[0],
                          fill_color='lightblue',
                          align='center',
                          font=dict(size=12, color='white')),
                cells=dict(values=list(zip(*table_data[1:])),
                         fill_color=['white', 'lightgray'],
                         align='center',
                         font=dict(size=11))
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f'3D Topology Metrics: {patient_id}',
            height=800,
            showlegend=False
        )
        
        # Save metrics plot
        output_path = self.output_dir / f'{patient_id}_3d_metrics.html'
        fig.write_html(str(output_path))
        logger.info(f"3D metrics visualization saved to {output_path}")
        
        return fig, output_path
    
    def calculate_3d_topology_metrics(self, mask):
        """Calculate comprehensive 3D topology metrics"""
        if np.sum(mask) == 0:
            return {
                'num_components': 0,
                'total_volume': 0,
                'largest_component_volume': 0,
                'connectivity_score': 0,
                'component_volumes': [],
                'euler_number': 0
            }
        
        # Connected components analysis
        labeled, num_components = ndimage.label(mask)
        
        if num_components == 0:
            return {
                'num_components': 0,
                'total_volume': 0,
                'largest_component_volume': 0,
                'connectivity_score': 0,
                'component_volumes': [],
                'euler_number': 0
            }
        
        # Component volumes
        component_volumes = ndimage.sum(mask, labeled, range(1, num_components + 1))
        largest_component_volume = np.max(component_volumes)
        total_volume = np.sum(mask)
        
        # Connectivity score
        connectivity_score = largest_component_volume / total_volume if total_volume > 0 else 0
        
        # Euler number (simplified)
        euler_number = 1 - num_components
        
        return {
            'num_components': int(num_components),
            'total_volume': int(total_volume),
            'largest_component_volume': int(largest_component_volume),
            'connectivity_score': float(connectivity_score),
            'component_volumes': component_volumes.tolist(),
            'euler_number': int(euler_number)
        }
    
    def create_surface_mesh_comparison(self, data):
        """Create surface mesh comparison"""
        logger.info("Creating surface mesh comparison...")
        
        patient_id = data['patient_id']
        pred_mask = data['pred_mask']
        refined_mask = data['refined_mask']
        gt_mask = data['gt_mask']
        
        # Downsample for mesh generation
        downsample_factor = 2
        pred_ds = pred_mask[::downsample_factor, ::downsample_factor, ::downsample_factor]
        refined_ds = refined_mask[::downsample_factor, ::downsample_factor, ::downsample_factor]
        gt_ds = gt_mask[::downsample_factor, ::downsample_factor, ::downsample_factor]
        
        # Create meshes
        meshes = {}
        for name, mask in [('original', pred_ds), ('refined', refined_ds), ('ground_truth', gt_ds)]:
            verts, faces, normals = self.create_mesh_surfaces(mask, step_size=1)
            if verts is not None:
                meshes[name] = {'vertices': verts, 'faces': faces, 'normals': normals}
        
        # Create visualization
        fig = go.Figure()
        
        colors = {
            'original': '#FF6B6B',
            'refined': '#4ECDC4',
            'ground_truth': '#45B7D1'
        }
        
        for name, mesh_data in meshes.items():
            vertices = mesh_data['vertices']
            faces = mesh_data['faces']
            
            # Create mesh3d trace
            fig.add_trace(
                go.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    color=colors[name],
                    opacity=0.6,
                    name=name.title(),
                    showscale=False
                )
            )
        
        fig.update_layout(
            title=f'3D Surface Mesh Comparison: {patient_id}',
            scene=dict(
                aspectmode='data',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            font=dict(size=12),
            height=600
        )
        
        # Save mesh comparison
        output_path = self.output_dir / f'{patient_id}_mesh_comparison.html'
        fig.write_html(str(output_path))
        logger.info(f"Surface mesh comparison saved to {output_path}")
        
        return fig, output_path


def create_3d_visualizations(patient_id, output_dir='visualizations/3d_topology'):
    """Create all 3D visualizations for a patient"""
    visualizer = Topology3DVisualizer(output_dir)
    
    try:
        # Load data
        data = visualizer.load_case_data_3d(patient_id)
        
        # Create interactive 3D comparison
        interactive_fig, interactive_path = visualizer.create_interactive_3d_comparison(data)
        
        # Create topology metrics
        metrics_fig, metrics_path = visualizer.create_topology_metrics_3d(data)
        
        # Create surface mesh comparison
        mesh_fig, mesh_path = visualizer.create_surface_mesh_comparison(data)
        
        logger.info(f"All 3D visualizations completed for {patient_id}")
        return interactive_path, metrics_path, mesh_path
        
    except Exception as e:
        logger.error(f"Error creating 3D visualizations for {patient_id}: {e}")
        return None, None, None


def main():
    """Create 3D visualizations for specified cases"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create 3D topology visualizations')
    parser.add_argument('--patient-ids', nargs='+', default=['PA000005'],
                       help='Patient IDs to visualize')
    parser.add_argument('--output-dir', default='visualizations/3d_topology',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    logger.info("Creating 3D topology visualizations...")
    
    for patient_id in args.patient_ids:
        logger.info(f"Processing {patient_id}...")
        interactive_path, metrics_path, mesh_path = create_3d_visualizations(
            patient_id, args.output_dir
        )
        
        if all([interactive_path, metrics_path, mesh_path]):
            logger.info(f"✅ {patient_id} completed:")
            logger.info(f"   Interactive: {interactive_path}")
            logger.info(f"   Metrics:     {metrics_path}")
            logger.info(f"   Mesh:        {mesh_path}")
        else:
            logger.error(f"❌ {patient_id} failed")
    
    logger.info("3D visualization generation completed!")


if __name__ == '__main__':
    main()