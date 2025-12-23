#!/usr/bin/env python
"""
Comprehensive graph visualization and analysis toolkit
"""

import sys
import numpy as np
import logging
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.graph_extraction import VascularGraph

def setup_style():
    """Setup plotting style"""
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10

def visualize_3d_graph(graph: VascularGraph, title: str = "Vascular Graph", 
                      show_edges: bool = True, show_labels: bool = False) -> plt.Figure:
    """Create 3D visualization of the graph"""
    
    # Extract node positions and types
    node_positions = []
    node_types = []
    node_radii = []
    
    for node in graph.nodes:
        if 'coordinates_voxel' in node:
            pos = node['coordinates_voxel']
            # Convert to numpy array if needed
            if isinstance(pos, (list, tuple, np.ndarray)):
                pos_array = np.array(pos)
                if pos_array.size >= 3:
                    node_positions.append(pos_array[:3])
                    node_types.append(node.get('type', 'unknown'))
                    node_radii.append(node.get('radius_voxels', 1))
    
    if not node_positions:
        return None
    
    node_positions = np.array(node_positions)
    
    # Create 3D plot
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color mapping for node types
    type_colors = {
        'bifurcation': 'red',
        'endpoint': 'blue', 
        'regular': 'green',
        'buffer': 'orange',
        'unknown': 'gray'
    }
    
    # Plot nodes with different colors for different types
    for node_type in set(node_types):
        mask = np.array(node_types) == node_type
        if np.any(mask):
            positions = node_positions[mask]
            radii = np.array(node_radii)[mask]
            
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                      c=type_colors.get(node_type, 'gray'),
                      s=radii * 20,  # Size based on radius
                      label=f'{node_type} ({np.sum(mask)})',
                      alpha=0.7)
    
    # Plot edges if requested
    if show_edges and len(graph.edges) > 0:
        # Create node index mapping
        node_id_to_pos = {}
        for i, node in enumerate(graph.nodes):
            node_id = node.get('node_id', i)
            if i < len(node_positions):
                node_id_to_pos[node_id] = node_positions[i]
        
        # Draw edges
        edge_count = 0
        for edge in graph.edges[:1000]:  # Limit to first 1000 edges for performance
            source_id = edge.get('source_node', edge.get('source'))
            target_id = edge.get('target_node', edge.get('target'))
            
            if source_id in node_id_to_pos and target_id in node_id_to_pos:
                source_pos = node_id_to_pos[source_id]
                target_pos = node_id_to_pos[target_id]
                
                ax.plot([source_pos[0], target_pos[0]],
                       [source_pos[1], target_pos[1]], 
                       [source_pos[2], target_pos[2]],
                       'k-', alpha=0.3, linewidth=0.5)
                edge_count += 1
        
        if len(graph.edges) > 1000:
            ax.text2D(0.02, 0.98, f"Showing {edge_count}/1000 edges", 
                     transform=ax.transAxes, fontsize=8)
    
    # Formatting
    ax.set_xlabel('X (voxels)')
    ax.set_ylabel('Y (voxels)')
    ax.set_zlabel('Z (voxels)')
    ax.set_title(title)
    ax.legend()
    
    # Add stats text
    stats_text = f"""Nodes: {len(node_positions)}
Edges: {len(graph.edges)}
Volume: {node_positions.max(axis=0) - node_positions.min(axis=0)}"""
    
    ax.text2D(0.02, 0.02, stats_text, transform=ax.transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
             fontsize=9)
    
    plt.tight_layout()
    return fig

def plot_graph_statistics(graph: VascularGraph) -> plt.Figure:
    """Create comprehensive statistics dashboard"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Graph Statistics Dashboard - {len(graph.nodes)} nodes, {len(graph.edges)} edges', 
                 fontsize=16, fontweight='bold')
    
    # 1. Node type distribution
    node_types = [node.get('type', 'unknown') for node in graph.nodes]
    type_counts = pd.Series(node_types).value_counts()
    
    axes[0,0].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
    axes[0,0].set_title('Node Type Distribution')
    
    # 2. Node degree distribution  
    nx_graph = graph.to_networkx()
    degrees = [nx_graph.degree(node) for node in nx_graph.nodes()]
    
    axes[0,1].hist(degrees, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,1].set_xlabel('Node Degree')
    axes[0,1].set_ylabel('Count')
    axes[0,1].set_title(f'Degree Distribution (avg: {np.mean(degrees):.1f})')
    
    # 3. Node radius distribution
    radii = [node.get('radius_voxels', 0) for node in graph.nodes if node.get('radius_voxels', 0) > 0]
    
    if radii:
        axes[0,2].hist(radii, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0,2].set_xlabel('Radius (voxels)')
        axes[0,2].set_ylabel('Count')
        axes[0,2].set_title(f'Radius Distribution (avg: {np.mean(radii):.1f})')
    else:
        axes[0,2].text(0.5, 0.5, 'No radius data', transform=axes[0,2].transAxes, 
                      ha='center', va='center')
        axes[0,2].set_title('Radius Distribution')
    
    # 4. Edge length distribution
    edge_lengths = [edge.get('euclidean_length', 0) for edge in graph.edges if edge.get('euclidean_length', 0) > 0]
    
    if edge_lengths:
        axes[1,0].hist(edge_lengths, bins=20, alpha=0.7, color='coral', edgecolor='black')
        axes[1,0].set_xlabel('Edge Length (voxels)')
        axes[1,0].set_ylabel('Count')
        axes[1,0].set_title(f'Edge Length Distribution (avg: {np.mean(edge_lengths):.1f})')
    else:
        axes[1,0].text(0.5, 0.5, 'No edge length data', transform=axes[1,0].transAxes,
                      ha='center', va='center')
        axes[1,0].set_title('Edge Length Distribution')
    
    # 5. Spatial distribution (2D projection)
    node_positions = []
    for node in graph.nodes:
        if 'coordinates_voxel' in node:
            pos = node['coordinates_voxel']
            # Convert to numpy array if needed
            if isinstance(pos, (list, tuple, np.ndarray)):
                pos_array = np.array(pos)
                if pos_array.size >= 3:
                    node_positions.append(pos_array[:3])
    
    if len(node_positions) > 0:
        node_positions = np.array(node_positions)
        scatter = axes[1,1].scatter(node_positions[:, 0], node_positions[:, 1], 
                                   c=node_positions[:, 2], cmap='viridis', alpha=0.6, s=20)
        axes[1,1].set_xlabel('X (voxels)')
        axes[1,1].set_ylabel('Y (voxels)')
        axes[1,1].set_title('Spatial Distribution (colored by Z)')
        plt.colorbar(scatter, ax=axes[1,1], label='Z (voxels)')
    else:
        axes[1,1].text(0.5, 0.5, 'No position data', transform=axes[1,1].transAxes,
                      ha='center', va='center')
        axes[1,1].set_title('Spatial Distribution')
    
    # 6. Quality metrics summary
    global_props = graph.global_properties
    metadata = graph.metadata
    
    quality_text = f"""
Total Length: {global_props.get('total_length_mm', 0):.1f} mm
Complexity: {global_props.get('complexity_score', 0):.3f}
Density: {global_props.get('density', 0):.3f}
Connected: {global_props.get('is_connected', False)}

Quality Score: {metadata.get('extraction_parameters', {}).get('quality_metrics', {}).get('overall_quality_score', 0):.3f}
"""
    
    axes[1,2].text(0.1, 0.5, quality_text, transform=axes[1,2].transAxes, 
                  fontsize=11, verticalalignment='center',
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    axes[1,2].set_title('Quality Metrics')
    axes[1,2].axis('off')
    
    plt.tight_layout()
    return fig

def compare_graphs(graphs: List[VascularGraph], names: List[str]) -> plt.Figure:
    """Compare multiple graphs side by side"""
    
    n_graphs = len(graphs)
    fig, axes = plt.subplots(2, n_graphs, figsize=(6*n_graphs, 12))
    
    if n_graphs == 1:
        axes = axes.reshape(2, 1)
    
    fig.suptitle('Graph Comparison', fontsize=16, fontweight='bold')
    
    for i, (graph, name) in enumerate(zip(graphs, names)):
        # Top row: 3D visualization (simplified)
        ax_3d = fig.add_subplot(2, n_graphs, i+1, projection='3d')
        
        # Extract node positions and types
        node_positions = []
        node_types = []
        
        for node in graph.nodes:
            if 'coordinates_voxel' in node:
                pos = node['coordinates_voxel']
                # Convert to numpy array if needed
                if isinstance(pos, (list, tuple, np.ndarray)):
                    pos_array = np.array(pos)
                    if pos_array.size >= 3:
                        node_positions.append(pos_array[:3])
                        node_types.append(node.get('type', 'unknown'))
        
        if len(node_positions) > 0:
            node_positions = np.array(node_positions)
            
            # Color by type
            type_colors = {'bifurcation': 'red', 'endpoint': 'blue', 
                          'regular': 'green', 'buffer': 'orange', 'unknown': 'gray'}
            
            colors = [type_colors.get(t, 'gray') for t in node_types]
            
            ax_3d.scatter(node_positions[:, 0], node_positions[:, 1], node_positions[:, 2],
                         c=colors, s=20, alpha=0.7)
            
            ax_3d.set_title(f'{name}\n{len(node_positions)} nodes')
            ax_3d.set_xlabel('X')
            ax_3d.set_ylabel('Y')
            ax_3d.set_zlabel('Z')
        
        # Bottom row: Statistics comparison
        stats = {
            'Nodes': len(graph.nodes),
            'Edges': len(graph.edges),
            'Bifurcations': graph.global_properties.get('node_type_counts', {}).get('bifurcations', 0),
            'Endpoints': graph.global_properties.get('node_type_counts', {}).get('endpoints', 0),
            'Regular': graph.global_properties.get('node_type_counts', {}).get('regular', 0),
            'Quality': graph.metadata.get('extraction_parameters', {}).get('quality_metrics', {}).get('overall_quality_score', 0)
        }
        
        # Bar chart of key statistics
        stats_names = list(stats.keys())[:-1]  # Exclude quality for bar chart
        stats_values = [stats[k] for k in stats_names]
        
        axes[1, i].bar(stats_names, stats_values, alpha=0.7)
        axes[1, i].set_title(f'{name}\nQuality: {stats["Quality"]:.3f}')
        axes[1, i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig

def create_richness_analysis(graph: VascularGraph) -> plt.Figure:
    """Analyze information richness visually"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Information Richness Analysis', fontsize=16, fontweight='bold')
    
    # 1. Spatial coverage grid
    node_positions = []
    for node in graph.nodes:
        if 'coordinates_voxel' in node:
            pos = node['coordinates_voxel']
            # Convert to numpy array if needed
            if isinstance(pos, (list, tuple, np.ndarray)):
                pos_array = np.array(pos)
                if pos_array.size >= 3:
                    node_positions.append(pos_array[:3])
    
    if len(node_positions) > 0:
        node_positions = np.array(node_positions)
        
        # Create 3D grid and count nodes per cell
        grid_size = 8
        min_coords = np.min(node_positions, axis=0)
        max_coords = np.max(node_positions, axis=0)
        
        # 2D projection for visualization
        grid_counts_2d = np.zeros((grid_size, grid_size))
        grid_spans = (max_coords[:2] - min_coords[:2]) / grid_size
        
        for pos in node_positions:
            grid_idx = ((pos[:2] - min_coords[:2]) / grid_spans).astype(int)
            grid_idx = np.clip(grid_idx, 0, grid_size-1)
            grid_counts_2d[grid_idx[1], grid_idx[0]] += 1  # Note: y,x for imshow
        
        im = axes[0,0].imshow(grid_counts_2d, cmap='YlOrRd', origin='lower')
        axes[0,0].set_title('Spatial Coverage Heatmap')
        axes[0,0].set_xlabel('X Grid')
        axes[0,0].set_ylabel('Y Grid')
        plt.colorbar(im, ax=axes[0,0], label='Node Count')
        
        # Calculate coverage metrics
        empty_cells = np.sum(grid_counts_2d == 0)
        coverage_ratio = 1 - (empty_cells / (grid_size**2))
        axes[0,0].text(0.02, 0.98, f'Coverage: {coverage_ratio:.1%}', 
                      transform=axes[0,0].transAxes, 
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 2. Node density along vessel length
    if len(node_positions) > 0:
        # Estimate vessel path and calculate local density
        total_length = graph.global_properties.get('total_length_mm', 0)
        num_nodes = len(node_positions)
        density = num_nodes / total_length if total_length > 0 else 0
        
        # Distance between consecutive nodes (simplified)
        distances = []
        for i in range(1, len(node_positions)):
            dist = np.linalg.norm(node_positions[i] - node_positions[i-1])
            distances.append(dist)
        
        if distances:
            axes[0,1].hist(distances, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
            axes[0,1].set_xlabel('Inter-node Distance (voxels)')
            axes[0,1].set_ylabel('Count')
            axes[0,1].set_title(f'Node Spacing\nDensity: {density:.3f} nodes/mm')
            axes[0,1].axvline(np.mean(distances), color='red', linestyle='--', label=f'Mean: {np.mean(distances):.1f}')
            axes[0,1].legend()
    
    # 3. Connectivity richness
    nx_graph = graph.to_networkx()
    degrees = [nx_graph.degree(node) for node in nx_graph.nodes()]
    
    if degrees:
        degree_counts = np.bincount(degrees)
        axes[1,0].bar(range(len(degree_counts)), degree_counts, alpha=0.7, color='lightgreen')
        axes[1,0].set_xlabel('Node Degree')
        axes[1,0].set_ylabel('Count')
        axes[1,0].set_title('Connectivity Distribution')
        
        # Add diversity metric
        unique_degrees = len(np.unique(degrees))
        entropy = -np.sum((degree_counts[degree_counts>0]/len(degrees)) * 
                         np.log(degree_counts[degree_counts>0]/len(degrees) + 1e-10))
        axes[1,0].text(0.98, 0.98, f'Diversity: {unique_degrees}\nEntropy: {entropy:.2f}', 
                      transform=axes[1,0].transAxes, ha='right', va='top',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 4. Overall richness score
    # Calculate richness components
    node_type_counts = graph.global_properties.get('node_type_counts', {})
    total_nodes = sum(node_type_counts.values()) if node_type_counts else 0
    
    richness_components = {
        'Spatial Coverage': coverage_ratio if 'coverage_ratio' in locals() else 0,
        'Node Density': min(1.0, density * 20) if 'density' in locals() else 0,  # Scale to 0-1
        'Regular Nodes': node_type_counts.get('regular', 0) / total_nodes if total_nodes > 0 else 0,
        'Connectivity': min(1.0, unique_degrees / 10) if 'unique_degrees' in locals() else 0
    }
    
    # Radar chart
    angles = np.linspace(0, 2*np.pi, len(richness_components), endpoint=False)
    values = list(richness_components.values())
    
    # Close the plot
    angles = np.concatenate((angles, [angles[0]]))
    values = np.concatenate((values, [values[0]]))
    
    ax_radar = fig.add_subplot(2, 2, 4, projection='polar')
    ax_radar.plot(angles, values, 'o-', linewidth=2, color='blue', alpha=0.7)
    ax_radar.fill(angles, values, alpha=0.25, color='blue')
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(list(richness_components.keys()))
    ax_radar.set_ylim(0, 1)
    ax_radar.set_title('Information Richness Components')
    
    # Overall score
    overall_richness = np.mean(list(richness_components.values()))
    ax_radar.text(0.5, 0.5, f'Overall\n{overall_richness:.3f}', 
                 transform=ax_radar.transAxes, ha='center', va='center',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
    
    plt.tight_layout()
    return fig

def visualize_available_graphs():
    """Visualize all available graphs"""
    
    setup_style()
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("=== Graph Visualization Analysis ===")
    
    # Find available graphs
    graph_dirs = [
        Path("test_output/single_sample"),
        Path("test_output/improved_configs"),
        Path("test_output/crops"),
        Path("extracted_graphs")  # For batch extracted graphs
    ]
    
    graph_files = []
    for dir_path in graph_dirs:
        if dir_path.exists():
            graph_files.extend(list(dir_path.glob("**/*.pkl")))
    
    if not graph_files:
        logger.error("No graph files found for visualization!")
        return False
    
    logger.info(f"Found {len(graph_files)} graph files")
    
    # Create output directory for visualizations
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # Process each graph
    for i, graph_file in enumerate(graph_files[:5]):  # Limit to first 5 for demo
        logger.info(f"Visualizing {graph_file.name}")
        
        try:
            graph = VascularGraph.load(graph_file)
            
            # 1. 3D visualization
            try:
                fig_3d = visualize_3d_graph(graph, title=f"{graph_file.stem} - 3D Structure")
                if fig_3d:
                    fig_3d.savefig(output_dir / f"{graph_file.stem}_3d.png", dpi=150, bbox_inches='tight')
                    plt.close(fig_3d)
            except Exception as e:
                logger.error(f"  Error in 3D visualization: {e}")
                import traceback
                traceback.print_exc()
            
            # 2. Statistics dashboard
            try:
                fig_stats = plot_graph_statistics(graph)
                fig_stats.savefig(output_dir / f"{graph_file.stem}_stats.png", dpi=150, bbox_inches='tight')
                plt.close(fig_stats)
            except Exception as e:
                logger.error(f"  Error in statistics dashboard: {e}")
                import traceback
                traceback.print_exc()
            
            # 3. Richness analysis
            try:
                fig_richness = create_richness_analysis(graph)
                fig_richness.savefig(output_dir / f"{graph_file.stem}_richness.png", dpi=150, bbox_inches='tight')
                plt.close(fig_richness)
            except Exception as e:
                logger.error(f"  Error in richness analysis: {e}")
                import traceback
                traceback.print_exc()
            
            logger.info(f"  ✅ Generated visualizations for {graph_file.stem}")
            
        except Exception as e:
            logger.error(f"  ❌ Failed to visualize {graph_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Create comparison if multiple graphs
    if len(graph_files) >= 2:
        try:
            logger.info("Creating comparison visualization...")
            
            # Load subset for comparison
            comparison_graphs = []
            comparison_names = []
            
            for graph_file in graph_files[:4]:  # Compare up to 4 graphs
                try:
                    graph = VascularGraph.load(graph_file)
                    comparison_graphs.append(graph)
                    comparison_names.append(graph_file.stem)
                except:
                    continue
            
            if len(comparison_graphs) >= 2:
                fig_compare = compare_graphs(comparison_graphs, comparison_names)
                fig_compare.savefig(output_dir / "graph_comparison.png", dpi=150, bbox_inches='tight')
                plt.close(fig_compare)
                logger.info("  ✅ Generated comparison visualization")
        
        except Exception as e:
            logger.error(f"  ❌ Failed to create comparison: {e}")
    
    logger.info(f"\n✅ Visualizations saved to: {output_dir}")
    logger.info("Generated files:")
    for viz_file in sorted(output_dir.glob("*.png")):
        logger.info(f"  - {viz_file.name}")
    
    return True

if __name__ == "__main__":
    success = visualize_available_graphs()
    sys.exit(0 if success else 1)