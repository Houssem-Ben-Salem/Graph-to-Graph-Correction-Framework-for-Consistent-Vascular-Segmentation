#!/usr/bin/env python
"""
Comprehensive statistics summary for all extracted graphs
"""

import sys
import numpy as np
import logging
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.graph_extraction import VascularGraph

def analyze_all_graphs(graph_dir: Path) -> Dict:
    """Analyze all extracted graphs and generate comprehensive statistics"""
    
    # Find all graph files
    graph_files = list(graph_dir.glob("**/*.pkl"))
    
    if not graph_files:
        return {"error": "No graph files found"}
    
    # Initialize collectors
    all_stats = []
    
    for graph_file in graph_files:
        try:
            # Load graph
            graph = VascularGraph.load(graph_file)
            
            # Extract statistics
            stats = {
                'patient_id': graph_file.parent.name,
                'graph_name': graph_file.stem,
                'num_nodes': graph.global_properties.get('num_nodes', 0),
                'num_edges': graph.global_properties.get('num_edges', 0),
                'total_length_mm': graph.global_properties.get('total_length_mm', 0),
                'is_connected': graph.global_properties.get('is_connected', False),
                'complexity_score': graph.global_properties.get('complexity_score', 0),
                'density': graph.global_properties.get('density', 0),
                'quality_score': graph.metadata.get('extraction_parameters', {}).get('quality_metrics', {}).get('overall_quality_score', 0),
            }
            
            # Node type breakdown
            node_types = graph.global_properties.get('node_type_counts', {})
            stats['bifurcations'] = node_types.get('bifurcations', 0)
            stats['endpoints'] = node_types.get('endpoints', 0)
            stats['regular'] = node_types.get('regular', 0)
            stats['buffer'] = node_types.get('buffer', 0)
            
            # Calculate ratios
            total_nodes = stats['num_nodes']
            if total_nodes > 0:
                stats['bifurcation_ratio'] = stats['bifurcations'] / total_nodes
                stats['endpoint_ratio'] = stats['endpoints'] / total_nodes
                stats['regular_ratio'] = stats['regular'] / total_nodes
                stats['buffer_ratio'] = stats['buffer'] / total_nodes
            else:
                stats['bifurcation_ratio'] = 0
                stats['endpoint_ratio'] = 0
                stats['regular_ratio'] = 0
                stats['buffer_ratio'] = 0
            
            # Additional metrics
            stats['avg_degree'] = graph.global_properties.get('avg_degree', 0)
            stats['node_density'] = stats['num_nodes'] / stats['total_length_mm'] if stats['total_length_mm'] > 0 else 0
            
            all_stats.append(stats)
            
        except Exception as e:
            logging.error(f"Failed to analyze {graph_file}: {e}")
    
    return {'graphs': all_stats, 'total_graphs': len(all_stats)}

def generate_summary_report(stats_data: Dict) -> Dict:
    """Generate comprehensive summary statistics"""
    
    if 'error' in stats_data or stats_data['total_graphs'] == 0:
        return stats_data
    
    df = pd.DataFrame(stats_data['graphs'])
    
    summary = {
        'total_graphs': stats_data['total_graphs'],
        'total_patients': df['patient_id'].nunique(),
        
        # Node statistics
        'nodes': {
            'total': int(df['num_nodes'].sum()),
            'mean': float(df['num_nodes'].mean()),
            'std': float(df['num_nodes'].std()),
            'min': int(df['num_nodes'].min()),
            'max': int(df['num_nodes'].max()),
            'median': float(df['num_nodes'].median())
        },
        
        # Edge statistics
        'edges': {
            'total': int(df['num_edges'].sum()),
            'mean': float(df['num_edges'].mean()),
            'std': float(df['num_edges'].std()),
            'min': int(df['num_edges'].min()),
            'max': int(df['num_edges'].max()),
            'median': float(df['num_edges'].median())
        },
        
        # Length statistics (in mm)
        'total_length_mm': {
            'total': float(df['total_length_mm'].sum()),
            'mean': float(df['total_length_mm'].mean()),
            'std': float(df['total_length_mm'].std()),
            'min': float(df['total_length_mm'].min()),
            'max': float(df['total_length_mm'].max()),
            'median': float(df['total_length_mm'].median())
        },
        
        # Node type statistics
        'node_types': {
            'bifurcations': {
                'total': int(df['bifurcations'].sum()),
                'mean': float(df['bifurcations'].mean()),
                'mean_ratio': float(df['bifurcation_ratio'].mean())
            },
            'endpoints': {
                'total': int(df['endpoints'].sum()),
                'mean': float(df['endpoints'].mean()),
                'mean_ratio': float(df['endpoint_ratio'].mean())
            },
            'regular': {
                'total': int(df['regular'].sum()),
                'mean': float(df['regular'].mean()),
                'mean_ratio': float(df['regular_ratio'].mean())
            },
            'buffer': {
                'total': int(df['buffer'].sum()),
                'mean': float(df['buffer'].mean()),
                'mean_ratio': float(df['buffer_ratio'].mean())
            }
        },
        
        # Quality metrics
        'quality': {
            'mean_quality_score': float(df['quality_score'].mean()),
            'std_quality_score': float(df['quality_score'].std()),
            'min_quality_score': float(df['quality_score'].min()),
            'max_quality_score': float(df['quality_score'].max()),
            'high_quality_count': int((df['quality_score'] > 0.8).sum()),
            'connected_graphs': int(df['is_connected'].sum()),
            'connectivity_rate': float(df['is_connected'].mean())
        },
        
        # Density metrics
        'density': {
            'mean_node_density': float(df['node_density'].mean()),
            'std_node_density': float(df['node_density'].std()),
            'mean_graph_density': float(df['density'].mean()),
            'mean_complexity': float(df['complexity_score'].mean())
        }
    }
    
    return summary

def create_visualization_summary(stats_data: Dict, output_dir: Path):
    """Create summary visualizations"""
    
    if 'error' in stats_data or stats_data['total_graphs'] == 0:
        return
    
    df = pd.DataFrame(stats_data['graphs'])
    
    # Setup style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Graph Extraction Summary - {stats_data["total_graphs"]} Graphs', fontsize=16, fontweight='bold')
    
    # 1. Node count distribution
    axes[0,0].hist(df['num_nodes'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].set_xlabel('Number of Nodes')
    axes[0,0].set_ylabel('Count')
    axes[0,0].set_title(f'Node Count Distribution\n(μ={df["num_nodes"].mean():.0f}, σ={df["num_nodes"].std():.0f})')
    axes[0,0].axvline(df['num_nodes'].mean(), color='red', linestyle='--', label='Mean')
    axes[0,0].legend()
    
    # 2. Regular node ratio distribution
    axes[0,1].hist(df['regular_ratio'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0,1].set_xlabel('Regular Node Ratio')
    axes[0,1].set_ylabel('Count')
    axes[0,1].set_title(f'Regular Node Ratio Distribution\n(μ={df["regular_ratio"].mean():.3f})')
    axes[0,1].axvline(df['regular_ratio'].mean(), color='red', linestyle='--', label='Mean')
    axes[0,1].legend()
    
    # 3. Quality score distribution
    axes[0,2].hist(df['quality_score'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0,2].set_xlabel('Quality Score')
    axes[0,2].set_ylabel('Count')
    axes[0,2].set_title(f'Quality Score Distribution\n(μ={df["quality_score"].mean():.3f})')
    axes[0,2].axvline(df['quality_score'].mean(), color='red', linestyle='--', label='Mean')
    axes[0,2].axvline(0.8, color='green', linestyle=':', label='High Quality Threshold')
    axes[0,2].legend()
    
    # 4. Node type composition
    node_type_means = [
        df['bifurcation_ratio'].mean(),
        df['endpoint_ratio'].mean(),
        df['regular_ratio'].mean(),
        df['buffer_ratio'].mean()
    ]
    node_type_labels = ['Bifurcations', 'Endpoints', 'Regular', 'Buffer']
    
    axes[1,0].pie(node_type_means, labels=node_type_labels, autopct='%1.1f%%', startangle=90)
    axes[1,0].set_title('Average Node Type Distribution')
    
    # 5. Node density vs graph size
    scatter = axes[1,1].scatter(df['num_nodes'], df['node_density'], 
                               c=df['quality_score'], cmap='viridis', alpha=0.6)
    axes[1,1].set_xlabel('Number of Nodes')
    axes[1,1].set_ylabel('Node Density (nodes/mm)')
    axes[1,1].set_title('Node Density vs Graph Size')
    plt.colorbar(scatter, ax=axes[1,1], label='Quality Score')
    
    # 6. Summary statistics text
    summary_text = f"""
Dataset Summary:
• Total Graphs: {stats_data['total_graphs']}
• Total Patients: {df['patient_id'].nunique()}
• Connected Graphs: {df['is_connected'].sum()} ({df['is_connected'].mean():.1%})

Average Graph Properties:
• Nodes: {df['num_nodes'].mean():.0f} ± {df['num_nodes'].std():.0f}
• Edges: {df['num_edges'].mean():.0f} ± {df['num_edges'].std():.0f}
• Length: {df['total_length_mm'].mean():.1f} ± {df['total_length_mm'].std():.1f} mm
• Node Density: {df['node_density'].mean():.3f} ± {df['node_density'].std():.3f} nodes/mm

Quality Metrics:
• Mean Quality Score: {df['quality_score'].mean():.3f}
• High Quality (>0.8): {(df['quality_score'] > 0.8).sum()} ({(df['quality_score'] > 0.8).mean():.1%})
"""
    
    axes[1,2].text(0.1, 0.5, summary_text, transform=axes[1,2].transAxes, 
                  fontsize=10, verticalalignment='center',
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    axes[1,2].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / 'dataset_summary.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create node type evolution plot
    fig2, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Sort by patient ID for consistent ordering
    df_sorted = df.sort_values('patient_id')
    
    x = range(len(df_sorted))
    ax.plot(x, df_sorted['bifurcation_ratio'], label='Bifurcations', marker='o', markersize=4)
    ax.plot(x, df_sorted['endpoint_ratio'], label='Endpoints', marker='s', markersize=4)
    ax.plot(x, df_sorted['regular_ratio'], label='Regular', marker='^', markersize=4)
    ax.plot(x, df_sorted['buffer_ratio'], label='Buffer', marker='d', markersize=4)
    
    ax.set_xlabel('Graph Index')
    ax.set_ylabel('Node Type Ratio')
    ax.set_title('Node Type Ratios Across Dataset')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'node_type_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to generate comprehensive statistics"""
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("=== Comprehensive Graph Dataset Statistics ===")
    
    # Find extracted graphs directory
    graph_dir = Path("extracted_graphs")
    
    if not graph_dir.exists():
        logger.error(f"Extracted graphs directory not found: {graph_dir}")
        return
    
    # Analyze all graphs
    logger.info("Analyzing all extracted graphs...")
    stats_data = analyze_all_graphs(graph_dir)
    
    if 'error' in stats_data:
        logger.error(stats_data['error'])
        return
    
    logger.info(f"Found {stats_data['total_graphs']} graphs")
    
    # Generate summary report
    summary = generate_summary_report(stats_data)
    
    # Save detailed statistics
    output_dir = Path("analysis_results")
    output_dir.mkdir(exist_ok=True)
    
    # Save raw statistics
    with open(output_dir / 'graph_statistics_raw.json', 'w') as f:
        json.dump(stats_data, f, indent=2)
    
    # Save summary
    with open(output_dir / 'graph_statistics_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create visualizations
    logger.info("Creating summary visualizations...")
    create_visualization_summary(stats_data, output_dir)
    
    # Print summary to console
    logger.info("\n" + "="*60)
    logger.info("DATASET SUMMARY")
    logger.info("="*60)
    
    logger.info(f"\nTotal Graphs: {summary['total_graphs']}")
    logger.info(f"Total Patients: {summary['total_patients']}")
    
    logger.info(f"\nNODE STATISTICS:")
    logger.info(f"  Total nodes: {summary['nodes']['total']:,}")
    logger.info(f"  Mean nodes per graph: {summary['nodes']['mean']:.1f} ± {summary['nodes']['std']:.1f}")
    logger.info(f"  Range: [{summary['nodes']['min']}, {summary['nodes']['max']}]")
    
    logger.info(f"\nNODE TYPE DISTRIBUTION:")
    logger.info(f"  Regular nodes: {summary['node_types']['regular']['mean_ratio']:.1%} (avg {summary['node_types']['regular']['mean']:.0f} per graph)")
    logger.info(f"  Bifurcations: {summary['node_types']['bifurcations']['mean_ratio']:.1%} (avg {summary['node_types']['bifurcations']['mean']:.0f} per graph)")
    logger.info(f"  Endpoints: {summary['node_types']['endpoints']['mean_ratio']:.1%} (avg {summary['node_types']['endpoints']['mean']:.0f} per graph)")
    logger.info(f"  Buffer nodes: {summary['node_types']['buffer']['mean_ratio']:.1%} (avg {summary['node_types']['buffer']['mean']:.0f} per graph)")
    
    logger.info(f"\nQUALITY METRICS:")
    logger.info(f"  Mean quality score: {summary['quality']['mean_quality_score']:.3f}")
    logger.info(f"  High quality graphs (>0.8): {summary['quality']['high_quality_count']} ({summary['quality']['high_quality_count']/summary['total_graphs']:.1%})")
    logger.info(f"  Connected graphs: {summary['quality']['connected_graphs']} ({summary['quality']['connectivity_rate']:.1%})")
    
    logger.info(f"\nDENSITY METRICS:")
    logger.info(f"  Mean node density: {summary['density']['mean_node_density']:.3f} nodes/mm")
    logger.info(f"  Mean complexity: {summary['density']['mean_complexity']:.3f}")
    
    logger.info(f"\n✅ Analysis complete! Results saved to: {output_dir}")
    logger.info("Generated files:")
    logger.info("  - graph_statistics_raw.json (detailed data)")
    logger.info("  - graph_statistics_summary.json (summary statistics)")
    logger.info("  - dataset_summary.png (visualization)")
    logger.info("  - node_type_evolution.png (node type trends)")

if __name__ == "__main__":
    main()