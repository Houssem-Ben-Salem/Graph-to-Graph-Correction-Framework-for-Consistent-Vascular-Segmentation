#!/usr/bin/env python3
"""
Master Visualization Script
Creates comprehensive visualization suite for topology improvement results
"""

import sys
sys.path.append('.')

import logging
import argparse
from pathlib import Path
import subprocess
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_visualization_script(script_name, args, description):
    """Run a visualization script with error handling"""
    logger.info(f"Running {description}...")
    
    try:
        # Build command
        cmd = [sys.executable, script_name] + args
        
        # Run the script
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logger.info(f"✅ {description} completed successfully")
            if result.stdout:
                logger.info(f"Output: {result.stdout.strip()}")
            return True
        else:
            logger.error(f"❌ {description} failed with return code {result.returncode}")
            if result.stderr:
                logger.error(f"Error: {result.stderr.strip()}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"❌ {description} timed out after 5 minutes")
        return False
    except Exception as e:
        logger.error(f"❌ {description} failed with exception: {e}")
        return False


def create_comprehensive_visualizations(patient_ids, output_dir, include_3d=True):
    """Create all visualizations for the specified patient IDs"""
    
    logger.info("="*80)
    logger.info("COMPREHENSIVE TOPOLOGY VISUALIZATION SUITE")
    logger.info("="*80)
    logger.info(f"Patient IDs: {', '.join(patient_ids)}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Include 3D visualizations: {include_3d}")
    logger.info("="*80)
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    results = {}
    start_time = time.time()
    
    # 1. Create 2D topology visualizations
    logger.info("\n" + "="*50)
    logger.info("STEP 1: 2D TOPOLOGY VISUALIZATIONS")
    logger.info("="*50)
    
    for patient_id in patient_ids:
        script_args = ['--patient-ids', patient_id, '--output-dir', f'{output_dir}/2d_topology']
        success = run_visualization_script(
            'scripts/create_topology_visualizations.py',
            script_args,
            f'2D topology visualization for {patient_id}'
        )
        results[f'2d_{patient_id}'] = success
    
    # 2. Create 3D visualizations (if requested)
    if include_3d:
        logger.info("\n" + "="*50)
        logger.info("STEP 2: 3D TOPOLOGY VISUALIZATIONS")
        logger.info("="*50)
        
        for patient_id in patient_ids:
            script_args = ['--patient-ids', patient_id, '--output-dir', f'{output_dir}/3d_topology']
            success = run_visualization_script(
                'scripts/create_3d_topology_visualizations.py',
                script_args,
                f'3D topology visualization for {patient_id}'
            )
            results[f'3d_{patient_id}'] = success
    
    # 3. Create publication figures
    logger.info("\n" + "="*50)
    logger.info("STEP 3: PUBLICATION-QUALITY FIGURES")
    logger.info("="*50)
    
    # Main figure for first patient
    main_patient = patient_ids[0]
    script_args = ['--patient-id', main_patient, '--output-dir', f'{output_dir}/publication']
    success = run_visualization_script(
        'scripts/generate_publication_figures.py',
        script_args,
        f'Publication main figure for {main_patient}'
    )
    results['publication_main'] = success
    
    # Batch summary figure
    script_args = ['--batch-only', '--output-dir', f'{output_dir}/publication']
    success = run_visualization_script(
        'scripts/generate_publication_figures.py',
        script_args,
        'Publication batch summary figure'
    )
    results['publication_batch'] = success
    
    # 4. Generate summary report
    logger.info("\n" + "="*50)
    logger.info("STEP 4: SUMMARY REPORT")
    logger.info("="*50)
    
    total_time = time.time() - start_time
    
    # Count successes and failures
    successful = sum(1 for success in results.values() if success)
    total = len(results)
    failed = total - successful
    
    logger.info(f"\nVISUALIZATION GENERATION COMPLETED!")
    logger.info(f"Total time: {total_time:.1f} seconds")
    logger.info(f"Success rate: {successful}/{total} ({successful/total*100:.1f}%)")
    
    if failed > 0:
        logger.warning(f"Failed visualizations: {failed}")
        for key, success in results.items():
            if not success:
                logger.warning(f"  ❌ {key}")
    
    # Generate file listing
    logger.info("\n" + "="*50)
    logger.info("GENERATED FILES")
    logger.info("="*50)
    
    output_path = Path(output_dir)
    if output_path.exists():
        for subfolder in ['2d_topology', '3d_topology', 'publication']:
            subfolder_path = output_path / subfolder
            if subfolder_path.exists():
                files = list(subfolder_path.glob('*.*'))
                logger.info(f"\n{subfolder.upper()}:")
                for file in sorted(files):
                    logger.info(f"  📄 {file.name}")
    
    # Create visualization index
    create_visualization_index(output_dir, patient_ids, results)
    
    return results


def create_visualization_index(output_dir, patient_ids, results):
    """Create HTML index of all visualizations"""
    logger.info("Creating visualization index...")
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Topology Improvement Visualizations</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .section {{
            margin: 20px 0;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }}
        .file-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .file-item {{
            background-color: white;
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #dee2e6;
            transition: transform 0.2s;
        }}
        .file-item:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        .file-item a {{
            text-decoration: none;
            color: #2c3e50;
            font-weight: 500;
        }}
        .file-item a:hover {{
            color: #3498db;
        }}
        .status {{
            margin-top: 20px;
            padding: 15px;
            border-radius: 6px;
        }}
        .success {{
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }}
        .warning {{
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
        }}
        .timestamp {{
            color: #6c757d;
            font-size: 0.9em;
            text-align: center;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 Graph-to-Graph Correction: Topology Visualizations</h1>
        
        <div class="status success">
            <strong>Visualization Suite Generated Successfully!</strong><br>
            Patient Cases: {', '.join(patient_ids)}<br>
            Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
        </div>
"""
    
    # Add sections for each visualization type
    sections = [
        ('2d_topology', '2D Topology Visualizations', 'Detailed 2D slice comparisons showing topology improvements'),
        ('3d_topology', '3D Interactive Visualizations', 'Interactive 3D visualizations of vascular structures'),
        ('publication', 'Publication Figures', 'High-quality figures ready for publication')
    ]
    
    for folder, title, description in sections:
        folder_path = Path(output_dir) / folder
        if folder_path.exists():
            files = list(folder_path.glob('*.*'))
            if files:
                html_content += f"""
        <div class="section">
            <h2>📊 {title}</h2>
            <p>{description}</p>
            <div class="file-grid">
"""
                for file in sorted(files):
                    relative_path = f"{folder}/{file.name}"
                    file_type = "🖼️" if file.suffix.lower() in ['.png', '.jpg', '.jpeg'] else \
                               "📄" if file.suffix.lower() == '.pdf' else \
                               "🌐" if file.suffix.lower() == '.html' else "📁"
                    
                    html_content += f"""
                <div class="file-item">
                    <a href="{relative_path}" target="_blank">
                        {file_type} {file.name}
                    </a>
                </div>
"""
                
                html_content += """
            </div>
        </div>
"""
    
    # Add status summary
    successful = sum(1 for success in results.values() if success)
    total = len(results)
    
    html_content += f"""
        <div class="section">
            <h2>📈 Generation Summary</h2>
            <p><strong>Success Rate:</strong> {successful}/{total} ({successful/total*100:.1f}%)</p>
            <p><strong>Patient Cases:</strong> {len(patient_ids)}</p>
            <p><strong>Visualization Types:</strong> 2D Topology, 3D Interactive, Publication Figures</p>
        </div>
        
        <div class="timestamp">
            Generated on {time.strftime('%Y-%m-%d at %H:%M:%S')} using Graph-to-Graph Correction Framework
        </div>
    </div>
</body>
</html>
"""
    
    # Save index file
    index_path = Path(output_dir) / 'index.html'
    with open(index_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Visualization index created: {index_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Create comprehensive topology visualizations')
    parser.add_argument('--patient-ids', nargs='+', 
                       default=['PA000005', 'PA000016', 'PA000026'],
                       help='Patient IDs to visualize')
    parser.add_argument('--output-dir', default='visualizations/comprehensive',
                       help='Output directory for all visualizations')
    parser.add_argument('--no-3d', action='store_true',
                       help='Skip 3D visualizations (faster)')
    parser.add_argument('--single-patient', 
                       help='Create visualizations for single patient only')
    
    args = parser.parse_args()
    
    # Handle single patient mode
    if args.single_patient:
        patient_ids = [args.single_patient]
    else:
        patient_ids = args.patient_ids
    
    # Create all visualizations
    results = create_comprehensive_visualizations(
        patient_ids=patient_ids,
        output_dir=args.output_dir,
        include_3d=not args.no_3d
    )
    
    # Final summary
    successful = sum(1 for success in results.values() if success)
    total = len(results)
    
    if successful == total:
        logger.info("🎉 All visualizations generated successfully!")
    elif successful > 0:
        logger.warning(f"⚠️ Partial success: {successful}/{total} visualizations completed")
    else:
        logger.error("❌ Visualization generation failed")
        return 1
    
    logger.info(f"📂 View results at: {args.output_dir}/index.html")
    return 0


if __name__ == '__main__':
    exit(main())