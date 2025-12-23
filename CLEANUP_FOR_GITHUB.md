# Cleanup Guide for GitHub Release

This document lists files and directories that should be removed before pushing to GitHub.
The `.gitignore` file already excludes most of these, but for a clean start, you may want to delete them manually.

## Files/Directories to DELETE

### Root Level - Debug/Temp Files
```bash
rm -f debug_features.py
rm -f examine_graph.py
rm -f calculate_anatomy_summary.py
rm -f anatomy_metrics_summary.json
rm -f correspondence_test.log
rm -f correspondence_test_results.json
rm -f graph_correction_test.log
rm -f graph_correction_test_results.json
rm -f graph_extraction.log
rm -f graph_extraction_predictions.log
rm -f extracted_graphs_test.log
rm -f exp1_anatomy_run.log
rm -f test_skeletonization.png
rm -f training_data_analysis.png
rm -f methodology_paper.md
rm -f graph_correction_strategy.md
rm -f results_impact_table.md
rm -f setup_nnunet.md
rm -f CBM2025_Houssem.pdf
```

### Directories to DELETE
```bash
rm -rf analysis_results/
rm -rf extracted_graphs/
rm -rf training_data/
rm -rf test_output/
rm -rf results/
rm -rf paper_figures/
rm -rf figures/
rm -rf visualizations/
rm -rf figure_generation/
rm -rf experiments/
rm -rf nnUNet/
rm -rf .serena/
rm -rf pulmonary-artery-seg-venv/
rm -rf pulmonary-artery-seg-server-venv/
```

### Scripts to DELETE (debug/test scripts)
Keep only these core scripts in `scripts/`:
- `train_unet.py`
- `train_graph_correction.py`
- `generate_predictions.py`
- `extract_all_graphs.py`
- `evaluate_full_pipeline.py`
- `train_examples.sh`

Delete all others:
```bash
cd scripts/
rm -f analyze_*.py
rm -f check_*.py
rm -f debug_*.py
rm -f quick_*.py
rm -f test_*.py
rm -f create_*.py
rm -f generate_*_figures.py
rm -f generate_*_visualizations.py
rm -f reevaluate_*.py
rm -f summarize_*.py
rm -f validate_*.py
rm -f visualize_*.py
rm -f batch_evaluate_*.py
rm -f collect_*.py
rm -f simple_*.py
rm -f setup_nnunet.py
rm -f prepare_nnunet_dataset.py
rm -f generate_nnunet_predictions.py
rm -f train_*.py  # except train_unet.py and train_graph_correction.py
rm -rf __pycache__/
```

### Configs to KEEP
Keep only essential configs:
- `unet_config.yaml`
- `graph_config.yaml`
- `enhanced_training_config.yaml`

Optionally remove experiment-specific configs:
```bash
cd configs/
rm -f exp2_topology_only_config.yaml
rm -f exp3_anatomy_only_config.yaml
rm -f graph_correction_balanced.yaml
rm -f graph_correction_progressive.yaml
rm -f regression_model_config.yaml
rm -f unet_cldice_config.yaml
rm -f enhanced_model_config.yaml
```

## Quick Cleanup Script

Run this script to clean everything at once:

```bash
#!/bin/bash
# cleanup_for_github.sh

# Navigate to project root
cd /path/to/MASTER_PROJECT

# Remove root-level temp files
rm -f debug_features.py examine_graph.py calculate_anatomy_summary.py
rm -f anatomy_metrics_summary.json
rm -f *.log correspondence_test_results.json graph_correction_test_results.json
rm -f test_skeletonization.png training_data_analysis.png
rm -f methodology_paper.md graph_correction_strategy.md results_impact_table.md setup_nnunet.md
rm -f CBM2025_Houssem.pdf

# Remove directories
rm -rf analysis_results extracted_graphs training_data test_output
rm -rf results paper_figures figures visualizations figure_generation
rm -rf experiments nnUNet .serena
rm -rf pulmonary-artery-seg-venv pulmonary-artery-seg-server-venv

# Clean scripts directory (keep only core scripts)
cd scripts/
find . -name "analyze_*.py" -delete
find . -name "check_*.py" -delete
find . -name "debug_*.py" -delete
find . -name "quick_*.py" -delete
find . -name "test_*.py" -delete
find . -name "create_*.py" -delete
find . -name "visualize_*.py" -delete
find . -name "batch_*.py" -delete
find . -name "collect_*.py" -delete
find . -name "reevaluate_*.py" -delete
find . -name "summarize_*.py" -delete
find . -name "validate_*.py" -delete
find . -name "simple_*.py" -delete
rm -f setup_nnunet.py prepare_nnunet_dataset.py generate_nnunet_predictions.py
rm -f generate_*_figures.py generate_*_visualizations.py generate_topology_showcase.py
rm -f train_multiple_unets.py train_regression_*.py train_nnunet_standalone.py train_unet_cldice.py train_enhanced_model.py
rm -rf __pycache__/
cd ..

# Remove notebooks if empty
rm -rf notebooks/

echo "Cleanup complete!"
```

## Final Directory Structure

After cleanup, your project should look like:

```
MASTER_PROJECT/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── CLAUDE.md                    # Can remove or keep as internal docs
├── configs/
│   ├── unet_config.yaml
│   ├── graph_config.yaml
│   └── enhanced_training_config.yaml
├── docs/
│   └── images/
│       └── pipeline_overview.png   # Add your figure here
├── src/
│   ├── __init__.py
│   ├── data/
│   ├── models/
│   ├── training/
│   └── utils/
├── scripts/
│   ├── train_unet.py
│   ├── train_graph_correction.py
│   ├── generate_predictions.py
│   ├── extract_all_graphs.py
│   ├── evaluate_full_pipeline.py
│   └── train_examples.sh
└── tests/
    └── (unit tests if any)
```

## Before Pushing

1. Run the cleanup script
2. Add a pipeline overview figure to `docs/images/pipeline_overview.png`
3. Verify all imports work: `python -c "from src.models.graph_correction import GraphCorrectionModel"`
4. Initialize git and push:

```bash
git init
git add .
git commit -m "Initial commit: Graph-to-Graph Correction Framework"
git remote add origin https://github.com/Houssem-Ben-Salem/Graph-to-Graph-Correction-Framework-for-Consistent-Vascular-Segmentation.git
git push -u origin main
```
