# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a medical imaging project implementing a novel **Graph-to-Graph Correction Framework** for enhancing vascular segmentation (specifically pulmonary artery segmentation). The system transforms imperfect U-Net predictions into anatomically and topologically correct segmentations by:

1. Converting segmentation masks to graph representations
2. Learning structural corrections in graph space using GNNs
3. Reconstructing enhanced volumetric masks from corrected graphs

## Project Structure

```
MASTER_PROJECT/
├── configs/                    # Configuration files
│   ├── unet_config.yaml       # U-Net training configuration
│   ├── graph_config.yaml      # Graph correction configuration
│   └── data_config.yaml       # Data processing configuration
│
├── src/                       # Source code
│   ├── data/                  # Data handling modules
│   │   ├── preprocessing/     # Image preprocessing utilities
│   │   ├── augmentation/      # Data augmentation strategies
│   │   └── loaders/          # PyTorch data loaders
│   │
│   ├── models/               # Model architectures
│   │   ├── unet/            # U-Net and variants (UNet3D, AttentionUNet3D)
│   │   ├── graph_extraction/ # Mask to graph conversion modules
│   │   ├── graph_correction/ # GNN correction models
│   │   └── reconstruction/   # Graph to mask conversion
│   │
│   ├── training/             # Training pipelines
│   │   ├── unet_trainer.py   # U-Net training logic
│   │   ├── graph_trainer.py  # Graph correction training
│   │   ├── losses.py         # Loss functions
│   │   └── synthetic_degradation.py # Synthetic graph degradation
│   │
│   ├── inference/            # Inference pipelines
│   │   ├── unet_predictor.py # U-Net inference
│   │   └── graph_enhancer.py # Full enhancement pipeline
│   │
│   ├── utils/                # Utility functions
│   │   ├── metrics.py        # Evaluation metrics
│   │   ├── visualization.py  # Visualization tools
│   │   └── graph_correspondence.py # Graph matching utilities
│   │
│   └── evaluation/           # Evaluation scripts
│       ├── topology_metrics.py # Topology preservation metrics
│       └── anatomy_metrics.py  # Anatomical consistency metrics
│
├── experiments/              # Experiment outputs
│   ├── unet/                # U-Net training runs
│   └── graph_correction/    # Graph correction runs
│
├── notebooks/               # Jupyter notebooks for exploration
│
├── scripts/                 # Executable scripts
│   ├── train_unet.py       # Train U-Net models
│   ├── train_graph_correction.py # Train graph correction
│   └── evaluate_pipeline.py # Evaluate full pipeline
│
├── tests/                   # Unit tests
│
├── DATASET/Parse_dataset/   # Medical imaging data (NIfTI format)
│   └── PA*/                # Patient directories
│
├── graph_correction_strategy.md # Technical specification
├── requirements.txt         # Python dependencies
├── README.md               # Project overview
└── pulmonary-artery-seg-venv/ # Virtual environment
```

## Key Technologies

- **PyTorch Geometric**: For graph neural network implementation
- **SimpleITK/nibabel**: For medical image I/O (NIfTI files)
- **scikit-image**: For image processing operations (skeletonization, morphology)
- **NetworkX**: For graph manipulation and visualization
- **MONAI**: Optional but recommended for medical imaging operations

## Common Development Commands

### Environment Setup
```bash
# Activate virtual environment
source pulmonary-artery-seg-venv/bin/activate  # Linux/Mac
# or
pulmonary-artery-seg-venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install torch-geometric  # Follow PyTorch Geometric installation guide for your CUDA version
```

### Data Processing
- Medical images are in NIfTI format (.nii.gz)
- Each patient has an image/ and label/ directory
- Use SimpleITK or nibabel to load and process these files

## Architecture Overview

The system consists of 5 main components:

### 1. Graph Extraction (Step 1)
- **Purpose**: Convert segmentation masks to graph representations
- **Key Operations**:
  - Morphological cleaning and preprocessing
  - 3D skeletonization for centerline extraction
  - Strategic node placement at bifurcations and along vessels
  - Edge creation following vessel connectivity
  - Node/edge attribute extraction (radius, curvature, confidence)

### 2. Graph Correspondence (Step 2)
- **Purpose**: Match nodes/edges between predicted and ground truth graphs
- **Key Algorithms**:
  - Spatial alignment and normalization
  - Multi-level correspondence matching
  - Topological feature comparison
  - Uncertainty quantification for matches

### 3. Graph-to-Graph Correction Network (Step 3)
- **Core Architecture**:
  - Multi-head Graph Attention Networks (GAT)
  - Specialized modules for topology and anatomy correction
  - Physiological constraint enforcement (Murray's law)
  - Multi-objective loss function

### 4. Training Strategy (Step 4)
- **Data Augmentation**: Synthetic graph degradation pipeline
- **Training Approach**: 
  - Pre-training on synthetic degradations
  - Fine-tuning on real U-Net predictions
  - Curriculum learning from simple to complex cases
  - Memory-efficient training with gradient checkpointing

### 5. Template-Based Reconstruction (Step 5)
- **Purpose**: Convert corrected graphs back to volumetric masks
- **Method**: 
  - Parameterized vessel templates (cylinders, bifurcations)
  - Signed Distance Field (SDF) based rendering
  - Integration with original U-Net predictions

## Implementation Considerations

### Graph Processing
- Typical graph size: 100-1000 nodes per vessel tree
- Maintain spatial indexing (KD-tree) for efficient queries
- Preserve topological features during all transformations

### Performance Optimization
- Use gradient checkpointing for memory efficiency
- Mixed precision training (autocast) where appropriate
- Batch processing strategies for graph operations

### Validation Metrics
- **Topology**: Connected component analysis, bifurcation detection accuracy
- **Anatomy**: Murray's law compliance, vessel tapering consistency
- **Overall**: Dice score, sensitivity, precision, Hausdorff distance

## Key Algorithms to Implement

1. **3D Skeletonization**: Lee's algorithm or Zhang-Suen for centerline extraction
2. **Graph Matching**: Hungarian algorithm with custom cost matrix
3. **Murray's Law Enforcement**: Physiological constraint for vessel radii at bifurcations
4. **SDF Rendering**: Efficient volumetric reconstruction from geometric primitives

## Development Workflow

1. Start with graph extraction module - ensure reliable conversion from masks to graphs
2. Implement correspondence matching - critical for supervised learning
3. Build GNN correction network incrementally - start with simple topology corrections
4. Develop synthetic degradation pipeline for training data
5. Implement template-based reconstruction last - can validate graph corrections first

## Training Workflow

### Phase 1: U-Net Training
1. **Data Preparation**: 
   - Use PulmonaryArteryDataset with patch-based training
   - Apply augmentations (rotation, scaling, elastic deformation)
   - Normalize images using percentile normalization

2. **Training Command**:
   ```bash
   python scripts/train_unet.py --config configs/unet_config.yaml
   ```

3. **Generate Predictions**:
   ```bash
   python scripts/generate_predictions.py \
     --model experiments/unet/best_model.pth \
     --output-dir experiments/unet/predictions
   ```

### Phase 2: Graph Correction Training
1. **Prerequisites**:
   - U-Net predictions must be generated first
   - Both prediction masks and confidence maps are used

2. **Training Command**:
   ```bash
   python scripts/train_graph_correction.py \
     --config configs/graph_config.yaml \
     --unet-predictions experiments/unet/predictions
   ```

3. **Curriculum Learning**:
   - Stage 1: Synthetic mild degradations (50 epochs)
   - Stage 2: Synthetic moderate degradations (50 epochs)  
   - Stage 3: Real U-Net predictions (200 epochs)

### Phase 3: Full Pipeline Evaluation
```bash
python scripts/evaluate_pipeline.py \
  --unet-model experiments/unet/best_model.pth \
  --graph-model experiments/graph_correction/best_model.pth \
  --test-data DATASET/Parse_dataset/test
```

## Important Notes

- The dataset contains sensitive medical data - handle with appropriate care
- Maintain reproducibility by storing all extraction parameters and random seeds
- Graph representations must preserve critical topological features
- Always validate that enhanced masks are better than original U-Net predictions