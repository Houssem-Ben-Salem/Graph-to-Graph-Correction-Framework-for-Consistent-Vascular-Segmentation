#!/bin/bash

# Multi-Architecture U-Net Training Examples
# ========================================

echo "Multi-Architecture U-Net Training Examples"
echo "==========================================="
echo

# Example 1: Train only the 3 PyTorch architectures (no nnU-Net)
echo "Example 1: Train UNet3D, Attention UNet3D, and UNet2D (no nnU-Net)"
echo "Command:"
echo "python scripts/train_multiple_unets.py \\"
echo "  --dataset DATASET/Parse_dataset \\"
echo "  --architectures unet3d attention_unet3d unet2d \\"
echo "  --output-dir experiments/pytorch_unets"
echo

# Example 2: Train only UNet3D and Attention UNet3D
echo "Example 2: Train only UNet3D and Attention UNet3D"
echo "Command:"
echo "python scripts/train_multiple_unets.py \\"
echo "  --dataset DATASET/Parse_dataset \\"
echo "  --architectures unet3d attention_unet3d \\"
echo "  --output-dir experiments/3d_unets_only"
echo

# Example 3: Train everything based on config file (currently excludes nnU-Net)
echo "Example 3: Train based on config file settings"
echo "Command:"
echo "python scripts/train_multiple_unets.py \\"
echo "  --dataset DATASET/Parse_dataset \\"
echo "  --output-dir experiments/config_based"
echo

# Example 4: Override config to disable UNet2D
echo "Example 4: Use config but disable UNet2D"
echo "Command:"
echo "python scripts/train_multiple_unets.py \\"
echo "  --dataset DATASET/Parse_dataset \\"
echo "  --disable-unet2d \\"
echo "  --output-dir experiments/no_2d"
echo

# Example 5: Train only UNet3D for quick testing
echo "Example 5: Train only UNet3D for quick testing"
echo "Command:"
echo "python scripts/train_multiple_unets.py \\"
echo "  --dataset DATASET/Parse_dataset \\"
echo "  --architectures unet3d \\"
echo "  --output-dir experiments/unet3d_only"
echo

# Example 6: When nnU-Net is working later
echo "Example 6: Train all architectures including nnU-Net (when installation is fixed)"
echo "Command:"
echo "python scripts/train_multiple_unets.py \\"
echo "  --dataset DATASET/Parse_dataset \\"
echo "  --architectures unet3d attention_unet3d unet2d nnunet \\"
echo "  --output-dir experiments/all_architectures"
echo

echo "Note: Set train_nnunet: true in configs/unet_config.yaml when nnU-Net is ready"
echo

# Example 7: Individual enable flags
echo "Example 7: Using individual enable flags"
echo "Command:"
echo "python scripts/train_multiple_unets.py \\"
echo "  --dataset DATASET/Parse_dataset \\"
echo "  --enable-unet3d \\"
echo "  --enable-attention-unet3d \\"
echo "  --output-dir experiments/individual_flags"
echo

echo "Example 8: Train with progress bars and WandB logging"
echo "Command:"
echo "python scripts/train_multiple_unets.py \\"
echo "  --dataset DATASET/Parse_dataset \\"
echo "  --architectures unet3d attention_unet3d unet2d \\"
echo "  --use-wandb \\"
echo "  --wandb-project my-pulmonary-project \\"
echo "  --output-dir experiments/tracked_training"
echo

echo "Example 9: Train with progress bars but no WandB"
echo "Command:"
echo "python scripts/train_multiple_unets.py \\"
echo "  --dataset DATASET/Parse_dataset \\"
echo "  --architectures unet3d attention_unet3d unet2d \\"
echo "  --output-dir experiments/progress_training"
echo

echo "Example 10: Parallel training on 2 GPUs (NEW!)"
echo "Command:"
echo "python scripts/train_multiple_unets.py \\"
echo "  --dataset DATASET/Parse_dataset \\"
echo "  --architectures unet3d attention_unet3d unet2d \\"
echo "  --parallel \\"
echo "  --gpu-ids 0 1 \\"
echo "  --output-dir experiments/parallel_training"
echo

echo "Example 11: Parallel training with WandB on specific GPUs"
echo "Command:"
echo "python scripts/train_multiple_unets.py \\"
echo "  --dataset DATASET/Parse_dataset \\"
echo "  --architectures unet3d attention_unet3d \\"
echo "  --parallel \\"
echo "  --gpu-ids 0 1 \\"
echo "  --use-wandb \\"
echo "  --wandb-project parallel-unet-training \\"
echo "  --output-dir experiments/parallel_tracked"
echo

echo "Example 12: Test parallel training with just 2 architectures"
echo "Command:"
echo "python scripts/train_multiple_unets.py \\"
echo "  --dataset DATASET/Parse_dataset \\"
echo "  --architectures unet3d attention_unet3d \\"
echo "  --parallel \\"
echo "  --output-dir experiments/parallel_test"
echo

echo "Note: Install wandb with: pip install wandb"
echo "Note: The script now automatically validates dataset and shows progress bars!"
echo "Note: Parallel training requires multiple GPUs and multiple architectures"
echo "Note: With --parallel, each architecture trains on a different GPU simultaneously"
echo

echo "RESULTS ANALYSIS EXAMPLES:"
echo "========================="
echo
echo "Example 13: Quick check of results"
echo "Command:"
echo "python scripts/quick_check.py experiments/multi_unet"
echo
echo "Example 14: Detailed analysis of results"
echo "Command:"
echo "python scripts/analyze_results.py --results-dir experiments/multi_unet --save-report"
echo
echo "Example 15: Check specific experiment"
echo "Command:"
echo "python scripts/quick_check.py experiments/parallel_training"
echo

echo "PREDICTION GENERATION EXAMPLES:"
echo "==============================="
echo
echo "Example 16: Generate predictions from specific model file (CURRENT TRAINING)"
echo "Command:"
echo "python scripts/generate_predictions.py \\"
echo "  --model-path experiments/parallel_training/attention_unet3d/fold_0/best_model.pth \\"
echo "  --architecture attention_unet3d \\"
echo "  --dataset DATASET/Parse_dataset \\"
echo "  --output-dir experiments/predictions_current \\"
echo "  --gpu-id 0 \\"
echo "  --min-dice 0.6"
echo
echo "Example 16b: Generate predictions with per-case analysis (best model only - when training complete)"
echo "Command:"
echo "python scripts/generate_predictions.py \\"
echo "  --results-dir experiments/parallel_training \\"
echo "  --dataset DATASET/Parse_dataset \\"
echo "  --output-dir experiments/predictions \\"
echo "  --best-only \\"
echo "  --gpu-id 0 \\"
echo "  --min-dice 0.6 \\"
echo "  --max-hausdorff 50"
echo
echo "Example 17: Generate predictions from all trained models with analysis"
echo "Command:"
echo "python scripts/generate_predictions.py \\"
echo "  --results-dir experiments/parallel_training \\"
echo "  --dataset DATASET/Parse_dataset \\"
echo "  --output-dir experiments/predictions \\"
echo "  --gpu-id 1 \\"
echo "  --min-dice 0.5 \\"
echo "  --max-hausdorff 100"
echo
echo "Example 18: Generate predictions for specific patients only"
echo "Command:"
echo "python scripts/generate_predictions.py \\"
echo "  --results-dir experiments/parallel_training \\"
echo "  --dataset DATASET/Parse_dataset \\"
echo "  --output-dir experiments/predictions_subset \\"
echo "  --best-only \\"
echo "  --patient-ids PA000070 PA000074 PA000080 PA000090"
echo
echo "Example 19: Extract graphs from good cases only (after analysis)"
echo "Command:"
echo "python scripts/extract_predicted_graphs.py \\"
echo "  --predictions-dir experiments/predictions/attention_unet3d \\"
echo "  --graphs-dir extracted_graphs \\"
echo "  --threshold 0.5 \\"
echo "  --patient-list experiments/predictions/attention_unet3d/good_cases.txt"
echo
echo "Choose the example that fits your current setup!"