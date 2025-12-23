"""
Helper Script: Find Best Slices for Figure 1

This script helps you:
1. Verify that all required data files exist
2. Find optimal slice indices with maximum vessel visibility
3. Preview slices before generating the final figure

Author: Generated for MASTER_PROJECT
Date: 2025-10-19
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage

# ============================================================================
# CONFIGURATION
# ============================================================================

CASES = {
    'easy': 'PA000016',
    'medium': 'PA000026',
    'hard': 'PA000005',
}

BASE_DIR = Path("/home/au89100@ens.ad.etsmtl.ca/MYWORK/MASTER_PROJECT/DATASET/Parse_dataset")

# ============================================================================
# DATA VERIFICATION
# ============================================================================

def verify_data_exists(base_dir):
    """
    Check if all required data files exist for the three cases.
    """
    print("=" * 80)
    print("DATA VERIFICATION")
    print("=" * 80)

    all_exist = True

    for case_type, case_id in CASES.items():
        print(f"\n{case_type.upper()} Case: {case_id}")
        print("-" * 80)

        # Define expected paths (matching create_figure1_qualitative_comparison.py)
        # Go up two levels: Parse_dataset -> DATASET -> MASTER_PROJECT
        project_root = base_dir.parent.parent

        paths = {
            'CT Scan': base_dir / case_id / 'image' / f'{case_id}.nii.gz',
            'Ground Truth': base_dir / case_id / 'label' / f'{case_id}.nii.gz',
            'Baseline (EXP-4)': project_root / 'experiments' / 'test_predictions' / f'{case_id}_pred.nii.gz',
            'Volumetric (EXP-1)': project_root / 'experiments' / 'ablation' / 'exp1_volumetric' / 'results' / 'predictions' / f'{case_id}_refined.nii.gz',
            'Topology-Only (EXP-2)': project_root / 'experiments' / 'ablation' / 'exp2_topology_only' / 'predictions_p90' / f'{case_id}_refined_p90.nii.gz',
            'Anatomy-Only (EXP-3)': project_root / 'experiments' / 'ablation' / 'exp3_anatomy_only' / 'predictions_p90' / f'{case_id}_refined_p90.nii.gz',
            'Full Model (EXP-2)': project_root / 'experiments' / 'ablation' / 'exp2_topology_only' / 'predictions_p90' / f'{case_id}_refined_p90.nii.gz',
        }

        for name, path in paths.items():
            exists = path.exists()
            status = "✓ Found" if exists else "✗ Missing"
            print(f"  {status:12s} {name:20s} {path}")
            if not exists:
                all_exist = False

    print("\n" + "=" * 80)
    if all_exist:
        print("SUCCESS: All required files found!")
    else:
        print("WARNING: Some files are missing. Please check the paths above.")
    print("=" * 80)

    return all_exist


# ============================================================================
# SLICE SELECTION HELPERS
# ============================================================================

def calculate_vessel_content(mask_slice):
    """
    Calculate a score representing vessel content in a slice.
    Higher score = more vessels visible.
    """
    # Percentage of non-zero voxels
    vessel_ratio = np.sum(mask_slice > 0) / mask_slice.size

    # Number of connected components (want moderate number, not too fragmented)
    labeled, num_features = ndimage.label(mask_slice > 0)

    # Ideal: reasonable coverage with not too many components
    # Penalize both very low and very high component counts
    component_penalty = abs(num_features - 10) / 10.0  # Normalize

    # Combined score (higher is better)
    score = vessel_ratio - 0.1 * component_penalty

    return score, vessel_ratio, num_features


def find_best_slices(base_dir, case_id, num_candidates=10):
    """
    Find the best slice indices for visualization based on vessel content.

    Parameters:
    -----------
    base_dir : Path
        Base directory
    case_id : str
        Case ID
    num_candidates : int
        Number of top candidate slices to return

    Returns:
    --------
    candidates : list
        List of (slice_idx, score, vessel_ratio, num_components) tuples
    """
    # Load ground truth mask
    gt_path = base_dir / case_id / 'label' / f'{case_id}.nii.gz'

    if not gt_path.exists():
        print(f"Warning: Ground truth not found for {case_id}")
        return []

    img = nib.load(str(gt_path))
    data = img.get_fdata()

    # Calculate scores for all slices
    slice_scores = []
    for slice_idx in range(data.shape[2]):
        mask_slice = data[:, :, slice_idx]
        score, vessel_ratio, num_comp = calculate_vessel_content(mask_slice)
        slice_scores.append((slice_idx, score, vessel_ratio, num_comp))

    # Sort by score (descending)
    slice_scores.sort(key=lambda x: x[1], reverse=True)

    # Return top candidates
    return slice_scores[:num_candidates]


def preview_slice_candidates(base_dir, case_id, num_candidates=5):
    """
    Create a preview figure showing top candidate slices.
    """
    candidates = find_best_slices(base_dir, case_id, num_candidates)

    if not candidates:
        print(f"Could not find candidates for {case_id}")
        return

    # Load CT and mask
    ct_path = base_dir / case_id / 'image' / f'{case_id}.nii.gz'
    gt_path = base_dir / case_id / 'label' / f'{case_id}.nii.gz'

    ct_img = nib.load(str(ct_path))
    gt_img = nib.load(str(gt_path))

    ct_data = ct_img.get_fdata()
    gt_data = gt_img.get_fdata()

    # Create preview figure
    fig, axes = plt.subplots(2, num_candidates, figsize=(15, 6))
    fig.suptitle(f'Top {num_candidates} Slice Candidates for {case_id}',
                 fontsize=14, fontweight='bold')

    for idx, (slice_idx, score, vessel_ratio, num_comp) in enumerate(candidates):
        # CT slice
        ct_slice = ct_data[:, :, slice_idx]
        ct_normalized = np.clip((ct_slice - (-500)) / (500 - (-500)), 0, 1)

        axes[0, idx].imshow(ct_slice, cmap='gray', vmin=-500, vmax=500)
        axes[0, idx].set_title(f'Slice {slice_idx}\nScore: {score:.3f}',
                              fontsize=10)
        axes[0, idx].axis('off')

        # Mask slice
        mask_slice = gt_data[:, :, slice_idx]
        axes[1, idx].imshow(mask_slice, cmap='hot')
        axes[1, idx].set_title(f'Vessels: {vessel_ratio*100:.1f}%\nComps: {num_comp}',
                              fontsize=9)
        axes[1, idx].axis('off')

    axes[0, 0].set_ylabel('CT Scan', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Vessel Mask', fontsize=11, fontweight='bold')

    plt.tight_layout()
    output_path = f'preview_slices_{case_id}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Preview saved: {output_path}")

    # Print candidate information
    print(f"\nTop {num_candidates} candidates for {case_id}:")
    print("-" * 60)
    print(f"{'Rank':<6} {'Slice':<8} {'Score':<10} {'Vessels':<12} {'Components':<12}")
    print("-" * 60)
    for rank, (slice_idx, score, vessel_ratio, num_comp) in enumerate(candidates, 1):
        print(f"{rank:<6} {slice_idx:<8} {score:<10.3f} {vessel_ratio*100:<11.1f}% {num_comp:<12}")
    print("-" * 60)

    return candidates


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print(" FIGURE 1 DATA PREPARATION HELPER")
    print("=" * 80 + "\n")

    # Step 1: Verify data exists
    data_exists = verify_data_exists(BASE_DIR)

    if not data_exists:
        print("\n⚠️  Some data files are missing. Please check the paths.")
        print("You may need to adjust BASE_DIR or generate missing predictions.\n")
        return

    print("\n" + "=" * 80)
    print(" FINDING OPTIMAL SLICE INDICES")
    print("=" * 80 + "\n")

    # Step 2: Find best slices for each case
    recommended_slices = {}

    for case_type, case_id in CASES.items():
        print(f"\nAnalyzing {case_type.upper()} case: {case_id}")
        print("-" * 80)

        candidates = preview_slice_candidates(BASE_DIR, case_id, num_candidates=5)

        if candidates:
            best_slice = candidates[0][0]
            recommended_slices[case_type] = best_slice
            print(f"✓ Recommended slice for {case_id}: {best_slice}\n")
        else:
            recommended_slices[case_type] = None
            print(f"✗ Could not find optimal slice for {case_id}\n")

    # Step 3: Print summary
    print("\n" + "=" * 80)
    print(" SUMMARY: RECOMMENDED SLICE INDICES")
    print("=" * 80 + "\n")

    print("Copy this into create_figure1_qualitative_comparison.py:\n")
    print("SLICE_INDICES = {")
    for case_type, slice_idx in recommended_slices.items():
        if slice_idx is not None:
            print(f"    '{case_type}': {slice_idx},")
        else:
            print(f"    '{case_type}': None,  # Not found")
    print("}")

    print("\n" + "=" * 80)
    print(" NEXT STEPS")
    print("=" * 80)
    print("\n1. Review the preview images: preview_slices_PA*.png")
    print("2. Choose your preferred slices from the candidates")
    print("3. Update SLICE_INDICES in create_figure1_qualitative_comparison.py")
    print("4. Run: python create_figure1_qualitative_comparison.py")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
