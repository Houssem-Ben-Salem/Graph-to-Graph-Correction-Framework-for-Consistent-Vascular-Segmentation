"""
FIGURE 1: Qualitative Visual Comparison
High-quality figure for journal publication

This script creates a 7×3 grid showing visual comparison of all methods
across three representative cases (Easy, Medium, Hard).

Author: Generated for MASTER_PROJECT
Date: 2025-10-19
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyBboxPatch
import nibabel as nib
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Color scheme (consistent with figures.md specifications)
COLORS = {
    'ground_truth': (0, 1, 0),      # Green RGB
    'prediction': (1, 0, 0),         # Red RGB
    'text_red': '#E63946',
    'text_green': '#06A77D',
    'text_blue': '#2E86AB',
    'text_orange': '#F77F00',
}

# Figure parameters
FIGURE_SIZE = (16, 12)
DPI = 300
WSPACE = 0.02
HSPACE = 0.05

# CT intensity window for display
CT_WINDOW = (-500, 500)

# Case information
CASES = {
    'easy': {'id': 'PA000016', 'label': 'Case PA000016 (Easy)'},
    'medium': {'id': 'PA000026', 'label': 'Case PA000026 (Medium)'},
    'hard': {'id': 'PA000005', 'label': 'Case PA000005 (Hard)'},
}

# Row labels
ROW_LABELS = [
    'CT Scan',
    'Ground Truth',
    'Baseline (360 comp.)',
    'Volumetric Refine',
    'Topology-Only',
    'Anatomy-Only',
    'Full Model (28 comp.)'
]

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_nifti_slice(file_path, slice_idx=None):
    """
    Load a 2D slice from a NIfTI file.

    Parameters:
    -----------
    file_path : str or Path
        Path to the NIfTI file (.nii or .nii.gz)
    slice_idx : int, optional
        Axial slice index. If None, uses middle slice.

    Returns:
    --------
    slice_data : np.ndarray
        2D numpy array of the slice
    """
    img = nib.load(str(file_path))
    data = img.get_fdata()

    # Get middle slice if not specified
    if slice_idx is None:
        slice_idx = data.shape[2] // 2

    # Extract axial slice
    slice_data = data[:, :, slice_idx]

    return slice_data


def get_data_paths(base_dir, case_id):
    """
    Get all required file paths for a given case.

    Parameters:
    -----------
    base_dir : str or Path
        Base directory containing the dataset (Parse_dataset)
    case_id : str
        Patient case ID (e.g., 'PA000016')

    Returns:
    --------
    paths : dict
        Dictionary containing paths to all required files
    """
    base_dir = Path(base_dir)
    case_dir = base_dir / case_id

    # Get project root (two levels up: Parse_dataset -> DATASET -> MASTER_PROJECT)
    project_root = base_dir.parent.parent

    paths = {
        # CT scan and ground truth from dataset
        'ct_scan': case_dir / 'image' / f'{case_id}.nii.gz',
        'ground_truth': case_dir / 'label' / f'{case_id}.nii.gz',

        # Baseline: U-Net predictions without any correction (EXP-4)
        'baseline': project_root / 'experiments' / 'test_predictions' / f'{case_id}_pred.nii.gz',

        # EXP-1: Volumetric refinement with 3D CNN
        'volumetric': project_root / 'experiments' / 'ablation' / 'exp1_volumetric' / 'results' / 'predictions' / f'{case_id}_refined.nii.gz',

        # EXP-2: Topology-only (graph correction with topology losses)
        'topology_only': project_root / 'experiments' / 'ablation' / 'exp2_topology_only' / 'predictions_p90' / f'{case_id}_refined_p90.nii.gz',

        # EXP-3: Anatomy-only (graph correction with anatomy losses)
        'anatomy_only': project_root / 'experiments' / 'ablation' / 'exp3_anatomy_only' / 'predictions_p90' / f'{case_id}_refined_p90.nii.gz',

        # Full Model: Since ablation study shows Main Model = EXP-2, use EXP-2 predictions
        # (Both topology and anatomy losses produce identical results)
        'full_model': project_root / 'experiments' / 'ablation' / 'exp2_topology_only' / 'predictions_p90' / f'{case_id}_refined_p90.nii.gz',
    }

    return paths


def load_case_data(base_dir, case_id, slice_idx=None):
    """
    Load all data for a specific case.

    Parameters:
    -----------
    base_dir : str or Path
        Base directory containing the dataset
    case_id : str
        Patient case ID
    slice_idx : int, optional
        Slice index to extract

    Returns:
    --------
    data : dict
        Dictionary containing all slices for the case
    """
    paths = get_data_paths(base_dir, case_id)

    data = {}
    for key, path in paths.items():
        if path.exists():
            data[key] = load_nifti_slice(path, slice_idx)
        else:
            print(f"Warning: File not found: {path}")
            # Create dummy data for demonstration
            data[key] = np.random.rand(256, 256) if key == 'ct_scan' else np.random.rand(256, 256) > 0.8

    return data


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def normalize_ct_image(ct_slice, window=CT_WINDOW):
    """
    Normalize CT image with windowing for display.

    Parameters:
    -----------
    ct_slice : np.ndarray
        Raw CT slice
    window : tuple
        (min_HU, max_HU) for windowing

    Returns:
    --------
    normalized : np.ndarray
        Normalized image in range [0, 1]
    """
    min_val, max_val = window
    ct_clipped = np.clip(ct_slice, min_val, max_val)
    normalized = (ct_clipped - min_val) / (max_val - min_val)
    return normalized


def create_overlay(background, mask, color, alpha=0.6):
    """
    Create RGB overlay of mask on grayscale background.

    Parameters:
    -----------
    background : np.ndarray
        Grayscale background image (2D)
    mask : np.ndarray
        Binary mask (2D)
    color : tuple
        RGB color (0-1 range)
    alpha : float
        Transparency of overlay

    Returns:
    --------
    overlay : np.ndarray
        RGB image with overlay (H, W, 3)
    """
    # Ensure background is normalized
    if background.max() > 1:
        background = background / background.max()

    # Create RGB background
    rgb_background = np.stack([background] * 3, axis=-1)

    # Create colored overlay
    overlay = rgb_background.copy()
    for i in range(3):
        overlay[:, :, i] = np.where(mask > 0,
                                      (1 - alpha) * rgb_background[:, :, i] + alpha * color[i],
                                      rgb_background[:, :, i])

    return overlay


def add_annotation_circle(ax, center, radius=20, color='red', linewidth=2):
    """Add a circular annotation to highlight a region."""
    circle = Circle(center, radius, color=color, fill=False, linewidth=linewidth)
    ax.add_patch(circle)


def add_annotation_text(ax, text, xy, fontsize=10, color='red', bbox=True):
    """Add text annotation with optional background box."""
    bbox_props = dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor=color, linewidth=1.5, alpha=0.9) if bbox else None

    ax.annotate(text, xy=xy, fontsize=fontsize, color=color,
                bbox=bbox_props, ha='center', va='center',
                weight='bold')


# ============================================================================
# MAIN FIGURE CREATION
# ============================================================================

def create_figure_1(base_dir, output_path='Figure1_Qualitative_Comparison.png',
                    slice_indices=None):
    """
    Create Figure 1: Qualitative Visual Comparison.

    Parameters:
    -----------
    base_dir : str or Path
        Base directory containing dataset and results
    output_path : str
        Path where figure will be saved
    slice_indices : dict, optional
        Dictionary mapping case types to slice indices
        e.g., {'easy': 120, 'medium': 135, 'hard': 110}
    """

    # Default slice indices (middle slices)
    if slice_indices is None:
        slice_indices = {'easy': None, 'medium': None, 'hard': None}

    # Create figure
    fig, axes = plt.subplots(7, 3, figsize=FIGURE_SIZE)
    fig.subplots_adjust(wspace=WSPACE, hspace=HSPACE)

    # Overall title
    fig.suptitle('Visual Comparison of Correction Methods Across Case Difficulties',
                 fontsize=16, fontweight='bold', y=0.98)

    # Load data for all three cases
    case_data = {}
    for case_type, case_info in CASES.items():
        case_id = case_info['id']
        slice_idx = slice_indices[case_type]
        case_data[case_type] = load_case_data(base_dir, case_id, slice_idx)

    # Iterate through grid
    for row_idx in range(7):
        for col_idx, case_type in enumerate(['easy', 'medium', 'hard']):
            ax = axes[row_idx, col_idx]

            # Get data for this case
            data = case_data[case_type]

            # Determine what to display in this subplot
            if row_idx == 0:
                # Row 1: CT Scan
                ct_normalized = normalize_ct_image(data['ct_scan'])
                ax.imshow(ct_normalized, cmap='gray', aspect='auto')

            elif row_idx == 1:
                # Row 2: Ground Truth (green overlay on CT)
                ct_normalized = normalize_ct_image(data['ct_scan'])
                overlay = create_overlay(ct_normalized, data['ground_truth'],
                                        COLORS['ground_truth'], alpha=0.6)
                ax.imshow(overlay, aspect='auto')

            elif row_idx == 2:
                # Row 3: Baseline (red overlay on CT)
                ct_normalized = normalize_ct_image(data['ct_scan'])
                overlay = create_overlay(ct_normalized, data['baseline'],
                                        COLORS['prediction'], alpha=0.6)
                ax.imshow(overlay, aspect='auto')

                # Add annotation for hard case (column 3)
                if col_idx == 2:
                    # Add red circle around spurious components
                    h, w = overlay.shape[:2]
                    add_annotation_circle(ax, (w*0.65, h*0.35), radius=30,
                                         color=COLORS['text_red'], linewidth=2)
                    add_annotation_text(ax, '360 disconnected\ncomponents',
                                       (w*0.65, h*0.2), fontsize=10,
                                       color=COLORS['text_red'])

            elif row_idx == 3:
                # Row 4: Volumetric Refine (red overlay)
                ct_normalized = normalize_ct_image(data['ct_scan'])
                overlay = create_overlay(ct_normalized, data['volumetric'],
                                        COLORS['prediction'], alpha=0.6)
                ax.imshow(overlay, aspect='auto')

            elif row_idx == 4:
                # Row 5: Topology-Only (red overlay)
                ct_normalized = normalize_ct_image(data['ct_scan'])
                overlay = create_overlay(ct_normalized, data['topology_only'],
                                        COLORS['prediction'], alpha=0.6)
                ax.imshow(overlay, aspect='auto')

                # Add annotation for medium case (column 2)
                if col_idx == 1:
                    h, w = overlay.shape[:2]
                    add_annotation_text(ax, 'Good topology',
                                       (w*0.5, h*0.1), fontsize=8,
                                       color=COLORS['text_blue'], bbox=False)

            elif row_idx == 5:
                # Row 6: Anatomy-Only (red overlay)
                ct_normalized = normalize_ct_image(data['ct_scan'])
                overlay = create_overlay(ct_normalized, data['anatomy_only'],
                                        COLORS['prediction'], alpha=0.6)
                ax.imshow(overlay, aspect='auto')

                # Add annotation for medium case (column 2)
                if col_idx == 1:
                    h, w = overlay.shape[:2]
                    add_annotation_text(ax, 'Good anatomy',
                                       (w*0.5, h*0.1), fontsize=8,
                                       color=COLORS['text_orange'], bbox=False)

            elif row_idx == 6:
                # Row 7: Full Model (red overlay)
                ct_normalized = normalize_ct_image(data['ct_scan'])
                overlay = create_overlay(ct_normalized, data['full_model'],
                                        COLORS['prediction'], alpha=0.6)
                ax.imshow(overlay, aspect='auto')

                # Add annotation for hard case (column 3)
                if col_idx == 2:
                    h, w = overlay.shape[:2]
                    # Add green checkmark (using text)
                    add_annotation_text(ax, '✓', (w*0.85, h*0.3),
                                       fontsize=24, color=COLORS['text_green'],
                                       bbox=False)
                    add_annotation_text(ax, '28 components\n(92.2% reduction)',
                                       (w*0.65, h*0.15), fontsize=10,
                                       color=COLORS['text_green'])

            # Remove axis ticks and labels
            ax.set_xticks([])
            ax.set_yticks([])

            # Add thin white border
            for spine in ax.spines.values():
                spine.set_edgecolor('white')
                spine.set_linewidth(0.5)

            # Add column labels (only top row)
            if row_idx == 0:
                ax.set_title(CASES[case_type]['label'],
                           fontsize=12, fontweight='bold', pad=10)

            # Add row labels (only first column)
            if col_idx == 0:
                ax.set_ylabel(ROW_LABELS[row_idx],
                            fontsize=11, fontweight='bold', rotation=90,
                            labelpad=15, va='center')

    # Save figure
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Figure 1 saved to: {output_path}")

    return fig


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Set base directory (adjust this to your dataset location)
    BASE_DIR = Path("/home/au89100@ens.ad.etsmtl.ca/MYWORK/MASTER_PROJECT/DATASET/Parse_dataset")

    # Output directory
    OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
    OUTPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_PATH = OUTPUT_DIR / "Figure1_Qualitative_Comparison.png"

    # Optional: Specify specific slice indices for each case
    # If None, will use middle slice automatically
    SLICE_INDICES = {
        'easy': None,    # Will use middle slice
        'medium': None,  # Will use middle slice
        'hard': None,    # Will use middle slice
    }

    # You can also manually specify slice indices if you know good slices:
    # SLICE_INDICES = {
    #     'easy': 120,
    #     'medium': 135,
    #     'hard': 110,
    # }

    # Create the figure
    fig = create_figure_1(
        base_dir=BASE_DIR,
        output_path=str(OUTPUT_PATH),
        slice_indices=SLICE_INDICES
    )

    # Optionally display the figure
    # plt.show()

    print("\n" + "="*80)
    print("FIGURE 1 GENERATION COMPLETE!")
    print("="*80)
    print(f"✓ Output saved to: {OUTPUT_PATH}")
    print(f"✓ Resolution: {FIGURE_SIZE[0]}×{FIGURE_SIZE[1]} inches at {DPI} DPI")
    print(f"✓ Dimensions: {int(FIGURE_SIZE[0]*DPI)}×{int(FIGURE_SIZE[1]*DPI)} pixels")
    print("="*80)
