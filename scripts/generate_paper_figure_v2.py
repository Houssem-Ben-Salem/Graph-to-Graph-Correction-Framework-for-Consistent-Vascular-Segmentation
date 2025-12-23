"""
Generate HIGH-QUALITY publication figure for medical imaging journal.
Professional styling matching Nature/IEEE TMI standards.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from scipy import ndimage
import os

# Configuration
DATASET_PATH = "/home/au89100@ens.ad.etsmtl.ca/MYWORK/MASTER_PROJECT/DATASET/Parse_dataset"
OUTPUT_PATH = "/home/au89100@ens.ad.etsmtl.ca/MYWORK/MASTER_PROJECT/figures"
PATIENT_ID = "PA000005"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# Professional color palettes
COLORS = {
    'vessel_cyan': '#00B4D8',      # Medical imaging cyan
    'vessel_teal': '#0077B6',      # Deep teal
    'vessel_gold': '#FFB703',      # Gold accent
    'vessel_orange': '#FB8500',    # Orange
    'contour_yellow': '#FFE066',   # Soft yellow for contours
    'nature_blue': '#1E88E5',      # Nature-style blue
    'nature_red': '#D32F2F',       # Nature-style red
}

def load_nifti(path):
    """Load NIfTI file."""
    img = nib.load(path)
    return img.get_fdata(), img.header.get_zooms()

def normalize_ct(ct_data, window_center=100, window_width=400):
    """Apply CT windowing."""
    min_val = window_center - window_width // 2
    max_val = window_center + window_width // 2
    ct_windowed = np.clip(ct_data, min_val, max_val)
    return (ct_windowed - min_val) / (max_val - min_val)

def find_optimal_slices(segmentation):
    """Find slices with best vessel visibility."""
    coords = np.where(segmentation > 0)
    if len(coords[0]) == 0:
        return segmentation.shape[0]//2, segmentation.shape[1]//2, segmentation.shape[2]//2

    # Use center of mass
    ax_slice = int(np.median(coords[0]))
    cor_slice = int(np.median(coords[1]))
    sag_slice = int(np.median(coords[2]))

    return ax_slice, cor_slice, sag_slice

def get_contour(binary_mask):
    """Extract contour from binary mask."""
    from skimage import measure
    contours = measure.find_contours(binary_mask, 0.5)
    return contours

def add_scale_bar(ax, pixel_spacing, length_mm=20, location='lower right'):
    """Add a professional scale bar."""
    pixels = length_mm / pixel_spacing
    fontprops = fm.FontProperties(size=8, weight='bold')
    scalebar = AnchoredSizeBar(
        ax.transData,
        pixels, f'{length_mm} mm',
        location,
        pad=0.5,
        color='white',
        frameon=False,
        size_vertical=2,
        fontproperties=fontprops
    )
    ax.add_artist(scalebar)

def create_professional_figure_v1(ct_data, seg_data, spacing, output_path):
    """
    Professional figure with contour overlay (Nature/Cell style).
    """
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 9,
        'axes.linewidth': 0.5,
        'axes.labelweight': 'bold',
    })

    ct_norm = normalize_ct(ct_data, window_center=150, window_width=500)
    ax_slice, cor_slice, sag_slice = find_optimal_slices(seg_data)

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5), dpi=300,
                             facecolor='white', edgecolor='none')

    # Vessel colormap (transparent to cyan)
    vessel_cmap = LinearSegmentedColormap.from_list(
        'vessel', ['#00000000', '#00B4D8CC'], N=256
    )

    slices_ct = [
        ct_norm[ax_slice, :, :].T,
        ct_norm[:, cor_slice, :].T,
        ct_norm[:, :, sag_slice].T
    ]

    slices_seg = [
        seg_data[ax_slice, :, :].T,
        seg_data[:, cor_slice, :].T,
        seg_data[:, :, sag_slice].T
    ]

    titles = ['Axial', 'Coronal', 'Sagittal']

    # Top row: CT only
    for i, (ct_slice, title) in enumerate(zip(slices_ct, titles)):
        ax = axes[0, i]
        ax.imshow(ct_slice, cmap='gray', origin='lower', aspect='equal')
        ax.set_title(title, fontsize=10, fontweight='bold', pad=8)
        ax.axis('off')

        # Add subtle border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#CCCCCC')
            spine.set_linewidth(0.5)

    # Bottom row: CT + Segmentation with contour
    for i, (ct_slice, seg_slice) in enumerate(zip(slices_ct, slices_seg)):
        ax = axes[1, i]
        ax.imshow(ct_slice, cmap='gray', origin='lower', aspect='equal')

        # Semi-transparent fill
        seg_masked = np.ma.masked_where(seg_slice == 0, seg_slice)
        ax.imshow(seg_masked, cmap=vessel_cmap, origin='lower',
                  aspect='equal', vmin=0, vmax=1)

        # Add contour lines
        contours = get_contour(seg_slice)
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], color='#00B4D8',
                   linewidth=0.8, alpha=0.9)

        ax.axis('off')

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#CCCCCC')
            spine.set_linewidth(0.5)

    # Row labels
    fig.text(0.01, 0.73, '(a)', fontsize=12, fontweight='bold',
             va='center', ha='left')
    fig.text(0.01, 0.27, '(b)', fontsize=12, fontweight='bold',
             va='center', ha='left')

    # Add legend
    legend_patch = mpatches.Patch(color='#00B4D8', alpha=0.6,
                                   label='Pulmonary Artery')
    fig.legend(handles=[legend_patch], loc='lower center',
               ncol=1, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout()
    plt.subplots_adjust(left=0.04, right=0.98, top=0.94, bottom=0.08,
                        wspace=0.08, hspace=0.15)

    # Save
    for ext in ['png', 'pdf', 'eps']:
        output_file = os.path.join(output_path, f'figure1_professional.{ext}')
        plt.savefig(output_file, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    plt.close()

    print(f"Version 1 (Cyan contour) saved")


def create_professional_figure_v2(ct_data, seg_data, spacing, output_path):
    """
    Professional figure with colormap overlay (IEEE TMI style).
    Uses a perceptually uniform colormap for the segmentation.
    """
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 9,
    })

    ct_norm = normalize_ct(ct_data, window_center=150, window_width=500)
    ax_slice, cor_slice, sag_slice = find_optimal_slices(seg_data)

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5), dpi=300,
                             facecolor='white')

    # Gold/Orange colormap - stands out well on grayscale
    vessel_cmap = LinearSegmentedColormap.from_list(
        'vessel_gold', ['#00000000', '#FB8500BB'], N=256
    )

    slices_ct = [
        ct_norm[ax_slice, :, :].T,
        ct_norm[:, cor_slice, :].T,
        ct_norm[:, :, sag_slice].T
    ]

    slices_seg = [
        seg_data[ax_slice, :, :].T,
        seg_data[:, cor_slice, :].T,
        seg_data[:, :, sag_slice].T
    ]

    titles = ['Axial', 'Coronal', 'Sagittal']

    # Top row: CT only
    for i, (ct_slice, title) in enumerate(zip(slices_ct, titles)):
        ax = axes[0, i]
        ax.imshow(ct_slice, cmap='gray', origin='lower', aspect='equal')
        ax.set_title(title, fontsize=10, fontweight='bold', pad=8)
        ax.axis('off')

    # Bottom row: CT + Segmentation
    for i, (ct_slice, seg_slice) in enumerate(zip(slices_ct, slices_seg)):
        ax = axes[1, i]
        ax.imshow(ct_slice, cmap='gray', origin='lower', aspect='equal')

        seg_masked = np.ma.masked_where(seg_slice == 0, seg_slice)
        ax.imshow(seg_masked, cmap=vessel_cmap, origin='lower',
                  aspect='equal', vmin=0, vmax=1)

        # Contour
        contours = get_contour(seg_slice)
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], color='#FB8500',
                   linewidth=0.6, alpha=1.0)

        ax.axis('off')

    # Labels
    fig.text(0.01, 0.73, '(a)', fontsize=12, fontweight='bold', va='center')
    fig.text(0.01, 0.27, '(b)', fontsize=12, fontweight='bold', va='center')

    # Legend
    legend_patch = mpatches.Patch(color='#FB8500', alpha=0.7,
                                   label='Pulmonary Artery')
    fig.legend(handles=[legend_patch], loc='lower center',
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout()
    plt.subplots_adjust(left=0.04, right=0.98, top=0.94, bottom=0.08,
                        wspace=0.08, hspace=0.15)

    for ext in ['png', 'pdf']:
        output_file = os.path.join(output_path, f'figure1_gold.{ext}')
        plt.savefig(output_file, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    plt.close()

    print(f"Version 2 (Gold) saved")


def create_professional_figure_v3(ct_data, seg_data, spacing, output_path):
    """
    Minimalist professional style with contour-only overlay.
    Clean, high-contrast appearance for print.
    """
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
    })

    ct_norm = normalize_ct(ct_data, window_center=150, window_width=500)
    ax_slice, cor_slice, sag_slice = find_optimal_slices(seg_data)

    fig, axes = plt.subplots(2, 3, figsize=(7.5, 5.2), dpi=300,
                             facecolor='white')

    slices_ct = [
        ct_norm[ax_slice, :, :].T,
        ct_norm[:, cor_slice, :].T,
        ct_norm[:, :, sag_slice].T
    ]

    slices_seg = [
        seg_data[ax_slice, :, :].T,
        seg_data[:, cor_slice, :].T,
        seg_data[:, :, sag_slice].T
    ]

    titles = ['Axial', 'Coronal', 'Sagittal']

    # Top row: CT only
    for i, (ct_slice, title) in enumerate(zip(slices_ct, titles)):
        ax = axes[0, i]
        ax.imshow(ct_slice, cmap='gray', origin='lower', aspect='equal')
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10,
                    color='#333333')
        ax.axis('off')

    # Bottom row: CT + Yellow/Green contour only (high visibility)
    for i, (ct_slice, seg_slice) in enumerate(zip(slices_ct, slices_seg)):
        ax = axes[1, i]
        ax.imshow(ct_slice, cmap='gray', origin='lower', aspect='equal')

        # Only contour, no fill - cleaner look
        contours = get_contour(seg_slice)
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], color='#00E676',
                   linewidth=1.2, alpha=0.95)

        ax.axis('off')

    # Minimalist labels
    fig.text(0.02, 0.74, 'a', fontsize=14, fontweight='bold',
             va='center', style='italic')
    fig.text(0.02, 0.28, 'b', fontsize=14, fontweight='bold',
             va='center', style='italic')

    plt.tight_layout()
    plt.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.04,
                        wspace=0.06, hspace=0.12)

    for ext in ['png', 'pdf']:
        output_file = os.path.join(output_path, f'figure1_contour.{ext}')
        plt.savefig(output_file, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    plt.close()

    print(f"Version 3 (Green contour) saved")


def create_professional_figure_v4(ct_data, seg_data, spacing, output_path):
    """
    High-end journal style (Nature Medicine / Radiology).
    Side-by-side panels with sophisticated color scheme.
    """
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.titleweight': 'bold',
    })

    ct_norm = normalize_ct(ct_data, window_center=150, window_width=500)
    ax_slice, cor_slice, sag_slice = find_optimal_slices(seg_data)

    # Create figure with specific size for two-column journal
    fig = plt.figure(figsize=(7.2, 4.8), dpi=300, facecolor='white')

    # Custom grid: 2 rows, 3 columns
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1],
                          hspace=0.12, wspace=0.08,
                          left=0.03, right=0.97, top=0.92, bottom=0.06)

    slices_ct = [
        ct_norm[ax_slice, :, :].T,
        ct_norm[:, cor_slice, :].T,
        ct_norm[:, :, sag_slice].T
    ]

    slices_seg = [
        seg_data[ax_slice, :, :].T,
        seg_data[:, cor_slice, :].T,
        seg_data[:, :, sag_slice].T
    ]

    titles = ['Axial', 'Coronal', 'Sagittal']

    # Blue colormap for vessels (medical standard)
    vessel_cmap = LinearSegmentedColormap.from_list(
        'vessel_blue', ['#00000000', '#1E88E5AA'], N=256
    )

    # Top row: CT Volume
    for i, (ct_slice, title) in enumerate(zip(slices_ct, titles)):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(ct_slice, cmap='gray', origin='lower', aspect='equal',
                  interpolation='bilinear')
        ax.set_title(title, pad=6, fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Bottom row: Segmentation overlay
    for i, (ct_slice, seg_slice) in enumerate(zip(slices_ct, slices_seg)):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(ct_slice, cmap='gray', origin='lower', aspect='equal',
                  interpolation='bilinear')

        # Overlay with transparency
        seg_masked = np.ma.masked_where(seg_slice == 0, seg_slice)
        ax.imshow(seg_masked, cmap=vessel_cmap, origin='lower',
                  aspect='equal', vmin=0, vmax=1, interpolation='bilinear')

        # Sharp contour
        contours = get_contour(seg_slice)
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], color='#1565C0',
                   linewidth=0.8)

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Row labels (Nature style - bold lowercase)
    fig.text(0.005, 0.72, 'a', fontsize=14, fontweight='bold', va='center')
    fig.text(0.005, 0.27, 'b', fontsize=14, fontweight='bold', va='center')

    # Compact legend
    legend_patch = mpatches.Patch(facecolor='#1E88E5', alpha=0.65,
                                   edgecolor='#1565C0', linewidth=1,
                                   label='Pulmonary Artery Segmentation')
    fig.legend(handles=[legend_patch], loc='lower center',
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.005))

    for ext in ['png', 'pdf', 'eps']:
        output_file = os.path.join(output_path, f'figure1_nature.{ext}')
        plt.savefig(output_file, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none', pad_inches=0.02)
    plt.close()

    print(f"Version 4 (Nature style - Blue) saved")


def main():
    print("=" * 60)
    print("Generating Professional Publication Figures")
    print("=" * 60)

    # Load data
    ct_path = os.path.join(DATASET_PATH, PATIENT_ID, "image", f"{PATIENT_ID}.nii.gz")
    seg_path = os.path.join(DATASET_PATH, PATIENT_ID, "label", f"{PATIENT_ID}.nii.gz")

    print(f"\nLoading data for {PATIENT_ID}...")
    ct_data, ct_spacing = load_nifti(ct_path)
    seg_data, _ = load_nifti(seg_path)
    seg_data = (seg_data > 0).astype(np.float32)

    print(f"Volume shape: {ct_data.shape}")
    print(f"Voxel spacing: {ct_spacing}")

    print("\nGenerating 4 style variants...")
    print("-" * 40)

    create_professional_figure_v1(ct_data, seg_data, ct_spacing, OUTPUT_PATH)
    create_professional_figure_v2(ct_data, seg_data, ct_spacing, OUTPUT_PATH)
    create_professional_figure_v3(ct_data, seg_data, ct_spacing, OUTPUT_PATH)
    create_professional_figure_v4(ct_data, seg_data, ct_spacing, OUTPUT_PATH)

    print("\n" + "=" * 60)
    print("All figures saved to:", OUTPUT_PATH)
    print("=" * 60)
    print("\nGenerated files:")
    print("  - figure1_professional.pdf  (Cyan, filled + contour)")
    print("  - figure1_gold.pdf          (Gold/Orange overlay)")
    print("  - figure1_contour.pdf       (Green contour only)")
    print("  - figure1_nature.pdf        (Blue, Nature/Cell style)")
    print("\nRecommendation: figure1_nature.pdf for CBM submission")


if __name__ == "__main__":
    main()
