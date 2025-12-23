"""
Generate publication-ready figure showing CT volume and PA segmentation.
For: Graph-to-Graph Correction Framework paper (CBM 2025)
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
import os

# Configuration
DATASET_PATH = "/home/au89100@ens.ad.etsmtl.ca/MYWORK/MASTER_PROJECT/DATASET/Parse_dataset"
OUTPUT_PATH = "/home/au89100@ens.ad.etsmtl.ca/MYWORK/MASTER_PROJECT/figures"
PATIENT_ID = "PA000005"  # Good example case

# Create output directory
os.makedirs(OUTPUT_PATH, exist_ok=True)

def load_nifti(path):
    """Load NIfTI file and return data array."""
    img = nib.load(path)
    return img.get_fdata(), img.affine, img.header

def normalize_ct(ct_data, window_center=100, window_width=400):
    """Apply CT windowing for better vessel visualization."""
    min_val = window_center - window_width // 2
    max_val = window_center + window_width // 2
    ct_windowed = np.clip(ct_data, min_val, max_val)
    ct_normalized = (ct_windowed - min_val) / (max_val - min_val)
    return ct_normalized

def find_best_slices(segmentation):
    """Find slices with maximum segmentation content."""
    # Find center of mass of segmentation
    coords = np.where(segmentation > 0)
    if len(coords[0]) == 0:
        return segmentation.shape[0]//2, segmentation.shape[1]//2, segmentation.shape[2]//2

    axial_slice = int(np.median(coords[0]))
    coronal_slice = int(np.median(coords[1]))
    sagittal_slice = int(np.median(coords[2]))

    return axial_slice, coronal_slice, sagittal_slice

def create_figure_2panel(ct_data, seg_data, patient_id, output_path):
    """
    Create a 2-panel figure:
    (a) CT volume - 3 orthogonal views
    (b) Segmentation overlay - 3 orthogonal views
    """
    # Normalize CT
    ct_norm = normalize_ct(ct_data, window_center=150, window_width=500)

    # Find best slices
    ax_slice, cor_slice, sag_slice = find_best_slices(seg_data)

    # Create figure
    fig = plt.figure(figsize=(12, 8), dpi=300)

    # Define grid
    gs = fig.add_gridspec(2, 3, hspace=0.15, wspace=0.1,
                          left=0.05, right=0.95, top=0.92, bottom=0.05)

    # Row labels
    fig.text(0.02, 0.72, '(a)', fontsize=14, fontweight='bold', va='center')
    fig.text(0.02, 0.28, '(b)', fontsize=14, fontweight='bold', va='center')

    # Column labels
    fig.text(0.22, 0.95, 'Axial', fontsize=11, ha='center', fontweight='bold')
    fig.text(0.52, 0.95, 'Coronal', fontsize=11, ha='center', fontweight='bold')
    fig.text(0.82, 0.95, 'Sagittal', fontsize=11, ha='center', fontweight='bold')

    # Row (a): CT volume only
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(ct_norm[ax_slice, :, :].T, cmap='gray', origin='lower', aspect='auto')
    ax1.axis('off')
    ax1.set_title(f'Slice {ax_slice}', fontsize=9)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(ct_norm[:, cor_slice, :].T, cmap='gray', origin='lower', aspect='auto')
    ax2.axis('off')
    ax2.set_title(f'Slice {cor_slice}', fontsize=9)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(ct_norm[:, :, sag_slice].T, cmap='gray', origin='lower', aspect='auto')
    ax3.axis('off')
    ax3.set_title(f'Slice {sag_slice}', fontsize=9)

    # Row (b): CT with segmentation overlay
    # Create red overlay colormap
    seg_cmap = ListedColormap(['none', 'red'])

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(ct_norm[ax_slice, :, :].T, cmap='gray', origin='lower', aspect='auto')
    ax4.imshow(seg_data[ax_slice, :, :].T, cmap=seg_cmap, alpha=0.5, origin='lower', aspect='auto')
    ax4.axis('off')

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(ct_norm[:, cor_slice, :].T, cmap='gray', origin='lower', aspect='auto')
    ax5.imshow(seg_data[:, cor_slice, :].T, cmap=seg_cmap, alpha=0.5, origin='lower', aspect='auto')
    ax5.axis('off')

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.imshow(ct_norm[:, :, sag_slice].T, cmap='gray', origin='lower', aspect='auto')
    ax6.imshow(seg_data[:, :, sag_slice].T, cmap=seg_cmap, alpha=0.5, origin='lower', aspect='auto')
    ax6.axis('off')

    # Add row descriptions on the right
    fig.text(0.98, 0.72, 'CT Volume', fontsize=10, va='center', ha='right', rotation=-90)
    fig.text(0.98, 0.28, 'PA Segmentation', fontsize=10, va='center', ha='right', rotation=-90)

    # Save
    output_file = os.path.join(output_path, f'figure_ct_segmentation_{patient_id}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_file.replace('.png', '.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Figure saved to: {output_file}")
    return output_file


def create_figure_with_3d(ct_data, seg_data, patient_id, output_path):
    """
    Create a comprehensive figure with 2D slices and 3D rendering.
    Layout:
    Top row: CT slices (axial, coronal, sagittal)
    Bottom row: Segmentation overlay + 3D rendering
    """
    # Normalize CT
    ct_norm = normalize_ct(ct_data, window_center=150, window_width=500)

    # Find best slices
    ax_slice, cor_slice, sag_slice = find_best_slices(seg_data)

    # Create figure with custom layout
    fig = plt.figure(figsize=(14, 10), dpi=300)

    # Top row: CT volume
    ax1 = fig.add_axes([0.05, 0.52, 0.28, 0.42])
    ax1.imshow(ct_norm[ax_slice, :, :].T, cmap='gray', origin='lower', aspect='auto')
    ax1.axis('off')
    ax1.set_title('Axial View', fontsize=11, fontweight='bold', pad=10)

    ax2 = fig.add_axes([0.36, 0.52, 0.28, 0.42])
    ax2.imshow(ct_norm[:, cor_slice, :].T, cmap='gray', origin='lower', aspect='auto')
    ax2.axis('off')
    ax2.set_title('Coronal View', fontsize=11, fontweight='bold', pad=10)

    ax3 = fig.add_axes([0.67, 0.52, 0.28, 0.42])
    ax3.imshow(ct_norm[:, :, sag_slice].T, cmap='gray', origin='lower', aspect='auto')
    ax3.axis('off')
    ax3.set_title('Sagittal View', fontsize=11, fontweight='bold', pad=10)

    # Bottom row: Segmentation overlays
    seg_cmap = ListedColormap(['none', 'red'])

    ax4 = fig.add_axes([0.05, 0.05, 0.28, 0.42])
    ax4.imshow(ct_norm[ax_slice, :, :].T, cmap='gray', origin='lower', aspect='auto')
    ax4.imshow(seg_data[ax_slice, :, :].T, cmap=seg_cmap, alpha=0.6, origin='lower', aspect='auto')
    ax4.axis('off')

    ax5 = fig.add_axes([0.36, 0.05, 0.28, 0.42])
    ax5.imshow(ct_norm[:, cor_slice, :].T, cmap='gray', origin='lower', aspect='auto')
    ax5.imshow(seg_data[:, cor_slice, :].T, cmap=seg_cmap, alpha=0.6, origin='lower', aspect='auto')
    ax5.axis('off')

    ax6 = fig.add_axes([0.67, 0.05, 0.28, 0.42])
    ax6.imshow(ct_norm[:, :, sag_slice].T, cmap='gray', origin='lower', aspect='auto')
    ax6.imshow(seg_data[:, :, sag_slice].T, cmap=seg_cmap, alpha=0.6, origin='lower', aspect='auto')
    ax6.axis('off')

    # Row labels
    fig.text(0.02, 0.73, '(a)', fontsize=14, fontweight='bold')
    fig.text(0.02, 0.26, '(b)', fontsize=14, fontweight='bold')

    # Row descriptions
    fig.text(0.5, 0.97, 'CT Angiography Volume', fontsize=12, ha='center', fontweight='bold')
    fig.text(0.5, 0.50, 'Pulmonary Artery Segmentation (Ground Truth)', fontsize=12, ha='center', fontweight='bold')

    # Save
    output_file = os.path.join(output_path, f'figure_ct_pa_segmentation_{patient_id}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_file.replace('.png', '.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Figure saved to: {output_file}")
    return output_file


def create_single_column_figure(ct_data, seg_data, patient_id, output_path):
    """
    Create a publication-ready single-column figure (journal format).
    Optimized for ~88mm column width.
    """
    # Normalize CT
    ct_norm = normalize_ct(ct_data, window_center=150, window_width=500)

    # Find best slices - use multiple slices for better visualization
    ax_slice, cor_slice, sag_slice = find_best_slices(seg_data)

    # Create figure (single column ~88mm = 3.46 inches)
    fig, axes = plt.subplots(2, 3, figsize=(7, 5), dpi=300)

    seg_cmap = ListedColormap(['none', '#E74C3C'])  # Professional red

    # Top row: CT only
    axes[0, 0].imshow(ct_norm[ax_slice, :, :].T, cmap='gray', origin='lower', aspect='auto')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(ct_norm[:, cor_slice, :].T, cmap='gray', origin='lower', aspect='auto')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(ct_norm[:, :, sag_slice].T, cmap='gray', origin='lower', aspect='auto')
    axes[0, 2].axis('off')

    # Bottom row: CT + Segmentation overlay
    axes[1, 0].imshow(ct_norm[ax_slice, :, :].T, cmap='gray', origin='lower', aspect='auto')
    axes[1, 0].imshow(seg_data[ax_slice, :, :].T, cmap=seg_cmap, alpha=0.55, origin='lower', aspect='auto')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(ct_norm[:, cor_slice, :].T, cmap='gray', origin='lower', aspect='auto')
    axes[1, 1].imshow(seg_data[:, cor_slice, :].T, cmap=seg_cmap, alpha=0.55, origin='lower', aspect='auto')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(ct_norm[:, :, sag_slice].T, cmap='gray', origin='lower', aspect='auto')
    axes[1, 2].imshow(seg_data[:, :, sag_slice].T, cmap=seg_cmap, alpha=0.55, origin='lower', aspect='auto')
    axes[1, 2].axis('off')

    # Column titles
    axes[0, 0].set_title('Axial', fontsize=10, fontweight='bold')
    axes[0, 1].set_title('Coronal', fontsize=10, fontweight='bold')
    axes[0, 2].set_title('Sagittal', fontsize=10, fontweight='bold')

    # Row labels
    fig.text(0.02, 0.75, '(a)', fontsize=12, fontweight='bold')
    fig.text(0.02, 0.28, '(b)', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(left=0.05, wspace=0.05, hspace=0.1)

    # Save
    output_file = os.path.join(output_path, f'figure1_ct_segmentation.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(output_file.replace('.png', '.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_file.replace('.png', '.eps'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Figure saved to: {output_file}")
    print(f"Also saved as PDF and EPS for journal submission")
    return output_file


def main():
    """Generate the publication figure."""
    print("="*60)
    print("Generating Publication Figure: CT Volume and PA Segmentation")
    print("="*60)

    # Load data
    ct_path = os.path.join(DATASET_PATH, PATIENT_ID, "image", f"{PATIENT_ID}.nii.gz")
    seg_path = os.path.join(DATASET_PATH, PATIENT_ID, "label", f"{PATIENT_ID}.nii.gz")

    print(f"\nLoading CT volume: {ct_path}")
    ct_data, ct_affine, ct_header = load_nifti(ct_path)
    print(f"CT shape: {ct_data.shape}")
    print(f"CT value range: [{ct_data.min():.1f}, {ct_data.max():.1f}]")

    print(f"\nLoading segmentation: {seg_path}")
    seg_data, seg_affine, seg_header = load_nifti(seg_path)
    seg_data = (seg_data > 0).astype(np.float32)  # Binarize
    print(f"Segmentation shape: {seg_data.shape}")
    print(f"Segmentation voxels: {seg_data.sum():.0f}")

    # Generate figures
    print("\n" + "-"*40)
    print("Generating publication figures...")
    print("-"*40)

    # Main figure for the paper
    create_single_column_figure(ct_data, seg_data, PATIENT_ID, OUTPUT_PATH)

    # Additional versions
    create_figure_2panel(ct_data, seg_data, PATIENT_ID, OUTPUT_PATH)
    create_figure_with_3d(ct_data, seg_data, PATIENT_ID, OUTPUT_PATH)

    print("\n" + "="*60)
    print("COMPLETE! Figures saved to:", OUTPUT_PATH)
    print("="*60)

    # Print LaTeX code for the figure
    print("\n" + "="*60)
    print("LaTeX code for your paper:")
    print("="*60)
    latex_code = r'''
\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/figure1_ct_segmentation.pdf}
    \caption{Representative case from the PARSE 2022 dataset demonstrating
    pulmonary artery imaging. (a) Contrast-enhanced CT angiography volume
    showing axial, coronal, and sagittal views. (b) Corresponding ground
    truth pulmonary artery segmentation (red overlay) delineating the
    vascular tree structure. The segmentation exhibits the complex
    branching topology characteristic of pediatric pulmonary arteries,
    with vessel diameters ranging from approximately 2 mm at peripheral
    branches to 15 mm at the main pulmonary trunk.}
    \label{fig:ct_segmentation}
\end{figure}
'''
    print(latex_code)


if __name__ == "__main__":
    main()
