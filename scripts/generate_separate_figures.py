"""
Generate 6 separate clean figures for publication.
Professional cyan style - no labels, titles, or annotations.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from skimage import measure
import os

# Configuration
DATASET_PATH = "/home/au89100@ens.ad.etsmtl.ca/MYWORK/MASTER_PROJECT/DATASET/Parse_dataset"
OUTPUT_PATH = "/home/au89100@ens.ad.etsmtl.ca/MYWORK/MASTER_PROJECT/figures/paper_figures"
PATIENT_ID = "PA000005"

os.makedirs(OUTPUT_PATH, exist_ok=True)

def load_nifti(path):
    """Load NIfTI file."""
    img = nib.load(path)
    return img.get_fdata(), img.header.get_zooms()

def normalize_ct(ct_data, window_center=150, window_width=500):
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

    ax_slice = int(np.median(coords[0]))
    cor_slice = int(np.median(coords[1]))
    sag_slice = int(np.median(coords[2]))

    return ax_slice, cor_slice, sag_slice

def get_contour(binary_mask):
    """Extract contour from binary mask."""
    contours = measure.find_contours(binary_mask, 0.5)
    return contours

def save_clean_figure(data, filename, cmap='gray', figsize=(6, 6)):
    """Save a clean figure without any axes, labels, or padding."""
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=300)
    ax.imshow(data, cmap=cmap, origin='lower', aspect='equal', interpolation='bilinear')
    ax.axis('off')

    # Remove all margins
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.margins(0, 0)

    # Save with tight bbox and no padding
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0,
                facecolor='white', edgecolor='none')
    plt.close()

def save_segmentation_figure(ct_slice, seg_slice, filename, figsize=(6, 6)):
    """Save CT with segmentation overlay (cyan professional style)."""
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=300)

    # CT background
    ax.imshow(ct_slice, cmap='gray', origin='lower', aspect='equal', interpolation='bilinear')

    # Cyan colormap for segmentation
    vessel_cmap = LinearSegmentedColormap.from_list(
        'vessel', ['#00000000', '#00B4D8CC'], N=256
    )

    # Overlay segmentation
    seg_masked = np.ma.masked_where(seg_slice == 0, seg_slice)
    ax.imshow(seg_masked, cmap=vessel_cmap, origin='lower',
              aspect='equal', vmin=0, vmax=1, interpolation='bilinear')

    # Add contour lines
    contours = get_contour(seg_slice)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], color='#00B4D8',
               linewidth=0.8, alpha=0.9)

    ax.axis('off')

    # Remove all margins
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.margins(0, 0)

    # Save with tight bbox and no padding
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0,
                facecolor='white', edgecolor='none')
    plt.close()

def main():
    print("=" * 60)
    print("Generating 6 Separate Clean Figures")
    print("=" * 60)

    # Load data
    ct_path = os.path.join(DATASET_PATH, PATIENT_ID, "image", f"{PATIENT_ID}.nii.gz")
    seg_path = os.path.join(DATASET_PATH, PATIENT_ID, "label", f"{PATIENT_ID}.nii.gz")

    print(f"\nLoading data for {PATIENT_ID}...")
    ct_data, _ = load_nifti(ct_path)
    seg_data, _ = load_nifti(seg_path)
    seg_data = (seg_data > 0).astype(np.float32)

    # Normalize CT
    ct_norm = normalize_ct(ct_data)

    # Find best slices
    ax_slice, cor_slice, sag_slice = find_optimal_slices(seg_data)
    print(f"Using slices - Axial: {ax_slice}, Coronal: {cor_slice}, Sagittal: {sag_slice}")

    # Prepare slices
    slices = {
        'axial': (ct_norm[ax_slice, :, :].T, seg_data[ax_slice, :, :].T),
        'coronal': (ct_norm[:, cor_slice, :].T, seg_data[:, cor_slice, :].T),
        'sagittal': (ct_norm[:, :, sag_slice].T, seg_data[:, :, sag_slice].T)
    }

    print("\nGenerating figures...")
    print("-" * 40)

    # Save CT-only figures (Row a)
    for view_name, (ct_slice, seg_slice) in slices.items():
        # CT only
        ct_filename = os.path.join(OUTPUT_PATH, f'ct_{view_name}.png')
        save_clean_figure(ct_slice, ct_filename)
        print(f"  Saved: ct_{view_name}.png")

        # Also save as PDF
        ct_pdf = os.path.join(OUTPUT_PATH, f'ct_{view_name}.pdf')
        save_clean_figure(ct_slice, ct_pdf)

    # Save Segmentation overlay figures (Row b)
    for view_name, (ct_slice, seg_slice) in slices.items():
        seg_filename = os.path.join(OUTPUT_PATH, f'segmentation_{view_name}.png')
        save_segmentation_figure(ct_slice, seg_slice, seg_filename)
        print(f"  Saved: segmentation_{view_name}.png")

        # Also save as PDF
        seg_pdf = os.path.join(OUTPUT_PATH, f'segmentation_{view_name}.pdf')
        save_segmentation_figure(ct_slice, seg_slice, seg_pdf)

    print("\n" + "=" * 60)
    print("All figures saved to:", OUTPUT_PATH)
    print("=" * 60)
    print("\nGenerated files:")
    print("  CT Volume (Row a):")
    print("    - ct_axial.png / .pdf")
    print("    - ct_coronal.png / .pdf")
    print("    - ct_sagittal.png / .pdf")
    print("  Segmentation (Row b):")
    print("    - segmentation_axial.png / .pdf")
    print("    - segmentation_coronal.png / .pdf")
    print("    - segmentation_sagittal.png / .pdf")

if __name__ == "__main__":
    main()
