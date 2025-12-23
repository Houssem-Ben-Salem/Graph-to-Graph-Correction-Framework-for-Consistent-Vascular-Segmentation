# Figure 1: Qualitative Visual Comparison

**High-quality figure for journal publication**

## Overview

This directory contains scripts to generate Figure 1 as specified in `figures.md`:
- **7 rows × 3 columns** grid showing all methods across three difficulty levels
- **Publication quality**: 16×12 inches at 300 DPI (4800×3600 pixels)
- **Three representative cases**: Easy (PA000016), Medium (PA000026), Hard (PA000005)

## Files

1. **`create_figure1_qualitative_comparison.py`** - Main script to generate Figure 1
2. **`helper_find_best_slices.py`** - Helper to find optimal slice indices
3. **`README_FIGURE1.md`** - This file

## Quick Start

### Step 1: Verify Data and Find Optimal Slices

```bash
cd /home/hous/Desktop/MASTER_PROJECT
python helper_find_best_slices.py
```

This script will:
- ✓ Verify all required data files exist
- ✓ Analyze all slices to find ones with maximum vessel visibility
- ✓ Generate preview images (`preview_slices_PA*.png`)
- ✓ Recommend optimal slice indices

**Output:**
```
SLICE_INDICES = {
    'easy': 120,
    'medium': 135,
    'hard': 110,
}
```

### Step 2: Review Previews

Open the generated preview images:
- `preview_slices_PA000016.png` - Easy case candidates
- `preview_slices_PA000026.png` - Medium case candidates
- `preview_slices_PA000005.png` - Hard case candidates

Each preview shows:
- **Top row**: CT scan slices
- **Bottom row**: Vessel masks with metrics
- **Rankings**: Best slices ranked by vessel content and complexity

### Step 3: Update Slice Indices (Optional)

Edit `create_figure1_qualitative_comparison.py` around line 360:

```python
SLICE_INDICES = {
    'easy': 120,      # Your chosen slice for PA000016
    'medium': 135,    # Your chosen slice for PA000026
    'hard': 110,      # Your chosen slice for PA000005
}
```

Or leave as `None` to auto-select middle slices.

### Step 4: Generate Figure 1

```bash
python create_figure1_qualitative_comparison.py
```

**Output:**
- `Figure1_Qualitative_Comparison.png` (4800×3600 pixels, ~5-10 MB)

## Data Requirements

The script expects the following directory structure:

```
MASTER_PROJECT/
├── DATASET/
│   └── Parse_dataset/
│       ├── PA000016/              # Easy case
│       │   ├── image/
│       │   │   └── PA000016.nii.gz
│       │   └── label/
│       │       └── PA000016.nii.gz
│       ├── PA000026/              # Medium case
│       └── PA000005/              # Hard case
│
└── experiments/
    ├── unet/
    │   └── predictions/
    │       ├── PA000016_pred.nii.gz
    │       ├── PA000026_pred.nii.gz
    │       └── PA000005_pred.nii.gz
    │
    ├── ablation/
    │   ├── exp1_volumetric/
    │   │   └── predictions/
    │   │       └── PA*_refined.nii.gz
    │   ├── exp2_topology_only/
    │   │   └── predictions/
    │   │       └── PA*_corrected.nii.gz
    │   └── exp3_anatomy_only/
    │       └── predictions/
    │           └── PA*_corrected.nii.gz
    │
    └── graph_correction/
        └── predictions/
            └── PA*_corrected.nii.gz
```

## Missing Data?

If some prediction files are missing, you need to:

1. **Generate predictions** using the respective models:
   ```bash
   # Example for EXP-2 (Topology-Only)
   python scripts/run_exp2_inference.py
   ```

2. **Or use placeholder data**: The script will generate random data if files are missing (for testing layout only)

## Customization

### Adjust CT Window

Edit line 23 in `create_figure1_qualitative_comparison.py`:
```python
CT_WINDOW = (-500, 500)  # Adjust min/max Hounsfield Units
```

### Change Colors

Edit lines 17-23:
```python
COLORS = {
    'ground_truth': (0, 1, 0),      # Green for GT
    'prediction': (1, 0, 0),         # Red for predictions
    # ... other colors
}
```

### Modify Annotations

The script includes annotations as specified:
- **Row 3, Column 3**: Red circle + "360 disconnected components" text
- **Row 7, Column 3**: Green checkmark + "28 components (92.2% reduction)" text
- **Row 5, Column 2**: "Good topology" in blue
- **Row 6, Column 2**: "Good anatomy" in orange

To modify, edit the respective sections in `create_figure_1()` function.

## Troubleshooting

### Issue: "File not found" errors
**Solution**: Run `helper_find_best_slices.py` to verify all paths. Adjust `BASE_DIR` if needed.

### Issue: Poor slice selection
**Solution**: Manually review preview images and choose different slices from candidates.

### Issue: Annotations not visible
**Solution**: Adjust annotation positions in the code (change `w*0.65` and `h*0.35` multipliers).

### Issue: Memory error
**Solution**: Process one case at a time or reduce image resolution temporarily.

## Expected Output

**Figure 1 Specifications:**
- **Size**: 16 inches × 12 inches
- **Resolution**: 300 DPI (4800 × 3600 pixels)
- **Format**: PNG
- **File size**: ~5-10 MB (depending on compression)
- **Quality**: Publication-ready for high-impact journals

**Grid Layout:**
- 7 rows (CT, GT, Baseline, Volumetric, Topo-Only, Anat-Only, Full Model)
- 3 columns (Easy, Medium, Hard cases)
- 21 total subplots with minimal spacing

## Verification Checklist

Before submitting to journal:

- [ ] All three cases display correctly
- [ ] CT windowing shows good vessel contrast
- [ ] Annotations are clearly visible and accurate
- [ ] Row and column labels are correct
- [ ] Image quality is sharp at 300 DPI
- [ ] Colors match specification (green for GT, red for predictions)
- [ ] File size is reasonable (<15 MB)
- [ ] No truncation or clipping of labels

## Performance

**Typical execution time:**
- Helper script: ~30 seconds
- Figure generation: ~1-2 minutes

**Memory usage:**
- ~2-4 GB RAM (depending on image sizes)

## Next Steps

After Figure 1 is complete:
1. ✓ Review visual quality
2. → Proceed to Figure 2 (Module Complementarity)
3. → Proceed to Figure 3 (3D Topology Correction)
4. → Proceed to Figure 4 (Anatomical Validation)
5. → Proceed to Figure 5 (Per-Case Performance)

---

**Questions or Issues?**
- Check `figures.md` for detailed specifications
- Verify data paths with helper script
- Ensure all experiments have generated predictions
