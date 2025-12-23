# Figure Generation for Journal Publication

**High-quality figures for the Graph-to-Graph Correction Framework paper**

This directory contains all scripts, helpers, and documentation for generating publication-ready figures as specified in `../figures.md`.

## 📁 Directory Structure

```
figure_generation/
├── README.md                           # This file
│
├── figure1_qualitative_comparison/     # Figure 1: Visual comparison
│   ├── create_figure1.py              # Main generation script
│   ├── helper_find_slices.py          # Slice optimization helper
│   └── README.md                       # Detailed instructions
│
├── figure2_module_complementarity/     # Figure 2: Dual-module analysis
│   └── (To be created)
│
├── figure3_3d_topology/                # Figure 3: 3D topology visualization
│   └── (To be created)
│
├── figure4_anatomical_validation/      # Figure 4: Anatomy metrics
│   └── (To be created)
│
├── figure5_percase_performance/        # Figure 5: Per-case consistency
│   └── (To be created)
│
└── outputs/                            # Generated figures go here
    ├── Figure1_Qualitative_Comparison.png
    ├── Figure2_Module_Complementarity.png
    ├── Figure3_Topology_Correction_3D.png
    ├── Figure4_Anatomical_Validation.png
    └── Figure5_PerCase_Performance.png
```

## 🎯 Figure Overview

| Figure | Title | Status | Size | Format |
|--------|-------|--------|------|--------|
| **Figure 1** | Qualitative Visual Comparison | ✅ Ready | 16×12 in | PNG 300 DPI |
| **Figure 2** | Module Complementarity Analysis | ⏳ Pending | 14×6 in | PNG 300 DPI |
| **Figure 3** | 3D Topology Correction | ⏳ Pending | 16×8 in | PNG 300 DPI |
| **Figure 4** | Anatomical Plausibility | ⏳ Pending | 14×10 in | PNG 300 DPI |
| **Figure 5** | Per-Case Performance | ⏳ Pending | 12×8 in | PNG 300 DPI |

## 🚀 Quick Start

### Generate All Figures

```bash
cd /home/hous/Desktop/MASTER_PROJECT/figure_generation

# Figure 1: Visual comparison
cd figure1_qualitative_comparison
python helper_find_slices.py          # Find optimal slices
python create_figure1.py               # Generate figure

# Figure 2: Module complementarity
cd ../figure2_module_complementarity
python create_figure2.py               # (when ready)

# ... and so on
```

### Output Location

All generated figures are automatically saved to `outputs/` directory for easy access and journal submission.

## 📋 Prerequisites

### Python Packages Required

```bash
pip install numpy matplotlib nibabel scipy seaborn
pip install scikit-image  # For Figure 3 (3D rendering)
```

### Data Requirements

Ensure you have:
- ✅ CT scans: `DATASET/Parse_dataset/PA*/image/*.nii.gz`
- ✅ Ground truth: `DATASET/Parse_dataset/PA*/label/*.nii.gz`
- ✅ Predictions: `experiments/*/predictions/*.nii.gz`
- ✅ Metrics: Results from anatomy and topology evaluations

## 📊 Figure Specifications

All figures follow strict specifications from `../figures.md`:

- **Resolution**: 300 DPI minimum
- **Format**: PNG with transparent background where applicable
- **Color scheme**: Consistent across all figures
  - Baseline/Errors: Gray (#808080) and Red (#E63946)
  - Topology-Only: Blue (#2E86AB)
  - Anatomy-Only: Orange (#F77F00)
  - Full Model: Green (#06A77D)
  - Success: Gold (#FFD700)
- **Fonts**: Arial or Helvetica, sizes as specified
- **Layout**: Professional spacing and alignment

## 🔄 Workflow

### For Each Figure:

1. **Navigate** to the figure's directory
2. **Read** the README.md for specific instructions
3. **Run helpers** (if available) to prepare data
4. **Generate** the figure
5. **Verify** output in `outputs/` directory
6. **Review** for journal quality

### Example: Figure 1

```bash
# Step 1: Navigate
cd figure1_qualitative_comparison

# Step 2: Prepare (find best slices)
python helper_find_slices.py

# Step 3: Review previews
ls preview_slices_*.png

# Step 4: Generate
python create_figure1.py

# Step 5: Check output
ls ../outputs/Figure1_*.png
```

## ✅ Quality Checklist

Before submitting figures to journal:

- [ ] All figures generated at 300 DPI
- [ ] File sizes reasonable (<15 MB each)
- [ ] No pixelation or artifacts
- [ ] Colors consistent across figures
- [ ] All labels clearly visible
- [ ] Annotations properly positioned
- [ ] Statistical markers (p-values, asterisks) included
- [ ] Figure legends complete
- [ ] Files named according to journal requirements

## 📝 Customization

### Global Color Scheme

Edit the `COLORS` dictionary in each script:

```python
COLORS = {
    'baseline': '#808080',    # Gray
    'topology': '#2E86AB',    # Blue
    'anatomy': '#F77F00',     # Orange
    'full': '#06A77D',        # Green
    'error': '#E63946',       # Red
    'perfect': '#FFD700',     # Gold
}
```

### Font Sizes

Adjust font sizes in individual scripts to match journal requirements.

### DPI and Size

Change in script parameters:
```python
DPI = 300  # or 600 for ultra-high quality
FIGURE_SIZE = (16, 12)  # inches (width, height)
```

## 🔧 Troubleshooting

### Issue: "Module not found"
```bash
pip install -r ../requirements.txt
```

### Issue: "File not found"
Check data paths in each script and verify predictions exist.

### Issue: Memory error
- Reduce DPI temporarily: `DPI = 150`
- Process figures one at a time
- Use fewer cases for testing

### Issue: Poor image quality
- Ensure DPI is at least 300
- Check CT windowing parameters
- Verify slice selection

## 📦 Deliverables

Upon completion, the `outputs/` folder will contain:

1. ✅ `Figure1_Qualitative_Comparison.png` - Visual comparison grid
2. ⏳ `Figure2_Module_Complementarity.png` - Dual-module analysis
3. ⏳ `Figure3_Topology_Correction_3D.png` - 3D topology visualization
4. ⏳ `Figure4_Anatomical_Validation.png` - Anatomy metrics
5. ⏳ `Figure5_PerCase_Performance.png` - Per-case results

All ready for journal submission!

## 🎓 Citation

When using these figures, cite:

> [Your Name] et al. (2025). "Graph-to-Graph Correction Framework for Vascular
> Segmentation: Simultaneous Topology and Anatomy Refinement."
> [Journal Name], [Volume]([Issue]), [Pages].

## 📧 Support

For questions or issues:
- Check individual figure README files
- Review `../figures.md` for detailed specifications
- Verify data and experiment outputs are complete

---

**Last Updated**: 2025-10-19
**Version**: 1.0
**Status**: Figure 1 Complete, Figures 2-5 Pending
