# Quick Start Guide - Figure Generation

## 📁 Organized Structure

```
figure_generation/
├── README.md                           # Overview and documentation
├── QUICK_START.md                      # This file
│
├── figure1_qualitative_comparison/     # ✅ READY
│   ├── create_figure1_qualitative_comparison.py
│   ├── helper_find_best_slices.py
│   └── README.md
│
├── figure2_module_complementarity/     # ⏳ Coming next
├── figure3_3d_topology/                # ⏳ Pending
├── figure4_anatomical_validation/      # ⏳ Pending
├── figure5_percase_performance/        # ⏳ Pending
│
└── outputs/                            # All figures save here
    └── (Generated PNG files)
```

## 🚀 Generate Figure 1 (3 Simple Steps)

### Step 1: Find Optimal Slices
```bash
cd /home/hous/Desktop/MASTER_PROJECT/figure_generation/figure1_qualitative_comparison
python helper_find_best_slices.py
```

**Expected output:**
- ✓ Verifies all data files exist
- ✓ Analyzes ~300 slices per case
- ✓ Creates preview images
- ✓ Recommends best slice indices

### Step 2: Review Previews
```bash
ls preview_slices_*.png
# Open these images to select your preferred slices
```

### Step 3: Generate Figure
```bash
python create_figure1_qualitative_comparison.py
```

**Output location:**
```
../outputs/Figure1_Qualitative_Comparison.png
```

## 📊 Expected Results

| Figure | Size | Resolution | File Size | Location |
|--------|------|------------|-----------|----------|
| Figure 1 | 16×12 in | 4800×3600 px | ~5-10 MB | `outputs/Figure1_Qualitative_Comparison.png` |
| Figure 2 | 14×6 in | 4200×1800 px | ~3-5 MB | `outputs/Figure2_Module_Complementarity.png` |
| Figure 3 | 16×8 in | 4800×2400 px | ~8-12 MB | `outputs/Figure3_Topology_Correction_3D.png` |
| Figure 4 | 14×10 in | 4200×3000 px | ~6-10 MB | `outputs/Figure4_Anatomical_Validation.png` |
| Figure 5 | 12×8 in | 3600×2400 px | ~4-8 MB | `outputs/Figure5_PerCase_Performance.png` |

## ⚡ One-Command Generation (After Setup)

```bash
# From figure_generation directory
cd figure1_qualitative_comparison && python create_figure1_qualitative_comparison.py && cd ..
```

## 🔍 Verify Output

```bash
# Check that figure was created
ls -lh outputs/Figure1_Qualitative_Comparison.png

# View figure dimensions
file outputs/Figure1_Qualitative_Comparison.png

# Expected: PNG image data, 4800 x 3600, 8-bit/color RGB
```

## 📝 Customization

### Change Slice Indices
Edit `create_figure1_qualitative_comparison.py` line ~410:
```python
SLICE_INDICES = {
    'easy': 125,      # Change these numbers
    'medium': 140,
    'hard': 115,
}
```

### Adjust CT Window
Edit line ~23:
```python
CT_WINDOW = (-500, 500)  # Change min/max HU values
```

### Modify Colors
Edit lines ~17-23:
```python
COLORS = {
    'ground_truth': (0, 1, 0),    # Green
    'prediction': (1, 0, 0),       # Red
    # ... customize as needed
}
```

## ✅ Quality Checklist

Before using for publication:

- [ ] All 21 subplots display correctly
- [ ] CT windowing shows good contrast
- [ ] Annotations visible and accurate
- [ ] Row/column labels correct
- [ ] 300 DPI resolution verified
- [ ] File size reasonable (<15 MB)
- [ ] Colors consistent with journal requirements
- [ ] No pixelation at full zoom

## 🐛 Troubleshooting

### Data Not Found
```bash
# Run helper first to check paths
python helper_find_best_slices.py
```

### Memory Error
```python
# In create_figure1_qualitative_comparison.py, reduce DPI temporarily:
DPI = 150  # Instead of 300
```

### Poor Slice Selection
```bash
# Review all 5 candidates in preview images
# Manually choose different slice from candidates
```

## 📧 Next Steps

After Figure 1:
1. Generate Figure 2 (Module Complementarity)
2. Generate Figure 3 (3D Topology Correction)
3. Generate Figure 4 (Anatomical Validation)
4. Generate Figure 5 (Per-Case Performance)

All figures will be saved to `outputs/` directory ready for journal submission!

---

**Last Updated**: 2025-10-19
**Status**: Figure 1 Complete ✅
