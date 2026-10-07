# Grad-CAM results (test set, seed 42)

Heatmap threshold 0.15; IoU against the union of ground-truth lesion boxes. Higher is better.

| Model | Grad-CAM IoU (mean) | IoU median | Energy inside boxes | IoU Fitzpatrick I–II | IoU III–V |
|---|---|---|---|---|---|
| A Baseline | 0.299 | 0.263 | 0.557 | 0.297 | 0.296 |
| B Fixed L*-CLAHE (β=2.0) | 0.294 | 0.264 | 0.540 | 0.286 | 0.302 |
| C Focal Loss | 0.191 | 0.103 | 0.236 | 0.168 | 0.215 |
| D Proposed (ITA L*-CLAHE + two-stage + Focal) | 0.230 | 0.192 | 0.334 | 0.208 | 0.250 |

| Metric | Friedman p | D vs A p (r) | D vs B p (r) | D vs C p (r) |
|---|---|---|---|---|
| gradcam_iou | 0.000 | 0.000 (-0.41) | 0.000 (-0.37) | 0.000 (+0.62) |
| energy_in_boxes | 0.000 | 0.000 (-0.97) | 0.000 (-0.92) | 0.000 (+0.92) |

Figure: figures/gradcam_examples_A_B_C_D.png (first single-disease test image of each class).