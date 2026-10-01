# Grad-CAM results (test set, seed 42)

Heatmap threshold 0.15; IoU against the union of ground-truth lesion boxes. Higher is better.

| Model | Grad-CAM IoU (mean) | IoU median | Energy inside boxes | IoU Fitzpatrick I–II | IoU III–V |
|---|---|---|---|---|---|
| A Baseline | 0.170 | 0.153 | 0.572 | 0.183 | 0.161 |
| B Fixed L*-CLAHE (β=2.0) | 0.163 | 0.147 | 0.564 | 0.174 | 0.161 |
| C Focal Loss | 0.197 | 0.102 | 0.238 | 0.173 | 0.220 |
| D Proposed (ITA L*-CLAHE + two-stage + Focal) | 0.197 | 0.148 | 0.350 | 0.189 | 0.209 |

| Metric | Friedman p | D vs A p (r) | D vs B p (r) | D vs C p (r) |
|---|---|---|---|---|
| gradcam_iou | 0.000 | 0.805 (+0.09) | 0.085 (+0.18) | 0.003 (+0.27) |
| energy_in_boxes | 0.000 | 0.000 (-0.90) | 0.000 (-0.88) | 0.000 (+0.90) |

Figure: figures/gradcam_examples_A_B_C_D.png (first single-disease test image of each class).