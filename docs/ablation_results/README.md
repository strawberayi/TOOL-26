# Ablation results (held-out test set)

Copied from `runs/ablation_yolov26/` after the overnight run (YOLO26n, seed 42,
200 test images). Calibration status: PROVISIONAL.

- `ablation_test_metrics.csv`: mAP@0.5, mAP@0.5:0.95, precision, recall per model
- `ablation_test_map50_95_by_bracket.csv`: mAP@0.5:0.95 per ITA bracket
- `*_per_seed.csv`: the same, per seed
- `figures/`: training curves and normalized confusion matrices
- `visual_proof/`: ground truth vs Models A, B, C, C′, D on 16 test photos

The executed notebooks (`notebooks/run_stage*.ipynb`, ~35 MB of training logs)
are kept locally and are not committed.
