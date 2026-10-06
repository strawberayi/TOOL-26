# Final ablation: manuscript Models A–D

| Model | What | Weights (`weights/final/`) |
|---|---|---|
| A | Baseline YOLO26n, raw images | `best_ModelA_raw_seed42.pt` |
| B | Fixed L*-CLAHE, β = 2.0 | `best_ModelC_fixed_l_clahe_seed42.pt` |
| C | Raw images + Focal Loss (α 0.25, γ 2.0) | `best_ModelC_focal_seed42.pt` |
| D | ITA-guided L*-CLAHE + two-stage + Focal Loss (α 0.25, γ 1.0). Stage 2: all layers trainable, lr0 0.00005 (round-2 validation search, `d_tuning_round2/validation.json`) | `best_ModelD_r2_unfreeze_seed{42,43,44}.pt` |

D's ITA → clip limit calibration: `backend/phase0_calibration_d2.json` (Darkest 5.0, Medium 3.5, Lightest 1.0).

## Results

- `sop_results_manuscript/SOP_RESULTS.md`: SOP 1–3 and H01/H02 on the 200 test images (final)
- `gradcam_results/GRADCAM_RESULTS.md`: Grad-CAM IoU, statistics and the example figure
- `d_tuning_validation.csv`, `d_tuning_choice.json`: grid search for Model D (validation set only)
- `sop_results/`, `sop_results_full_D/`: earlier comparisons (RGB CLAHE, C′, D2, D with γ 2.0)

## Notebooks and scripts

| File | What |
|---|---|
| `train_model_D_full.ipynb` (`_done` = executed) | Stage 2 of Model D (γ 2.0) |
| `tune_model_D.ipynb` (`_done` = executed) | Grid search γ / frozen layers on validation, winner seeds 43–44, final test report |
| `train_remaining.ipynb` | Model C (Focal Loss) and extra seeds for A, B |
| `sop_analysis.py` | SOP tables and statistical tests |
| `gradcam_analysis.py` | Grad-CAM (same method as the app) |
| `build_*_notebook.py` | Generators of the notebooks above |

These were run from `ablation_training/final/` (next to `TOOL-26`), which holds the datasets and all
training runs; the datasets and runs are not in this repository.
