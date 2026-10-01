# Plan after the full Model D training

## 0. Model list (follow the manuscript)

| Paper model | What | Trained model | Status |
|---|---|---|---|
| A | Baseline, raw images | ModelA_raw | seed 42 done; seeds 43, 44 to train |
| B | Fixed L*-CLAHE, β = 2.0 | ModelC_fixed_l_clahe (our old "C") | seed 42 done; seeds 43, 44 to train |
| C | Raw images + Focal Loss (α 0.25, γ 2.0) | ModelC_focal (new) | seeds 42, 43, 44 to train |
| D | ITA L*-CLAHE + two-stage + Focal Loss | ModelD_full (Stage 2 from D2) | training now (train_model_D_full.ipynb) |

RGB CLAHE, C′ and D2 become supplementary (appendix), not part of A–D.

## 1. Training (one notebook, headless, after D full) ~5–6 h

- `train_remaining.ipynb`: A s43/s44, B s43/s44, C-focal s42/s43/s44; same settings (300 epochs, patience 50, batch 8, 640, deterministic).
- Focal Loss uses the same verified patch as full D, applied only to Model C.

## 2. Evaluation (no training)

- `sop_analysis.py` switched to the manuscript mapping; 3 seeds for every model.
- Outputs: SOP 1–2 tables, per disease, within-cluster rate, Friedman → Wilcoxon (D vs A, B, C) + Bonferroni + rank-biserial,
  SOP 3 ΔAP50 by Fitzpatrick group (I–II vs III–V, VI excluded).
- Model D is chosen on validation. Test results are reported as they are, even if D is not best.
- Real Grad-CAM for the paper figures + Grad-CAM IoU against the lesion boxes (replaces "Not measured").
- Efficiency: latency, FPS, GFLOPs, memory for each model (including the ITA/CLAHE preprocessing time).

## 3. App changes

| # | Change | Note |
|---|---|---|
| 1 | Model buttons: A "Baseline (Raw Images)", B "Fixed L*-CLAHE (β=2.0)", C "Focal Loss Optimization", D "Proposed: ITA-CLAHE + Two-Stage + Focal Loss" | New ONNX files for B, C, D; RGB CLAHE/C′/D2 removed from the main buttons |
| 2 | Benchmark table, confusion matrices, per-disease results from the new evaluation | Numbers come only from `sop_results`, never typed by hand |
| 3 | Skin-Tone Comparison: Fitzpatrick I–II vs III–V, ΔAP50 (D − A), p-value | Show the real sign (+/−) and "not significant" if p ≥ 0.05 |
| 4 | Diagnostic & Computational Insights: "ITA x° → L*-CLAHE β = y" | Use the final D calibration (D2 β: Darkest 5.0, Medium 3.5, Lightest 1.0) |
| 5 | Patient tab: "Other possible diseases (same cluster)" under "Most likely to be" | Export per-class scores in the ONNX. YOLO class scores are independent (sigmoid): they do **not** add up to 100 %, so no "remaining %" wording |
| 6 | Role separation: Patient sees Home + Patient only; Clinician sees Workspace, Panel, XAI too | `updateAuthUI()` hides tabs by role (desktop + mobile) |
| 7 | Warning under Upload: "⚠️ Capture images directly from real skin under natural light. Photos of monitors or laptop screens can lower accuracy." | Home tab, near the Data Privacy box |
| 8 | Real Grad-CAM in XAI / Explainability Audit | Precomputed Grad-CAM examples + measured IoU; live heatmap labelled as approximation or removed |
| 9 | Rebuild APK (`phone-development/build-apk.sh`), test on phone | |

## 4. Paper

- SOP, hypotheses, scope: keep the manuscript A–D (now trained); add Stage 2 settings (freeze layers 0–10, AdamW lr 0.0002, α 0.25, γ 2.0).
- Methodology: per-image matched observations (200 test images), ITA → Fitzpatrick proxy, 3 seeds.
- Results chapter: tables from `sop_results`, Grad-CAM figures, efficiency table.
- Limitations: training on two laptops, D gets a second training stage (by design), small classes (Tinea pedis n = 8).

## Defense talking points must match the results

Say "Model D improved …" or "the models reduced confusion because of decoupled training and Focal Loss"
only if the final tables show it. If D is not best, present the honest result and the skin-tone trend.
