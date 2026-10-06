# Statement of the Problem: results

Test set: 200 images (never used in training or calibration). Models A, B, C, D = A Baseline YOLOv26, B Fixed L*-CLAHE (β = 2.0), C Focal Loss Optimization, D ITA-guided L*-CLAHE + two-stage + Focal Loss (proposed).
mAP, precision, recall and F1 tables are the official Ultralytics test evaluation (P and R at the confidence that maximizes F1). Per-image scores for the tests and the confusion use confidence >= 0.25 and IoU >= 0.5. Paired tests use seed 42 for every model; models with more finished seeds also show mean ± SD.

## SOP 1: localization (mAP@50, mAP@50-95) and SOP 2: classification (P, R, F1)

| Model | mAP@50 | mAP@50-95 | Precision | Recall | F1 |
|---|---|---|---|---|---|
| A Baseline YOLOv26 (seed 42) | 42.6 | 19.0 | 45.2 | 47.9 | 46.5 |
| B Fixed L*-CLAHE (β = 2.0) (seed 42) | 39.5 | 20.3 | 50.2 | 41.6 | 45.5 |
| C Focal Loss Optimization (seed 42) | 36.8 | 17.3 | 43.0 | 39.0 | 40.9 |
| D ITA-guided L*-CLAHE + two-stage + Focal Loss (proposed) (seed 42) | 42.5 | 19.9 | 51.5 | 43.8 | 47.3 |
| D ITA-guided L*-CLAHE + two-stage + Focal Loss (proposed) (3 seeds) | 41.2 ± 1.2 | 19.9 ± 0.5 | 50.1 ± 1.2 | 43.3 ± 0.8 | 46.5 ± 0.8 |

### Per disease: mAP@50 (seed 42)

| Disease | n | A | B | C | D | Best |
|---|---|---|---|---|---|---|
| Warts | 110 | 47.0 | 48.5 | 46.3 | 46.0 | B |
| Molluscum | 213 | 54.3 | 50.4 | 57.4 | 58.4 | D |
| Varicella | 329 | 40.3 | 23.6 | 23.0 | 32.9 | A |
| HFMD | 368 | 32.2 | 27.9 | 24.0 | 25.2 | A |
| Tinea versicolor | 233 | 14.5 | 11.5 | 15.1 | 15.4 | D |
| Tinea corporis | 42 | 59.2 | 63.1 | 43.8 | 60.5 | B |
| Tinea pedis | 8 | 61.5 | 62.5 | 60.0 | 64.6 | D |
| Impetigo | 132 | 31.7 | 28.1 | 24.7 | 36.8 | D |

### Per disease: mAP@50-95 (seed 42)

| Disease | n | A | B | C | D | Best |
|---|---|---|---|---|---|---|
| Warts | 110 | 18.4 | 24.2 | 19.8 | 20.5 | B |
| Molluscum | 213 | 23.3 | 22.8 | 24.3 | 23.5 | C |
| Varicella | 329 | 14.5 | 8.3 | 8.6 | 11.4 | A |
| HFMD | 368 | 11.2 | 9.3 | 8.2 | 8.8 | A |
| Tinea versicolor | 233 | 6.1 | 5.0 | 5.8 | 6.2 | D |
| Tinea corporis | 42 | 29.6 | 37.6 | 29.6 | 34.4 | B |
| Tinea pedis | 8 | 36.7 | 45.0 | 32.3 | 42.5 | B |
| Impetigo | 132 | 12.1 | 10.3 | 10.1 | 12.2 | D |

### Per disease: F1 (seed 42)

| Disease | n | A | B | C | D | Best |
|---|---|---|---|---|---|---|
| Warts | 110 | 50.0 | 55.1 | 47.5 | 48.7 | B |
| Molluscum | 213 | 54.4 | 55.0 | 51.0 | 60.7 | D |
| Varicella | 329 | 42.3 | 33.3 | 31.4 | 37.6 | A |
| HFMD | 368 | 40.7 | 35.9 | 32.6 | 31.9 | A |
| Tinea versicolor | 233 | 20.1 | 17.4 | 24.2 | 21.6 | C |
| Tinea corporis | 42 | 58.5 | 60.2 | 44.0 | 61.4 | D |
| Tinea pedis | 8 | 54.5 | 63.4 | 53.1 | 63.8 | D |
| Impetigo | 132 | 39.3 | 36.7 | 33.0 | 43.4 | D |

### Within-cluster misclassification rate (seed 42)

Share of detected lesions (IoU >= 0.5) assigned to a *different disease in the same cluster*. Lower is better.

| Cluster | A | B | C | D |
|---|---|---|---|---|
| Vesiculopapular/Eruptive | 7.0% (32/457) | 10.3% (37/360) | 15.2% (54/356) | 4.7% (16/338) |
| Papulosquamous/Verrucous | 0.0% (0/112) | 2.9% (3/105) | 2.6% (3/116) | 1.8% (2/111) |

## H01: difference among Models A, B, C, D

Friedman test on the 200 matched test images (per-image score), then Wilcoxon signed-rank D vs A, B, C (Bonferroni-corrected p = p × 3) with rank-biserial correlation r (positive = D better).

| Metric | Friedman χ² | p | D vs A p (r) | D vs B p (r) | D vs C p (r) | Decision |
|---|---|---|---|---|---|---|
| AP50 | 5.77 | 0.124 | 0.449 (-0.14) | 0.685 (+0.12) | 1.000 (+0.07) | fail to reject H01 |
| AP50_95 | 0.85 | 0.838 | 1.000 (-0.00) | 1.000 (-0.01) | 1.000 (-0.01) | fail to reject H01 |
| precision | 14.72 | 0.002 | 0.782 (+0.12) | 1.000 (+0.06) | 0.005 (+0.33) | reject H01 |
| recall | 15.93 | 0.001 | 1.000 (-0.10) | 0.987 (+0.12) | 0.007 (+0.36) | reject H01 |
| F1 | 16.78 | 0.001 | 1.000 (-0.00) | 1.000 (+0.09) | 0.000 (+0.39) | reject H01 |

## SOP 3 / H02: ΔAP50 (D − A) by skin type

Fitzpatrick proxy from ITA: I–II = ITA > 41, III–V = −30 < ITA ≤ 41; VI (ITA ≤ −30, 11 images) and images without ITA (2) excluded.

| Group | Images | mAP@50 A | mAP@50 D | ΔmAP@50 | Median per-image ΔAP50 | Wilcoxon D vs A p (r) |
|---|---|---|---|---|---|---|
| I-II | 85 | 50.7 | 48.9 | -1.8 | +0.0 | 0.325 (-0.15) |
| III-V | 102 | 37.6 | 38.3 | +0.7 | +0.0 | 0.420 (-0.11) |

Mann-Whitney U (per-image ΔAP50, III–V vs I–II): U = 4565, p = 0.526, rank-biserial r = +0.05 (positive = III–V gained more). Difference in ΔmAP@50 (III–V − I–II) = +2.5 points, bootstrap 95% CI [-5.1, +12.3]. Decision: fail to reject H02 (not significant).
