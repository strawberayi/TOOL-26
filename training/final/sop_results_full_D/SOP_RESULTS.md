# Statement of the Problem: results

Test set: 200 images (never used in training or calibration). Models A, B, C, D = A Baseline YOLOv26, B RGB CLAHE, C Fixed L*-CLAHE (C'), D ITA-guided L*-CLAHE + two-stage + Focal Loss (full D).
mAP, precision, recall and F1 tables are the official Ultralytics test evaluation (P and R at the confidence that maximizes F1). Per-image scores for the tests and the confusion use confidence >= 0.25 and IoU >= 0.5. Paired tests use seed 42 for every model; C and D also show the mean ± SD of seeds 42-44.

## SOP 1: localization (mAP@50, mAP@50-95) and SOP 2: classification (P, R, F1)

| Model | mAP@50 | mAP@50-95 | Precision | Recall | F1 |
|---|---|---|---|---|---|
| A Baseline YOLOv26 (seed 42) | 42.6 | 19.0 | 45.2 | 47.9 | 46.5 |
| B RGB CLAHE (seed 42) | 43.4 | 18.8 | 55.4 | 42.8 | 48.3 |
| C Fixed L*-CLAHE (C') (seed 42) | 42.2 | 20.7 | 47.8 | 43.6 | 45.6 |
| D ITA-guided L*-CLAHE + two-stage + Focal Loss (full D) (seed 42) | 41.4 | 20.8 | 51.8 | 43.0 | 47.0 |
| C Fixed L*-CLAHE (C') (3 seeds) | 41.1 ± 1.1 | 19.6 ± 1.1 | 50.6 ± 3.9 | 41.7 ± 1.8 | 45.6 ± 0.7 |
| D ITA-guided L*-CLAHE + two-stage + Focal Loss (full D) (3 seeds) | 40.3 ± 1.0 | 20.3 ± 0.6 | 45.9 ± 5.7 | 43.8 ± 0.7 | 44.7 ± 2.4 |

### Per disease: mAP@50 (seed 42)

| Disease | n | A | B | C | D | Best |
|---|---|---|---|---|---|---|
| Warts | 110 | 47.0 | 48.1 | 40.5 | 41.0 | B |
| Molluscum | 213 | 54.3 | 58.6 | 55.8 | 62.7 | D |
| Varicella | 329 | 40.3 | 32.9 | 23.7 | 31.0 | A |
| HFMD | 368 | 32.2 | 30.0 | 27.7 | 25.6 | A |
| Tinea versicolor | 233 | 14.5 | 17.2 | 12.3 | 13.1 | B |
| Tinea corporis | 42 | 59.2 | 62.1 | 66.3 | 59.8 | C |
| Tinea pedis | 8 | 61.5 | 63.2 | 74.1 | 66.7 | C |
| Impetigo | 132 | 31.7 | 35.0 | 36.8 | 31.5 | C |

### Per disease: mAP@50-95 (seed 42)

| Disease | n | A | B | C | D | Best |
|---|---|---|---|---|---|---|
| Warts | 110 | 18.4 | 21.6 | 19.0 | 19.9 | B |
| Molluscum | 213 | 23.3 | 25.6 | 23.5 | 24.9 | B |
| Varicella | 329 | 14.5 | 11.5 | 7.6 | 10.7 | A |
| HFMD | 368 | 11.2 | 9.5 | 9.2 | 9.1 | A |
| Tinea versicolor | 233 | 6.1 | 7.4 | 5.1 | 5.4 | B |
| Tinea corporis | 42 | 29.6 | 37.2 | 38.6 | 37.9 | C |
| Tinea pedis | 8 | 36.7 | 26.1 | 49.8 | 46.8 | C |
| Impetigo | 132 | 12.1 | 11.8 | 12.3 | 12.0 | C |

### Per disease: F1 (seed 42)

| Disease | n | A | B | C | D | Best |
|---|---|---|---|---|---|---|
| Warts | 110 | 50.0 | 50.4 | 43.4 | 43.6 | B |
| Molluscum | 213 | 54.4 | 59.9 | 56.1 | 60.0 | D |
| Varicella | 329 | 42.3 | 40.4 | 30.4 | 40.4 | A |
| HFMD | 368 | 40.7 | 36.6 | 33.2 | 35.4 | A |
| Tinea versicolor | 233 | 20.1 | 24.1 | 19.6 | 21.1 | B |
| Tinea corporis | 42 | 58.5 | 66.1 | 65.3 | 57.3 | B |
| Tinea pedis | 8 | 54.5 | 62.1 | 66.4 | 66.3 | C |
| Impetigo | 132 | 39.3 | 41.4 | 43.2 | 38.6 | C |

### Within-cluster misclassification rate (seed 42)

Share of detected lesions (IoU >= 0.5) assigned to a *different disease in the same cluster*. Lower is better.

| Cluster | A | B | C | D |
|---|---|---|---|---|
| Vesiculopapular/Eruptive | 7.0% (32/457) | 5.7% (26/457) | 5.3% (18/340) | 6.2% (27/435) |
| Papulosquamous/Verrucous | 0.0% (0/112) | 0.8% (1/132) | 3.7% (4/109) | 1.8% (2/113) |

## H01: difference among Models A, B, C, D

Friedman test on the 200 matched test images (per-image score), then Wilcoxon signed-rank D vs A, B, C (Bonferroni-corrected p = p × 3) with rank-biserial correlation r (positive = D better).

| Metric | Friedman χ² | p | D vs A p (r) | D vs B p (r) | D vs C p (r) | Decision |
|---|---|---|---|---|---|---|
| AP50 | 5.72 | 0.126 | 0.006 (-0.31) | 0.360 (-0.15) | 0.848 (-0.11) | fail to reject H01 |
| AP50_95 | 0.45 | 0.930 | 1.000 (+0.04) | 1.000 (+0.02) | 1.000 (-0.01) | fail to reject H01 |
| precision | 3.91 | 0.271 | 0.903 (-0.11) | 0.924 (-0.11) | 1.000 (-0.04) | fail to reject H01 |
| recall | 17.36 | 0.001 | 1.000 (+0.05) | 1.000 (+0.02) | 0.007 (+0.38) | reject H01 |
| F1 | 7.14 | 0.068 | 1.000 (-0.04) | 1.000 (-0.07) | 1.000 (+0.10) | fail to reject H01 |

## SOP 3 / H02: ΔAP50 (D − A) by skin type

Fitzpatrick proxy from ITA: I–II = ITA > 41, III–V = −30 < ITA ≤ 41; VI (ITA ≤ −30, 11 images) and images without ITA (2) excluded.

| Group | Images | mAP@50 A | mAP@50 D | ΔmAP@50 | Median per-image ΔAP50 | Wilcoxon D vs A p (r) |
|---|---|---|---|---|---|---|
| I-II | 85 | 50.7 | 46.6 | -4.1 | +0.0 | 0.006 (-0.42) |
| III-V | 102 | 37.6 | 38.3 | +0.7 | +0.0 | 0.119 (-0.22) |

Mann-Whitney U (per-image ΔAP50, III–V vs I–II): U = 5026, p = 0.057, rank-biserial r = +0.16 (positive = III–V gained more). Difference in ΔmAP@50 (III–V − I–II) = +4.8 points, bootstrap 95% CI [-2.6, +13.5]. Decision: fail to reject H02 (not significant).
