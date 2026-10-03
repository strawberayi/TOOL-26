# Plan: app revision (SOP alignment, traceable results, system checks)

Goal: the app answers SOP 1–3 and every number in it comes from a real computation or model output,
is labelled for what it is, and can be traced through a "How this result was calculated" section.

## 0. What the paper defines (checked in `Final Thesis Paper.docx`)
| Item | In the paper? | Use in app |
|---|---|---|
| 8 diseases | Yes | Only these 8 classes |
| Two clusters: Vesiculopapular/Eruptive (Varicella, HFMD, Molluscum, Impetigo); Papulosquamous/Verrucous (Tinea corporis, Tinea versicolor, Warts, Tinea pedis) | Yes ("analytical groupings, not official medical categories") | Group the 4 candidate diseases; label as analysis grouping |
| Viral (Molluscum, Varicella, HFMD, Warts) / Fungal (3 tinea) / Bacterial (Impetigo) | Yes (scope section) | Allowed as a label |
| Models A–D, mAP50, mAP50-95, P, R, F1, within-cluster rate, ΔAP50 I–II vs III–V, Friedman/Wilcoxon/Bonferroni/rank-biserial, Mann-Whitney | Yes | Benchmark view |
| Fitzpatrick type | Only as an "ITA-derived proxy", type VI excluded | Never show "Fitzpatrick Type IV" as a fact; show "ITA group (proxy for Fitzpatrick I–II / III–V)" |
| Phase 0 (ITA calibration), Stage 1 (localization learning), Stage 2 (fine-grained, frozen backbone + Focal Loss) | Yes, as **training** steps | Only Phase 0 runs per photo; Stage 1/2 are described as how the model was trained, not as live status |
| Grad-CAM | Yes, as visual evidence of where the model looks ("not a diagnostic algorithm") | Research/XAI view only, never as accuracy |
| Confidence cutoff 0.25, NMS IoU 0.7 | **Not in the paper** (Ultralytics defaults) | Must be stated in the methodology, or chosen on validation and documented |
| Inference time / GFLOPs | Not an SOP; only in the literature review | Measure and report as an extra system check |
| ITA bracket cutoffs and β per bracket | In the code/calibration file; check the paper text | Add to the methodology if missing |

## 1. SOP → screen map
| SOP | Question | Screen / function |
|---|---|---|
| SOP 1 | Localization of A–D: mAP@50, mAP@50–95 | Benchmark view: table A–D + per-disease AP50 + computation |
| SOP 2 | Classification of A–D: P, R, F1, within-cluster misclassification | Benchmark view: table + confusion matrices + within-cluster computation |
| SOP 3 | ΔAP50 (D − A) I–II vs III–V, Mann-Whitney | Benchmark view: ITA-group table, U, p, r; Technical details shows the ITA group of the current photo |
| H01 | Friedman → Wilcoxon (D vs A, B, C), Bonferroni, rank-biserial | Benchmark view: p-value table |
| Demo of the framework | One photo through Model D (Phase 0 → detection → candidates) | Image Workspace + Analysis Results |
| Ablation on one photo | Same photo through A–D | Image Workspace: compare control (models actually tested) |

## 2. Audit of the current app (to fix)
| Where | Shown | Problem → action |
|---|---|---|
| `PRESETS` in app.js | 94.2%, P 92.4%, R 89.1%, F1 0.907, 12.4 ms, 16.5 GFLOPs, attribution 73.1%, IoU 78.2% | Hard-coded → remove; presets run the real pipeline on bundled sample photos |
| Patient: "Disease split within the cluster" | 4 diseases scaled to 100% | Forced 100% → show the raw sigmoid class scores of the 4 candidates |
| Current run: "Differential Diagnosis Probabilities", "Class Overlap Profiler" | share of summed confidences (sums to 100%) | Not probabilities → remove (duplicate of candidate scores) |
| Current run: "Contrast Gain" | shows β | Wrong label → remove or replace with a measured L* contrast change |
| Current run: Stage 1 "FROZEN", Stage 2 "ACTIVE HEAD" | static badges | Training steps, not runtime → move to the methodology text |
| Current run: P/R/F1/latency/GFLOPs/attribution | "—" or fake | Test-set metrics belong only to the Benchmark view |
| Patient: ITA chip "Fitzpatrick Type IV \| ITA 22.4° • Clip 3.4" | ITA + Fitzpatrick | Remove from patient view; ITA ≠ Fitzpatrick |
| Architecture: `clamp(4.4 − 0.045 × ITA, 2.0, 4.0)` | old formula | Replace with the bracket rule (Darkest 5.0, Medium 3.5, Lightest 1.0) |
| "Typical Lesion Features (reference)", "How to tell apart", "Differential key" | text per disease | Not computed → hold (remove) per instruction |
| "How the AI got this result" / "why the AI chose this" | explanation | Keep only parts backed by model output (scores, z, σ) |
| "Diagnostic & Computational Insights" | 2 lines | Rename to "Technical details" |
| Benchmark charts on several pages (desktop, poster, panel) | duplicates | One full Benchmark view; a short summary on the main page |

## 3. Views (kept separate)
1. **Image Workspace:** original, enhanced (L*-CLAHE) image, boxes, labels, confidence scores; compare
   control for A–D only.
2. **Analysis Results (patient-facing):** top prediction, the 4 candidate diseases of its cluster with raw
   scores and the margin to the 2nd, plain-language disclaimer. No ITA, no β, no Grad-CAM.
   Expandable "How this result was calculated":
   - box confidence = σ(z) = 1/(1+e^−z), z shown
   - candidate scores = the 8 sigmoid outputs at the main box (independent, not summing to 100%), 4 shown
   - decision = highest summed box confidence; ranking explained only by these numbers
3. **Lesion feature extraction (item 10, like the frog-app reference):** per detected box, a crop + its
   confidence, sorted. Below each, only measured values with formulas: lesion vs surrounding-skin CIELAB
   (L*, a*, b*), ΔE*ab, box size (px, % of photo). Roughness kept only if its formula is in the methodology,
   else held.
4. **Technical details (clinician/research):** Phase 0 for this photo (skin mask pixels, mean L*, b*,
   ITA formula and value, ITA group, β from the calibration file), detection settings (cutoff, NMS IoU,
   input size), measured runtime per stage.
5. **XAI / research view:** Grad-CAM per box and combined, legend "where the model focused, not proof of
   correctness"; test-set Grad-CAM table (IoU, energy in boxes, reviewed examples).
6. **Benchmark view (only place with test-set metrics):** SOP 1–3 tables, per-disease AP50, confusion
   matrices, ITA-group results, p-values, and **per-model computations (item 2)**: TP, FP, FN →
   P = TP/(TP+FP), R = TP/(TP+FN), F1 = 2PR/(P+R); mAP50 = mean per-class AP; within-cluster rate =
   within-cluster errors / detected. Same 200 test images for all models. Runtime table (device, input size).
   Needs `sop_analysis.py` to save TP/FP/FN and per-class AP; `export_app_benchmark.py` to export them.

7. **"Analyzing on device" modal dialog (UI/UX + animation):** replaces the current plain text overlay
   (`setAnalyzingOverlay` in app.js) and the `alert()` on errors.
   - Centered modal over a dimmed background, with the photo thumbnail and an animated scan line over it.
   - Step list driven by **real progress events** from `ondevice.js` (a `onStep` callback), not a fake
     timer or fake % bar: Loading model → Skin mask & ITA (Phase 0) → L*-CLAHE (β) → Detecting lesions
     (YOLO26) → Grad-CAM → Measuring lesion features. Each step: pending → active (pulse) → done (check)
     with its **measured time in ms**, which also feeds the runtime check (section 4.2).
   - Compare mode shows the same list per model (A, B, C, D).
   - First run note: "Loading the model (only the first time)".
   - Errors and "No lesion detected" shown inside the modal with a clear next step (retake photo, better
     light), instead of `alert()`.
   - Cancel button (ignores the result if pressed), focus trapped in the dialog, `aria-live` step updates,
     animations off when `prefers-reduced-motion` is set, no emojis.
   - The page must actually repaint between steps: yield to the browser after each step (or run inference
     in the Web Worker from 4.2), otherwise the animation freezes while the model runs.

## 4. System checks
1. **Run consistency (item 4):** analyse the same file 10× with identical settings → outputs must be
   identical (ITA, β, boxes, scores). Pipeline is seeded; verify and fix any difference. Separately explain
   that a new camera capture is a different photo, and that ITA near a bracket edge switches β; show
   "borderline ITA group" when within ±3° of an edge.
2. **Runtime (item 5):** measure per stage (load, ITA K-means, CLAHE, YOLO, Grad-CAM) on the emulator and a
   real phone; report mean ± SD with device, image size (max side 1280, 640 input). Speed-ups that keep
   results identical: warm-up at start, Web Worker, run A–C only when Compare is opened.
3. **No lesion detected (item 3):** do not lower the cutoff silently. Options: (a) state the 0.25 cutoff in
   the methodology; (b) choose per-model cutoffs on validation and rerun the SOP; (c) show the best box
   below the cutoff as "low-confidence, not counted". Add photo-quality checks (blur, exposure).
4. **Grad-CAM review (item 3):** keep the paper's method (one-to-many head, `gradcam_analysis.py`). Review the
   8 gallery examples per model; keep only clear ones in the XAI view. Any method change is rerun in
   Python and reported.

## 5. Answers to the questions
- **ITA in patient view (6):** not needed there. ITA matters for SOP 3 and for choosing β, so show it in
  Technical details only.
- **"Diagnostic and Computational Insights" (7):** rename to "Technical details". Phase 0 (ITA → β → CLAHE)
  and localization (detection boxes) are real per-photo steps and stay; Stage 1/2 are training steps and
  move to the methodology text.
- **Current run vs SOP benchmark (8):** different. Current run = one photo; SOP benchmark = 200 test images
  (the paper's results). Current run becomes Technical details; all metrics live only in Benchmark.

## 6. Hold / remove now
- Hard-coded or sample percentages
- Duplicate benchmark charts or scores across pages
- Feature text without computation (typical features, differential key)
- Grad-CAM presented as accuracy
- Undefined thresholds or stages; Fitzpatrick type shown as fact

## 7. Order of work
1. Remove/relabel (sections 2, 6) → 2. Views restructure (3) → 3. Benchmark computations (sop_analysis
   export) → 4. Feature extraction → 5. Analyzing modal + step timing (3.7) → 6. System checks (4) → 7. Parity tests (Python vs app on test photos
   and your photos), grep for any untraced `%`, rebuild APK, emulator test, commit and push.

## 8. Decisions needed
1. Confidence cutoff: (a) keep 0.25 and add it to the methodology, (b) choose per-model cutoffs on
   validation (reruns SOP tables), or (c) also show below-cutoff candidates labelled "not counted".
2. Roughness feature: keep (add formula to methodology) or hold.
3. Keep viral/fungal/bacterial label in Analysis Results? (defined in the paper's scope)

## Status (2026-10-04)
Done in `frontend/app.html`, `app.js`, `app.css`, `ondevice.js`, `backend/export_app_benchmark.py`:
- One app for phone and laptop with five views: Home, Workspace, Results (patients and clinicians),
  Benchmark and Research (clinicians). Removed: Phone Simulator, 9-Screen Poster, the old desktop
  workstation, the triage quiz and every hard-coded preset number (all still in git history).
- Results: top prediction = label of the highest-confidence box; the 4 candidate diseases of its cluster with
  their raw sigmoid scores from the same box (not scaled to 100%), margin, best score outside the cluster;
  below-cutoff candidate shown as "not counted"; expandable "How this result was calculated" with z, σ(z),
  all 8 scores, cutoff/NMS and data sources. No ITA, Grad-CAM or reference text in the patient view.
- Lesion feature extraction: per box crop + confidence + measured CIELAB ΔE, Δa*, Δb*, ΔL*, texture ratio,
  box size, each with its formula. Verbal labels that needed invented thresholds were removed.
- Benchmark: SOP map, SOP 1–3 tables, per-disease values, computations (mAP = mean AP, P/R macro means,
  F1 = 2PR/(P+R), within-cluster counts), confusion matrix, ITA groups + Mann-Whitney, H01 table,
  inference time (RTX 3050: `training/final/runtime_benchmark.py`) and on-device measurement.
- Research: Phase 0 and localization details of the current photo (ITA formula, bracket, distance to edge,
  β, SOP 3 group as ITA proxy), all box scores, per-step timing, run-consistency check (5 repeats),
  Grad-CAM of the photo and of A–D, test-set Grad-CAM table labelled "not an accuracy measure".
- "Analyzing on device" modal driven by real step events with measured times; also used for Compare,
  consistency and runtime checks.
- Checks: 5/5 repeated runs identical; app scores match Python predict (≤ 0.0003); laptop Chrome
  (WASM, 1 thread) ≈ 1.4 s per photo.

Not done (by decision):
- Photo-quality gating (blur/exposure) and a "borderline ITA" flag: they need thresholds the methodology
  does not define. The distance to the nearest ITA bracket edge is shown instead.
- Web Worker: the dialog yields between steps so it animates; the model step itself still blocks ~0.2–1 s.
- Paper text: add the 0.25 cutoff / NMS 0.7 (Ultralytics defaults), the top-box decision rule, runtime
  results and the Grad-CAM wording.
