# Plan: changes from the voice feedback (2026-10-06)

| # | Feedback | Change |
|---|---|---|
| 1 | The heatmap was removed | Bring back the Grad-CAM overlay in the Workspace (toggle + opacity slider), labelled "where the model focused, not accuracy". |
| 2 | Results change when the same photo is tried again | Re-test the same photo several times on the phone build; keep the "Repeat 5 times" check in Research. Show in Results that the same photo gives the same result. |
| 3 | Grad-CAM shows heat but the app says "No lesion detected" | Make Grad-CAM follow the result: with boxes, the heatmap explains those boxes; with no box, show the strongest candidate as a dashed box labelled "below 25%, not counted" with its heatmap, so the picture and the text agree. |
| 4 | Computations aligned with the 3 SOPs | Keep the Benchmark computations (SOP 1–3); simplify the wording (item 9). |
| 5 | Clean the code, remove the comments | Remove comments from the app code (frontend), backend and training scripts; check that everything still runs. |
| 6 | Why the model focuses on the lesion (Gemini note) | Add a short plain-language explanation to the Grad-CAM section. |
| 7 | Morphological features instead of lesion-feature computations | Workspace: replace the box list ("1. Warts 66% …") with the morphological features (surface texture, crust and exudate, border and distribution, how to tell apart). Results: remove the measured values and formulas from lesion feature extraction (keep only each lesion crop and its confidence). |
| 8 | "80%, the others 0.5%: why not 100%?" | Add "Why don't these add up to 100%?" under the candidate scores, in simple words. |
| 9 | Panel dashboard terms too technical | Plain-language titles and one-line meanings for every metric (mAP, precision, recall, F1, within-cluster, p-value, ITA group); keep formulas inside "Computation" expanders. |

Order: 7 → 1 → 3 → 8 → 9 → 6 → 5 → test (browser and emulator, same photo repeated) → APK → push.

## Status (2026-10-06)
1. Grad-CAM overlay back in the Workspace (toggle, opacity), labelled as focus, not accuracy. Done.
2. Same photo analysed repeatedly: identical (5/5 in the browser check; 67.6% twice on the emulator). Done.
3. No-lesion case: dashed "Not counted" box + heatmap + text explaining the warm areas are weak signs below the cutoff. Done.
4. SOP computations kept in Benchmark. Done.
5. Comments removed from 45 source files (frontend, backend, training, scripts); Python checked by comparing syntax trees, JS by syntax check and the app tests. Done.
6. "Why does the heatmap focus on the lesion?" added to Research. Done.
7. Workspace: box list replaced by the morphological features; Results: measured values and formulas removed, only lesion crops with confidence kept. Done.
8. "Why don't these add up to 100%?" added under the candidate scores. Done.
9. Benchmark in plain language (titles, row meanings, Yes/No for significance, glossary; formulas in expanders). Done.
