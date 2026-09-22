# Member 1 AI Handoff

Updated: 2026-09-22 (Asia/Manila)

## Project and role

Project directory: `/home/eney/Documents/TOOL2026/TOOL-26`

The user is Member 1, responsible for Phase 0 data preparation, healthy-skin masking,
and ITA computation. Phase 0 is offline preprocessing calibration only. It must not train
YOLOv26, apply Focal Loss, calculate mAP, update model weights, or use validation,
Test Set A, Test Set B, respondent images, or clinical hold-out images for calibration.

All eight required disease classes are:

1. Warts
2. Molluscum
3. Varicella
4. HFMD
5. Tinea versicolor
6. Tinea corporis
7. Tinea pedis
8. Impetigo

## Authoritative technical requirements

- Every image needs a manifest row containing image path/ID, disease label, source
  repository, split, image-quality status, optional proxy skin-tone group A (I–II) or B
  (III–V), and duplicate-removal status.
- Raw images must remain unchanged. Processed images and masks are separate copies.
- Accepted formats are single-frame JPEG, PNG, WebP, and TIFF.
- Apply EXIF orientation, ICC-to-sRGB conversion when available, and explicit RGB/BGR
  handling.
- Provisional minimum short side is 256 pixels. Quality checks also cover blur,
  exposure clipping, low contrast, invalid formats, transparency, and unreadable data.
- OpenCV LAB must be corrected exactly:

  ```text
  L* = L_cv × 100 / 255
  a* = a_cv − 128
  b* = b_cv − 128
  ```

- Test K-means with `k = 2, 3, 4, 5` using corrected, unstandardized LAB features.
- Use Silhouette Score to choose k; tie goes to the smaller k.
- Use fixed `random_state=42` and `n_init=20`.
- Sampling/resizing is used only to choose k. The selected k is refit using every
  original-resolution pixel for the final mask.
- Choose the largest cluster passing the provisional skin-plausibility rules.
- If primary masking fails, rerun on the centered 60% × 60% crop and restore the mask
  to original-image dimensions.
- If both attempts fail, return `MASK_FAILED`, exclude the image, and never calculate
  ITA. In the app, show: `No plausible skin detected; please retake image.`
- Successful-mask ITA formula:

  ```text
  ITA = arctan((mean L* − 50) / mean b*) × 180 / π
  ```

- Reject ITA as unstable if `abs(mean b*) < 0.001` instead of substituting epsilon.
- Brackets:
  - Darkest: ITA < 28
  - Medium: 28 ≤ ITA ≤ 41
  - Lightest: ITA > 41
- ITA brackets and manual proxy Fitzpatrick groups must remain separate fields.
- No manual per-image k/mask override is allowed. Phase 0 failures are logged and
  excluded.

## Implementation completed

Important files:

- `backend/masking_ita.py`
  - EXIF/ICC-aware image loading
  - Image-quality checks before K-means
  - Corrected CIELAB conversion
  - Dynamic k/Silhouette selection
  - Original-resolution final K-means and mask
  - Cluster plausibility audit
  - Center-crop fallback
  - ITA and bracket calculation
  - Backward-compatible result fields
- `backend/run_member1_batch.py`
  - Safe ZIP extraction
  - Archive and supplemental-archive inventory
  - Standardized PNG generation
  - Exact and near-duplicate detection
  - Lineage-group-aware 70/20/10 split
  - Deterministic 160-image pilot builder
  - Phase 0 masks, audit JSON, CSV, metadata, and summaries
- `backend/member1_phase0_config.json`
  - Provisional quality/mask settings
  - Status intentionally remains `PROVISIONAL_UNTIL_160_IMAGE_PILOT_APPROVAL`
- `backend/api.py`
  - Preserves `POST /analyze`
  - Adds EXIF-aware loading, quality handling, and retake messages
- `docs/member1_phase0_protocol.md`
  - Complete methodology and operating instructions
- `tests/test_member1.py`
  - Eleven passing tests
- `backend/requirements.txt`
- `README.md`

## Source archives used

Workspace archives:

- `/home/eney/Documents/TOOL2026/SPLITTING-20260922T134834Z-1-001.zip`
- `/home/eney/Documents/TOOL2026/ANNOTATED-20260922T134955Z-1-001.zip`
- `/home/eney/Documents/TOOL2026/2. Annotated-20260922T141609Z-1-001.zip`

Additional archives found in Downloads:

- `/home/eney/Downloads/3. Impetigo-20260917T140925Z-1-001.zip`
  - Contains 636 Impetigo images.
- `/home/eney/Downloads/3. Impetigo-20260917T140901Z-1-001.zip`
  - Contains JSON only; not used as an image source.
- `/home/eney/Downloads/Warts-20260907T075157Z-1-001.zip`
  - Contains 509 Warts images.
- `/home/eney/Downloads/FILTERED-20260907T064914Z-1-001.zip`
  - Supplied additional Tinea pedis images and overlapping Warts records.

No Kaggle dataset was downloaded or used.

Source repository is currently recorded at traceable local-archive level, such as
`LOCAL_ARCHIVE:<zip filename>`. The research team must replace or supplement this with
the true original repository (DermNet, Kaggle dataset name, Roboflow project, etc.) when
known before final calibration/publication.

## Dataset preparation results

Final source inventory:

- 3,886 total records
- All eight disease classes present

Quality and duplicate results:

- 2,694 passed automatic image-quality screening
- 1,192 rejected by quality checks
- 707 exact duplicates excluded
- 129 near-duplicates excluded
- 1,858 independent, quality-passing images retained

Leakage-safe split:

- Training: 1,292
- Validation: 376
- Test Set A: 190
- Excluded (quality/duplicate): 2,028

Training rows by class:

- HFMD: 85
- Impetigo: 214
- Molluscum: 152
- Tinea corporis: 103
- Tinea pedis: 138
- Tinea versicolor: 147
- Varicella: 127
- Warts: 326

Key manifest:

`phase0_work/prepared/cleaned_split_manifest.csv`

## Pilot results

Pilot manifest:

`phase0_work/pilot/member1_pilot_160.csv`

The pilot has 160 images: 20 per disease.

Important limitation: every `proxy_skin_tone_group` is currently blank/UNKNOWN. The
pilot is disease-balanced but not yet demonstrably balanced between proxy groups A and
B. Researchers must assign those labels independently using the study protocol.

Provisional pilot output directory:

`phase0_outputs/member1_pilot/`

Main output:

`phase0_outputs/member1_pilot/image_ita_manifest.csv`

Pilot results:

- Total: 160
- Calibration eligible: 145
- Primary-mask success: 132
- Center-crop fallback success: 13
- `MASK_FAILED`: 15
- Darkest: 69
- Medium: 23
- Lightest: 53
- No bracket because masking failed: 15
- Selected k counts:
  - k=2: 127
  - k=3: 23
  - k=4: 7
  - k=5: 3
- All 15 failures were:
  `PRIMARY_FAILED:NO_PLAUSIBLE_CLUSTER;FALLBACK_FAILED:NO_PLAUSIBLE_CLUSTER`

Other pilot outputs:

- `phase0_outputs/member1_pilot/masks/` — 145 accepted binary masks
- `phase0_outputs/member1_pilot/audits/` — 160 per-image audit JSON files
- `phase0_outputs/member1_pilot/failed_images.csv`
- `phase0_outputs/member1_pilot/summary.json`
- `phase0_outputs/member1_pilot/run_metadata.json`
- `phase0_outputs/member1_pilot/member1_config_used.json`

## Tests and verification

Run from the project root:

```bash
cd /home/eney/Documents/TOOL2026/TOOL-26
python -m unittest discover -s tests -v
python -m py_compile backend/*.py tests/*.py
git diff --check
```

Current result: all 11 tests pass. A real Ringworm image was also processed
successfully with a complete mask/ITA audit.

## Required next work

Do not immediately freeze the configuration or run final full-dataset calibration.
Complete these actions first:

1. Two researchers independently review all 160 pilot masks.
2. Fill these existing reviewer columns in the pilot/ITA manifest:
   - `reviewer_1_mask_acceptable`
   - `reviewer_2_mask_acceptable`
   - `reviewer_1_contamination`
   - `reviewer_2_contamination`
   - `consensus_mask_acceptable`
   - `consensus_notes`
3. Assign image-based proxy skin-tone group A or B where defensible. Use two
   independent reviewers with consensus/tie-breaking; do not derive proxy group from
   ITA bracket.
4. Confirm original repository provenance and licensing for each archive.
5. Resolve Warts and Impetigo annotation/image matching before YOLO training. This does
   not invalidate Phase 0 ITA processing, but it matters for later detection training.
6. Review the 15 failed masks and accepted masks for background/shadow errors. Do not
   manually override individual masks; use evidence only to tune global configuration.
7. If threshold changes are approved, update `backend/member1_phase0_config.json`,
   increment its version, and rerun the same pilot from clean outputs.
8. After adviser/research-team approval, change configuration status to `FROZEN`.
9. Run the complete cleaned training manifest without `--allow-provisional`:

   ```bash
   python backend/run_member1_batch.py run-phase0 \
     --manifest phase0_work/prepared/cleaned_split_manifest.csv \
     --output-dir phase0_outputs/member1_final
   ```

10. Give Member 2 the final `image_ita_manifest.csv`, masks, frozen configuration, and
    runtime metadata. The output already includes Member 2-compatible fields:
    `image_id`, `image_path`, `mask_path`, `ITA`, `bracket`, and
    `calibration_eligible`.

## Important safeguards for the next AI

- Do not use validation/test/clinical images for Phase 0.
- Do not use the 15 `MASK_FAILED` images for ITA or CLAHE calibration.
- Do not infer missing proxy tone from ITA.
- Do not silently loosen thresholds to increase pass rate.
- Do not overwrite or edit raw source images.
- Do not use frontend demo images as research data.
- Do not describe the current configuration as frozen or biologically validated.
- Generated `phase0_work/` and `phase0_outputs/` are ignored by Git but exist locally.
- The Git worktree also contains unrelated pre-existing Android/CLAHE changes. Do not
  revert or overwrite them.

## Short prompt for another AI

Use this if a short continuation prompt is needed:

> Continue the Member 1 Phase 0 work in
> `/home/eney/Documents/TOOL2026/TOOL-26`. Read `MEMBER1_AI_HANDOFF.md` and
> `docs/member1_phase0_protocol.md` first. The implementation and 160-image
> provisional pilot are complete. Do not rerun preparation unnecessarily. Help conduct
> or organize the two-reviewer mask QC, proxy A/B labeling, provenance verification,
> threshold review, configuration freeze, and final training-only Phase 0 run. Preserve
> all raw images and unrelated worktree changes.
