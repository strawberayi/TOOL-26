# Member 1 Phase 0 Protocol

## Scope

Phase 0 calibrates preprocessing; it does not train YOLO, update weights, apply Focal
Loss, or calculate mAP. Only cleaned global public training images from all eight
disease classes are eligible. Validation, Test Set A, Test Set B, respondent images,
and clinical hold-out images are prohibited.

The canonical classes are Warts, Molluscum, Varicella, HFMD, Tinea versicolor,
Tinea corporis, Tinea pedis, and Impetigo. The image-based proxy group (`A` for
Fitzpatrick I–II, `B` for III–V) remains separate from the computed ITA bracket.

## Running the pipeline

The supplied archives can first be inventoried and safely extracted:

```bash
python backend/run_member1_batch.py build-archive-manifest \
  --images-zip ../SPLITTING-20260922T134834Z-1-001.zip \
  --annotations-zip ../ANNOTATED-20260922T134955Z-1-001.zip \
  --output-dir phase0_work/archive_inventory
```

The inventory records the ZIP's original split for auditing, but preprocessing recreates
the 70/20/10 split after duplicate removal. Replace the placeholder source repository
and add proxy-tone metadata where available before continuing.

If a later annotation archive also contains actual images, merge those images with:

```bash
python backend/run_member1_batch.py add-supplemental-archive \
  --archive "../2. Annotated-20260922T141609Z-1-001.zip" \
  --base-manifest phase0_work/archive_inventory/archive_source_manifest.csv \
  --output-dir phase0_work/combined_inventory
```

```bash
python backend/run_member1_batch.py prepare-manifest \
  --input-manifest phase0_work/archive_inventory/archive_source_manifest.csv \
  --output-dir phase0_work/prepared

python backend/run_member1_batch.py build-pilot-manifest \
  --manifest phase0_work/prepared/cleaned_split_manifest.csv \
  --output phase0_work/pilot/member1_pilot_160.csv

python backend/run_member1_batch.py run-phase0 \
  --manifest phase0_work/pilot/member1_pilot_160.csv \
  --output-dir phase0_outputs/member1_pilot \
  --allow-provisional
```

All thresholds come from `backend/member1_phase0_config.json`. Its initial values are
provisional and must not be described as validated biological limits.
`--allow-provisional` is only for the 160-image pilot. Final calibration refuses to run
until the configuration status is changed to `FROZEN` after approval.
After freezing, rerun `run-phase0` without `--allow-provisional` against the complete
cleaned split manifest.

## Data preparation

Inputs require `image_path`, `disease_label`, and `source_repository`. Stable image IDs
are retained when supplied and otherwise derived from repository and path. Exact
duplicates use a hash of the orientation-corrected RGB pixels. Near duplicates require
both perceptual-hash Hamming distance at most 6 and resized grayscale correlation at
least 0.95. The sharpest, then largest, then lexically highest-ID image is retained.
Conflicting disease labels in one duplicate or lineage group stop the run.

Related records use `source_item_id`/`lineage_id`. A deterministic seed-42, group-aware
split targets 70% train, 20% validation, and 10% Test A within disease and proxy-tone
strata. A lineage never crosses splits. Raw images are never changed.

## Standardization and quality

Single-frame JPEG, PNG, WebP, and TIFF images are accepted. EXIF orientation is
applied, ICC-tagged inputs are converted to sRGB, and output is three-channel RGB.
Animated/multipage files and transparent images are rejected. Standardized copies are
lossless PNG.

The provisional automatic gates are: short side at least 256 pixels, Laplacian variance
at least 50 after deterministic metric resizing, at most 30% pixels near black or white,
and luminance P95–P5 span at least 20. Manual curation must separately exclude faces,
obstruction, extreme angles, framing failures, and insufficient surrounding skin.

## CIELAB, masking, and fallback

After `cv2.cvtColor(RGB, COLOR_RGB2LAB)`, values are corrected exactly:

```text
L* = L_cv × 100 / 255
a* = a_cv − 128
b* = b_cv − 128
```

K-means tests k=2,3,4,5 on unstandardized corrected LAB. It uses seed 42, 20
initializations, and Lloyd's algorithm. A maximum 10,000-pixel deterministic sample
from a copy capped at 512 pixels selects k; Silhouette Score uses at most 5,000 pixels.
The highest score wins and a tie selects smaller k. If scoring is impossible, k=3 is
attempted and explicitly logged. The selected k is then refit using every pixel from the
original-resolution corrected LAB image. Sampling and resizing affect k selection only;
they never affect the final full-resolution mask or its ITA statistics.

A candidate cluster must pass the frozen mean LAB ranges, occupy 5–85% of the image,
contain at least 1,000 pixels, contain at least 85% individually plausible pixels, and
contain no more than 10% shadow or 10% highlight pixels. Border contact is allowed and
audited. The largest passing cluster is selected.

If the full-image attempt fails, the complete algorithm runs on the centered 60% × 60%
crop. Its mask is restored to original dimensions. A second failure becomes
`MASK_FAILED`; Phase 0 excludes the image and the app displays “No plausible skin
detected; please retake image.” Manual per-image k or mask overrides are prohibited.

## ITA and brackets

For successful masks only, mean corrected L* and b* produce:

```text
ITA = arctan((mean L* − 50) / mean b*) × 180 / π
```

An absolute mean b* below 0.001 is unstable and fails rather than using an epsilon.
Darkest is ITA below 28, Medium is 28 through 41 inclusive, and Lightest is above 41.

## Pilot and freeze

Before final calibration, two researchers independently review a deterministic
160-image public-training pilot: ten images per disease × proxy-tone cell where
available, balanced across repositories. Unavailable cells are redistributed within the
same disease and documented. Reviewers judge healthy-skin isolation and contamination,
resolve disagreements by consensus, and report agreement. Only configuration-level
thresholds may change. After adviser approval, change the configuration status to
`FROZEN`, version it, record its SHA-256, and rerun the entire training split from clean
outputs.

Each final run exports standardized images, binary masks, per-image cluster audit JSON,
the ITA manifest, failed-image log, aggregate summary, frozen configuration, manifest
hash, environment/package versions, CPU/runtime details, and timestamp.
