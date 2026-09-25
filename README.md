# TOOL-26

Skin-tone masking and **ITA (Individual Typology Angle)** analysis for images, exposed both as a reusable Python module and as a small HTTP API.

Given an image, TOOL-26 isolates the most plausible skin region using dynamic K-Means clustering in CIELAB space, validates that region against configurable biological-plausibility rules, computes the ITA angle, and assigns a calibration bracket (Darkest / Medium / Lightest).

---

## Contents

| Path | Description |
| --- | --- |
| [backend/masking_ita.py](backend/masking_ita.py) | The full pipeline: colour conversion, clustering, mask validation, ITA, brackets. Importable, no server required. |
| [backend/api.py](backend/api.py) | FastAPI wrapper exposing a single `POST /analyze` upload endpoint. |
| [backend/run_member1_batch.py](backend/run_member1_batch.py) | Archive inventory, manifest cleaning/splitting, duplicate control, and Phase 0 batch export. |
| [backend/clahe_calibration.py](backend/clahe_calibration.py) | Member 2 beta calibration: per-bracket β_high/β_mid/β_low plus the pooled `beta_global` used by the fixed-β ablation control. |
| [backend/build_yolo_dataset.py](backend/build_yolo_dataset.py) | Builds `datasets/source_yolo` (YOLO layout) from the cleaned split manifest and the AnyLabeling JSON annotations. |
| [notebooks/ablation_study_yolov26.ipynb](notebooks/ablation_study_yolov26.ipynb) | Five-model YOLOv26 ablation (A raw, B RGB-CLAHE, C fixed L\*-CLAHE, C′ fixed L\*-CLAHE at `beta_global`, D ITA-based L\*-CLAHE). |
| [docs/member1_phase0_protocol.md](docs/member1_phase0_protocol.md) | Frozen-scope Member 1 methodology and pilot protocol. |
| [frontend/](frontend/) | Web application interface (Clinical Workstation, Mobile Simulator, Landing & Auth). |
| [phone-development/](phone-development/) | Android mobile app development workspace, build scripts, and compiled APK. |

---

## How it works

1. **Colour conversion** — RGB is converted to OpenCV CIELAB and rescaled to true CIELAB ranges:

   ```
   L* = Lcv × 100 / 255
   a* = acv − 128
   b* = bcv − 128
   ```

2. **Dynamic K-Means** — the pipeline clusters Lab pixels for `k = 2, 3, 4, 5` and keeps the `k` with the highest **Silhouette Score**. Scoring is computed on a deterministic pixel sample (default 10,000) so large images stay tractable.

3. **Cluster selection** — the *largest* cluster whose mean `L*`, `a*`, `b*` and area fall inside the configured plausible-skin limits becomes the mask.

4. **Validation** — the resulting mask is re-checked for area percentage and mean luminance.

5. **Fallback** — if the primary pass fails at any stage, the image is re-processed through a center-weighted crop (default 60% × 60%) and the same steps run again.

6. **ITA** — computed from the masked mean values:

   ```
   ITA = atan((L* − 50) / b*) × 180 / π
   ```

7. **Bracket assignment**

   | Bracket | Condition |
   | --- | --- |
   | Darkest | `ITA < 28` |
   | Medium | `28 ≤ ITA ≤ 41` |
   | Lightest | `ITA > 41` |

Ordinary image-processing failures never raise — they return a structured result with `calibration_eligible = False` and a `failure_reason`, so batch runs never abort mid-dataset.

---

## Installation

```bash
pip install numpy opencv-python scikit-learn
```

For the API, additionally:

```bash
pip install fastapi uvicorn python-multipart
```

The complete reproducible dependency list is in `backend/requirements.txt`.

---

## Usage — Python module

### Single image

```python
from masking_ita import MaskingITAProcessor

processor = MaskingITAProcessor()
result = processor.process_file("image_001.jpg")

print(result.status)                # MASK_SUCCESS / MASK_FALLBACK / MASK_FAILED / IMAGE_FAILED
print(result.ita)                   # e.g. 34.7
print(result.bracket)               # "Medium"
print(result.calibration_eligible)  # True
```

### Batch directory

```python
processor = MaskingITAProcessor()
results = processor.process_directory("images")

for result in results:
    print(result.filename, result.status, result.ita, result.bracket)
```

Files are processed in sorted filename order. Supported extensions default to `.jpg`, `.jpeg`, `.png`, `.webp`, `.tif`, `.tiff`.

### Export to CSV

```python
import pandas as pd
from masking_ita import MaskingITAProcessor, results_to_rows

results = MaskingITAProcessor().process_directory("images")
pd.DataFrame(results_to_rows(results)).to_csv("masking_ita_results.csv", index=False)
```

### Command line

```bash
python backend/masking_ita.py path/to/image.jpg
```

Prints the full result as indented JSON.

---

## Usage — HTTP API

The API imports `masking_ita` as a top-level module, so start it **from inside `backend/`**:

```bash
cd backend
uvicorn api:app --reload
```

### `POST /analyze`

Multipart upload of a single image file.

```bash
curl -X POST http://127.0.0.1:8000/analyze -F "file=@test_image.jpg"
```

Response — the serialized result object:

```json
{
  "filename": "test_image.jpg",
  "status": "MASK_SUCCESS",
  "mask_method": "primary",
  "selected_k": 3,
  "silhouette_score": 0.6142,
  "mask_area_pixels": 148230,
  "mask_area_percent": 37.4,
  "mean_l": 62.8,
  "mean_a": 12.4,
  "mean_b": 18.9,
  "ita": 34.7,
  "bracket": "Medium",
  "calibration_eligible": true,
  "failure_reason": null
}
```

Interactive docs are served at `http://127.0.0.1:8000/docs`.

---

## YOLOv26 ablation workflow

The source is the team archive `Data Set-*.zip`, extracted to `phase0_work/team_dataset/`. It provides `1. Filtered` images and `2. Annotated` JSON. Its `3. Stratefied-Split` folder is **not** used, because it contains exact and near-duplicate images across train/val/test.

Run from the project root, in order:

```bash
# 0. Verified image + JSON pairs (file name and image size must match)
python backend/build_team_source_manifest.py \
  --dataset-root "phase0_work/team_dataset/Data Set" \
  --source-repository "TEAM_DATASET:Data Set-20260925T122828Z-1-001.zip" \
  --manual-exclusions docs/team_dataset_manual_exclusions.csv \
  --output phase0_work/team_inventory/team_source_manifest.csv

# 1. Quality screening, duplicate removal, leakage-safe 70/20/10 split
python backend/run_member1_batch.py prepare-split-manifest --quality-scope calibration \
  --input-manifest phase0_work/team_inventory/team_source_manifest.csv \
  --output-dir phase0_work/team_prepared

# 2. Masking + ITA on the TRAIN split
python backend/run_member1_batch.py run-phase0 \
  --manifest phase0_work/team_prepared/cleaned_split_manifest.csv \
  --output-dir phase0_outputs/member1_team_train --allow-provisional

# 3. Beta calibration; copy the result into backend/
python backend/clahe_calibration.py \
  phase0_outputs/member1_team_train/image_ita_manifest.csv \
  --output phase0_outputs/member2_calibration
cp phase0_outputs/member2_calibration/phase0_calibration.json backend/phase0_calibration.json

# 4. YOLO dataset from the same split
python backend/build_yolo_dataset.py \
  --manifest phase0_work/team_prepared/cleaned_split_manifest.csv \
  --output datasets/source_yolo
```

With `--quality-scope calibration`, every readable image is deduplicated and split, and the quality gate (blur and resolution) only decides which **train** images are used for ITA/β calibration. The protocol default (`dataset`) removes quality-rejected images from every split.

Then open `notebooks/ablation_study_yolov26.ipynb` with the **TOOL-26 (.venv)** kernel.

Beta selection (`CLAHECalibrationConfig.selection_rule`):

- `knee` (default) chooses the point of diminishing returns on the admissible contrast-gain curve. Because local contrast always rises with the clip limit, the older `max_gain` rule selected whatever β the noise ceiling allowed, which produced identical β for every bracket.
- `max_gain` is the original rule and is kept for comparison.

Annotations are matched to images by filename **and** image dimensions, because several archives reuse filenames for different photos. Classes with fewer than 30 matched training images are excluded and listed in `datasets/source_yolo/classes.json`.

---

## Mobile app: fully on-device inference

The APK runs everything on the phone, with no server or network:

- `frontend/ondevice.js` implements the Python pipeline in JavaScript: K-means skin mask + ITA, bracket β, L*-CLAHE, and YOLO26 (ONNX).
- `frontend/vendor/` holds OpenCV.js 5.0 (the same OpenCV as Python) and ONNX Runtime Web 1.30.
- `frontend/models/` holds the five ablation models plus `manifest.json` (classes, betas, masking config).
- `frontend/benchmark_data.json` holds the real test-set results shown in the app's Ablation Benchmark.

After retraining, regenerate the app assets and rebuild:

```bash
.venv/bin/python backend/export_app_models.py     # ONNX models + manifest
.venv/bin/python backend/export_app_benchmark.py  # test-set results for the app
phone-development/build-apk.sh
```

Parity with Python was checked on 40 test images (5 per class) in Chrome:
- CLAHE pixels were identical (40/40).
- YOLO detections were identical (same count and classes, boxes within 0.0002).
- ITA brackets agreed (40/40).

K-means is re-implemented in JavaScript, so ITA can differ by a few degrees on ambiguous images. `backend/api.py` (FastAPI) remains as an optional laptop server.

---

## Result fields

| Field | Meaning |
| --- | --- |
| `filename` | Source filename. |
| `status` | `MASK_SUCCESS`, `MASK_FALLBACK`, `MASK_FAILED`, or `IMAGE_FAILED`. |
| `mask_method` | `primary`, `center_weighted_crop`, or `primary_then_center_weighted_crop`. |
| `selected_k` | K chosen by Silhouette Score. |
| `silhouette_score` | Score for the selected clustering. |
| `mask_area_pixels` / `mask_area_percent` | Size of the accepted mask. |
| `mean_l` / `mean_a` / `mean_b` | Mean CIELAB values inside the mask. |
| `ita` | Individual Typology Angle, in degrees. |
| `bracket` | `Darkest`, `Medium`, or `Lightest`. |
| `calibration_eligible` | `True` only when a mask passed every validation step. |
| `failure_reason` | Populated on failure, e.g. `NO_PLAUSIBLE_CLUSTER`, `RESOLUTION_TOO_LOW`, or `ITA_UNSTABLE_B_ZERO`. |

---

## Configuration

Every threshold is exposed through `MaskingITAConfig`. **The defaults are conservative starting points, not validated values** — the plausibility limits should be calibrated against your own dataset and documented alongside your results.

```python
from masking_ita import MaskingITAConfig, MaskingITAProcessor

config = MaskingITAConfig(
    k_values=(2, 3, 4, 5),
    random_state=42,
    n_init=20,
    max_silhouette_samples=5_000,

    # plausible-skin Lab limits
    l_min=20.0, l_max=90.0,
    a_min=-5.0, a_max=35.0,
    b_min=0.0,  b_max=50.0,

    # mask area limits (% of image)
    mask_area_min_percent=5.0,
    mask_area_max_percent=85.0,

    # center-weighted fallback crop
    center_crop_width_ratio=0.60,
    center_crop_height_ratio=0.60,

    # numerical guard for ITA
    minimum_abs_b=0.001,

    # Kept for backward compatibility; the protocol requires False.
    use_standardized_features=False,
)

processor = MaskingITAProcessor(config)
```

Notes:

- `random_state` and the deterministic silhouette sampling make runs reproducible.
- The frozen protocol uses corrected, unstandardized LAB features; setting `use_standardized_features=True` is rejected.
- Invalid configurations (e.g. changing the required k set, inverted area limits, or crop ratios outside `(0, 1]`) raise `ValueError` at construction time.

---

## License

Not yet specified.
