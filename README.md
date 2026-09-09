# TOOL-26

Skin-tone masking and **ITA (Individual Typology Angle)** analysis for images, exposed both as a reusable Python module and as a small HTTP API.

Given an image, TOOL-26 isolates the most plausible skin region using dynamic K-Means clustering in CIELAB space, validates that region against configurable biological-plausibility rules, computes the ITA angle, and assigns a calibration bracket (Darkest / Medium / Lightest).

---

## Contents

| Path | Description |
| --- | --- |
| [backend/masking_ita.py](backend/masking_ita.py) | The full pipeline: colour conversion, clustering, mask validation, ITA, brackets. Importable, no server required. |
| [backend/api.py](backend/api.py) | FastAPI wrapper exposing a single `POST /analyze` upload endpoint. |
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

Files are processed in sorted filename order. Supported extensions default to `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`.

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
| `failure_reason` | Populated on failure, e.g. `NO_PLAUSIBLE_CLUSTER`, `MASK_AREA_OUT_OF_RANGE`, `LUMINANCE_OUT_OF_RANGE`, `EMPTY_MASK`. |

---

## Configuration

Every threshold is exposed through `MaskingITAConfig`. **The defaults are conservative starting points, not validated values** — the plausibility limits should be calibrated against your own dataset and documented alongside your results.

```python
from masking_ita import MaskingITAConfig, MaskingITAProcessor

config = MaskingITAConfig(
    k_values=(2, 3, 4, 5),
    random_state=42,
    n_init=10,
    max_silhouette_samples=10_000,

    # plausible-skin Lab limits
    l_min=20.0, l_max=90.0,
    a_min=-5.0, a_max=35.0,
    b_min=0.0,  b_max=50.0,

    # mask area limits (% of image)
    mask_area_min_percent=2.0,
    mask_area_max_percent=70.0,

    # center-weighted fallback crop
    center_crop_width_ratio=0.60,
    center_crop_height_ratio=0.60,

    # numerical guard for ITA
    minimum_abs_b=1e-6,

    # standardize L*, a*, b* before K-Means
    use_standardized_features=False,
)

processor = MaskingITAProcessor(config)
```

Notes:

- `random_state` and the deterministic silhouette sampling make runs reproducible.
- `use_standardized_features` changes the geometry K-Means sees — keep it constant across a calibration dataset.
- Invalid configurations (e.g. `k < 2`, inverted area limits, crop ratios outside `(0, 1]`) raise `ValueError` at construction time.

---

## License

Not yet specified.
