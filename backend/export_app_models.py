"""
Export the ablation models for fully on-device inference in the app.

Writes frontend/models/<id>.onnx (end-to-end YOLO26, no NMS needed) and
frontend/models/manifest.json with the class names, calibrated betas and the
Phase 0 masking configuration, so the JavaScript pipeline uses exactly the
same parameters as the Python pipeline.

    .venv/bin/python backend/export_app_models.py
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "weights" / "ablation_yolov26"
OUTPUT = ROOT / "frontend" / "models"

MODELS = [
    ("A", "best_ModelA_raw.pt", "raw", "Raw image"),
    ("B", "best_ModelB_rgb_clahe.pt", "rgb_clahe_fixed", "RGB CLAHE, β=2.0"),
    ("C", "best_ModelC_fixed_l_clahe.pt", "l_clahe_fixed", "L*-CLAHE, β=2.0"),
    ("C2", "best_ModelC2_fixed_l_clahe_global.pt", "l_clahe_global", "L*-CLAHE, β=beta_global (no ITA)"),
    ("D", "best_ModelD_proposed.pt", "l_clahe_ita", "K-means + ITA → L*-CLAHE"),
]


def main() -> None:
    from ultralytics import YOLO

    calibration = json.loads((ROOT / "backend" / "phase0_calibration.json").read_text(encoding="utf-8"))
    member1 = json.loads((ROOT / "backend" / "member1_phase0_config.json").read_text(encoding="utf-8"))
    classes = json.loads((ROOT / "datasets" / "source_yolo" / "classes.json").read_text())["classes"]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    exported = []
    for model_id, weights_name, preprocessing, description in MODELS:
        weights = WEIGHTS / weights_name
        if not weights.is_file():
            print(f"skip {model_id}: {weights} not found")
            continue
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / f"{model_id}.pt"
            shutil.copy2(weights, local)
            # nms=False keeps the end-to-end (one-to-one) head, matching PyTorch predict.
            onnx_path = YOLO(str(local)).export(format="onnx", imgsz=640, opset=17, simplify=True, nms=False)
            shutil.copy2(onnx_path, OUTPUT / f"{model_id}.onnx")
        exported.append({
            "id": "C′" if model_id == "C2" else model_id,
            "file": f"{model_id}.onnx",
            "preprocessing": preprocessing,
            "description": description.replace("beta_global", str(calibration["beta_global"])),
        })
        print(f"exported {model_id}")

    manifest = {
        "classes": classes,
        "imgsz": 640,
        "confidence": 0.25,
        "fixed_beta": 2.0,
        "tile_grid_size": calibration["CLAHE"]["tile_grid_size"],
        "calibration": {k: calibration[k] for k in ("beta_high", "beta_mid", "beta_low", "beta_global", "status")},
        "masking": member1["masking"],
        "models": exported,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT / 'manifest.json'}")


if __name__ == "__main__":
    main()
