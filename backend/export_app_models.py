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
import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAINING = ROOT.parent / "ablation_training"
# Final manuscript models (seed 42) from ablation_training; ABLATION_WEIGHTS overrides the folder.
WEIGHTS = Path(os.environ.get("ABLATION_WEIGHTS", TRAINING / "weights" / "ablation_split"))
# Model D's ITA -> clip limit calibration (lesion-visibility, train images only).
D_CALIBRATION = TRAINING / "backend" / "phase0_calibration_d2.json"
OUTPUT = ROOT / "frontend" / "models"

# (id, weights, preprocessing, description): the manuscript's four-model ablation.
MODELS = [
    ("A", "best_ModelA_raw_seed42.pt", "raw", "Baseline (raw images)"),
    ("B", "best_ModelC_fixed_l_clahe_seed42.pt", "l_clahe_fixed", "Fixed L*-CLAHE, β=2.0"),
    ("C", "best_ModelC_focal_seed42.pt", "raw", "Focal Loss (raw images)"),
    ("D", "best_ModelD_g1.0_f11_seed42.pt", "l_clahe_ita", "Proposed: ITA L*-CLAHE + two-stage + Focal Loss"),
]


def add_cam_outputs(onnx_path: Path, cam_path: Path) -> None:
    """Expose what on-device Grad-CAM needs, without changing output0.

    The last layer of each class head (cv3, the one-to-many head used for evaluation) is a linear 1x1 convolution
    (64 features -> class logits). The gradient of a class logit with respect to
    those 64 features is therefore that layer's weight row, so Grad-CAM at this
    layer is exact without backpropagation: the app needs the features, the
    logits (to pick the detections to explain) and the weights.
    """
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(str(onnx_path))
    inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    nodes = {n.name: n for n in model.graph.node}
    layers = []
    for i in range(3):
        head = f"/model.23/cv3.{i}/cv3.{i}"
        feature = f"{head}.1/cv3.{i}.1.1/act/Mul_output_0"
        final = nodes[f"{head}.2/Conv"]
        weight, bias = inits[final.input[1]], inits[final.input[2]]
        for name in (feature, final.output[0]):
            model.graph.output.append(helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))
        layers.append({"feature": feature, "logits": final.output[0],
                       "weight": weight.reshape(weight.shape[0], weight.shape[1]).round(6).tolist(),
                       "bias": bias.round(6).tolist()})
    onnx.save(model, str(onnx_path))
    cam_path.write_text(json.dumps({"layers": layers}), encoding="utf-8")


def main() -> None:
    from ultralytics import YOLO

    calibration = json.loads((ROOT / "backend" / "phase0_calibration.json").read_text(encoding="utf-8"))
    d_betas = json.loads(D_CALIBRATION.read_text(encoding="utf-8"))["betas"]
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
            # Same inference as the test-set evaluation (Ultralytics val/predict):
            # - one-to-many head with NMS (nms left at None; the app runs NMS, iou 0.7),
            #   not the NMS-free one-to-one head;
            # - dynamic=True: rectangular input padded to a multiple of 32, not a 640x640 square.
            onnx_path = YOLO(str(local)).export(format="onnx", imgsz=640, opset=17, simplify=True, dynamic=True)
            shutil.copy2(onnx_path, OUTPUT / f"{model_id}.onnx")
        add_cam_outputs(OUTPUT / f"{model_id}.onnx", OUTPUT / f"{model_id}_cam.json")
        exported.append({"id": model_id, "file": f"{model_id}.onnx", "cam": f"{model_id}_cam.json",
                         "preprocessing": preprocessing, "description": description})
        print(f"exported {model_id}")

    manifest = {
        "classes": classes,
        "imgsz": 640,
        "confidence": 0.25,
        "nms_iou": 0.7,  # Ultralytics default (val and predict)
        # Grad-CAM: detections to explain (same rule as ablation_training/final/gradcam_analysis.py).
        "gradcam": {"confidence": 0.25, "max_targets": 10, "iou_threshold": 0.15},
        "fixed_beta": 2.0,
        "tile_grid_size": calibration["CLAHE"]["tile_grid_size"],
        # beta_high = Darkest, beta_mid = Medium, beta_low = Lightest; no ITA -> Medium beta.
        "calibration": {"beta_high": d_betas["Darkest"], "beta_mid": d_betas["Medium"], "beta_low": d_betas["Lightest"],
                        "beta_fallback": d_betas["Medium"], "beta_global": calibration["beta_global"],
                        "status": "D2 lesion-visibility calibration (train images)"},
        "masking": member1["masking"],
        "models": exported,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT / 'manifest.json'}")


if __name__ == "__main__":
    main()
