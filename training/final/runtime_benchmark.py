"""Inference time and GFLOPs of Models A-D on the 200 test images (each model's own preprocessed images).

    ../TOOL-26/.venv/bin/python final/runtime_benchmark.py

Batch 1, imgsz 640 (rectangular letterbox, as in the evaluation), confidence 0.25, NMS IoU 0.7.
The first 10 images are a warm-up and are not timed. Writes runtime_results/runtime.json and .csv.
"""

import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "datasets" / "ablation_yolov26"
WEIGHTS = ROOT / "weights" / "ablation_split"
OUT = HERE / "runtime_results"
D_WINNER = json.loads((HERE / "d_tuning_choice.json").read_text())["winner"]
MODELS = {
    "A": ("best_ModelA_raw_seed42.pt", "ModelA_raw"),
    "B": ("best_ModelC_fixed_l_clahe_seed42.pt", "ModelC_fixed_l_clahe"),
    "C": ("best_ModelC_focal_seed42.pt", "ModelA_raw"),
    "D": (f"best_{D_WINNER}_seed42.pt", "ModelD2_lesion_ita"),
}
WARMUP = 10


def main() -> None:
    OUT.mkdir(exist_ok=True)
    device = 0 if torch.cuda.is_available() else "cpu"
    rows, summary = [], {}
    for key, (weights, dataset) in MODELS.items():
        model = YOLO(str(WEIGHTS / weights))
        images = sorted((DATA / dataset / "images" / "test").iterdir())
        gflops = get_flops(model.model, 640)
        params = sum(p.numel() for p in model.model.parameters())
        times = []
        for i, image in enumerate(images):
            r = model.predict(str(image), imgsz=640, conf=0.25, iou=0.7, device=device, verbose=False)[0]
            if i >= WARMUP:
                times.append(r.speed)
                rows.append({"model": key, "image": image.stem, **r.speed})
        df = pd.DataFrame(times)
        total = df.sum(axis=1)
        summary[key] = {
            "weights": weights, "gflops_640": round(gflops, 2), "params_million": round(params / 1e6, 3),
            "images_timed": len(df),
            "preprocess_ms": round(df["preprocess"].mean(), 2),
            "inference_ms_mean": round(df["inference"].mean(), 2), "inference_ms_sd": round(df["inference"].std(), 2),
            "postprocess_ms": round(df["postprocess"].mean(), 2),
            "total_ms_mean": round(total.mean(), 2), "total_ms_sd": round(total.std(), 2),
        }
        print(key, summary[key])
    hardware = {
        "device": torch.cuda.get_device_name(0) if device == 0 else platform.processor(),
        "framework": f"PyTorch {torch.__version__}, Ultralytics",
        "batch": 1, "imgsz": 640, "letterbox": "rectangular (stride 32)", "confidence": 0.25, "nms_iou": 0.7,
        "warmup_images": WARMUP, "test_images": 200,
        "note": "Ultralytics r.speed per image: preprocess, inference, postprocess (ms). L*-CLAHE preprocessing "
                "of B and D is already applied in their test images, so it is not included here.",
    }
    (OUT / "runtime.json").write_text(json.dumps({"hardware": hardware, "models": summary}, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT / "runtime_per_image.csv", index=False)
    print("Wrote", OUT / "runtime.json")


if __name__ == "__main__":
    main()
