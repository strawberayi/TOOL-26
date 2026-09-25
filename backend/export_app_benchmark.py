"""
Export the real ablation results to frontend/benchmark_data.json for the app's
"Ablation Benchmark" views. Run after the notebook's test evaluation:

    .venv/bin/python backend/export_app_benchmark.py

Metrics come from runs/ablation_yolov26/*.csv written by the notebook. The
confusion matrices are recomputed from the primary-seed weights on the same
held-out test split (evaluation only; nothing is tuned).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs" / "ablation_yolov26"
WEIGHTS = ROOT / "weights" / "ablation_yolov26"
ABLATION = ROOT / "datasets" / "ablation_yolov26"
OUTPUT = ROOT / "frontend" / "benchmark_data.json"

calibration = json.loads((ROOT / "backend" / "phase0_calibration.json").read_text(encoding="utf-8"))

MODELS = [
    ("ModelA_raw", "A", "Raw", "Raw images, no enhancement"),
    ("ModelB_rgb_clahe", "B", "RGB CLAHE", "RGB CLAHE, fixed β=2.0"),
    ("ModelC_fixed_l_clahe", "C", "L*-CLAHE β=2", "CIELAB L*-CLAHE, fixed β=2.0"),
    ("ModelC2_fixed_l_clahe_global", "C′", "L*-CLAHE β global",
     f"CIELAB L*-CLAHE, fixed β={calibration['beta_global']} (no ITA)"),
    ("ModelD_proposed", "D", "Proposed",
     f"K-means mask + ITA → β {calibration['beta_high']}/{calibration['beta_mid']}/{calibration['beta_low']}"),
]
# App codes use the local disease names.
CLASS_CODES = {
    "Warts": ("KU", "Kulugo (Warts)"),
    "Molluscum": ("MC", "Molluscum"),
    "Varicella": ("BT", "Bulutong (Varicella)"),
    "HFMD": ("HF", "HFMD"),
    "Tinea versicolor": ("AN", "An-an (T. versicolor)"),
    "Tinea corporis": ("BU", "Buni (T. corporis)"),
    "Tinea pedis": ("AL", "Alipunga (T. pedis)"),
    "Impetigo": ("MA", "Mamaso (Impetigo)"),
}
BRACKETS = [("Darkest", "ITA < 28°"), ("Medium", "28° ≤ ITA ≤ 41°"), ("Lightest", "ITA > 41°")]


def percent(value: float) -> float:
    return round(float(value) * 100, 1)


def main() -> None:
    from ultralytics import YOLO

    per_seed = pd.read_csv(RUNS / "ablation_test_metrics_per_seed.csv")
    summary = per_seed.groupby("model")[["mAP@0.5", "mAP@0.5:0.95", "precision", "recall"]].mean()
    seeds = sorted(per_seed["seed"].unique().tolist())

    class_names = json.loads((ROOT / "datasets" / "source_yolo" / "classes.json").read_text())["classes"]

    models, confusion = [], {}
    for folder, model_id, short, name in MODELS:
        row = summary.loc[folder]
        p, r = float(row["precision"]), float(row["recall"])
        models.append({
            "id": model_id, "short": short, "name": name,
            "map50": percent(row["mAP@0.5"]), "map5095": percent(row["mAP@0.5:0.95"]),
            "precision": percent(p), "recall": percent(r),
            "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0,
        })
        # Ultralytics stores matrix[predicted, actual] with background last.
        # plots=True is required: the confusion matrix is only filled when plotting.
        results = YOLO(str(WEIGHTS / f"best_{folder}.pt")).val(
            data=str(ABLATION / folder / "data.yaml"), split="test", imgsz=640, batch=8,
            plots=True, verbose=False, project=str(RUNS / "app_export"), name=folder, exist_ok=True,
        )
        matrix = results.confusion_matrix.matrix
        n = len(class_names)
        confusion[model_id] = [[int(matrix[pred][actual]) for pred in range(n)] for actual in range(n)]

    bracket_rows = pd.read_csv(RUNS / "ablation_test_metrics_by_bracket_per_seed.csv")
    bracket_means = bracket_rows.groupby(["model", "bracket"])["mAP@0.5:0.95"].mean()
    ita = pd.read_csv(ABLATION / "ita_table.csv")
    test_rows = ita[ita["split"] == "test"]
    test_counts = test_rows["bracket"].value_counts()
    fairness = []
    for bracket, rule in BRACKETS:
        scores = {
            model_id: percent(bracket_means[(folder, bracket)])
            for folder, model_id, _, _ in MODELS if (folder, bracket) in bracket_means.index
        }
        fairness.append({"group": bracket, "range": rule, "images": int(test_counts.get(bracket, 0)), "scores": scores})

    payload = {
        "status": "final" if calibration.get("status") == "FROZEN" else "provisional",
        "generated": datetime.now().isoformat(timespec="minutes"),
        "note": (f"Held-out test set ({len(test_rows)} images), YOLO26n, "
                 f"{len(seeds)} seed(s): {', '.join(map(str, seeds))}. "
                 f"Calibration {calibration.get('status', 'unknown')}."),
        "models": models,
        "classes": [CLASS_CODES[c][0] for c in class_names],
        "classLegend": [CLASS_CODES[c][1] for c in class_names],
        "confusion": confusion,
        "fairness": fairness,
    }
    if not any(any(any(row) for row in matrix) for matrix in confusion.values()):
        raise RuntimeError("All confusion matrices are empty")
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    print(json.dumps({m["id"]: (m["map50"], m["map5095"], m["f1"]) for m in models}, ensure_ascii=False))


if __name__ == "__main__":
    main()
