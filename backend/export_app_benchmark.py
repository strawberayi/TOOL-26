"""
Export the final SOP results to frontend/benchmark_data.json for the app's benchmark views.

Reads the test-set analysis written by ablation_training/final/sop_analysis.py
(manuscript Models A-D, seed 42), so the app shows exactly the numbers in the paper:

    .venv/bin/python backend/export_app_benchmark.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(os.environ.get("SOP_RESULTS", ROOT.parent / "ablation_training" / "final" / "sop_results_manuscript"))
OUTPUT = ROOT / "frontend" / "benchmark_data.json"

MODELS = [
    ("A", "Baseline", "Baseline YOLOv26 (raw images)"),
    ("B", "Fixed L*-CLAHE", "Fixed L*-CLAHE (β = 2.0)"),
    ("C", "Focal Loss", "Focal Loss Optimization (raw images)"),
    ("D", "Proposed", "ITA L*-CLAHE + two-stage + Focal Loss"),
]
NAMES = ["Warts", "Molluscum", "Varicella", "HFMD", "Tinea versicolor", "Tinea corporis", "Tinea pedis", "Impetigo"]
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
GROUPS = [("I-II", "Fitzpatrick I–II", "ITA > 41°"), ("III-V", "Fitzpatrick III–V", "−30° < ITA ≤ 41°")]


def percent(value: float) -> float:
    return round(float(value) * 100, 1)


def main() -> None:
    overall = pd.read_csv(RESULTS / "sop1_2_overall_per_seed.csv")
    overall = overall[overall["seed"] == 42].set_index("model")
    clusters = pd.read_csv(RESULTS / "sop2_within_cluster_per_seed.csv")
    clusters = clusters[clusters["seed"] == 42]
    groups = pd.read_csv(RESULTS / "sop3_group_map50_all_models.csv").set_index(["model", "group"])
    h02 = pd.read_csv(RESULTS / "h02_mann_whitney.csv").iloc[0]
    h01 = pd.read_csv(RESULTS / "h01_friedman_wilcoxon_per_image.csv").set_index("metric")

    models, confusion = [], {}
    for model_id, short, name in MODELS:
        row = overall.loc[model_id]
        wc = clusters[clusters["model"] == model_id].set_index("cluster")["within_cluster_rate_of_detected"]
        models.append({
            "id": model_id, "short": short, "name": name,
            "map50": percent(row["mAP50"]), "map5095": percent(row["mAP50_95"]),
            "precision": percent(row["precision"]), "recall": percent(row["recall"]), "f1": round(float(row["F1"]), 3),
            "withinEruptive": percent(wc["Vesiculopapular/Eruptive"]),
            "withinScaly": percent(wc["Papulosquamous/Verrucous"]),
        })
        # sop_analysis stores rows = predicted (+ background), columns = actual (+ background FP).
        matrix = pd.read_csv(RESULTS / f"confusion_{model_id}_seed42.csv", index_col=0)
        confusion[model_id] = [[int(matrix.iloc[pred, actual]) for pred in range(len(NAMES))]
                               for actual in range(len(NAMES))]

    fairness = [{"group": label, "range": rule, "images": int(groups.loc[("A", key), "n_images"]),
                 "scores": {m: percent(groups.loc[(m, key), "mAP50"]) for m, _, _ in MODELS}}
                for key, label, rule in GROUPS]

    payload = {
        "status": "final",
        "generated": datetime.now().isoformat(timespec="minutes"),
        "note": ("Held-out test set (200 images), YOLO26n, seed 42. P/R at the F1-maximizing confidence; "
                 "within-cluster rates at confidence 0.25, IoU 0.5. Friedman → Wilcoxon (D vs A, B, C, Bonferroni)."),
        "models": models,
        "classes": [CLASS_CODES[c][0] for c in NAMES],
        "classLegend": [CLASS_CODES[c][1] for c in NAMES],
        "confusion": confusion,
        "fairnessMetric": "mAP@50",
        "fairness": fairness,
        "fairnessTest": {"test": "Mann-Whitney U on per-image ΔAP50 (D − A), III–V vs I–II",
                         "p": round(float(h02["p"]), 3), "r": round(float(h02["rank_biserial"]), 2),
                         "excludedVI": int(h02["excluded_VI"])},
        "friedman": {m: round(float(h01.loc[m, "friedman_p"]), 4) for m in h01.index},
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    print(json.dumps({m["id"]: (m["map50"], m["map5095"], m["f1"]) for m in models}, ensure_ascii=False))


if __name__ == "__main__":
    main()
