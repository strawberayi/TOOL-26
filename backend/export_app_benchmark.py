from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FINAL = ROOT.parent / "ablation_training" / "final"
RESULTS = Path(os.environ.get("SOP_RESULTS", FINAL / "sop_results_manuscript"))
RUNTIME = FINAL / "runtime_results" / "runtime.json"
OUTPUT = ROOT / "frontend" / "benchmark_data.json"

MODELS = [
    ("A", "Baseline", "Baseline YOLOv26 (raw images)"),
    ("B", "Fixed L*-CLAHE", "YOLOv26 with fixed-parameter L*-CLAHE (clip limit 2.0)"),
    ("C", "Focal Loss", "YOLOv26 with Focal Loss optimization (raw images)"),
    ("D", "Proposed", "ITA-guided adaptive L*-CLAHE + two-stage decoupled training + Focal Loss"),
]
NAMES = ["Warts", "Molluscum", "Varicella", "HFMD", "Tinea versicolor", "Tinea corporis", "Tinea pedis", "Impetigo"]
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
GROUPS = [("I-II", "Fitzpatrick I–II (ITA proxy)", "ITA > 41°"),
          ("III-V", "Fitzpatrick III–V (ITA proxy)", "−30° < ITA ≤ 41°")]
METRIC_LABELS = {"AP50": "AP@50", "AP50_95": "AP@50–95", "precision": "Precision", "recall": "Recall", "F1": "F1"}


def percent(value: float) -> float:
    return round(float(value) * 100, 1)


def r4(value: float) -> float:
    return round(float(value), 4)


def main() -> None:
    overall = pd.read_csv(RESULTS / "sop1_2_overall_per_seed.csv")
    overall = overall[overall["seed"] == 42].set_index("model")
    per_class = pd.read_csv(RESULTS / "sop1_2_per_class_per_seed.csv")
    per_class = per_class[per_class["seed"] == 42]
    clusters = pd.read_csv(RESULTS / "sop2_within_cluster_per_seed.csv")
    clusters = clusters[clusters["seed"] == 42]
    groups = pd.read_csv(RESULTS / "sop3_group_map50_all_models.csv").set_index(["model", "group"])
    group_d = pd.read_csv(RESULTS / "sop3_group_dAP50.csv").set_index("group")
    h02 = pd.read_csv(RESULTS / "h02_mann_whitney.csv").iloc[0]
    h01 = pd.read_csv(RESULTS / "h01_friedman_wilcoxon_per_image.csv").set_index("metric")

    models, confusion = [], {}
    for model_id, short, name in MODELS:
        row = overall.loc[model_id]
        rows = per_class[per_class["model"] == model_id].set_index("class").reindex(NAMES)
        for column, key in [("mAP50", "mAP50"), ("mAP50_95", "mAP50_95"), ("precision", "precision"), ("recall", "recall")]:
            assert abs(rows[column].mean() - row[key]) < 1e-9, (model_id, column)
        wc = clusters[clusters["model"] == model_id].set_index("cluster")
        models.append({
            "id": model_id, "short": short, "name": name,
            "map50": percent(row["mAP50"]), "map5095": percent(row["mAP50_95"]),
            "precision": percent(row["precision"]), "recall": percent(row["recall"]), "f1": round(float(row["F1"]), 3),
            "withinEruptive": percent(wc.loc["Vesiculopapular/Eruptive", "within_cluster_rate_of_detected"]),
            "withinScaly": percent(wc.loc["Papulosquamous/Verrucous", "within_cluster_rate_of_detected"]),
            "exact": {"mAP50": r4(row["mAP50"]), "mAP50_95": r4(row["mAP50_95"]), "precision": r4(row["precision"]),
                      "recall": r4(row["recall"]), "F1": r4(row["F1"])},
            "perClass": [{"class": c, "instances": int(rows.loc[c, "instances"]),
                          "precision": r4(rows.loc[c, "precision"]), "recall": r4(rows.loc[c, "recall"]),
                          "F1": r4(rows.loc[c, "F1"]), "AP50": r4(rows.loc[c, "mAP50"]),
                          "AP50_95": r4(rows.loc[c, "mAP50_95"])} for c in NAMES],
            "clusters": [{"cluster": c, "gt": int(wc.loc[c, "gt_lesions"]), "detected": int(wc.loc[c, "detected"]),
                          "correct": int(wc.loc[c, "correct"]), "within": int(wc.loc[c, "within_cluster_errors"]),
                          "cross": int(wc.loc[c, "cross_cluster_errors"]), "missed": int(wc.loc[c, "missed"])}
                         for c in ["Vesiculopapular/Eruptive", "Papulosquamous/Verrucous"]],
        })
        matrix = pd.read_csv(RESULTS / f"confusion_{model_id}_seed42.csv", index_col=0)
        confusion[model_id] = [[int(matrix.iloc[pred, actual]) for pred in range(len(NAMES))]
                               for actual in range(len(NAMES))]

    fairness = [{"group": label, "key": key, "range": rule, "images": int(groups.loc[("A", key), "n_images"]),
                 "scores": {m: percent(groups.loc[(m, key), "mAP50"]) for m, _, _ in MODELS},
                 "dAP50": r4(group_d.loc[key, "dmAP50"]),
                 "meanPerImageDAP50": r4(group_d.loc[key, "mean_per_image_dAP50"]),
                 "medianPerImageDAP50": r4(group_d.loc[key, "median_per_image_dAP50"])}
                for key, label, rule in GROUPS]

    tests = []
    for metric in h01.index:
        t = h01.loc[metric]
        tests.append({
            "metric": METRIC_LABELS.get(metric, metric), "n": int(t["n_images"]),
            "friedmanChi2": round(float(t["friedman_chi2"]), 3), "friedmanP": r4(t["friedman_p"]),
            "means": {m: r4(t[f"mean_{m}"]) for m, _, _ in MODELS},
            "pairs": [{"vs": other, "p": r4(t[f"D_vs_{other}_p"]), "pBonf": r4(t[f"D_vs_{other}_p_bonf"]),
                       "r": round(float(t[f"D_vs_{other}_rank_biserial"]), 3)} for other in ("A", "B", "C")],
        })

    runtime = json.loads(RUNTIME.read_text()) if RUNTIME.is_file() else None

    payload = {
        "status": "final",
        "generated": datetime.now().isoformat(timespec="minutes"),
        "testImages": 200,
        "seed": 42,
        "note": ("Held-out test set (200 images), YOLO26n, seed 42, the same images for every model. "
                 "mAP, P, R: official Ultralytics evaluation (P and R at the confidence that maximizes F1, "
                 "averaged over the 8 diseases). Within-cluster rates and confusion matrices: boxes with "
                 "confidence ≥ 0.25 matched to labels at IoU ≥ 0.5."),
        "models": models,
        "classes": [CLASS_CODES[c][0] for c in NAMES],
        "classNames": NAMES,
        "classLegend": [CLASS_CODES[c][1] for c in NAMES],
        "confusion": confusion,
        "fairnessMetric": "mAP@50",
        "fairness": fairness,
        "fairnessTest": {"test": "Mann-Whitney U on per-image ΔAP50 (D − A), III–V vs I–II",
                         "n1": int(h02["n_I_II"]), "n2": int(h02["n_III_V"]),
                         "U": float(h02["mann_whitney_U"]), "p": round(float(h02["p"]), 4),
                         "r": round(float(h02["rank_biserial"]), 3),
                         "ciLow": r4(h02["bootstrap_ci_low"]), "ciHigh": r4(h02["bootstrap_ci_high"]),
                         "excludedVI": int(h02["excluded_VI"]), "excludedNoIta": int(h02["excluded_no_ITA"])},
        "tests": tests,
        "friedman": {m: round(float(h01.loc[m, "friedman_p"]), 4) for m in h01.index},
        "runtime": runtime,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    print(json.dumps({m["id"]: (m["map50"], m["map5095"], m["f1"]) for m in models}, ensure_ascii=False))


if __name__ == "__main__":
    main()
