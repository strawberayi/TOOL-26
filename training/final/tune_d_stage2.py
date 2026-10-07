import csv
import json
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.utils import loss as uloss

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_YAML = ROOT / "datasets" / "ablation_yolov26" / "ModelD2_lesion_ita" / "data.yaml"
WEIGHTS = ROOT / "weights" / "ablation_split"
RUNS = ROOT / "runs" / "ablation_split" / "training"
OUT = HERE / "d_tuning_round2"
BASELINE = "ModelD_g1.0_f11"

BASE = dict(epochs=150, patience=50, imgsz=640, batch=8, freeze=11, optimizer="AdamW", lr0=0.0002, warmup_epochs=1)
CANDIDATES = {
    "ModelD_r2_unfreeze": dict(freeze=0, lr0=0.00005),
    "ModelD_r2_cos": dict(cos_lr=True, lr0=0.0001, warmup_epochs=3),
    "ModelD_r2_nomosaic": dict(mosaic=0.0),
    "ModelD_r2_alpha05": dict(focal_alpha=0.5),
    "ModelD_r2_box10": dict(box=10.0),
    "ModelD_r2_img800": dict(imgsz=800, batch=4),
    "ModelD_r3_freeze3": dict(freeze=3, lr0=0.00005),
    "ModelD_r3_freeze5": dict(freeze=5, lr0=0.00005),
    "ModelD_r3_freeze7": dict(freeze=7, lr0=0.00005),
}
FOCAL_GAMMA = 1.0


class ElementwiseFocalLoss(nn.Module):
    def __init__(self, gamma, alpha):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, pred, label):
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        prob = pred.sigmoid()
        p_t = label * prob + (1 - label) * (1 - prob)
        loss = loss * (1.0 - p_t) ** self.gamma
        if self.alpha > 0:
            loss = loss * (label * self.alpha + (1 - label) * (1 - self.alpha))
        return loss


def patch_focal(alpha):
    if not hasattr(uloss.v8DetectionLoss, "_init_without_focal"):
        uloss.v8DetectionLoss._init_without_focal = uloss.v8DetectionLoss.__init__

    def init(self, *args, **kwargs):
        uloss.v8DetectionLoss._init_without_focal(self, *args, **kwargs)
        self.bce = ElementwiseFocalLoss(FOCAL_GAMMA, alpha)

    uloss.v8DetectionLoss.__init__ = init


def train(name, seed):
    destination = WEIGHTS / f"best_{name}_seed{seed}.pt"
    if destination.is_file():
        print(f"{name} seed {seed}: already finished")
        return
    settings = {**BASE, **CANDIDATES[name]}
    patch_focal(settings.pop("focal_alpha", 0.25))
    run = RUNS / f"{name}_seed{seed}"
    last = run / "weights" / "last.pt"
    if last.is_file():
        YOLO(str(last)).train(resume=True)
    else:
        YOLO(str(WEIGHTS / f"best_ModelD2_lesion_ita_seed{seed}.pt")).train(
            data=str(DATA_YAML), seed=seed, deterministic=True, project=str(RUNS), name=f"{name}_seed{seed}",
            exist_ok=True, plots=True, verbose=True, workers=4, **settings)
    shutil.copy2(run / "weights" / "best.pt", destination)


def best_validation(name, seed=42):
    rows = list(csv.DictReader(open(RUNS / f"{name}_seed{seed}" / "results.csv")))
    fitness = lambda r: 0.9 * float(r["metrics/mAP50-95(B)"]) + 0.1 * float(r["metrics/mAP50(B)"])
    b = max(rows, key=fitness)
    m50, m5095 = 100 * float(b["metrics/mAP50(B)"]), 100 * float(b["metrics/mAP50-95(B)"])
    return {"candidate": name, "best_epoch": int(b["epoch"]), "epochs_run": len(rows),
            "val_P": round(100 * float(b["metrics/precision(B)"]), 2), "val_R": round(100 * float(b["metrics/recall(B)"]), 2),
            "val_mAP50": round(m50, 2), "val_mAP50_95": round(m5095, 2), "score": round((m50 + m5095) / 2, 3)}


def select():
    OUT.mkdir(exist_ok=True)
    rows = [best_validation(BASELINE)]
    for name in CANDIDATES:
        if (RUNS / f"{name}_seed42" / "results.csv").is_file() and (WEIGHTS / f"best_{name}_seed42.pt").is_file():
            rows.append(best_validation(name))
    rows.sort(key=lambda r: r["score"], reverse=True)
    for r in rows:
        print(r)
    winner = rows[0]["candidate"]
    (OUT / "validation.json").write_text(json.dumps({
        "rule": "mean of validation mAP50 and mAP50-95 at the best epoch, seed 42 (fixed before training)",
        "baseline": BASELINE, "candidates": rows, "winner": winner,
        "settings": {name: {**BASE, **CANDIDATES[name]} for name in CANDIDATES}}, indent=2))
    print("winner on validation:", winner)


if __name__ == "__main__":
    assert torch.cuda.is_available(), "No GPU"
    if sys.argv[1] == "train":
        train(sys.argv[2], int(sys.argv[3]))
    elif sys.argv[1] == "select":
        select()
