
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
DATA = ROOT / "datasets" / "ablation_yolov26" / "ModelD2_lesion_ita"
ITA = ROOT / "datasets" / "ablation_yolov26" / "ita_table.csv"
WEIGHTS = ROOT / "weights" / "ablation_split"
RUNS = ROOT / "runs" / "ablation_split" / "training"
OUT = HERE / "darkskin_retrain"
NAME = "ModelD_dark2"
BASELINE = "ModelD_g1.0_f11"
REPEAT_DARKEST = 2
FOCAL_GAMMA, FOCAL_ALPHA = 1.0, 0.25
NAMES = ["Warts", "Molluscum", "Varicella", "HFMD", "Tinea versicolor", "Tinea corporis", "Tinea pedis", "Impetigo"]


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


def patch_focal():
    if not hasattr(uloss.v8DetectionLoss, "_init_without_focal"):
        uloss.v8DetectionLoss._init_without_focal = uloss.v8DetectionLoss.__init__

    def init(self, *args, **kwargs):
        uloss.v8DetectionLoss._init_without_focal(self, *args, **kwargs)
        self.bce = ElementwiseFocalLoss(FOCAL_GAMMA, FOCAL_ALPHA)

    uloss.v8DetectionLoss.__init__ = init


def write_data_yaml():
    OUT.mkdir(exist_ok=True)
    brackets = {r["image_id"]: r["bracket"] for r in csv.DictReader(open(ITA)) if r["split"] == "train"}
    images = sorted((DATA / "images" / "train").iterdir())
    lines, dark = [], 0
    for image in images:
        copies = REPEAT_DARKEST if brackets.get(image.stem) == "Darkest" else 1
        dark += copies > 1
        lines += [str(image)] * copies
    (OUT / "train_darkest_x2.txt").write_text("\n".join(lines) + "\n")
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(NAMES))
    yaml = OUT / "data.yaml"
    yaml.write_text(f"path: {DATA}\ntrain: {OUT / 'train_darkest_x2.txt'}\nval: images/val\ntest: images/test\nnames:\n{names}\n")
    print(f"train list: {len(images)} images, {dark} Darkest shown x{REPEAT_DARKEST} -> {len(lines)} per epoch")
    return yaml


def train(seed):
    destination = WEIGHTS / f"best_{NAME}_seed{seed}.pt"
    if destination.is_file():
        print(f"{NAME} seed {seed}: already finished")
        return destination
    patch_focal()
    run = RUNS / f"{NAME}_seed{seed}"
    last = run / "weights" / "last.pt"
    if last.is_file():
        YOLO(str(last)).train(resume=True)
    else:
        YOLO(str(WEIGHTS / f"best_ModelD2_lesion_ita_seed{seed}.pt")).train(
            data=str(write_data_yaml()), epochs=150, patience=50, imgsz=640, batch=8, seed=seed, deterministic=True,
            freeze=11, optimizer="AdamW", lr0=0.0002, warmup_epochs=1, project=str(RUNS), name=f"{NAME}_seed{seed}",
            exist_ok=True, plots=True, verbose=True, workers=4,
        )
    shutil.copy2(run / "weights" / "best.pt", destination)
    return destination


def best_validation(name, seed=42):
    rows = list(csv.DictReader(open(RUNS / f"{name}_seed{seed}" / "results.csv")))
    fitness = lambda r: 0.9 * float(r["metrics/mAP50-95(B)"]) + 0.1 * float(r["metrics/mAP50(B)"])
    b = max(rows, key=fitness)
    m50, m5095 = 100 * float(b["metrics/mAP50(B)"]), 100 * float(b["metrics/mAP50-95(B)"])
    return {"model": name, "best_epoch": int(b["epoch"]), "val_mAP50": round(m50, 2), "val_mAP50_95": round(m5095, 2),
            "score": round((m50 + m5095) / 2, 3)}


def val_details(name, seed=42):
    model = YOLO(str(WEIGHTS / f"best_{name}_seed{seed}.pt"))
    m = model.val(data=str(DATA / "data.yaml"), split="val", imgsz=640, batch=4, rect=True, plots=False,
                  verbose=False, workers=0)
    cm = m.confusion_matrix.matrix
    v, mo = NAMES.index("Varicella"), NAMES.index("Molluscum")
    return {"val_P": round(100 * m.box.mp, 2), "val_R": round(100 * m.box.mr, 2),
            "varicella_as_molluscum": int(cm[mo, v]), "molluscum_as_varicella": int(cm[v, mo]),
            "varicella_correct": int(cm[v, v])}


def compare():
    rows = []
    for name in (BASELINE, NAME):
        row = best_validation(name)
        row.update(val_details(name))
        rows.append(row)
        print(row)
    winner = max(rows, key=lambda r: r["score"])["model"]
    decision = {"rule": "mean of validation mAP50 and mAP50-95 at the best epoch, seed 42 (fixed before training)",
                "candidates": rows, "winner": winner}
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2))
    print("winner on validation:", winner)
    return winner


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "train"
    assert torch.cuda.is_available(), "No GPU"
    if step == "train":
        train(42)
    elif step == "compare":
        compare()
    elif step == "seeds":
        for seed in (43, 44):
            train(seed)
