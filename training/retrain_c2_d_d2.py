from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NOTEBOOK = HERE / "daniel_train_C2_D.ipynb"
CAL = ROOT / "calibration_d2"
D2_CONFIG = ROOT / "backend" / "phase0_calibration_d2.json"
OUT = ROOT / "runs" / "ablation_split" / "d2_seeds"

C2, D, D2 = "ModelC2_fixed_l_clahe_global", "ModelD_proposed", "ModelD2_lesion_ita"
SEEDS = (42, 43, 44)
BETA_GRID = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
BRACKETS = ("Darkest", "Medium", "Lightest")


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def banner(text):
    print("\n" + "=" * 72 + f"\n  {text}\n" + "=" * 72, flush=True)


def lesion_visibility():
    import cv2
    import numpy as np
    import pandas as pd
    sys.path.insert(0, str(ROOT / "backend"))
    from clahe_calibration import apply_clahe

    images = ROOT / "datasets" / "source_yolo" / "images" / "train"
    labels = ROOT / "datasets" / "source_yolo" / "labels" / "train"
    rows = []
    for _, r in pd.read_csv(CAL / "train_ita_masks.csv").iterrows():
        label = labels / f"{r.image_id}.txt"
        if not label.is_file():
            continue
        rgb = cv2.cvtColor(cv2.imread(str(images / f"{r.image_id}.png")), cv2.COLOR_BGR2RGB)
        skin = cv2.imread(str(CAL / "masks" / f"{r.image_id}.png"), cv2.IMREAD_GRAYSCALE) > 0
        h, w = skin.shape
        lesion = np.zeros_like(skin)
        for line in label.read_text().splitlines():
            _, x, y, bw, bh = map(float, line.split())
            lesion[max(0, int((y - bh / 2) * h)):int((y + bh / 2) * h),
                   max(0, int((x - bw / 2) * w)):int((x + bw / 2) * w)] = True
        healthy = skin & ~lesion
        if lesion.sum() < 50 or healthy.sum() < 500:
            continue
        for beta in [0.0] + BETA_GRID:
            image = rgb if beta == 0 else apply_clahe(rgb, beta=beta, tile_grid_size=(8, 8))
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
            lab[..., 0] *= 100 / 255
            lab[..., 1:] -= 128
            rows.append({"image_id": r.image_id, "bracket": r.bracket, "beta": beta,
                         "delta_e": float(np.linalg.norm(lab[lesion].mean(0) - lab[healthy].mean(0)))})
    return pd.DataFrame(rows)


def calibrate_d2() -> dict:
    import pandas as pd

    if D2_CONFIG.is_file():
        config = json.loads(D2_CONFIG.read_text())
        print(f"Using the existing D2 calibration ({D2_CONFIG.name}): {config['betas']}")
        return config
    banner("Step 1/4: D2 calibration (lesion visibility, train images only; ~3-5 min)")
    curves = lesion_visibility()
    OUT.mkdir(parents=True, exist_ok=True)
    curves.to_csv(OUT / "d2_lesion_visibility_per_image.csv", index=False)
    median = curves.groupby(["bracket", "beta"])["delta_e"].median().unstack(0)
    median.round(3).to_csv(OUT / "d2_lesion_visibility_median.csv")
    counts = {k: int(v) for k, v in curves[curves.beta == 0].bracket.value_counts().items()}

    noise = pd.read_csv(CAL / "beta_search_summary.csv")
    admissible = {b: set(noise[(noise.bracket == b) & (noise.acceptance_rate >= 0.8)].beta.round(2))
                  | {x for x in BETA_GRID if x < 2.0} for b in BRACKETS}

    target = float(median.loc[0.0, "Lightest"])
    betas, reasons = {}, {}
    for b in BRACKETS:
        options = [x for x in BETA_GRID if round(x, 2) in admissible[b]]
        reaching = [x for x in options if median.loc[x, b] >= target]
        if reaching:
            betas[b] = min(reaching)
            reasons[b] = f"smallest beta reaching the target ({median.loc[betas[b], b]:.2f} >= {target:.2f})"
        else:
            betas[b] = max(options, key=lambda x: (round(median.loc[x, b], 3), -x))
            reasons[b] = f"target not reachable; highest lesion visibility ({median.loc[betas[b], b]:.2f})"

    print(f"Train images with lesion boxes: {counts}")
    print("Median lesion visibility (delta-E), beta 0 = no CLAHE:")
    print(median.round(2).to_string())
    print(f"\nTarget = lightest skin without CLAHE: {target:.2f}")
    for b in BRACKETS:
        print(f"  {b:9s} beta = {betas[b]:.1f}   ({reasons[b]})")

    config = {
        "name": "D2 lesion-visibility equalization",
        "status": "PROVISIONAL (needs adviser approval)",
        "created": time.strftime("%Y-%m-%dT%H:%M"),
        "rule": ("For each ITA bracket, the smallest CLAHE clip limit whose median lesion-vs-healthy-skin "
                 "CIELAB delta-E reaches the lightest bracket's delta-E without CLAHE; if unreachable "
                 "within the noise limit, the clip limit with the highest delta-E."),
        "data": "calibration-eligible TRAIN images with lesion boxes only",
        "images_per_bracket": counts,
        "target_delta_e": target,
        "betas": betas,
        "no_ita_fallback": "Medium bracket beta",
        "reasons": reasons,
    }
    D2_CONFIG.write_text(json.dumps(config, indent=2) + "\n")
    print(f"\nSaved {D2_CONFIG}")
    return config


def notebook_namespace(d2_betas: dict) -> dict:
    cells = ["".join(c["source"]) for c in json.loads(NOTEBOOK.read_text())["cells"] if c["cell_type"] == "code"]

    def cell(marker):
        match = [c for c in cells if marker in c]
        assert len(match) == 1, marker
        return match[0]

    def set_line(source, prefix, line):
        return "\n".join(line if l.startswith(prefix) else l for l in source.splitlines())

    os.chdir(HERE)
    ns = {"__name__": "__notebook__"}
    config = set_line(cell("MODEL_CHECKPOINT = "), "SEEDS = ", f"SEEDS = {SEEDS}")
    config = set_line(config, "TRAIN_MODELS = ", f"TRAIN_MODELS = {(C2, D, D2)!r}")
    for source in (config, cell("IMAGE_SUFFIXES = "), cell("ITA_CACHE = ")):
        exec(compile(source, "notebook", "exec"), ns)

    dataset_cell = cell("def make_dataset(")
    exec(compile(dataset_cell[: dataset_cell.index("for model_name in TRAIN_MODELS:")], "notebook", "exec"), ns)

    apply_clahe, tile = ns["apply_clahe"], ns["TILE_GRID_SIZE"]
    original_transform = ns["transform"]

    def transform(model_name, rgb, image_id):
        if model_name != D2:
            return original_transform(model_name, rgb, image_id)
        bracket = ns["ITA_BY_ID"].loc[image_id, "bracket"]
        bracket = bracket if isinstance(bracket, str) and bracket else "Medium"
        beta = d2_betas[bracket]
        return apply_clahe(rgb, beta=beta, tile_grid_size=tile), {"beta": beta, "bracket": bracket}

    ns["transform"] = transform
    ns["MODELS"] = tuple(ns["MODELS"]) + (D2,)
    for model in ns["TRAIN_MODELS"]:
        print(f"Preparing {model}...", flush=True)
        ns["make_dataset"](model)
    exec(compile(cell("def write_data_yaml("), "notebook", "exec"), ns)

    training = set_line(cell("def train_one("), "TRAIN_ONLY = ", "TRAIN_ONLY = TRAIN_MODELS")
    ns["_training_source"] = training
    return ns


def evaluate_all(ns) -> None:
    import pandas as pd
    from ultralytics import YOLO

    rows = []
    for model in (C2, D, D2):
        for seed in SEEDS:
            weights = ns["WEIGHTS_ROOT"] / f"best_{model}_seed{seed}.pt"
            targets = [("val", "all", ns["DATA_YAMLS"][model], "val"),
                       ("test", "all", ns["DATA_YAMLS"][model], "test")]
            targets += [("test", b, ns["BRACKET_YAMLS"][(model, b)], "test") for b in BRACKETS]
            for split, subset, yaml_path, yolo_split in targets:
                r = YOLO(str(weights)).val(data=str(yaml_path), split=yolo_split, imgsz=ns["IMG_SIZE"], batch=8,
                                           plots=False, verbose=False, project=str(OUT / "eval"),
                                           name=f"{model}_s{seed}_{split}_{subset}", exist_ok=True)
                p, rc = float(r.box.mp), float(r.box.mr)
                rows.append({"model": model, "seed": seed, "split": split, "subset": subset,
                             "mAP50": r.box.map50, "mAP50-95": r.box.map, "precision": p, "recall": rc,
                             "F1": 2 * p * rc / (p + rc) if p + rc else 0.0})
    per_run = pd.DataFrame(rows)
    per_run.to_csv(OUT / "comparison_per_run.csv", index=False)

    names = {C2: "C′ (fixed β 4.0, no ITA)", D: "D (ITA, original β)", D2: "D2 (ITA, lesion-visibility β)"}
    lines = ["# C′ vs D vs D2 (mean ± SD over seeds 42, 43, 44)", ""]
    sections = [("val", "all", "Validation: use this to decide"), ("test", "all", "Test: final report")]
    sections += [("test", b, f"Test, {b} skin") for b in BRACKETS]
    for split, subset, title in sections:
        part = per_run[(per_run.split == split) & (per_run.subset == subset)]
        metrics = ["mAP50", "mAP50-95", "precision", "recall", "F1"] if subset == "all" else ["mAP50-95"]
        lines += [f"## {title}", "", "| Model | " + " | ".join(metrics) + " |", "|---" * (len(metrics) + 1) + "|"]
        for model in (C2, D, D2):
            m = part[part.model == model]
            cells = [f"{m[k].mean():.3f} ± {m[k].std(ddof=1):.3f}" if k == "F1"
                     else f"{100 * m[k].mean():.1f} ± {100 * m[k].std(ddof=1):.1f}" for k in metrics]
            lines.append(f"| {names[model]} | " + " | ".join(cells) + " |")
        lines.append("")
    (OUT / "comparison_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    log = (OUT / "run_log.txt").open("a", encoding="utf-8")
    log.write(f"\n##### run started {time.strftime('%Y-%m-%d %H:%M:%S')} #####\n")
    sys.stdout = Tee(sys.__stdout__, log)
    sys.stderr = Tee(sys.__stderr__, log)
    os.environ.setdefault("MPLBACKEND", "Agg")
    epochs = os.environ.get("ABLATION_SMOKE_EPOCHS")
    start = time.time()

    d2 = calibrate_d2()
    banner("Steps 2-3/4: Training C′, D (seeds 43, 44) and D2 (seeds 42, 43, 44)")
    print("About 45 minutes per model and seed; finished runs are skipped.", flush=True)
    ns = notebook_namespace(d2["betas"])
    training = ns["_training_source"]
    if epochs:
        ns["EPOCHS"] = int(epochs)
    exec(compile(training, "training", "exec"), ns)

    banner("Step 4/4: Evaluating C′, D, D2 on validation and test")
    evaluate_all(ns)
    banner(f"DONE in {(time.time() - start) / 3600:.1f} h")
    print(f"Tables: {OUT / 'comparison_summary.md'}\nAll numbers: {OUT / 'comparison_per_run.csv'}")


if __name__ == "__main__":
    main()
