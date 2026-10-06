from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
DATASETS = ROOT / "datasets" / "ablation_yolov26"
WEIGHTS = ROOT / "weights" / "ablation_split"
OUT = HERE / "gradcam_results"
APP_ASSETS = ROOT.parent / "TOOL-26" / "frontend" / "assets" / "gradcam"

NAMES = ["Warts", "Molluscum", "Varicella", "HFMD", "Tinea versicolor", "Tinea corporis", "Tinea pedis", "Impetigo"]
D_WEIGHTS = "ModelD_full"
_choice = HERE / "d_tuning_choice.json"
if _choice.is_file():
    D_WEIGHTS = json.loads(_choice.read_text())["winner"]
MODELS = {
    "A": ("Baseline", "ModelA_raw", "ModelA_raw"),
    "B": ("Fixed L*-CLAHE (β=2.0)", "ModelC_fixed_l_clahe", "ModelC_fixed_l_clahe"),
    "C": ("Focal Loss", "ModelA_raw", "ModelC_focal"),
    "D": ("Proposed (ITA L*-CLAHE + two-stage + Focal)", "ModelD2_lesion_ita", D_WEIGHTS),
}
SEED = 42
IMGSZ = 640
N_SCALES = 3
CONF = 0.25
MAX_TARGETS = 10
CAM_THRESHOLD = 0.15


def letterbox(img: np.ndarray, stride: int = 32) -> tuple[np.ndarray, float, tuple[int, int]]:
    h, w = img.shape[:2]
    r = min(IMGSZ / h, IMGSZ / w)
    nw, nh = round(w * r), round(h * r)
    dw, dh = ((IMGSZ - nw) % stride) / 2, ((IMGSZ - nh) % stride) / 2
    top, bottom, left, right = round(dh - 0.1), round(dh + 0.1), round(dw - 0.1), round(dw + 0.1)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) else img
    canvas = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return canvas, r, (top, left)


class GradCAM:
    def __init__(self, weights: Path):
        from ultralytics import YOLO
        self.model = YOLO(str(weights)).model.float().eval()
        self.acts, self.grads = {}, {}
        head = self.model.model[-1].cv3
        for i in range(N_SCALES):
            head[i][1].register_forward_hook(self._hook(i))

    def _hook(self, i):
        def fn(_module, _inp, out):
            self.acts[i] = out
            out.register_hook(lambda g: self.grads.__setitem__(i, g))
        return fn

    def __call__(self, img_bgr: np.ndarray) -> tuple[np.ndarray, list[int]]:
        h, w = img_bgr.shape[:2]
        lb, r, (top, left) = letterbox(img_bgr)
        x = torch.from_numpy(lb[:, :, ::-1].copy()).permute(2, 0, 1)[None].float() / 255.0
        x.requires_grad_(True)
        self.model.zero_grad()
        _, preds = self.model(x)
        scores = preds["one2many"]["scores"][0]
        probs = scores.sigmoid()
        best_p, best_c = probs.max(0)
        idx = torch.nonzero(best_p >= CONF).flatten()
        if len(idx) == 0:
            idx = best_p.argmax()[None]
        idx = idx[best_p[idx].argsort(descending=True)][:MAX_TARGETS]
        target = scores[best_c[idx], idx].sum()
        target.backward()

        ph, pw = lb.shape[:2]
        cam_total = np.zeros((ph, pw), np.float32)
        for i in range(N_SCALES):
            a, g = self.acts[i][0], self.grads[i][0]
            weights = g.mean(dim=(1, 2), keepdim=True)
            cam = torch.relu((weights * a).sum(0)).detach().numpy()
            if cam.max() > 0:
                cam = cam / cam.max()
            cam_total += cv2.resize(cam, (pw, ph), interpolation=cv2.INTER_LINEAR)
        cam_total /= N_SCALES
        nh, nw = round(h * r), round(w * r)
        cam_img = cv2.resize(cam_total[top:top + nh, left:left + nw], (w, h), interpolation=cv2.INTER_LINEAR)
        if cam_img.max() > 0:
            cam_img = cam_img / cam_img.max()
        return cam_img, sorted({int(c) for c in best_c[idx].tolist()})


def gt_mask(label: Path, h: int, w: int) -> tuple[np.ndarray, np.ndarray, list[tuple]]:
    rows = np.loadtxt(label, ndmin=2) if label.stat().st_size else np.zeros((0, 5))
    mask = np.zeros((h, w), bool)
    boxes = []
    for c, xc, yc, bw, bh in rows:
        x1, y1 = int(max(0, (xc - bw / 2) * w)), int(max(0, (yc - bh / 2) * h))
        x2, y2 = int(min(w, (xc + bw / 2) * w)), int(min(h, (yc + bh / 2) * h))
        mask[y1:y2, x1:x2] = True
        boxes.append((int(c), x1, y1, x2, y2))
    return mask, rows[:, 0].astype(int), boxes


def overlay(img: np.ndarray, cam: np.ndarray, boxes: list[tuple] | None = None) -> np.ndarray:
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET).astype(np.float32)
    alpha = (0.65 * np.clip(cam, 0, 1))[..., None]
    out = (img.astype(np.float32) * (1 - alpha) + heat * alpha).astype(np.uint8)
    for _, x1, y1, x2, y2 in boxes or []:
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), max(1, img.shape[1] // 250))
    return out


def rank_biserial_paired(diff: np.ndarray) -> float:
    d = diff[diff != 0]
    if len(d) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    return float((ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum())


def main() -> None:
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    ita = pd.read_csv(DATASETS / "ita_table.csv", dtype={"image_id": str})
    ita = ita[ita["split"] == "test"].set_index("image_id")["ita"]

    stems = sorted(p.stem for p in (DATASETS / "ModelA_raw" / "images" / "test").iterdir())
    rows, cams = [], {}
    for key, (label, folder, weights_name) in MODELS.items():
        weights = WEIGHTS / f"best_{weights_name}_seed{SEED}.pt"
        print(f"Model {key} ({label}): Grad-CAM on {len(stems)} test images...", flush=True)
        explainer = GradCAM(weights)
        img_dir = DATASETS / folder / "images" / "test"
        for n, stem in enumerate(stems, 1):
            img_path = next(img_dir.glob(stem + ".*"))
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            mask, gt_cls, boxes = gt_mask(DATASETS / folder / "labels" / "test" / f"{stem}.txt", h, w)
            cam, target_cls = explainer(img)
            hot = cam >= CAM_THRESHOLD
            union = (hot | mask).sum()
            iou = float((hot & mask).sum() / union) if union else 0.0
            energy = float(cam[mask].sum() / cam.sum()) if cam.sum() > 0 else 0.0
            rows.append({"model": key, "image_id": stem, "gradcam_iou": iou, "energy_in_boxes": energy,
                         "box_area_share": float(mask.mean()), "gt_classes": ";".join(NAMES[c] for c in sorted(set(gt_cls))),
                         "explained_classes": ";".join(NAMES[c] for c in target_cls), "ita": ita.get(stem, np.nan)})
            cams[(key, stem)] = cam
            if n % 50 == 0:
                print(f"  {n}/{len(stems)}", flush=True)
        del explainer

    df = pd.DataFrame(rows)
    df["group"] = np.select([df["ita"] > 41, df["ita"] > -30], ["I-II", "III-V"], "VI")
    df.loc[df["ita"].isna(), "group"] = "no ITA"
    df.to_csv(OUT / "gradcam_per_image.csv", index=False)

    summary = df.groupby("model").agg(gradcam_iou_mean=("gradcam_iou", "mean"), gradcam_iou_median=("gradcam_iou", "median"),
                                      energy_in_boxes_mean=("energy_in_boxes", "mean")).reset_index()
    summary.insert(1, "name", summary["model"].map(lambda k: MODELS[k][0]))
    by_group = df[df["group"].isin(["I-II", "III-V"])].pivot_table(index="model", columns="group",
                                                                   values="gradcam_iou", aggfunc="mean")
    summary = summary.merge(by_group.add_prefix("iou_").reset_index(), on="model")
    summary.to_csv(OUT / "gradcam_summary.csv", index=False)

    tests = []
    for metric in ("gradcam_iou", "energy_in_boxes"):
        table = df.pivot(index="image_id", columns="model", values=metric)
        chi2, p = stats.friedmanchisquare(*[table[k] for k in MODELS])
        row = {"metric": metric, "friedman_chi2": chi2, "friedman_p": p}
        for other in "ABC":
            diff = (table["D"] - table[other]).to_numpy()
            pw = stats.wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
            row[f"D_vs_{other}_p_bonf"] = min(1.0, 3 * pw)
            row[f"D_vs_{other}_rank_biserial"] = rank_biserial_paired(diff)
        tests.append(row)
    tests = pd.DataFrame(tests)
    tests.to_csv(OUT / "gradcam_statistics.csv", index=False)

    examples = []
    raw_dir = DATASETS / "ModelA_raw" / "images" / "test"
    single = df[(df["model"] == "A") & ~df["gt_classes"].str.contains(";")]
    for name in NAMES:
        pick = single[single["gt_classes"] == name].sort_values("image_id")
        if len(pick):
            examples.append((name, pick.iloc[0]["image_id"]))
    APP_ASSETS.mkdir(parents=True, exist_ok=True)
    tiles, app_items = [], []
    for name, stem in examples:
        raw = cv2.imread(str(next(raw_dir.glob(stem + ".*"))))
        h, w = raw.shape[:2]
        _, _, boxes = gt_mask(DATASETS / "ModelA_raw" / "labels" / "test" / f"{stem}.txt", h, w)
        def cell(image):
            s_ = min(300 / image.shape[1], 240 / image.shape[0])
            small_ = cv2.resize(image, (round(image.shape[1] * s_), round(image.shape[0] * s_)))
            canvas = np.full((240, 300, 3), 255, np.uint8)
            y0, x0 = (240 - small_.shape[0]) // 2, (300 - small_.shape[1]) // 2
            canvas[y0:y0 + small_.shape[0], x0:x0 + small_.shape[1]] = small_
            return cv2.copyMakeBorder(canvas, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255))

        row_imgs = [cell(overlay(raw, np.zeros((h, w), np.float32), boxes))]
        item = {"image_id": stem, "disease": name, "models": {}}
        small = 480 / max(h, w)
        cv2.imwrite(str(APP_ASSETS / f"{stem}_original.jpg"),
                    cv2.resize(raw, (round(w * small), round(h * small))), [cv2.IMWRITE_JPEG_QUALITY, 85])
        for key in MODELS:
            ov = overlay(raw, cams[(key, stem)], boxes)
            row_imgs.append(cell(ov))
            cv2.imwrite(str(APP_ASSETS / f"{stem}_{key}.jpg"),
                        cv2.resize(ov, (round(w * small), round(h * small))), [cv2.IMWRITE_JPEG_QUALITY, 85])
            r = df[(df["model"] == key) & (df["image_id"] == stem)].iloc[0]
            item["models"][key] = {"iou": round(r.gradcam_iou, 3), "energy": round(r.energy_in_boxes, 3)}
        tiles.append(np.hstack(row_imgs))
        app_items.append(item)
    width = tiles[0].shape[1]
    grid = np.vstack(tiles)
    header = np.full((36, width, 3), 255, np.uint8)
    cols = ["Original + lesion boxes"] + [f"Model {k}" for k in MODELS]
    col_w = tiles[0].shape[1] // len(cols)
    for i, text in enumerate(cols):
        cv2.putText(header, text, (i * col_w + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(OUT / "figures" / "gradcam_examples_A_B_C_D.png"), np.vstack([header, grid]))

    app = {"method": "Grad-CAM at the last feature layer of the class head (cv3, one-to-many; P3/P4/P5), "
                     "targets = detections with confidence >= 0.25",
           "iou_threshold": CAM_THRESHOLD, "seed": SEED, "d_weights": D_WEIGHTS,
           "summary": {r.model: {"iou_mean": round(r.gradcam_iou_mean, 3), "energy_mean": round(r.energy_in_boxes_mean, 3),
                                 "iou_I_II": round(r["iou_I-II"], 3), "iou_III_V": round(r["iou_III-V"], 3)}
                       for _, r in summary.iterrows()},
           "statistics": {t.metric: {"friedman_p": round(t.friedman_p, 4),
                                     **{f"D_vs_{o}_p_bonf": round(t[f"D_vs_{o}_p_bonf"], 4) for o in "ABC"},
                                     **{f"D_vs_{o}_r": round(t[f"D_vs_{o}_rank_biserial"], 3) for o in "ABC"}}
                          for _, t in tests.iterrows()},
           "examples": app_items}
    (APP_ASSETS / "gradcam.json").write_text(json.dumps(app, indent=1), encoding="utf-8")

    lines = ["# Grad-CAM results (test set, seed 42)", "",
             f"Heatmap threshold {CAM_THRESHOLD}; IoU against the union of ground-truth lesion boxes. Higher is better.", "",
             "| Model | Grad-CAM IoU (mean) | IoU median | Energy inside boxes | IoU Fitzpatrick I–II | IoU III–V |",
             "|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        lines.append(f"| {r.model} {r['name']} | {r.gradcam_iou_mean:.3f} | {r.gradcam_iou_median:.3f} | "
                     f"{r.energy_in_boxes_mean:.3f} | {r['iou_I-II']:.3f} | {r['iou_III-V']:.3f} |")
    lines += ["", "| Metric | Friedman p | D vs A p (r) | D vs B p (r) | D vs C p (r) |", "|---|---|---|---|---|"]
    for _, t in tests.iterrows():
        cells = " | ".join(f"{t[f'D_vs_{o}_p_bonf']:.3f} ({t[f'D_vs_{o}_rank_biserial']:+.2f})" for o in "ABC")
        lines.append(f"| {t.metric} | {t.friedman_p:.3f} | {cells} |")
    lines += ["", "Figure: figures/gradcam_examples_A_B_C_D.png (first single-disease test image of each class)."]
    (OUT / "GRADCAM_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
