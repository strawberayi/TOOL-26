"""
Answer the paper's Statement of the Problem with the trained models (no training).

Paper models (all seed 42 for the paired tests; C and D also have seeds 43, 44):
    A  Baseline YOLOv26                         ModelA_raw
    B  RGB CLAHE                                ModelB_rgb_clahe
    C  Fixed L*-CLAHE (global beta)             ModelC2_fixed_l_clahe_global   (C')
    D  ITA-guided adaptive L*-CLAHE (proposed)  ModelD2_lesion_ita             (D2)
       with SOP_MODEL_D=full: D2 + Stage 2 (frozen backbone) + Focal Loss, weights best_ModelD_full_seed*.pt

SOP 1  mAP@50, mAP@50-95 (overall and per disease)
SOP 2  Precision, Recall, F1 (overall and per disease), within-cluster misclassification rate
SOP 3  dAP50 (D - A) for Fitzpatrick I-II (ITA > 41) vs III-V (-30 < ITA <= 41); VI (ITA <= -30) excluded
H01    Friedman test (A, B, C, D) on the 200 matched test images, then Wilcoxon signed-rank
       D vs A, B, C with Bonferroni correction (x3) and rank-biserial correlation
H02    Mann-Whitney U on per-image dAP50 between the two skin-type groups, rank-biserial correlation

    ../TOOL-26/.venv/bin/python final/sop_analysis.py      (from ablation_training/)

Writes final/sop_results/*.csv and final/sop_results/SOP_RESULTS.md
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets" / "ablation_yolov26"
WEIGHTS = ROOT / "weights" / "ablation_split"
# SOP_MODELS=manuscript (default): the manuscript's A-D. SOP_MODELS=supplementary: RGB CLAHE, C', D2.
# SOP_MODEL_D=full with supplementary: D = full D instead of D2.
SET = os.environ.get("SOP_MODELS", "manuscript")
FULL_D = os.environ.get("SOP_MODEL_D", "d2") == "full"
OUT = Path(__file__).resolve().parent / ("sop_results_manuscript" if SET == "manuscript"
                                          else "sop_results_full_D" if FULL_D else "sop_results")

MODELS = {
    "A": ("Baseline YOLOv26", "ModelA_raw", "ModelA_raw"),
    "B": ("RGB CLAHE", "ModelB_rgb_clahe", "ModelB_rgb_clahe"),
    "C": ("Fixed L*-CLAHE (C')", "ModelC2_fixed_l_clahe_global", "ModelC2_fixed_l_clahe_global"),
    "D": ("ITA-guided adaptive L*-CLAHE (D2)", "ModelD2_lesion_ita", "ModelD2_lesion_ita"),
}
if SET == "manuscript":
    MODELS = {
        "A": ("Baseline YOLOv26", "ModelA_raw", "ModelA_raw"),
        "B": ("Fixed L*-CLAHE (β = 2.0)", "ModelC_fixed_l_clahe", "ModelC_fixed_l_clahe"),
        "C": ("Focal Loss Optimization", "ModelA_raw", "ModelC_focal"),
        "D": ("ITA-guided L*-CLAHE + two-stage + Focal Loss (proposed)", "ModelD2_lesion_ita", "ModelD_full"),
    }
    # tune_model_D.ipynb writes the validation-selected Stage 2 setting here.
    _choice = Path(__file__).resolve().parent / "d_tuning_choice.json"
    if _choice.is_file():
        import json
        MODELS["D"] = (MODELS["D"][0], "ModelD2_lesion_ita", json.loads(_choice.read_text())["winner"])
elif FULL_D:
    MODELS["D"] = ("ITA-guided L*-CLAHE + two-stage + Focal Loss (full D)", "ModelD2_lesion_ita", "ModelD_full")
# Seeds with finished weights are used; paired tests always use seed 42.
MULTI_SEED = {k: (42, 43, 44) for k in "ABCD"}
NAMES = ["Warts", "Molluscum", "Varicella", "HFMD", "Tinea versicolor", "Tinea corporis", "Tinea pedis", "Impetigo"]
CLUSTERS = {
    "Vesiculopapular/Eruptive": ["Varicella", "HFMD", "Molluscum", "Impetigo"],
    "Papulosquamous/Verrucous": ["Tinea corporis", "Tinea versicolor", "Warts", "Tinea pedis"],
}
CONF = 0.25  # threshold for per-image P, R, F1 and the confusion matrix (Ultralytics default)
IOUS = np.linspace(0.5, 0.95, 10)
ALPHA = 0.05


# ------------------------------------------------------------------ geometry and AP

def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def average_precision(tp: np.ndarray, conf: np.ndarray, n_gt: int) -> np.ndarray:
    """AP per IoU threshold, computed exactly as Ultralytics does (utils.metrics.compute_ap)."""
    from ultralytics.utils.metrics import compute_ap
    if n_gt == 0:
        return np.full(tp.shape[1], np.nan)
    if len(tp) == 0:
        return np.zeros(tp.shape[1])
    order = np.argsort(-conf, kind="stable")
    tpc = tp[order].cumsum(0)
    fpc = (~tp[order]).cumsum(0)
    recall, precision = tpc / n_gt, tpc / (tpc + fpc)
    return np.array([compute_ap(recall[:, t], precision[:, t])[0] for t in range(tp.shape[1])])


# ------------------------------------------------------------------ predictions

def run_validator(weights: Path, dataset: Path) -> tuple[dict, dict]:
    """Run the official Ultralytics test evaluation and keep every image's matched predictions."""
    from ultralytics.models.yolo.detect import DetectionValidator
    records = {}
    validator = DetectionValidator(args=dict(model=str(weights), data=str(dataset / "data.yaml"), split="test",
                                             batch=8, imgsz=640, rect=True, workers=0, plots=False, verbose=False))
    original = validator._process_batch

    def capture(preds, batch):
        out = original(preds, batch)
        records[Path(batch["im_file"]).stem] = (
            out["tp"], preds["conf"].cpu().numpy(), preds["cls"].cpu().numpy().astype(int),
            preds["bboxes"].cpu().numpy(), batch["cls"].cpu().numpy().astype(int), batch["bboxes"].cpu().numpy())
        return out

    validator._process_batch = capture
    validator()
    box = validator.metrics.box
    official = {"mAP50": box.map50, "mAP50_95": box.map, "precision": box.mp, "recall": box.mr, "per_class": {}}
    for i, c in enumerate(box.ap_class_index):
        official["per_class"][NAMES[c]] = {"precision": box.p[i], "recall": box.r[i], "F1": box.f1[i],
                                           "mAP50": box.ap50[i], "mAP50_95": box.ap[i]}
    return records, official


def evaluate(records: dict, official: dict) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, dict]:
    """Per-image metrics, per-class metrics (official), confusion matrix (pred x true, last = background)."""
    nc = len(NAMES)
    n_gt = np.zeros(nc, int)
    confusion = np.zeros((nc + 1, nc + 1), int)
    per_image, cache = [], {}
    for stem, (tp, pconf, pcls, pbox, gcls, gbox) in sorted(records.items()):
        cache[stem] = (tp[:, :1], pconf, pcls, gcls)
        n_gt += np.bincount(gcls, minlength=nc)

        # Per-image AP: mean over the classes present in the image.
        aps = [average_precision(tp[pcls == c], pconf[pcls == c], int((gcls == c).sum())) for c in np.unique(gcls)]
        ap = np.nanmean(aps, 0) if aps else np.full(len(IOUS), np.nan)

        keep = pconf >= CONF
        tp50 = int(tp[keep, 0].sum())
        fp, fn = int(keep.sum()) - tp50, len(gcls) - tp50
        p = tp50 / (tp50 + fp) if tp50 + fp else (1.0 if fn == 0 else 0.0)
        r = tp50 / len(gcls) if len(gcls) else 1.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        per_image.append({"image_id": stem, "AP50": ap[0], "AP50_95": ap.mean(),
                          "precision": p, "recall": r, "F1": f1})

        # Confusion at CONF, IoU >= 0.5 regardless of class (one-to-one, highest IoU first).
        kb, kc = pbox[keep], pcls[keep]
        ious = iou_matrix(gbox, kb)
        pairs = sorted(((ious[g, k], g, k) for g in range(len(gcls)) for k in range(len(kc)) if ious[g, k] >= 0.5),
                       reverse=True)
        used_g, used_k = set(), set()
        for _, g, k in pairs:
            if g not in used_g and k not in used_k:
                used_g.add(g); used_k.add(k)
                confusion[kc[k], gcls[g]] += 1
        for g in set(range(len(gcls))) - used_g:
            confusion[nc, gcls[g]] += 1
        for k in set(range(len(kc))) - used_k:
            confusion[kc[k], nc] += 1

    per_class = [{"class": name, "instances": int(n_gt[c]), **official["per_class"][name]}
                 for c, name in enumerate(NAMES)]
    return pd.DataFrame(per_image), pd.DataFrame(per_class), confusion, cache


def subset_map50(cache: dict, stems) -> float:
    """Dataset-level mAP@50 on a subset of images (pooled over images, mean over classes present)."""
    tp = np.concatenate([cache[s][0] for s in stems])
    conf = np.concatenate([cache[s][1] for s in stems])
    cls = np.concatenate([cache[s][2] for s in stems])
    n_gt = np.bincount(np.concatenate([cache[s][3] for s in stems]), minlength=len(NAMES))
    aps = [average_precision(tp[cls == c], conf[cls == c], int(n_gt[c]))[0]
           for c in range(len(NAMES)) if n_gt[c]]
    return float(np.mean(aps))


# ------------------------------------------------------------------ statistics

def rank_biserial_paired(diff: np.ndarray) -> float:
    d = diff[diff != 0]
    if len(d) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    return float((ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum())


def cluster_rates(confusion: np.ndarray) -> pd.DataFrame:
    idx = {n: i for i, n in enumerate(NAMES)}
    rows = []
    for cluster, members in CLUSTERS.items():
        ids = [idx[m] for m in members]
        others = [i for i in range(len(NAMES)) if i not in ids]
        gt_total = confusion[:, ids].sum()
        within = sum(confusion[p, t] for t in ids for p in ids if p != t)
        cross = confusion[np.ix_(others, ids)].sum()
        correct = sum(confusion[t, t] for t in ids)
        detected = confusion[:len(NAMES), ids].sum()
        rows.append({"cluster": cluster, "gt_lesions": int(gt_total), "detected": int(detected),
                     "correct": int(correct), "within_cluster_errors": int(within), "cross_cluster_errors": int(cross),
                     "missed": int(confusion[len(NAMES), ids].sum()),
                     "within_cluster_rate_of_detected": within / detected if detected else np.nan,
                     "within_cluster_rate_of_all_gt": within / gt_total if gt_total else np.nan})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ita = pd.read_csv(DATASETS / "ita_table.csv", dtype={"image_id": str})
    ita = ita[ita["split"] == "test"].set_index("image_id")["ita"]

    results = {}
    for key, (label, folder, weights_name) in MODELS.items():
        seeds = [s for s in MULTI_SEED.get(key, (42,)) if (WEIGHTS / f"best_{weights_name}_seed{s}.pt").is_file()]
        for seed in seeds:
            print(f"Model {key} ({label}) seed {seed}: evaluating the 200 test images...", flush=True)
            records, official = run_validator(WEIGHTS / f"best_{weights_name}_seed{seed}.pt", DATASETS / folder)
            per_image, per_class, confusion, cache = evaluate(records, official)
            results[(key, seed)] = {"cache": cache, "official": official, "per_image": per_image,
                                    "per_class": per_class, "confusion": confusion}

    # ---------------- SOP 1 and 2: overall and per-disease metrics
    overall, per_class_rows, cluster_rows = [], [], []
    for (key, seed), r in results.items():
        pc, o = r["per_class"], r["official"]
        overall.append({"model": key, "name": MODELS[key][0], "seed": seed, "mAP50": o["mAP50"],
                        "mAP50_95": o["mAP50_95"], "precision": o["precision"], "recall": o["recall"],
                        "F1": 2 * o["precision"] * o["recall"] / (o["precision"] + o["recall"])})
        per_class_rows.append(pc.assign(model=key, seed=seed))
        cluster_rows.append(cluster_rates(r["confusion"]).assign(model=key, seed=seed))
        pd.DataFrame(r["confusion"], index=[f"pred {n}" for n in NAMES] + ["pred background"],
                     columns=NAMES + ["background (FP)"]).to_csv(OUT / f"confusion_{key}_seed{seed}.csv")
    overall = pd.DataFrame(overall)
    per_class = pd.concat(per_class_rows)
    clusters = pd.concat(cluster_rows)
    overall.to_csv(OUT / "sop1_2_overall_per_seed.csv", index=False)
    per_class.to_csv(OUT / "sop1_2_per_class_per_seed.csv", index=False)
    clusters.to_csv(OUT / "sop2_within_cluster_per_seed.csv", index=False)

    metrics = ["mAP50", "mAP50_95", "precision", "recall", "F1"]
    seed42 = overall[overall["seed"] == 42].set_index("model")[metrics]
    mean_sd = overall.groupby("model")[metrics].agg(["mean", "std"])

    # ---------------- H01: Friedman + Wilcoxon on matched per-image metrics (seed 42)
    img = {k: results[(k, 42)]["per_image"].set_index("image_id") for k in MODELS}
    tests = []
    for metric in ["AP50", "AP50_95", "precision", "recall", "F1"]:
        table = pd.concat({k: img[k][metric] for k in MODELS}, axis=1).dropna()
        chi2, p_f = stats.friedmanchisquare(*[table[k] for k in MODELS])
        row = {"metric": metric, "n_images": len(table), "friedman_chi2": chi2, "friedman_p": p_f,
               **{f"median_{k}": table[k].median() for k in MODELS},
               **{f"mean_{k}": table[k].mean() for k in MODELS}}
        for other in ("A", "B", "C"):
            diff = (table["D"] - table[other]).to_numpy()
            p_w = stats.wilcoxon(diff, zero_method="wilcox").pvalue if np.any(diff != 0) else 1.0
            row[f"D_vs_{other}_p"] = p_w
            row[f"D_vs_{other}_p_bonf"] = min(1.0, p_w * 3)
            row[f"D_vs_{other}_rank_biserial"] = rank_biserial_paired(diff)
        tests.append(row)
    tests = pd.DataFrame(tests)
    tests.to_csv(OUT / "h01_friedman_wilcoxon_per_image.csv", index=False)

    # ---------------- SOP 3 / H02: dAP50 (D - A) by Fitzpatrick group (seed 42)
    groups = pd.Series(np.select([ita > 41, ita > -30], ["I-II", "III-V"], "VI"), index=ita.index)
    groups[ita.isna()] = "no ITA"
    delta = (img["D"]["AP50"] - img["A"]["AP50"]).rename("dAP50").to_frame()
    delta["AP50_A"], delta["AP50_D"] = img["A"]["AP50"], img["D"]["AP50"]
    delta["ITA"], delta["group"] = ita.reindex(delta.index), groups.reindex(delta.index)
    delta.to_csv(OUT / "sop3_per_image_dAP50.csv")
    light = delta.loc[delta["group"] == "I-II", "dAP50"].dropna()
    dark = delta.loc[delta["group"] == "III-V", "dAP50"].dropna()
    u, p_u = stats.mannwhitneyu(dark, light, alternative="two-sided")
    rb_u = 1 - 2 * u / (len(dark) * len(light))  # >0 means III-V improved less; sign flipped below
    rb_u = -rb_u  # positive = III-V gained more than I-II

    rA, rD = results[("A", 42)], results[("D", 42)]
    group_rows = []
    for g in ("I-II", "III-V"):
        stems = delta.index[delta["group"] == g].tolist()
        a, d = subset_map50(rA["cache"], stems), subset_map50(rD["cache"], stems)
        vals = delta.loc[stems, "dAP50"].dropna()
        w = stats.wilcoxon(vals).pvalue if np.any(vals != 0) else 1.0
        group_rows.append({"group": g, "n_images": len(stems), "mAP50_A": a, "mAP50_D": d, "dmAP50": d - a,
                           "mean_per_image_dAP50": vals.mean(), "median_per_image_dAP50": vals.median(),
                           "wilcoxon_D_vs_A_p": w, "rank_biserial_D_vs_A": rank_biserial_paired(vals.to_numpy())})

    # Bootstrap 95% CI of (dmAP50 III-V) - (dmAP50 I-II) at dataset level.
    rng = np.random.default_rng(42)
    boot = []
    light_ids = delta.index[delta["group"] == "I-II"].to_numpy()
    dark_ids = delta.index[delta["group"] == "III-V"].to_numpy()
    for _ in range(2000):
        ls, ds = rng.choice(light_ids, len(light_ids)), rng.choice(dark_ids, len(dark_ids))
        dl = subset_map50(rD["cache"], list(ls)) - subset_map50(rA["cache"], list(ls))
        dd = subset_map50(rD["cache"], list(ds)) - subset_map50(rA["cache"], list(ds))
        boot.append(dd - dl)
    group_df = pd.DataFrame(group_rows)
    group_df.to_csv(OUT / "sop3_group_dAP50.csv", index=False)
    # mAP@50 per skin-type group for every model (seed 42), for the app's Skin-Tone Comparison.
    all_groups = [{"model": k, "group": g, "n_images": int((delta["group"] == g).sum()),
                   "mAP50": subset_map50(results[(k, 42)]["cache"], delta.index[delta["group"] == g].tolist())}
                  for k in MODELS for g in ("I-II", "III-V")]
    pd.DataFrame(all_groups).to_csv(OUT / "sop3_group_map50_all_models.csv", index=False)
    h02 = {"n_I_II": len(light), "n_III_V": len(dark), "median_dAP50_I_II": light.median(),
           "median_dAP50_III_V": dark.median(), "mann_whitney_U": u, "p": p_u, "rank_biserial": rb_u,
           "diff_dmAP50_III_V_minus_I_II": group_df["dmAP50"].iloc[1] - group_df["dmAP50"].iloc[0],
           "bootstrap_ci_low": np.percentile(boot, 2.5), "bootstrap_ci_high": np.percentile(boot, 97.5),
           "excluded_VI": int((delta["group"] == "VI").sum()), "excluded_no_ITA": int((delta["group"] == "no ITA").sum())}
    pd.DataFrame([h02]).to_csv(OUT / "h02_mann_whitney.csv", index=False)

    write_report(seed42, mean_sd, overall.groupby("model")["seed"].count(), per_class, clusters, tests, group_df, h02)
    print(f"\nDone. Results in {OUT}")


# ------------------------------------------------------------------ report

def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def verdict(p: float) -> str:
    return "significant" if p < ALPHA else "not significant"


def write_report(seed42, mean_sd, seed_counts, per_class, clusters, tests, group_df, h02) -> None:
    L = ["# Statement of the Problem: results", "",
         "Test set: 200 images (never used in training or calibration). Models A, B, C, D = "
         + ", ".join(f"{k} {v[0]}" for k, v in MODELS.items()) + ".",
         "mAP, precision, recall and F1 tables are the official Ultralytics test evaluation (P and R at the "
         f"confidence that maximizes F1). Per-image scores for the tests and the confusion use confidence >= {CONF} "
         "and IoU >= 0.5. "
         "Paired tests use seed 42 for every model; models with more finished seeds also show mean ± SD.", ""]

    L += ["## SOP 1: localization (mAP@50, mAP@50-95) and SOP 2: classification (P, R, F1)", "",
          "| Model | mAP@50 | mAP@50-95 | Precision | Recall | F1 |", "|---|---|---|---|---|---|"]
    for k in MODELS:
        s = seed42.loc[k]
        L.append(f"| {k} {MODELS[k][0]} (seed 42) | {pct(s.mAP50)} | {pct(s.mAP50_95)} | {pct(s.precision)} | "
                 f"{pct(s.recall)} | {pct(s.F1)} |")
    for k in MODELS:
        if seed_counts[k] < 2:
            continue
        m = mean_sd.loc[k]
        cells = " | ".join(f"{pct(m[(c, 'mean')])} ± {pct(m[(c, 'std')])}"
                           for c in ["mAP50", "mAP50_95", "precision", "recall", "F1"])
        L.append(f"| {k} {MODELS[k][0]} ({seed_counts[k]} seeds) | {cells} |")

    for metric, title in [("mAP50", "mAP@50"), ("mAP50_95", "mAP@50-95"), ("F1", "F1")]:
        L += ["", f"### Per disease: {title} (seed 42)", "", "| Disease | n | " + " | ".join(MODELS) + " | Best |",
              "|---|---|" + "---|" * (len(MODELS) + 1)]
        pc = per_class[per_class["seed"] == 42]
        for name in NAMES:
            vals = {k: pc[(pc["model"] == k) & (pc["class"] == name)][metric].iloc[0] for k in MODELS}
            n = pc[pc["class"] == name]["instances"].iloc[0]
            best = max(vals, key=vals.get)
            L.append(f"| {name} | {n} | " + " | ".join(pct(v) for v in vals.values()) + f" | {best} |")

    L += ["", "### Within-cluster misclassification rate (seed 42)", "",
          "Share of detected lesions (IoU >= 0.5) assigned to a *different disease in the same cluster*. Lower is better.",
          "", "| Cluster | " + " | ".join(MODELS) + " |", "|---|" + "---|" * len(MODELS)]
    c42 = clusters[clusters["seed"] == 42]
    for cluster in CLUSTERS:
        cells = []
        for k in MODELS:
            row = c42[(c42["model"] == k) & (c42["cluster"] == cluster)].iloc[0]
            cells.append(f"{pct(row.within_cluster_rate_of_detected)}% ({row.within_cluster_errors}/{row.detected})")
        L.append(f"| {cluster} | " + " | ".join(cells) + " |")

    L += ["", "## H01: difference among Models A, B, C, D", "",
          "Friedman test on the 200 matched test images (per-image score), then Wilcoxon signed-rank D vs A, B, C "
          "(Bonferroni-corrected p = p × 3) with rank-biserial correlation r (positive = D better).", "",
          "| Metric | Friedman χ² | p | D vs A p (r) | D vs B p (r) | D vs C p (r) | Decision |",
          "|---|---|---|---|---|---|---|"]
    for _, t in tests.iterrows():
        pw = " | ".join(f"{t[f'D_vs_{o}_p_bonf']:.3f} ({t[f'D_vs_{o}_rank_biserial']:+.2f})" for o in "ABC")
        L.append(f"| {t.metric} | {t.friedman_chi2:.2f} | {t.friedman_p:.3f} | {pw} | "
                 f"{'reject H01' if t.friedman_p < ALPHA else 'fail to reject H01'} |")

    L += ["", "## SOP 3 / H02: ΔAP50 (D − A) by skin type", "",
          "Fitzpatrick proxy from ITA: I–II = ITA > 41, III–V = −30 < ITA ≤ 41; "
          f"VI (ITA ≤ −30, {h02['excluded_VI']} images) and images without ITA ({h02['excluded_no_ITA']}) excluded.", "",
          "| Group | Images | mAP@50 A | mAP@50 D | ΔmAP@50 | Median per-image ΔAP50 | Wilcoxon D vs A p (r) |",
          "|---|---|---|---|---|---|---|"]
    for _, g in group_df.iterrows():
        L.append(f"| {g.group} | {g.n_images} | {pct(g.mAP50_A)} | {pct(g.mAP50_D)} | {100 * g.dmAP50:+.1f} | "
                 f"{100 * g.median_per_image_dAP50:+.1f} | {g.wilcoxon_D_vs_A_p:.3f} ({g.rank_biserial_D_vs_A:+.2f}) |")
    L += ["", f"Mann-Whitney U (per-image ΔAP50, III–V vs I–II): U = {h02['mann_whitney_U']:.0f}, "
          f"p = {h02['p']:.3f}, rank-biserial r = {h02['rank_biserial']:+.2f} (positive = III–V gained more). "
          f"Difference in ΔmAP@50 (III–V − I–II) = {100 * h02['diff_dmAP50_III_V_minus_I_II']:+.1f} points, "
          f"bootstrap 95% CI [{100 * h02['bootstrap_ci_low']:+.1f}, {100 * h02['bootstrap_ci_high']:+.1f}]. "
          f"Decision: {'reject H02' if h02['p'] < ALPHA else 'fail to reject H02'} ({verdict(h02['p'])}).", ""]
    (OUT / "SOP_RESULTS.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
