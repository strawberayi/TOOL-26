import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics.data.augment import LetterBox

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT.parent / "ablation_training" / "datasets" / "ablation_yolov26"
RAW = DATA / "ModelA_raw"
ENHANCED = DATA / "ModelD2_lesion_ita"
MODEL = ROOT / "frontend" / "models" / "D.onnx"
OUTPUT = ROOT / "frontend" / "assets" / "morphology_reference.json"
MAX_SIDE = 1280
CLASSES = ["Warts", "Molluscum", "Varicella", "HFMD", "Tinea versicolor", "Tinea corporis", "Tinea pedis", "Impetigo"]
CLUSTERS = [["Varicella", "HFMD", "Molluscum", "Impetigo"], ["Tinea corporis", "Tinea versicolor", "Warts", "Tinea pedis"]]


def py_round(x):
    return int(round(x))


def resize_max(rgb):
    h, w = rgb.shape[:2]
    scale = min(1.0, MAX_SIDE / max(h, w))
    if scale >= 1:
        return rgb
    return cv2.resize(rgb, (max(1, py_round(w * scale)), max(1, py_round(h * scale))), interpolation=cv2.INTER_AREA)


def measure(rgb, boxes):
    h, w = rgb.shape[:2]
    inside = np.zeros((h, w), bool)
    outer = np.zeros((h, w), bool)
    for x1, y1, x2, y2 in boxes:
        x1, y1, x2, y2 = (py_round(v) for v in (x1 * w, y1 * h, x2 * w, y2 * h))
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        inside[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = True
        outer[max(0, py_round(y1 - bh / 4)):min(h, py_round(y2 + bh / 4)),
              max(0, py_round(x1 - bw / 4)):min(w, py_round(x2 + bw / 4))] = True
    ring = outer & ~inside
    if not inside.any() or not ring.any():
        return None
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab).astype(np.float64)
    L, a, b = lab[..., 0] * 100 / 255, lab[..., 1] - 128, lab[..., 2] - 128
    lap = cv2.Laplacian(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.CV_32F).astype(np.float64)
    rough_in, rough_ring = lap[inside].std(), lap[ring].std()
    dL, da, db = (c[inside].mean() - c[ring].mean() for c in (L, a, b))
    return {
        "texture": rough_in / rough_ring if rough_ring > 0 else None,
        "crust": db,
        "edge": float(np.sqrt(dL ** 2 + da ** 2 + db ** 2)),
    }


def yolo_boxes(label_file):
    boxes = []
    for line in label_file.read_text().splitlines():
        parts = line.split()
        if len(parts) == 5:
            _, cx, cy, bw, bh = map(float, parts)
            boxes.append((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
    return boxes


def margin(session, bgr):
    x = LetterBox((640, 640), auto=True, stride=32)(image=bgr)[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255
    out = session.run(None, {session.get_inputs()[0].name: np.ascontiguousarray(x)})[0][0]
    scores = out[4:]
    j = int(scores.max(axis=0).argmax())
    s = scores[:, j]
    top = int(s.argmax())
    if s[top] < 0.25:
        return None
    cluster = next(c for c in CLUSTERS if CLASSES[top] in c)
    rest = sorted((s[CLASSES.index(c)] for c in cluster if c != CLASSES[top]), reverse=True)
    return float(s[top] - rest[0])


def quantiles(values):
    v = np.array([x for x in values if x is not None and np.isfinite(x)])
    return {"n": int(len(v)), "q": [round(float(x), 4) for x in np.quantile(v, np.linspace(0, 1, 101))]}


def main():
    session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    feats = {"texture": [], "crust": [], "edge": [], "differential": []}
    for image in sorted((RAW / "images" / "train").iterdir()):
        boxes = yolo_boxes(RAW / "labels" / "train" / f"{image.stem}.txt")
        if not boxes:
            continue
        rgb = resize_max(cv2.cvtColor(cv2.imread(str(image)), cv2.COLOR_BGR2RGB))
        m = measure(rgb, boxes)
        if m:
            for k in ("texture", "crust", "edge"):
                feats[k].append(m[k])
        enhanced = next((ENHANCED / "images" / "train").glob(f"{image.stem}.*"), None)
        if enhanced is not None:
            feats["differential"].append(margin(session, cv2.imread(str(enhanced))))
    payload = {
        "source": "Training set (1370 images): labelled lesion boxes for texture, crust and edge; Model D's top box for the differential key",
        "features": {k: quantiles(v) for k, v in feats.items()},
    }
    OUTPUT.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print({k: (v["n"], v["q"][50]) for k, v in payload["features"].items()})


if __name__ == "__main__":
    main()
