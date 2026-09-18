import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
SPLITTING_DIR = PROJECT_DIR / "SPLITTING"
ANNOTATED_DIR = PROJECT_DIR / "ANNOTATED"
OUTPUT_DIR = PROJECT_DIR / "micro_dataset_B"

IMAGES_PER_CLASS = 10
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

CLASSES = [
    ("70% AF", "7. Athlete_s foot", "AF", 0),
    ("70% AN-AN", "5. An-an", "AN_AN", 1),
    ("70% HFMD", "1. HFMD", "HFMD", 2),
    ("70% MOLLUSCUM", "2. Molluscum", "MOLLUSCUM", 3),
    ("70% ringworm", "4. Ringworm", "RINGWORM", 4),
]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

IMAGE_OUT = OUTPUT_DIR / "images"
LABEL_OUT = OUTPUT_DIR / "labels"
IMAGE_OUT.mkdir(parents=True, exist_ok=True)
LABEL_OUT.mkdir(parents=True, exist_ok=True)

try:
    import cv2
except ImportError:
    print("ERROR: OpenCV is not installed.")
    print("Run: python -m pip install opencv-python")
    raise SystemExit(1)

def find_json(img, folder):
    exact = folder / f"{img.stem}.json"
    if exact.exists():
        return exact
    target = img.stem.lower()
    for p in folder.rglob("*.json"):
        if p.stem.lower() == target:
            return p
    return None

def box_to_yolo(points, w, h, class_id):
    if not isinstance(points, list) or len(points) < 2:
        return None
    x1, y1 = float(points[0][0]), float(points[0][1])
    x2, y2 = float(points[1][0]), float(points[1][1])
    left, right = max(0, min(x1, x2)), min(w, max(x1, x2))
    top, bottom = max(0, min(y1, y2)), min(h, max(y1, y2))
    bw, bh = right - left, bottom - top
    if bw <= 0 or bh <= 0:
        return None
    cx = (left + right) / 2 / w
    cy = (top + bottom) / 2 / h
    return f"{class_id} {cx:.6f} {cy:.6f} {bw/w:.6f} {bh/h:.6f}"

def convert_json(json_path, w, h, class_id):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    lines = []
    for shape in data.get("shapes", []):
        if str(shape.get("shape_type", "")).lower() != "rectangle":
            continue
        line = box_to_yolo(shape.get("points", []), w, h, class_id)
        if line:
            lines.append(line)
    return lines

print("=" * 70)
print("YOLOv26 OPTION B - FIXED L*-CLAHE MICRO-DATASET")
print("=" * 70)
print(f"CLAHE clip limit: {CLAHE_CLIP_LIMIT}")
print(f"Output: {OUTPUT_DIR}")

total = 0
for split_name, ann_name, class_name, class_id in CLASSES:
    split = SPLITTING_DIR / split_name
    ann = ANNOTATED_DIR / ann_name
    print("\n" + "-" * 70)
    print(f"CLASS: {class_name}")
    print(f"Images: {split}")
    print(f"JSON:   {ann}")

    if not split.exists():
        print("[ERROR] 70% folder not found.")
        continue
    if not ann.exists():
        print("[ERROR] annotation folder not found.")
        continue

    images = sorted(p for p in split.rglob("*")
                    if p.is_file() and p.suffix.lower() in EXTS)
    made = 0

    for img_path in images:
        if made >= IMAGES_PER_CLASS:
            break
        json_path = find_json(img_path, ann)
        if json_path is None:
            print(f"[NO JSON] {img_path.name}")
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[SKIP] Cannot read {img_path.name}")
            continue

        h, w = image.shape[:2]
        labels = convert_json(json_path, w, h, class_id)
        if not labels:
            print(f"[SKIP] No valid rectangle in {json_path.name}")
            continue

        # Option B: fixed L*-CLAHE, clipLimit = 2.0
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=CLAHE_TILE_GRID_SIZE
        )
        l2 = clahe.apply(l)
        enhanced = cv2.cvtColor(
            cv2.merge([l2, a, b]),
            cv2.COLOR_LAB2BGR
        )

        stem = f"{class_name}_{made+1:03d}"
        out_img = IMAGE_OUT / f"{stem}.jpg"
        out_txt = LABEL_OUT / f"{stem}.txt"

        cv2.imwrite(
            str(out_img), enhanced,
            [cv2.IMWRITE_JPEG_QUALITY, 95]
        )
        out_txt.write_text("\n".join(labels) + "\n", encoding="utf-8")

        made += 1
        total += 1
        print(f"[OK] {img_path.name} + {json_path.name} -> {stem}")

    print(f"Created: {made}/{IMAGES_PER_CLASS}")

yaml = OUTPUT_DIR / "data.yaml"
yaml.write_text(
    f"""path: {OUTPUT_DIR.as_posix()}
train: images
val: images

names:
  0: AF
  1: AN_AN
  2: HFMD
  3: MOLLUSCUM
  4: RINGWORM
""",
    encoding="utf-8"
)

print("\n" + "=" * 70)
print("FINAL RESULT")
print("=" * 70)
print(f"Total image/label pairs created: {total}")
print(f"Expected: {len(CLASSES) * IMAGES_PER_CLASS}")
print(f"Output: {OUTPUT_DIR}")

if total == len(CLASSES) * IMAGES_PER_CLASS:
    print("SUCCESS - Option B micro-dataset is ready.")
else:
    print("NOT COMPLETE - do not train yet.")
