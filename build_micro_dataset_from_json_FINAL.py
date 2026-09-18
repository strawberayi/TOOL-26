import json
import shutil
from pathlib import Path

# ============================================================
# YOLOv26 DRY-RUN MICRO DATASET BUILDER
# ============================================================
# Project:
# C:\TOOL-26-main\
#
# Uses:
#   SPLITTING\70% ...       -> training images
#   ANNOTATED\...           -> AnyLabeling JSON annotations
#
# Creates:
#   micro_dataset\
#       images\
#       labels\
#
# It selects 10 correctly matched image + JSON pairs per class.
# It converts AnyLabeling rectangle points into YOLO .txt format.
#
# IMPORTANT:
# - Original images and JSON files are NOT modified.
# - Only the 70% training folders are used.
# - No CLAHE, K-Means, ITA, beta search, or augmentation is done.
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

SPLITTING_DIR = PROJECT_DIR / "SPLITTING"
ANNOTATED_DIR = PROJECT_DIR / "ANNOTATED"
OUTPUT_DIR = PROJECT_DIR / "micro_dataset"

IMAGES_PER_CLASS = 10

# YOLO class IDs for the 5-class dry run
# 0 = AF
# 1 = AN-AN
# 2 = HFMD
# 3 = MOLLUSCUM
# 4 = RINGWORM
CLASSES = [
    {
        "split_folder": "70% AF",
        "annotated_folder": "7. Athlete_s foot",
        "name": "AF",
        "class_id": 0,
    },
    {
        "split_folder": "70% AN-AN",
        "annotated_folder": "5. An-an",
        "name": "AN_AN",
        "class_id": 1,
    },
    {
        "split_folder": "70% HFMD",
        "annotated_folder": "1. HFMD",
        "name": "HFMD",
        "class_id": 2,
    },
    {
        "split_folder": "70% MOLLUSCUM",
        "annotated_folder": "2. Molluscum",
        "name": "MOLLUSCUM",
        "class_id": 3,
    },
    {
        "split_folder": "70% ringworm",
        "annotated_folder": "4. Ringworm",
        "name": "RINGWORM",
        "class_id": 4,
    },
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

IMAGE_OUT = OUTPUT_DIR / "images"
LABEL_OUT = OUTPUT_DIR / "labels"

IMAGE_OUT.mkdir(parents=True, exist_ok=True)
LABEL_OUT.mkdir(parents=True, exist_ok=True)


def load_json(json_path):
    """Load AnyLabeling/LabelMe-style JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_image_size(image_path):
    """Get image width and height using OpenCV."""
    try:
        import cv2
    except ImportError:
        print("\nERROR: OpenCV is not installed.")
        print("Run: python -m pip install opencv-python")
        raise SystemExit(1)

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    height, width = image.shape[:2]
    return width, height


def find_matching_json(image_path, annotated_folder):
    """
    Match image stem to JSON stem.

    Example:
      HFMD(93).jpg -> HFMD(93).json
    """
    json_path = annotated_folder / f"{image_path.stem}.json"

    if json_path.exists():
        return json_path

    # Case-insensitive fallback
    target = image_path.stem.lower()
    for candidate in annotated_folder.rglob("*.json"):
        if candidate.stem.lower() == target:
            return candidate

    return None


def rectangle_to_yolo(points, image_width, image_height, class_id):
    """
    Convert AnyLabeling rectangle:
        [[x1,y1], [x2,y2]]
    into YOLO:
        class_id center_x center_y width height

    Coordinates are normalized to 0-1.
    """

    if len(points) < 2:
        return None

    x1 = float(points[0][0])
    y1 = float(points[0][1])
    x2 = float(points[1][0])
    y2 = float(points[1][1])

    left = min(x1, x2)
    right = max(x1, x2)
    top = min(y1, y2)
    bottom = max(y1, y2)

    # Clamp to image boundaries
    left = max(0.0, min(left, float(image_width)))
    right = max(0.0, min(right, float(image_width)))
    top = max(0.0, min(top, float(image_height)))
    bottom = max(0.0, min(bottom, float(image_height)))

    box_width = right - left
    box_height = bottom - top

    if box_width <= 0 or box_height <= 0:
        return None

    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0

    # Normalize
    center_x /= image_width
    center_y /= image_height
    box_width /= image_width
    box_height /= image_height

    return (
        f"{class_id} "
        f"{center_x:.6f} "
        f"{center_y:.6f} "
        f"{box_width:.6f} "
        f"{box_height:.6f}"
    )


def convert_json_to_yolo(json_path, image_width, image_height, class_id):
    """Convert all rectangle shapes in one AnyLabeling JSON."""
    data = load_json(json_path)

    shapes = data.get("shapes", [])

    yolo_lines = []

    for shape in shapes:

        shape_type = str(
            shape.get("shape_type", "")
        ).lower()

        # Your screenshot shows shape_type = rectangle.
        if shape_type != "rectangle":
            continue

        points = shape.get("points", [])

        result = rectangle_to_yolo(
            points,
            image_width,
            image_height,
            class_id
        )

        if result is not None:
            yolo_lines.append(result)

    return yolo_lines


print("=" * 70)
print("YOLOv26 MICRO-DATASET BUILDER")
print("=" * 70)

print(f"\nProject : {PROJECT_DIR}")
print(f"Images  : {SPLITTING_DIR}")
print(f"JSON    : {ANNOTATED_DIR}")
print(f"Output  : {OUTPUT_DIR}")

if not SPLITTING_DIR.exists():
    print(f"\nERROR: Missing folder:\n{SPLITTING_DIR}")
    raise SystemExit(1)

if not ANNOTATED_DIR.exists():
    print(f"\nERROR: Missing folder:\n{ANNOTATED_DIR}")
    raise SystemExit(1)


total_created = 0
total_missing_json = 0
total_bad_json = 0
class_counts = {}

# ============================================================
# PROCESS EACH CLASS
# ============================================================

for config in CLASSES:

    split_folder = SPLITTING_DIR / config["split_folder"]
    annotated_folder = ANNOTATED_DIR / config["annotated_folder"]

    class_name = config["name"]
    class_id = config["class_id"]

    print("\n" + "-" * 70)
    print(f"CLASS: {class_name}")
    print(f"Image folder: {split_folder}")
    print(f"JSON folder : {annotated_folder}")

    if not split_folder.exists():
        print("[ERROR] 70% image folder not found.")
        class_counts[class_name] = 0
        continue

    if not annotated_folder.exists():
        print("[ERROR] Annotated JSON folder not found.")
        class_counts[class_name] = 0
        continue

    image_files = sorted(
        p for p in split_folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    print(f"Images found: {len(image_files)}")

    created_for_class = 0

    for image_path in image_files:

        if created_for_class >= IMAGES_PER_CLASS:
            break

        json_path = find_matching_json(
            image_path,
            annotated_folder
        )

        if json_path is None:
            total_missing_json += 1
            continue

        try:
            width, height = get_image_size(image_path)

            yolo_lines = convert_json_to_yolo(
                json_path,
                width,
                height,
                class_id
            )

            if not yolo_lines:
                total_bad_json += 1
                print(
                    f"[SKIP] No rectangle annotation: "
                    f"{image_path.name}"
                )
                continue

            new_stem = (
                f"{class_name}_{created_for_class + 1:03d}"
            )

            new_image_path = (
                IMAGE_OUT
                / f"{new_stem}{image_path.suffix.lower()}"
            )

            new_label_path = (
                LABEL_OUT
                / f"{new_stem}.txt"
            )

            shutil.copy2(
                image_path,
                new_image_path
            )

            with open(
                new_label_path,
                "w",
                encoding="utf-8"
            ) as f:
                f.write(
                    "\n".join(yolo_lines)
                    + "\n"
                )

            created_for_class += 1
            total_created += 1

            print(
                f"[OK] {image_path.name}"
                f" + {json_path.name}"
                f" -> {new_stem}"
            )

        except Exception as error:
            total_bad_json += 1
            print(
                f"[SKIP] {image_path.name} "
                f"because: {error}"
            )

    class_counts[class_name] = created_for_class

    print(
        f"Created for {class_name}: "
        f"{created_for_class}/{IMAGES_PER_CLASS}"
    )


# ============================================================
# CREATE DATASET YAML
# ============================================================

yaml_path = OUTPUT_DIR / "data.yaml"

yaml_text = f"""path: {OUTPUT_DIR.as_posix()}
train: images
val: images

names:
  0: AF
  1: AN_AN
  2: HFMD
  3: MOLLUSCUM
  4: RINGWORM
"""

with open(
    yaml_path,
    "w",
    encoding="utf-8"
) as f:
    f.write(yaml_text)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL RESULT")
print("=" * 70)

for class_name, count in class_counts.items():
    print(
        f"{class_name:12s}: "
        f"{count}/{IMAGES_PER_CLASS}"
    )

print("-" * 70)
print(f"Total images created : {total_created}")
print(f"Missing JSON matches : {total_missing_json}")
print(f"Bad/empty JSON       : {total_bad_json}")

print("\nOutput:")
print(OUTPUT_DIR)

print("\nImages:")
print(IMAGE_OUT)

print("\nLabels:")
print(LABEL_OUT)

print("\nYOLO YAML:")
print(yaml_path)

expected = len(CLASSES) * IMAGES_PER_CLASS

if total_created == expected:
    print("\nSUCCESS!")
    print(
        f"You now have {expected} image + YOLO label pairs."
    )
    print(
        "The micro-dataset is ready for the YOLOv26 "
        "50-epoch dry run."
    )
else:
    print("\nNOT YET COMPLETE.")
    print(
        f"Expected: {expected} pairs"
    )
    print(
        f"Created : {total_created} pairs"
    )
    print(
        "Do NOT start training yet."
    )
    print(
        "Check the messages above for missing JSON "
        "or annotation problems."
    )

print("\nIMPORTANT:")
print("- Original ANNOTATED JSON files were not modified.")
print("- Original SPLITTING images were not modified.")
print("- Only 70% folders were used.")
print("- JSON rectangle coordinates were converted to YOLO format.")
print("- No CLAHE, ITA, K-Means, beta search, or augmentation was used.")
