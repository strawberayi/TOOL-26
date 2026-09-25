"""
Build the YOLO detection dataset used by the YOLOv26 ablation notebook.

Source of truth is the leakage-safe cleaned split manifest produced by
``run_member1_batch.py prepare-manifest``. The same split therefore feeds
Phase 0 calibration (train only) and YOLO training/validation/testing.

Images are the standardized lossless PNGs (EXIF orientation applied,
ICC converted to sRGB). Every ablation model starts from these identical
pixels, so preprocessing is the only experimental variable.

AnyLabeling/LabelMe JSON files are matched to images by filename stem
within the same disease. A match is accepted only when the JSON
imageWidth/imageHeight equal the standardized image size; several
archives reuse filenames such as ``An-an(14).jpg`` for different photos,
so a name-only match is not trustworthy. Nothing is matched by image size
alone.

Usage:

    python backend/build_yolo_dataset.py \\
        --manifest phase0_work/prepared/cleaned_split_manifest.csv \\
        --annotations phase0_work/archive_inventory/archive_contents/annotations/ANNOTATED \\
        --annotations "phase0_work/combined_inventory/supplemental_archive_contents/2. Annotated" \\
        --output datasets/source_yolo
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

# Canonical thesis order. Class ids follow this order after empty classes
# are removed.
CLASS_ORDER = [
    "Warts",
    "Molluscum",
    "Varicella",
    "HFMD",
    "Tinea versicolor",
    "Tinea corporis",
    "Tinea pedis",
    "Impetigo",
]

# Annotation shape labels accepted for each disease (lower-case).
LABEL_ALIASES = {
    "Warts": {"warts", "wart"},
    "Molluscum": {"molluscum"},
    "Varicella": {"chickenpox", "varicella"},
    "HFMD": {"hfmd"},
    "Tinea versicolor": {"an-an", "tinea versicolor"},
    "Tinea corporis": {"ringworm", "tinea corporis"},
    "Tinea pedis": {"athlete's foot", "athletes foot", "tinea pedis"},
    "Impetigo": {"impetigo"},
}

SPLIT_NAMES = {"train": "train", "validation": "val", "test_a": "test"}


def index_annotations(roots: list[Path]) -> dict[str, list[dict]]:
    index: dict[str, list[dict]] = defaultdict(list)
    for root in roots:
        for path in sorted(root.rglob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            index[path.stem.lower()].append({
                "path": path,
                "width": data.get("imageWidth"),
                "height": data.get("imageHeight"),
                "shapes": data.get("shapes", []),
            })
    return index


def shapes_to_yolo(shapes: list[dict], width: int, height: int, disease: str, class_id: int):
    """Return (yolo lines, number of shapes rejected for a foreign label)."""
    lines, foreign = [], 0
    for shape in shapes:
        if str(shape.get("label", "")).strip().lower() not in LABEL_ALIASES[disease]:
            foreign += 1
            continue
        points = shape.get("points") or []
        if str(shape.get("shape_type", "")).lower() not in {"rectangle", "polygon"} or len(points) < 2:
            continue
        xs = [min(max(float(p[0]), 0.0), width) for p in points]
        ys = [min(max(float(p[1]), 0.0), height) for p in points]
        left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
        if right - left <= 0 or bottom - top <= 0:
            continue
        lines.append(
            f"{class_id} {(left + right) / 2 / width:.6f} {(top + bottom) / 2 / height:.6f} "
            f"{(right - left) / width:.6f} {(bottom - top) / height:.6f}"
        )
    return lines, foreign


def load_annotation(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {"path": path, "width": data.get("imageWidth"), "height": data.get("imageHeight"),
            "shapes": data.get("shapes", [])}


def match_annotation(row: dict, index: dict[str, list[dict]]) -> tuple[str, dict | None]:
    # A manifest that already pairs each image with its JSON (e.g. from
    # build_team_source_manifest.py) is used directly; the size check still applies.
    supplied = row.get("annotation_path", "").strip()
    candidates = (
        [load_annotation(Path(supplied))] if supplied and Path(supplied).is_file()
        else index.get(Path(row["image_path"]).stem.lower(), [])
    )
    if not candidates:
        return "NO_ANNOTATION", None
    width, height = Image.open(row["processed_image_path"]).size
    sized = [c for c in candidates if (c["width"], c["height"]) == (width, height)]
    if not sized:
        return "SIZE_MISMATCH", None
    distinct = {json.dumps(c["shapes"], sort_keys=True) for c in sized}
    if len(distinct) > 1:
        return "CONFLICTING_ANNOTATIONS", None
    return "MATCHED", sized[0]


def build(manifest: Path, annotation_roots: list[Path], output: Path, min_train_images: int) -> dict:
    rows = [
        row for row in csv.DictReader(manifest.open(encoding="utf-8"))
        if row.get("split") in SPLIT_NAMES
    ]
    if not rows:
        raise ValueError(f"No train/validation/test_a rows in {manifest}")

    # A stem shared by several retained images of one disease cannot be
    # matched unambiguously.
    stem_counts = Counter(
        (row["disease_label"], Path(row["image_path"]).stem.lower()) for row in rows
    )
    index = index_annotations(annotation_roots)

    matched = []
    records = []
    for row in rows:
        key = (row["disease_label"], Path(row["image_path"]).stem.lower())
        if stem_counts[key] > 1 and not row.get("annotation_path", "").strip():
            status, annotation = "DUPLICATE_IMAGE_STEM", None
        else:
            status, annotation = match_annotation(row, index)
        record = {
            "image_id": row["image_id"],
            "split": SPLIT_NAMES[row["split"]],
            "disease_label": row["disease_label"],
            "source_image": row["processed_image_path"],
            "raw_image_path": row["image_path"],
            "annotation_path": str(annotation["path"]) if annotation else "",
            "match_status": status,
        }
        records.append(record)
        if annotation:
            matched.append((record, annotation))

    train_counts = Counter(r["disease_label"] for r, _ in matched if r["split"] == "train")
    classes = [name for name in CLASS_ORDER if train_counts[name] >= min_train_images]
    excluded = {name: train_counts[name] for name in CLASS_ORDER if name not in classes}
    class_ids = {name: i for i, name in enumerate(classes)}

    if output.exists():
        shutil.rmtree(output)
    for split in SPLIT_NAMES.values():
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)

    for record, annotation in matched:
        disease = record["disease_label"]
        if disease not in class_ids:
            record["match_status"] = "CLASS_EXCLUDED_TOO_FEW_TRAIN_LABELS"
            continue
        lines, foreign = shapes_to_yolo(
            annotation["shapes"], annotation["width"], annotation["height"],
            disease, class_ids[disease],
        )
        record["foreign_label_shapes"] = foreign
        if not lines:
            record["match_status"] = "NO_VALID_BOXES"
            continue
        split = record["split"]
        image_target = output / "images" / split / f"{record['image_id']}.png"
        shutil.copy2(record["source_image"], image_target)
        (output / "labels" / split / f"{record['image_id']}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        record.update({
            "match_status": "INCLUDED",
            "class_id": class_ids[disease],
            "boxes": len(lines),
            "yolo_image": str(image_target.resolve()),
        })

    included = [r for r in records if r["match_status"] == "INCLUDED"]
    fields = [
        "image_id", "split", "disease_label", "class_id", "boxes", "match_status",
        "foreign_label_shapes", "source_image", "raw_image_path", "annotation_path", "yolo_image",
    ]
    with (output / "source_yolo_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    split_counts = Counter(r["split"] for r in included)
    summary = {
        "source_manifest": str(manifest.resolve()),
        "annotation_roots": [str(root.resolve()) for root in annotation_roots],
        "classes": classes,
        "excluded_classes_train_label_counts": excluded,
        "min_train_images_per_class": min_train_images,
        "included_by_split": dict(split_counts),
        "included_split_shares": {
            k: round(v / max(1, len(included)), 3) for k, v in split_counts.items()
        },
        "included_by_split_and_class": {
            split: dict(Counter(r["disease_label"] for r in included if r["split"] == split))
            for split in SPLIT_NAMES.values()
        },
        "status_by_class": {
            name: dict(Counter(r["match_status"] for r in records if r["disease_label"] == name))
            for name in CLASS_ORDER
        },
    }
    (output / "classes.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, action="append", default=[],
                        help="Folders searched for JSON when the manifest has no annotation_path.")
    parser.add_argument("--output", type=Path, default=Path("datasets/source_yolo"))
    parser.add_argument("--min-train-images", type=int, default=30,
                        help="Classes with fewer matched training images are excluded (default 30).")
    args = parser.parse_args()
    summary = build(args.manifest, args.annotations, args.output, args.min_train_images)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
