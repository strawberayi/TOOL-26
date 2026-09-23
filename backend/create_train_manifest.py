from pathlib import Path
import csv
import hashlib

TRAIN_DIR = Path(r"C:\TOOL-26-main\Skin Lesion Datasets\train")

OUTPUT = Path(r"C:\TOOL-26-main\backend\train_source_manifest.csv")

CLASS_MAP = {
    "0. CHICKENPOX": "Varicella",
    "1. HFMD": "HFMD",
    "2. MOLLUSCUM": "Molluscum",
    "3. IMPETIGO": "Impetigo",
    "4. RINGWORM": "Tinea corporis",
    "5. AN-AN": "Tinea versicolor",
    "6. WARTS": "Warts",
    "7. ATHLETE_S FOOT": "Tinea pedis",
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".tif",
    ".tiff",
}


def make_id(path):
    return hashlib.sha256(
        str(path.resolve()).encode("utf-8")
    ).hexdigest()[:24]


rows = []

for folder_name, disease in CLASS_MAP.items():

    folder = TRAIN_DIR / folder_name

    if not folder.exists():
        print(f"[WARNING] Missing folder: {folder}")
        continue

    images = [
        p for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    print(f"{disease}: {len(images)} images")

    for image_path in images:

        rows.append({
            "image_id": make_id(image_path),
            "image_path": str(image_path.resolve()),
            "disease_label": disease,
            "source_repository": "LOCAL_TRAIN_DATASET",
            "original_split": "train",
        })


if not rows:
    raise RuntimeError("No training images were found.")


OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "image_id",
            "image_path",
            "disease_label",
            "source_repository",
            "original_split",
        ],
    )

    writer.writeheader()
    writer.writerows(rows)


print()
print("===================================")
print("TRAIN MANIFEST CREATED")
print("===================================")
print(f"Total images: {len(rows)}")
print(f"Output: {OUTPUT}")
print()

for disease in sorted(set(r["disease_label"] for r in rows)):
    count = sum(
        1 for r in rows
        if r["disease_label"] == disease
    )
    print(f"{disease}: {count}")