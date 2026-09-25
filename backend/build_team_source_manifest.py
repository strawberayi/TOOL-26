"""
Build the source manifest for the team's annotated dataset archive
("Data Set-*.zip": "1. Filtered" images + "2. Annotated" AnyLabeling JSON).

Only images with a verified annotation are listed: the JSON imagePath (or
file name) must match the image file name within the same class folder AND
the JSON imageWidth/imageHeight must equal the EXIF-oriented image size.
Everything else is written to a separate exclusion log with the reason.

The team's "3. Stratefied-Split" folder is intentionally NOT used: it has
exact and near-duplicate images across train/val/test. The output of this
script goes to ``run_member1_batch.py prepare-split-manifest``, which removes
duplicates before creating the 70/20/10 split.

Usage:

    python backend/build_team_source_manifest.py \\
        --dataset-root "phase0_work/team_dataset/Data Set" \\
        --source-repository "TEAM_DATASET:Data Set-20260925T122828Z-1-001.zip" \\
        --output phase0_work/team_inventory/team_source_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps

# Class folders are numbered identically in "1. Filtered" and "2. Annotated".
FOLDER_PREFIX_TO_DISEASE = {
    "0": "Varicella",
    "1": "HFMD",
    "2": "Molluscum",
    "3": "Impetigo",
    "4": "Tinea corporis",
    "5": "Tinea versicolor",
    "6": "Warts",
    "7": "Tinea pedis",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def disease_for(folder: str) -> str:
    return FOLDER_PREFIX_TO_DISEASE[folder.strip()[0]]


def oriented_size(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as image:
            return ImageOps.exif_transpose(image).size
    except Exception:
        return None


def read_manual_exclusions(path: Path | None) -> dict[str, str]:
    """Map "1. Filtered"-relative image path -> documented reason."""
    if path is None:
        return {}
    with path.open(encoding="utf-8") as handle:
        return {row["image"].strip(): row["reason"].strip() for row in csv.DictReader(handle)}


def build(dataset_root: Path, source_repository: str, output: Path, manual_exclusions: Path | None = None) -> dict:
    filtered = dataset_root / "1. Filtered"
    annotated = dataset_root / "2. Annotated"

    images: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for path in sorted(filtered.rglob("*")):
        if path.suffix.lower() in IMAGE_SUFFIXES:
            disease = disease_for(path.relative_to(filtered).parts[0])
            images[(disease, path.stem.lower())].append(path)

    manual = read_manual_exclusions(manual_exclusions)
    rows, excluded, used = [], [], set()
    for json_path in sorted(annotated.rglob("*.json")):
        disease = disease_for(json_path.relative_to(annotated).parts[0])
        data = json.loads(json_path.read_text(encoding="utf-8"))
        stem = Path(str(data.get("imagePath", "")).replace("\\", "/")).stem.lower()
        candidates = images.get((disease, stem)) or images.get((disease, json_path.stem.lower()), [])
        # Tinea pedis reuses file names across dataset1 / dataset1-(fork) /
        # dataset2, so prefer the image in the same sub-folder as the JSON.
        subfolder = json_path.parent.relative_to(annotated).parts[1:]
        same_folder = [p for p in candidates if p.parent.relative_to(filtered).parts[1:] == subfolder]
        candidates = same_folder or candidates
        size = (data.get("imageWidth"), data.get("imageHeight"))
        matched = [p for p in candidates if oriented_size(p) == size]
        if not candidates:
            reason = "NO_IMAGE"
        elif not matched:
            reason = "UNREADABLE_OR_SIZE_MISMATCH"
        elif len(matched) > 1:
            reason = "AMBIGUOUS_IMAGE"
        elif matched[0] in used:
            reason = "IMAGE_ALREADY_MATCHED"
        else:
            used.add(matched[0])
            relative = str(matched[0].relative_to(filtered))
            rows.append({
                "image_path": str(matched[0].resolve()),
                "disease_label": disease,
                "source_repository": source_repository,
                "annotation_path": str(json_path.resolve()),
                "annotation_match_status": "MATCHED",
                "proxy_skin_tone_group": "",
                # prepare-split-manifest rejects any value other than PASS.
                "manual_quality_status": manual.get(relative, "PASS"),
            })
            continue
        excluded.append({"annotation_path": str(json_path), "disease_label": disease, "reason": reason})

    for (disease, _), paths in images.items():
        for path in paths:
            if path not in used:
                excluded.append({"image_path": str(path), "disease_label": disease, "reason": "IMAGE_WITHOUT_VERIFIED_ANNOTATION"})

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    exclusion_path = output.with_name(output.stem + "_excluded.csv")
    with exclusion_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["disease_label", "reason", "image_path", "annotation_path"])
        writer.writeheader()
        writer.writerows(excluded)

    summary = {
        "manual_exclusions": {row["image_path"]: row["manual_quality_status"]
                              for row in rows if row["manual_quality_status"] != "PASS"},
        "matched_by_class": dict(Counter(row["disease_label"] for row in rows)),
        "excluded_by_class_and_reason": {
            f"{e['disease_label']}:{e['reason']}": n
            for e, n in ((dict(zip(("disease_label", "reason"), k)), v)
                         for k, v in Counter((e["disease_label"], e["reason"]) for e in excluded).items())
        },
        "manifest": str(output.resolve()),
        "exclusions": str(exclusion_path.resolve()),
    }
    output.with_name(output.stem + "_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-repository", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manual-exclusions", type=Path,
                        help="CSV with columns image (path relative to '1. Filtered') and reason.")
    args = parser.parse_args()
    print(json.dumps(build(args.dataset_root, args.source_repository, args.output, args.manual_exclusions), indent=2))


if __name__ == "__main__":
    main()
