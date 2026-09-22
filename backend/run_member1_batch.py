"""Member 1 manifest preparation and Phase 0 batch commands."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import PIL
import sklearn
from PIL import Image

from masking_ita import MaskingITAConfig, MaskingITAProcessor


DISEASE_LABELS = {
    "Warts", "Molluscum", "Varicella", "HFMD", "Tinea versicolor",
    "Tinea corporis", "Tinea pedis", "Impetigo",
}
ACCEPTED_FORMATS = {"JPEG", "PNG", "WEBP", "TIFF"}
ARCHIVE_CLASS_MAP = {
    "AF": "Tinea pedis",
    "AN-AN": "Tinea versicolor",
    "HFMD": "HFMD",
    "MOLLUSCUM": "Molluscum",
    "RINGWORM": "Tinea corporis",
}
ANNOTATED_CLASS_MAP = {
    "0. CHICKENPOX": "Varicella",
    "1. HFMD": "HFMD",
    "2. MOLLUSCUM": "Molluscum",
    "3. IMPETIGO": "Impetigo",
    "4. RINGWORM": "Tinea corporis",
    "5. AN-AN": "Tinea versicolor",
    "6. WARTS": "Warts",
    "7. ATHLETE_S FOOT": "Tinea pedis",
    "7. TINEA PEDIS": "Tinea pedis",
    "WARTS": "Warts",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot write an empty manifest.")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_id(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:24]


def perceptual_hash(rgb: np.ndarray) -> int:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)[:8, :8]
    values = dct.flatten()[1:]
    return int("".join("1" if value > np.median(values) else "0" for value in values), 2)


def visual_signature(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1)


def signature_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = left - left.mean()
    right = right - right.mean()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else float(np.array_equal(left, right))


def quality_metrics(rgb: np.ndarray, quality: dict[str, Any]) -> tuple[str, str, dict[str, float]]:
    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    scale = min(1.0, 1024.0 / max(height, width))
    metric_gray = cv2.resize(
        gray, (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    ) if scale < 1 else gray
    blur = float(cv2.Laplacian(metric_gray, cv2.CV_64F).var())
    black = float((metric_gray <= quality["black_luma_max"]).mean())
    white = float((metric_gray >= quality["white_luma_min"]).mean())
    span = float(np.percentile(metric_gray, 95) - np.percentile(metric_gray, 5))
    metrics = {
        "width": width, "height": height, "short_side": min(height, width),
        "laplacian_variance": blur, "black_fraction": black,
        "white_fraction": white, "luminance_p95_p5": span,
    }
    if min(height, width) < quality["minimum_short_side"]:
        return "REJECTED", "RESOLUTION_TOO_LOW", metrics
    if blur < quality["minimum_laplacian_variance"]:
        return "REJECTED", "SEVERE_BLUR", metrics
    if black > quality["maximum_clipped_fraction"]:
        return "REJECTED", "EXCESSIVE_BLACK_CLIPPING", metrics
    if white > quality["maximum_clipped_fraction"]:
        return "REJECTED", "EXCESSIVE_WHITE_CLIPPING", metrics
    if span < quality["minimum_luminance_span"]:
        return "REJECTED", "SEVERE_LOW_CONTRAST", metrics
    return "PASS", "", metrics


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    config = json.loads(raw)
    shared_quality_keys = {
        "minimum_short_side", "minimum_laplacian_variance", "black_luma_max",
        "white_luma_min", "maximum_clipped_fraction", "minimum_luminance_span",
    }
    if "quality" in config and "masking" in config:
        mismatched = [
            key for key in shared_quality_keys
            if config["quality"].get(key) != config["masking"].get(key)
        ]
        if mismatched:
            raise ValueError(f"Quality settings must match masking/API settings: {sorted(mismatched)}")
    return config, hashlib.sha256(raw).hexdigest()


def safe_extract(archive_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination != target and destination not in target.parents:
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
        archive.extractall(destination)


def archive_lineage_id(filename: str) -> str:
    """Group Roboflow-style derivatives of the same source image."""
    stem = Path(filename).stem
    stem = re.sub(r"\.rf\.[0-9a-f]+$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"_(?:jpg|jpeg|png)$", "", stem, flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]+", "", stem.lower())


def build_archive_manifest(images_zip: Path, annotations_zip: Path, output_dir: Path) -> Path:
    """Extract the supplied archives safely and create a source inventory CSV."""
    extracted = output_dir / "archive_contents"
    images_root, annotations_root = extracted / "images", extracted / "annotations"
    safe_extract(images_zip, images_root)
    safe_extract(annotations_zip, annotations_root)
    annotation_index: dict[str, list[Path]] = defaultdict(list)
    for annotation in annotations_root.rglob("*.json"):
        annotation_index[archive_lineage_id(annotation.name)].append(annotation.resolve())
    rows: list[dict[str, Any]] = []
    folder_pattern = re.compile(r"^(70|20|10)_\s*(.+)$", re.IGNORECASE)
    split_names = {"70": "train", "20": "validation", "10": "test_a"}
    for image_path in sorted(path for path in images_root.rglob("*") if path.is_file()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}:
            continue
        parent = image_path.parent.name.strip()
        match = folder_pattern.match(parent)
        if not match:
            continue
        folder_class = match.group(2).strip().upper()
        disease = ARCHIVE_CLASS_MAP.get(folder_class)
        if disease is None:
            raise ValueError(f"Unrecognized class folder in archive: {parent}")
        lineage = archive_lineage_id(image_path.name)
        annotations = annotation_index.get(lineage, [])
        rows.append({
            "image_path": str(image_path.resolve()),
            "image_id": stable_id("splitting_archive", str(image_path.relative_to(images_root))),
            "disease_label": disease,
            "source_repository": f"LOCAL_ARCHIVE:{images_zip.name}",
            "source_item_id": f"{disease}:{lineage}",
            "proxy_skin_tone_group": "",
            "original_split": split_names[match.group(1)],
            "annotation_path": str(annotations[0]) if len(annotations) == 1 else "",
            "annotation_match_status": "MATCHED" if len(annotations) == 1 else ("AMBIGUOUS" if annotations else "NOT_FOUND"),
        })
    if not rows:
        raise ValueError("No supported images found in the splitting archive.")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "archive_source_manifest.csv"
    write_csv(output_path, rows)
    coverage = {
        "image_count": len(rows),
        "classes_present": sorted({row["disease_label"] for row in rows}),
        "classes_missing": sorted(DISEASE_LABELS - {row["disease_label"] for row in rows}),
        "original_split_counts": dict(Counter(row["original_split"] for row in rows)),
        "annotation_match_counts": dict(Counter(row["annotation_match_status"] for row in rows)),
        "warning": "The local archive name is retained as provisional provenance; record the original repository before final calibration when known.",
    }
    (output_dir / "archive_inventory_summary.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    return output_path


def build_supplemental_manifest(
    archive_path: Path,
    base_manifest_path: Path,
    output_dir: Path,
) -> Path:
    """Add actual images found inside an annotated-folder archive to a source manifest."""
    extracted_root = output_dir / "supplemental_archive_contents"
    safe_extract(archive_path, extracted_root)
    json_index: dict[str, list[Path]] = defaultdict(list)
    for annotation in extracted_root.rglob("*.json"):
        try:
            payload = json.loads(annotation.read_text(encoding="utf-8"))
            image_name = Path(str(payload.get("imagePath", ""))).name
            if image_name:
                json_index[image_name.lower()].append(annotation.resolve())
        except (OSError, json.JSONDecodeError):
            continue
    additions: list[dict[str, Any]] = []
    for image_path in sorted(path for path in extracted_root.rglob("*") if path.is_file()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}:
            continue
        disease = None
        class_folder = None
        for parent in image_path.parents:
            key = parent.name.strip().upper()
            if key in ANNOTATED_CLASS_MAP:
                disease = ANNOTATED_CLASS_MAP[key]
                class_folder = key
                break
            if parent == extracted_root:
                break
        if disease is None:
            continue
        annotations = json_index.get(image_path.name.lower(), [])
        additions.append({
            "image_path": str(image_path.resolve()),
            "image_id": stable_id("supplemental_annotated_archive", str(image_path.relative_to(extracted_root))),
            "disease_label": disease,
            "source_repository": f"LOCAL_ARCHIVE:{archive_path.name}",
            "source_item_id": f"{disease}:{archive_lineage_id(image_path.name)}",
            "proxy_skin_tone_group": "",
            "original_split": "UNASSIGNED",
            "annotation_path": str(annotations[0]) if len(annotations) == 1 else "",
            "annotation_match_status": "MATCHED" if len(annotations) == 1 else ("AMBIGUOUS" if annotations else "NOT_FOUND"),
        })
    if not additions:
        raise ValueError("No supported images were found in the supplemental archive.")
    base_rows = read_csv(base_manifest_path)
    existing_ids = {row.get("image_id") for row in base_rows}
    additions = [row for row in additions if row["image_id"] not in existing_ids]
    combined = base_rows + additions
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "combined_source_manifest.csv"
    write_csv(output_path, combined)
    present = {row["disease_label"] for row in combined}
    summary = {
        "base_count": len(base_rows), "supplemental_images_added": len(additions),
        "combined_count": len(combined),
        "supplemental_by_class": dict(Counter(row["disease_label"] for row in additions)),
        "supplemental_annotation_matches": dict(Counter(row["annotation_match_status"] for row in additions)),
        "classes_present": sorted(present), "classes_missing": sorted(DISEASE_LABELS - present),
        "warning": "Local archive names are provisional provenance; record original repositories before final calibration when known.",
    }
    (output_dir / "combined_inventory_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output_path


def prepare_manifest(input_path: Path, output_dir: Path, config_path: Path) -> Path:
    config, config_hash = load_config(config_path)
    rows = read_csv(input_path)
    required = {"image_path", "disease_label", "source_repository"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Input CSV must contain: {sorted(required)}")
    present_classes = {row["disease_label"].strip() for row in rows}
    if config.get("require_all_eight_classes", True) and present_classes != DISEASE_LABELS:
        raise ValueError(
            f"Phase 0 requires all eight classes; missing={sorted(DISEASE_LABELS - present_classes)}, "
            f"unexpected={sorted(present_classes - DISEASE_LABELS)}"
        )
    supplied_ids = [row.get("image_id", "").strip() for row in rows if row.get("image_id", "").strip()]
    if len(supplied_ids) != len(set(supplied_ids)):
        raise ValueError("Input manifest contains duplicate image_id values.")
    ratios = config["split"]["ratios"]
    if set(ratios) != {"train", "validation", "test_a"} or not np.isclose(sum(ratios.values()), 1.0):
        raise ValueError("Split ratios must contain train/validation/test_a and sum to 1.0.")

    standardized_dir = output_dir / "standardized"
    standardized_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    hashes: list[str | None] = []
    phashes: list[int | None] = []
    signatures: list[np.ndarray | None] = []

    for source_row in rows:
        raw_path = Path(source_row["image_path"]).expanduser().resolve()
        disease = source_row["disease_label"].strip()
        repository = source_row["source_repository"].strip()
        if disease not in DISEASE_LABELS:
            raise ValueError(f"Unknown disease label: {disease}")
        if not repository:
            raise ValueError(f"Missing source_repository for {raw_path}")
        image_id = source_row.get("image_id", "").strip() or stable_id(repository, str(raw_path))
        record: dict[str, Any] = dict(source_row)
        record.update({
            "image_id": image_id, "unique_id": image_id, "raw_image_path": str(raw_path),
            "data_origin": "public", "disease_label": disease,
            "source_repository": repository, "processed_image_path": "",
            "image_quality_status": "REJECTED", "quality_reason": "",
            "duplicate_status": "NOT_EVALUATED", "canonical_image_id": "",
            "lineage_group_id": "", "split": "", "calibration_eligible": False,
        })
        try:
            with Image.open(raw_path) as probe:
                source_format = (probe.format or "").upper()
            if source_format not in ACCEPTED_FORMATS:
                raise ValueError(f"UNSUPPORTED_FORMAT:{source_format or 'UNKNOWN'}")
            rgb = MaskingITAProcessor.load_image(raw_path)
            status, reason, metrics = quality_metrics(rgb, config["quality"])
            manual_status = source_row.get("manual_quality_status", "PASS").strip().upper() or "PASS"
            if manual_status != "PASS":
                status, reason = "REJECTED", f"MANUAL_{manual_status}"
            processed_path = standardized_dir / f"{image_id}.png"
            Image.fromarray(rgb).save(processed_path, format="PNG", optimize=False)
            normalized_hash = hashlib.sha256(
                f"{rgb.shape}".encode("ascii") + rgb.tobytes()
            ).hexdigest()
            record.update(metrics)
            record.update({
                "source_format": source_format, "processed_image_path": str(processed_path.resolve()),
                "image_quality_status": status, "quality_reason": reason,
                "normalized_sha256": normalized_hash,
            })
            hashes.append(normalized_hash)
            phashes.append(perceptual_hash(rgb))
            signatures.append(visual_signature(rgb))
        except Exception as exc:
            record["quality_reason"] = str(exc)
            hashes.append(None); phashes.append(None); signatures.append(None)
        records.append(record)

    candidates = [index for index, row in enumerate(records) if row["image_quality_status"] == "PASS"]
    groups = DisjointSet(len(records))
    exact_seen: dict[str, int] = {}
    for index in candidates:
        digest = hashes[index]
        if digest in exact_seen:
            groups.union(index, exact_seen[digest])
        else:
            exact_seen[digest] = index
    for position, left in enumerate(candidates):
        for right in candidates[position + 1:]:
            if groups.find(left) == groups.find(right):
                continue
            if (phashes[left] ^ phashes[right]).bit_count() <= config["deduplication"]["phash_hamming_max"]:
                if signature_correlation(signatures[left], signatures[right]) >= config["deduplication"]["minimum_correlation"]:
                    groups.union(left, right)

    components: dict[int, list[int]] = defaultdict(list)
    for index in candidates:
        components[groups.find(index)].append(index)
    for component in components.values():
        labels = {records[index]["disease_label"] for index in component}
        if len(labels) != 1:
            raise ValueError(f"Duplicate group has conflicting disease labels: {[records[i]['image_id'] for i in component]}")
        representative = max(
            component,
            key=lambda index: (
                float(records[index].get("laplacian_variance", 0)),
                int(records[index].get("short_side", 0)),
                records[index]["image_id"],
            ),
        )
        for index in component:
            records[index]["canonical_image_id"] = records[representative]["image_id"]
            if index == representative:
                records[index]["duplicate_status"] = "UNIQUE"
            else:
                records[index]["duplicate_status"] = (
                    "EXACT_DUPLICATE_EXCLUDED" if hashes[index] == hashes[representative]
                    else "NEAR_DUPLICATE_EXCLUDED"
                )
                records[index]["calibration_eligible"] = False
    for index, record in enumerate(records):
        if record["image_quality_status"] != "PASS":
            record["duplicate_status"] = "NOT_EVALUATED"

    eligible = [row for row in records if row["image_quality_status"] == "PASS" and row["duplicate_status"] == "UNIQUE"]
    for row in eligible:
        supplied = row.get("source_item_id", "").strip() or row.get("lineage_id", "").strip()
        row["lineage_group_id"] = supplied or stable_id("lineage", row["canonical_image_id"])
    lineage_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        lineage_rows[row["lineage_group_id"]].append(row)
    for lineage, members in lineage_rows.items():
        if len({member["disease_label"] for member in members}) != 1:
            raise ValueError(f"Lineage {lineage} contains conflicting labels.")

    strata: dict[tuple[str, str], list[tuple[str, list[dict[str, Any]]]]] = defaultdict(list)
    for lineage, members in lineage_rows.items():
        tones = {member.get("proxy_skin_tone_group", "").strip().upper() for member in members}
        tones.discard("")
        tone = next(iter(tones)) if len(tones) == 1 else "UNKNOWN"
        if tone not in {"A", "B", "UNKNOWN"}:
            raise ValueError(f"Invalid proxy_skin_tone_group in lineage {lineage}")
        strata[(members[0]["disease_label"], tone)].append((lineage, members))
    seed = int(config["random_seed"])
    for groups_in_stratum in strata.values():
        groups_in_stratum.sort(key=lambda item: hashlib.sha256(f"{seed}:{item[0]}".encode()).hexdigest())
        total = sum(len(members) for _, members in groups_in_stratum)
        targets = {name: total * float(ratio) for name, ratio in ratios.items()}
        assigned = {name: 0 for name in ratios}
        for _, members in groups_in_stratum:
            split = max(ratios, key=lambda name: (targets[name] - assigned[name]) / max(targets[name], 1.0))
            for member in members:
                member["split"] = split
                member["calibration_eligible"] = split == "train"
            assigned[split] += len(members)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "cleaned_split_manifest.csv"
    write_csv(output_path, records)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "global_public_dataset", "config_sha256": config_hash,
        "input_manifest_sha256": file_sha256(input_path), "counts": dict(Counter(row["split"] or "excluded" for row in records)),
    }
    (output_dir / "manifest_preparation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_path


def runtime_metadata(config_hash: str, manifest_path: Path, random_seed: int) -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config_hash, "input_manifest_sha256": file_sha256(manifest_path),
        "random_seed": random_seed, "python": sys.version, "platform": platform.platform(),
        "machine": platform.machine(), "processor": platform.processor(), "cpu_count": os.cpu_count(),
        "packages": {"opencv": cv2.__version__, "numpy": np.__version__, "scikit_learn": sklearn.__version__, "pillow": PIL.__version__},
    }


def balanced_take(rows: list[dict[str, str]], count: int, seed: int, salt: str) -> list[dict[str, str]]:
    """Deterministically round-robin across repositories."""
    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_source[row["source_repository"]].append(row)
    for source, source_rows in by_source.items():
        source_rows.sort(
            key=lambda row: hashlib.sha256(f"{seed}:{salt}:{source}:{row['image_id']}".encode()).hexdigest()
        )
    selected: list[dict[str, str]] = []
    sources = sorted(by_source)
    while len(selected) < count and sources:
        remaining: list[str] = []
        for source in sources:
            if by_source[source] and len(selected) < count:
                selected.append(by_source[source].pop(0))
            if by_source[source]:
                remaining.append(source)
        sources = remaining
    return selected


def build_pilot_manifest(manifest_path: Path, output_path: Path, config_path: Path) -> Path:
    config, config_hash = load_config(config_path)
    rows = read_csv(manifest_path)
    candidates = [
        row for row in rows
        if row.get("data_origin", "").lower() == "public"
        and row.get("split") == "train"
        and row.get("image_quality_status") == "PASS"
        and row.get("duplicate_status") == "UNIQUE"
    ]
    selected: list[dict[str, str]] = []
    seed = int(config["random_seed"])
    allocation: dict[str, dict[str, int]] = {}
    for disease in sorted(DISEASE_LABELS):
        disease_rows = [row for row in candidates if row.get("disease_label") == disease]
        chosen: list[dict[str, str]] = []
        cell_counts: dict[str, int] = {}
        for tone in ("A", "B"):
            cell = [row for row in disease_rows if row.get("proxy_skin_tone_group", "").strip().upper() == tone]
            cell_choice = balanced_take(cell, 10, seed, f"{disease}:{tone}")
            chosen.extend(cell_choice)
            cell_counts[tone] = len(cell_choice)
        remaining = [row for row in disease_rows if row["image_id"] not in {item["image_id"] for item in chosen}]
        chosen.extend(balanced_take(remaining, 20 - len(chosen), seed, f"{disease}:redistributed"))
        if len(chosen) != 20:
            raise ValueError(f"Pilot requires 20 eligible training images for {disease}; found {len(chosen)}.")
        for order, row in enumerate(chosen, start=1):
            pilot_row = dict(row)
            pilot_row.update({
                "pilot_order_within_disease": order,
                "reviewer_1_mask_acceptable": "", "reviewer_2_mask_acceptable": "",
                "reviewer_1_contamination": "", "reviewer_2_contamination": "",
                "consensus_mask_acceptable": "", "consensus_notes": "",
            })
            selected.append(pilot_row)
        allocation[disease] = {**cell_counts, "redistributed": 20 - sum(cell_counts.values())}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output_path, selected)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "count": len(selected),
        "config_sha256": config_hash, "input_manifest_sha256": file_sha256(manifest_path),
        "allocation": allocation,
    }
    output_path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_path


def run_phase0(manifest_path: Path, output_dir: Path, config_path: Path, allow_provisional: bool = False) -> Path:
    config, config_hash = load_config(config_path)
    if config.get("status") != "FROZEN" and not allow_provisional:
        raise ValueError(
            "Configuration is not FROZEN. Use --allow-provisional only for the documented pilot run."
        )
    rows = read_csv(manifest_path)
    required = {"image_id", "processed_image_path", "disease_label", "source_repository", "split", "image_quality_status", "duplicate_status", "data_origin"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Prepared manifest is missing columns: {sorted(required - set(rows[0] if rows else []))}")
    processor = MaskingITAProcessor(MaskingITAConfig(**config["masking"]))
    masks_dir, audits_dir = output_dir / "masks", output_dir / "audits"
    masks_dir.mkdir(parents=True, exist_ok=True); audits_dir.mkdir(parents=True, exist_ok=True)
    input_rows = [
        row for row in rows
        if (
            row["data_origin"].lower() == "public" and row["split"] == "train"
            and row["image_quality_status"] == "PASS" and row["duplicate_status"] == "UNIQUE"
        )
    ]
    present_training_classes = {row["disease_label"] for row in input_rows}
    if config.get("require_all_eight_classes", True) and present_training_classes != DISEASE_LABELS:
        raise ValueError(f"Eligible training rows are missing classes: {sorted(DISEASE_LABELS - present_training_classes)}")
    output_rows: list[dict[str, Any]] = []
    for row in input_rows:
        processed_path = Path(row["processed_image_path"])
        result = processor.process_file(processed_path)
        mask_path = ""
        if result.calibration_eligible and result.mask is not None:
            target = masks_dir / f"{row['image_id']}.png"
            if not cv2.imwrite(str(target), result.mask.astype(np.uint8) * 255):
                raise OSError(f"Could not write mask: {target}")
            mask_path = str(target.resolve())
        audit = {
            "image_id": row["image_id"], "selected_k": result.selected_k,
            "selection_method": result.selection_method, "silhouette_score": result.silhouette_score,
            "k_scores": result.k_scores, "clusters": result.clusters,
            "status": result.status, "failure_reason": result.failure_reason,
        }
        audit_path = audits_dir / f"{row['image_id']}.json"
        audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
        out = dict(row)
        out.update({
            "image_path": str(processed_path.resolve()), "mask_path": mask_path,
            "status": result.status, "mask_status": result.status,
            "mask_method": result.mask_method or "", "selected_k": result.selected_k,
            "silhouette_score": result.silhouette_score, "selection_method": result.selection_method or "",
            "mask_area_pixels": result.mask_area_pixels, "mask_area_percent": result.mask_area_percent,
            "mean_l": result.mean_l, "mean_a": result.mean_a, "mean_b": result.mean_b,
            "ita": result.ita, "ITA": result.ita, "bracket": result.bracket,
            "ita_bracket": result.bracket, "skin_tone_group": result.bracket,
            "calibration_eligible": result.calibration_eligible,
            "failure_reason": result.failure_reason or "", "audit_path": str(audit_path.resolve()),
        })
        output_rows.append(out)
    if not output_rows:
        raise ValueError("No eligible public training rows were found.")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "image_ita_manifest.csv"
    write_csv(output_path, output_rows)
    failures = [row for row in output_rows if str(row["calibration_eligible"]).lower() != "true"]
    if failures:
        write_csv(output_dir / "failed_images.csv", failures)
    summary = {
        "total": len(output_rows), "eligible": len(output_rows) - len(failures), "failed": len(failures),
        "by_status": dict(Counter(row["status"] for row in output_rows)),
        "by_disease": dict(Counter(row["disease_label"] for row in output_rows)),
        "by_proxy_group": dict(Counter(row.get("proxy_skin_tone_group", "") or "UNKNOWN" for row in output_rows)),
        "by_source": dict(Counter(row["source_repository"] for row in output_rows)),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "run_metadata.json").write_text(
        json.dumps(runtime_metadata(config_hash, manifest_path, int(config["random_seed"])), indent=2), encoding="utf-8"
    )
    config_bytes = config_path.read_bytes()
    (output_dir / "member1_config_used.json").write_bytes(config_bytes)
    if config.get("status") == "FROZEN":
        (output_dir / "frozen_member1_config.json").write_bytes(config_bytes)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Member 1 Phase 0 pipeline")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("member1_phase0_config.json"))
    subparsers = parser.add_subparsers(dest="command", required=True)
    inventory = subparsers.add_parser("build-archive-manifest")
    inventory.add_argument("--images-zip", type=Path, required=True)
    inventory.add_argument("--annotations-zip", type=Path, required=True)
    inventory.add_argument("--output-dir", type=Path, required=True)
    supplement = subparsers.add_parser("add-supplemental-archive")
    supplement.add_argument("--archive", type=Path, required=True)
    supplement.add_argument("--base-manifest", type=Path, required=True)
    supplement.add_argument("--output-dir", type=Path, required=True)
    prepare = subparsers.add_parser("prepare-manifest")
    prepare.add_argument("--input-manifest", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    pilot = subparsers.add_parser("build-pilot-manifest")
    pilot.add_argument("--manifest", type=Path, required=True)
    pilot.add_argument("--output", type=Path, required=True)
    run = subparsers.add_parser("run-phase0")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--allow-provisional", action="store_true", help="Permit the documented 160-image pilot before thresholds are frozen.")
    args = parser.parse_args()
    if args.command == "build-archive-manifest":
        result = build_archive_manifest(args.images_zip, args.annotations_zip, args.output_dir)
    elif args.command == "add-supplemental-archive":
        result = build_supplemental_manifest(args.archive, args.base_manifest, args.output_dir)
    elif args.command == "prepare-manifest":
        result = prepare_manifest(args.input_manifest, args.output_dir, args.config)
    elif args.command == "build-pilot-manifest":
        result = build_pilot_manifest(args.manifest, args.output, args.config)
    else:
        result = run_phase0(args.manifest, args.output_dir, args.config, args.allow_provisional)
    print(result)


if __name__ == "__main__":
    main()
