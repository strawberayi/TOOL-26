from pathlib import Path
import csv

from masking_ita import MaskingITAProcessor


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(r"C:\THESIS")

DATASET_DIR = BASE_DIR / "datasets" / "FILTERED"
OUTPUT_DIR = BASE_DIR / "outputs"

MANIFEST_PATH = OUTPUT_DIR / "image_ita_manifest.csv"


# ============================================================
# DISEASE FOLDERS
# ============================================================

DISEASE_FOLDERS = {
    "1. Warts": "Warts",
    "2. Molluscum": "Molluscum",
    "3. Varicella": "Varicella",
    "4. HFMD": "HFMD",
    "5. Tinea versicolor": "Tinea versicolor",
    "6. Tinea corporis": "Tinea corporis",
    "7. Tinea pedis": "Tinea pedis",
    "8. Impetigo": "Impetigo",
}


# ============================================================
# SUPPORTED IMAGE TYPES
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("MEMBER 1 - MASKING + ITA BATCH PROCESSING")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    processor = MaskingITAProcessor()

    rows = []

    total_images = 0

    for folder_name, disease_label in DISEASE_FOLDERS.items():

        folder_path = DATASET_DIR / folder_name

        if not folder_path.exists():
            print(f"\nWARNING: Folder not found: {folder_path}")
            continue

        print(f"\nProcessing: {disease_label}")
        print(f"Folder: {folder_path}")

        image_files = sorted(
            [
                path
                for path in folder_path.iterdir()
                if path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
            ]
        )

        print(f"Images found: {len(image_files)}")

        for image_path in image_files:

            total_images += 1

            result = processor.process_file(image_path)

            row = {
                "image_path": str(image_path),
                "filename": image_path.name,
                "unique_id": image_path.stem,
                "disease_label": disease_label,

                "status": result.status,
                "mask_method": result.mask_method,

                "selected_k": result.selected_k,
                "silhouette_score": result.silhouette_score,

                "mask_area_pixels": result.mask_area_pixels,
                "mask_area_percent": result.mask_area_percent,

                "mean_l": result.mean_l,
                "mean_a": result.mean_a,
                "mean_b": result.mean_b,

                "ita": result.ita,
                "skin_tone_group": result.bracket,

                "calibration_eligible": result.calibration_eligible,
                "failure_reason": result.failure_reason,

                # These will be completed later when dataset
                # metadata / splitting / duplicate checking is done.
                "source_repository": "",
                "split": "",
                "image_quality_status": "",
                "duplicate_status": "",
            }

            rows.append(row)

            print(
                f"  [{total_images}] "
                f"{image_path.name} -> "
                f"{result.status} | "
                f"ITA={result.ita} | "
                f"Group={result.bracket}"
            )

    # ========================================================
    # SAVE CSV
    # ========================================================

    if not rows:
        print("\nERROR: No images were found.")
        return

    fieldnames = list(rows[0].keys())

    with open(
        MANIFEST_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    # ========================================================
    # SUMMARY
    # ========================================================

    eligible = sum(
        1
        for row in rows
        if row["calibration_eligible"]
    )

    failed = len(rows) - eligible

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)

    print(f"Total images processed : {len(rows)}")
    print(f"Calibration eligible   : {eligible}")
    print(f"Failed / excluded      : {failed}")

    print("\nManifest created at:")
    print(MANIFEST_PATH)

    print("\nOpen this folder:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()