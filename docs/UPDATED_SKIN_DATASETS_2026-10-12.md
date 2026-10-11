# Updated Skin Datasets — 12 October 2026

## Overview

- **Archive Name:** `UPDATED_SKIN_DATASETS_2026-10-12.zip`
- **Archive Size:** 12,907,237,362 bytes (~12.9 GB)
- **Verification Status:** PASS (all 28,114 ZIP entries CRC32-verified)
- **Primary Storage:** Google Drive (raw archives exceed GitHub's 100 MB per-file and 2 GB Git LFS limits)

---

## Dataset Summary & Logical Counts

The updated dataset comprises **5,103 logical examples**:
- **4,803 disease examples** across 8 classes:
  - Chickenpox (Varicella): 600
  - HFMD: 600
  - Molluscum: 600
  - Impetigo: 603
  - Ringworm (Tinea corporis): 600
  - An-an (Tinea versicolor): 600
  - Warts: 600
  - Athlete's foot (Tinea pedis): 600
- **300 healthy skin negative background examples**:
  - Source: MSKCC Skin Tone Labeling Dataset (DOI: 10.34970/962049), Memorial Sloan Kettering Cancer Center (CC-BY).
  - Source-recorded Fitzpatrick types: I–II: 50, III–V: 200, VI: 50.
  - Formatted as background examples with blank YOLO `.txt` annotations (not a ninth class).

---

## Split Distribution

- **Disease Splits:** Stratified ~70% training / 20% validation / 10% testing per class. Data augmentation was applied exclusively to training images; validation and test sets contain unaugmented source images only.
- **Healthy Background Allocation:**
  - Training: 175 photos (FST I–V)
  - Validation: 50 photos (FST I–V)
  - Test: 25 photos (FST I–V)
  - Stress Test: 50 photos (FST VI, evaluated separately)

### Final YOLO Dataset Counts

| Split | Disease Images | Healthy Background | Total |
|---|---:|---:|---:|
| **Train** | 3,362 | 175 | 3,537 |
| **Validation** | 961 | 50 | 1,011 |
| **Test** | 480 | 25 | 505 |
| **Stress Test** | 0 | 50 | 50 |
| **Total** | 4,803 | 300 | 5,103 |

---

## Packaged Folders Inventory

| Folder | File Count | Uncompressed Size | Description |
|---|---:|---:|---|
| `Combined_dataset_600/` | 16,934 | 4.53 GB | Curated disease images, AnyLabeling JSONs, stratified split folders |
| `Healthy_skin_300/` | 927 | 459.90 MB | 300 original body-only healthy skin photos, annotations, source metadata |
| `Combined_dataset_600_YOLO/` | 10,211 | 7.94 GB | Ready-to-train YOLO images and label txt files (`data.yaml`, `stress_test.yaml`) |
| `Combined_dataset_600_reports/` | 16 | 11.15 MB | Disease curation, counts, split, augmentation and verification reports |
| `A_NEW_DATASETS_support/` | 24 | 3.39 MB | Supporting new-dataset curation and candidate tracking records |
| **Total** | **28,112** | **12.94 GB** | **Packaged into single archive `UPDATED_SKIN_DATASETS_2026-10-12.zip`** |

---

## Verification Manifest

The complete CRC32 and file verification record is preserved in:
`docs/UPDATED_SKIN_DATASETS_2026-10-12_verification.json`
