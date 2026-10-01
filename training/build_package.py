"""
Create the self-contained split-training folder next to TOOL-26:

    ~/Documents/TOOL2026/ablation_training/
        izzy/     izzy_train_A_B_C.ipynb, combine_and_evaluate.ipynb
        daniel/   daniel_train_C2_D.ipynb
        backend/  the pipeline code the notebooks import
        datasets/ source_yolo (images + labels) and the shared ITA table
        training/ split_utils.py, fingerprints.json
        results/  result zips (results/incoming/ for the other trainer's)

and ablation_training.zip to send to Daniel.

    .venv/bin/python training/build_package.py
"""

from __future__ import annotations

import os
import shutil
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOL26 = HERE.parent
# ABLATION_PACKAGE_DEST builds elsewhere (e.g. when the default folder already holds results).
DEST = Path(os.environ.get("ABLATION_PACKAGE_DEST", TOOL26.parent / "ablation_training"))
ZIP = DEST.parent / "ablation_training.zip"

sys.path.insert(0, str(HERE))
import build_notebooks as nb  # noqa: E402

BACKEND_FILES = ["masking_ita.py", "clahe_calibration.py", "phase0_calibration.json", "member1_phase0_config.json"]
CHECKPOINT = TOOL26 / "weights" / "yolo26n.pt"

REQUIREMENTS = """\
ultralytics==8.4.163
torch==2.14.0
torchvision==0.29.0
opencv-python==5.0.0.93
scikit-learn==1.9.1
numpy==2.5.3
pillow==12.3.0
pandas
pyyaml
matplotlib
ipykernel
"""

README = """\
# Ablation training: Izzy (A, B, C) and Daniel (C′, D)

Izzy trains Models **A, B, C**; Daniel trains Models **C′, D**. Izzy then combines and evaluates all five.
Settings are the same for everyone: YOLO26n, seed 42, batch 8, image size 640, 300 epochs, patience 50.

## Who does what

| Order | Izzy | Daniel |
|---|---|---|
| 1 | Send `ablation_training.zip` to Daniel | Download the zip |
| 2 | — | Set up the laptop (driver, Python, kernel) |
| 3 | **Run All** on `izzy/izzy_train_A_B_C.ipynb` (~2.5–3.5 h) | **Run All** on `daniel/daniel_train_C2_D.ipynb` (~1.5–2.5 h) |
| 4 | Wait for Daniel's result | Send `results/results_daniel.zip` to Izzy |
| 5 | Put it in `results/incoming/` | Done |
| 6 | **Run All** on `izzy/combine_and_evaluate.ipynb` (~15 min) | — |
| 7 | Collect the results (optional: update the app) | — |

Step 3 runs at the same time on both laptops.

## Izzy's tasks

1. **Send the zip to Daniel.** Upload `ablation_training.zip` (~600 MB, next to this folder) to Google Drive and share the link.
2. **Train A, B, C.**
   - In VS Code, open this `ablation_training` folder, then `izzy/izzy_train_A_B_C.ipynb`.
   - Select the **TOOL-26 (.venv)** kernel (the existing one; nothing to install) and click **Run All**.
   - When it finishes, `results/results_izzy.zip` appears. Leave it there.
3. **Receive Daniel's `results_daniel.zip`** and put it in `results/incoming/`.
4. **Combine and evaluate.** Open `izzy/combine_and_evaluate.ipynb`, select the same kernel, and click **Run All**.
   It does not train; it imports all five models, checks Daniel's data matched, and tests everything on the test set.
5. **Results** are in `runs/ablation_split/`:
   - `ablation_test_metrics.csv`: mAP, precision, recall of A, B, C, C′, D
   - `ablation_test_map50_95_by_bracket.csv`: per skin tone
   - `trained_by.csv`: who trained each model, on which GPU
   - `figures/` and `visual_proof/`: confusion matrices, curves, Model A vs D pictures
6. **Optional, update the app.** Set `UPDATE_APP = True` in the last code cell and run it (needs `TOOL-26`
   next to this folder), then rebuild the APK with `TOOL-26/phone-development/build-apk.sh`.

## Daniel's tasks

1. **Prepare the laptop (once).**
   - **NVIDIA driver.** Windows: nvidia.com/drivers or the NVIDIA App. Ubuntu: `sudo ubuntu-drivers install`.
   - **Python 3.14** (the version Izzy used).
   - **VS Code** with the **Python** and **Jupyter** extensions.
2. **Download and unzip** `ablation_training.zip`.
3. **Create the kernel (once)** in a terminal inside the `ablation_training` folder:
   ```bash
   python3 -m venv .venv
   .venv/bin/python -m pip install ipykernel
   .venv/bin/python -m ipykernel install --user --name tool26 --display-name "TOOL-26 (.venv)"
   ```
   - On Windows use `.venv\\Scripts\\python` instead of `.venv/bin/python`.
   - On Ubuntu, if `python3 -m venv` says ensurepip is missing, run `sudo apt install python3.14-venv` first.
4. **Train C′ and D.** Open `daniel/daniel_train_C2_D.ipynb`, select the **TOOL-26 (.venv)** kernel, and click **Run All**.
   - The first time, the **GPU setup** cell downloads PyTorch with CUDA (~2–3 GB) and the other packages,
     then stops with "Restart the kernel". Click **Restart**, then **Run All** again.
5. **Send `results/results_daniel.zip`** (~20 MB) to Izzy.

## Rules (both)

- **Do not change anything in the notebooks or the data.**
- Keep the laptop plugged in and awake, and do not close VS Code while training.
- If it stops, **Run All** again: finished models are skipped and an interrupted one resumes.
- The first cell (**GPU setup**) installs PyTorch with CUDA only if it is missing. After it installs anything, restart the kernel and Run All again.
- The notebook stops with an error (**[BAD]**) if package versions, data or preprocessed images differ from Izzy's laptop. Send the message to Izzy.
- `requirements.txt` lists the same packages for manual installation.

## Limitation to report

Models A, B, C are trained on Izzy's laptop and C′, D on Daniel's. Different GPUs can shift results slightly even with the same seed, so state this in the manuscript. `runs/ablation_split/trained_by.csv` records who trained each model and on which GPU. All models are evaluated on the same machine.

## About this folder

This folder is generated from `TOOL-26/training/build_package.py`; regenerate it there instead of editing it here.
"""


def copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def add_d2_retraining(dest: Path) -> None:
    """Daniel's D2 re-training: the script plus the Phase 0 inputs its calibration needs."""
    import pandas as pd

    phase0 = TOOL26 / "phase0_outputs"
    manifest = pd.read_csv(phase0 / "member2_calibration" / "image_ita_manifest.csv")
    eligible = manifest[manifest["calibration_eligible"].astype(str) == "True"]
    cal = dest / "calibration_d2"
    (cal / "masks").mkdir(parents=True, exist_ok=True)
    for _, row in eligible.iterrows():
        shutil.copy2(row["mask_path"], cal / "masks" / f"{row['image_id']}.png")
    eligible[["image_id", "bracket", "ITA"]].to_csv(cal / "train_ita_masks.csv", index=False)
    copy(phase0 / "member2_calibration" / "beta_search_summary.csv", cal / "beta_search_summary.csv")
    copy(HERE / "retrain_c2_d_d2.py", dest / "daniel" / "retrain_c2_d_d2.py")
    copy(HERE / "daniel_retrain_C2_D_D2.ipynb", dest / "daniel" / "daniel_retrain_C2_D_D2.ipynb")


def main() -> None:
    if DEST.exists():
        existing = [p for p in (DEST / "runs", DEST / "weights") if p.exists()]
        if existing:
            raise SystemExit(f"{DEST} already has training results {existing}; not overwriting. Move it away first.")
        shutil.rmtree(DEST)

    for name in BACKEND_FILES:
        copy(TOOL26 / "backend" / name, DEST / "backend" / name)
    for name in ("split_utils.py", "fingerprints.json"):
        copy(HERE / name, DEST / "training" / name)

    source = TOOL26 / "datasets" / "source_yolo"
    for path in source.rglob("*"):
        rel = path.relative_to(source)
        if path.is_file() and (rel.parts[0] in ("images", "labels") or rel.name == "classes.json") \
                and not path.name.endswith(".cache"):
            copy(path, DEST / "datasets" / "source_yolo" / rel)
    copy(TOOL26 / "datasets" / "ablation_yolov26" / "ita_table.csv",
         DEST / "datasets" / "ablation_yolov26" / "ita_table.csv")

    # Same pretrained checkpoint for everyone (the notebooks load 'yolo26n.pt' from their folder).
    for trainer in ("izzy", "daniel"):
        copy(CHECKPOINT, DEST / trainer / "yolo26n.pt")

    add_d2_retraining(DEST)

    nb.write(DEST / "izzy" / "izzy_train_A_B_C.ipynb", nb.trainer_notebook("Izzy"))
    nb.write(DEST / "izzy" / "combine_and_evaluate.ipynb", nb.combine_notebook())
    nb.write(DEST / "daniel" / "daniel_train_C2_D.ipynb", nb.trainer_notebook("Daniel"))

    (DEST / "results" / "incoming").mkdir(parents=True, exist_ok=True)
    (DEST / "README.md").write_text(README, encoding="utf-8")
    (DEST / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")

    # PNGs are already compressed; storing keeps zipping fast.
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_STORED) as zf:
        for path in sorted(DEST.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(DEST.parent))
    print(f"Folder: {DEST}")
    print(f"Zip for Daniel: {ZIP} ({ZIP.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
