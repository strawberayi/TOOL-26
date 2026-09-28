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

import shutil
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOL26 = HERE.parent
DEST = TOOL26.parent / "ablation_training"
ZIP = TOOL26.parent / "ablation_training.zip"

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

Each person runs **one notebook, once**. Izzy then combines and evaluates everything.

| Who | Open this notebook | Time |
|---|---|---|
| Izzy | `izzy/izzy_train_A_B_C.ipynb` | ~2.5–3.5 h |
| Daniel | `daniel/daniel_train_C2_D.ipynb` | ~1.5–2.5 h |
| Izzy, after both finish | `izzy/combine_and_evaluate.ipynb` | ~15 min |

Both training notebooks can run at the same time on the two laptops.
Settings are the same for everyone: YOLO26n, seed 42, batch 8, image size 640, 300 epochs, patience 50.

## 1. Setup (once per laptop)

This folder already contains the code and the data. The first cell of every
notebook (**GPU setup**) checks the NVIDIA driver and PyTorch, and downloads the
CUDA build of PyTorch 2.14.0 and the other pinned packages **only if they are
missing**. If it installs anything, restart the kernel and click Run All again.

- **Izzy:** nothing to prepare. Use the existing **TOOL-26 (.venv)** kernel.
- **Daniel:** needs an **NVIDIA driver** (Windows: nvidia.com/drivers or the NVIDIA App;
  Ubuntu: `sudo ubuntu-drivers install`) and **Python 3.14** (the version Izzy used).
  Unzip `ablation_training.zip`, then inside the `ablation_training` folder create the kernel once:
  ```bash
  python3 -m venv .venv
  .venv/bin/python -m pip install ipykernel
  .venv/bin/python -m ipykernel install --user --name tool26 --display-name "TOOL-26 (.venv)"
  ```
  On Windows use `.venv\\Scripts\\python`. On Ubuntu, if `python3 -m venv` says ensurepip is
  missing, run `sudo apt install python3.14-venv` first. The notebook installs everything else.
  (`requirements.txt` lists the same packages for manual installation.)

## 2. Train

Open your notebook in VS Code or Jupyter, select the **TOOL-26 (.venv)** kernel, and click **Run All**.

- Keep the laptop plugged in and awake.
- If it stops, **Run All** again: finished models are skipped and an interrupted one resumes.
- The notebook stops with an error if the package versions, the data or the preprocessed images differ from Izzy's laptop. That keeps the comparison fair.

When it finishes:
- **Daniel:** send `results/results_daniel.zip` (~20 MB) to Izzy.
- **Izzy:** keep `results/results_izzy.zip` where it is.

## 3. Combine (Izzy)

1. Put Daniel's `results_daniel.zip` in `results/incoming/`.
2. Open `izzy/combine_and_evaluate.ipynb` and click **Run All**.
3. Results: `runs/ablation_split/`, including `ablation_test_metrics.csv`, the per-skin-tone table, figures, and `trained_by.csv`.
4. Optional: to put the new models in the app, set `UPDATE_APP = True` in the last code cell. This needs `TOOL-26` next to this folder. Then rebuild the APK.

## Limitation to report

Models A, B, C are trained on Izzy's laptop and C′, D on Daniel's. Different GPUs can shift results slightly even with the same seed, so state this in the manuscript. `runs/ablation_split/trained_by.csv` records who trained each model and on which GPU. All models are evaluated on the same machine.

## Rules

- **Do not change anything in the notebooks or the data.**
- This folder is generated from `TOOL-26/training/build_package.py`; regenerate it there instead of editing it here.
"""


def copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


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
