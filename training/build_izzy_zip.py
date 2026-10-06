from __future__ import annotations

import os
import shutil
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOL2026 = HERE.parent.parent
BUILD = TOOL2026 / "_for_izzy_build"
os.environ["ABLATION_PACKAGE_DEST"] = str(BUILD / "ablation_training")
sys.path.insert(0, str(HERE))
import build_package

OUT = TOOL2026 / "for_izzy_ablation_training.zip"
DANIEL_CANDIDATES = [
    TOOL2026 / "ablation_training" / "results" / "incoming" / "results_daniel.zip",
    TOOL2026 / "ablation_training" / "results" / "results_daniel.zip",
]

BAT = """@echo off
rem Izzy: double-click this file. It installs what is needed, trains A, B, C,
rem then combines with Daniel's C' and D. Run it again if it stops.
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  echo Python is not installed.
  echo Install Python 3.14 from https://www.python.org/downloads/ and tick "Add python.exe to PATH".
  pause
  exit /b 1
)
py -3.14 -c "pass" >nul 2>nul
if errorlevel 1 (
  echo Python 3.14 not found; using the default Python.
  py run_izzy.py
) else (
  py -3.14 run_izzy.py
)
if errorlevel 1 (
  echo.
  echo The run stopped. Read the message above or run_izzy_log.txt, then double-click again.
) else (
  echo.
  echo Finished. Results are in runs\\ablation_split
)
pause
"""

README = """\
# Izzy: one-click ablation training

Daniel has **finished Models C′ and D**. Their results are already in `results/incoming/`.
You train **Models A, B, C**, and the same run combines and evaluates all five.

## Before you start (once)

1. **NVIDIA driver:** https://www.nvidia.com/Download/index.aspx (or the NVIDIA App).
2. **Python 3.14:** https://www.python.org/downloads/. During installation, tick **"Add python.exe to PATH"**.
3. A good internet connection: the first run downloads PyTorch with CUDA (~2–3 GB).

## Run

**Double-click `RUN_ME_IZZY.bat`.** That's it. It:
1. checks Python and the NVIDIA driver,
2. installs PyTorch with CUDA and the other packages (only if missing),
3. trains Models A, B, C (about 2.5–3.5 hours),
4. combines them with Daniel's C′ and D and tests all five on the test set.

- Keep the laptop **plugged in**. The script stops Windows from sleeping while it runs.
- Don't close the black window until it says **Finished**.
- If it stops (power loss, window closed), **double-click again**: finished models are skipped and an
  interrupted one resumes.
- Everything is written to `run_izzy_log.txt`. If it stops with an error, send that file to Daniel.

## Results

In `runs/ablation_split/`:
- `ablation_test_metrics.csv`: mAP, precision, recall of A, B, C, C′, D
- `ablation_test_map50_95_by_bracket.csv`: per skin tone
- `trained_by.csv`: who trained each model, on which GPU
- `figures/` and `visual_proof/`: confusion matrices, curves, Model A vs D pictures

**Zip the `runs/ablation_split` folder and send it back to Daniel.**

## Notes

- Do not change any files in this folder.
- Same settings as Daniel: YOLO26n, seed 42, batch 8, image size 640, 300 epochs, patience 50.
- The notebooks in `izzy/` are what the script runs; you can also open them in VS Code instead.
- Limitation for the manuscript: A, B, C are trained on Izzy's laptop and C′, D on Daniel's.
"""


def main() -> None:
    daniel = next((p for p in DANIEL_CANDIDATES if p.is_file()), None)
    if daniel is None:
        raise SystemExit(f"results_daniel.zip not found in {DANIEL_CANDIDATES}")

    shutil.rmtree(BUILD, ignore_errors=True)
    build_package.main()
    dest = BUILD / "ablation_training"
    (dest / "results" / "incoming").mkdir(parents=True, exist_ok=True)
    shutil.copy2(daniel, dest / "results" / "incoming" / "results_daniel.zip")
    shutil.copy2(HERE / "run_izzy.py", dest / "run_izzy.py")
    (dest / "RUN_ME_IZZY.bat").write_bytes(BAT.replace("\n", "\r\n").encode("ascii"))
    (dest / "README.md").write_text(README, encoding="utf-8")

    OUT.unlink(missing_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_STORED) as zf:
        for path in sorted(dest.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(BUILD))
    shutil.rmtree(BUILD)
    print(f"Zip for Izzy: {OUT} ({OUT.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
