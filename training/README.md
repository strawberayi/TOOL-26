# Split training (source)

This folder generates the separate, self-contained training folder
`~/Documents/TOOL2026/ablation_training/` for Izzy (Models A, B, C) and
Daniel (Models C′, D), plus `ablation_training.zip` to send to Daniel.

```bash
cd ~/Documents/TOOL2026/TOOL-26
.venv/bin/python training/build_package.py
```

The instructions for Izzy and Daniel are in `ablation_training/README.md`.

- `build_package.py`: copies the code, data and pretrained checkpoint, and writes the notebooks.
- `build_notebooks.py`: builds the notebooks from `notebooks/ablation_study_yolov26.ipynb`, so they use the same code.
- `split_utils.py`: version checks, data fingerprints, packaging and importing results.
- `fingerprints.json`: reference fingerprints of the data on Izzy's laptop. Do not edit.
