"""Write train_model_D_full.ipynb (the paper's full Model D: Stage 2 with frozen backbone + Focal Loss).

    ../TOOL-26/.venv/bin/python final/build_full_d_notebook.py
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

INTRO = """# Model D (full): ITA-guided L*-CLAHE + two-stage decoupled training + Focal Loss

This is the complete Model D described in the paper.

| Stage | What | Status |
|---|---|---|
| Stage 1: localization | YOLO26n trained on ITA-adaptive L*-CLAHE images (normal loss) | **Done**: D2 weights, seeds 42, 43, 44 |
| Stage 2: fine-grained classification | Start from Stage 1, **freeze backbone layers 0–10**, train with **Focal Loss** (α = 0.25, γ = 2.0) | **This notebook** |

**Run All once** (or headless, see the last cell). About 30–45 min per seed, 3 seeds.
If it stops, Run All again: finished seeds are skipped and an interrupted one resumes.

Rules kept for a fair comparison:
- Same data, split, image size (640), batch (8) and seeds as Models A, B, C.
- The best epoch is chosen on the **validation** set; the **test** set is only used at the end, once.
- All three seeds are reported (mean ± SD); no seed is picked by its test score.
"""

SETUP = r"""import os, sys, shutil
from pathlib import Path
os.environ.setdefault('MPLBACKEND', 'Agg')

HERE = Path.cwd().resolve()
if not (HERE / 'sop_analysis.py').is_file():
    HERE = HERE / 'final'
assert (HERE / 'sop_analysis.py').is_file(), 'Open this notebook from ablation_training/final.'
ROOT = HERE.parent
DATA_YAML = ROOT / 'datasets' / 'ablation_yolov26' / 'ModelD2_lesion_ita' / 'data.yaml'
WEIGHTS_ROOT = ROOT / 'weights' / 'ablation_split'
RUNS = ROOT / 'runs' / 'ablation_split' / 'training'

# ---- Stage 2 settings (from the paper; do not change between seeds)
SEEDS = (42, 43, 44)
FOCAL_ALPHA = 0.25   # paper: alpha = 0.25
FOCAL_GAMMA = 2.0    # paper: gamma = 2.0
FREEZE = 11          # freeze layers 0-10 = the YOLO26n backbone (Conv ... SPPF, C2PSA)
EPOCHS = 150         # Stage 2 is fine-tuning; early stopping usually ends it sooner
PATIENCE = 50        # same patience as every other model
LR0 = 0.0002         # lower than Stage 1 (AdamW 0.000833) so the localization features are not disturbed
BATCH, IMG_SIZE = 8, 640

import torch
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT AVAILABLE')
assert torch.cuda.is_available(), 'No GPU: select the TOOL-26 (.venv) kernel.'
assert DATA_YAML.is_file(), f'Missing D2 dataset: {DATA_YAML}'
for seed in SEEDS:
    stage1 = WEIGHTS_ROOT / f'best_ModelD2_lesion_ita_seed{seed}.pt'
    assert stage1.is_file(), f'Missing Stage 1 weights: {stage1}'
print('Stage 1 weights found for seeds', SEEDS)
"""

FOCAL = r"""# Focal Loss replaces the BCE class loss of YOLO26 (both the one-to-many and one-to-one heads).
# Ultralytics sums the element-wise class loss itself, so the focal term must stay element-wise.
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils import loss as uloss


class ElementwiseFocalLoss(nn.Module):
    def __init__(self, gamma: float, alpha: float):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, pred, label):
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction='none')
        prob = pred.sigmoid()
        p_t = label * prob + (1 - label) * (1 - prob)
        loss = loss * (1.0 - p_t) ** self.gamma
        if self.alpha > 0:
            loss = loss * (label * self.alpha + (1 - label) * (1 - self.alpha))
        return loss


if not hasattr(uloss.v8DetectionLoss, '_init_without_focal'):
    uloss.v8DetectionLoss._init_without_focal = uloss.v8DetectionLoss.__init__

def _init_with_focal(self, *args, **kwargs):
    uloss.v8DetectionLoss._init_without_focal(self, *args, **kwargs)
    self.bce = ElementwiseFocalLoss(FOCAL_GAMMA, FOCAL_ALPHA)

uloss.v8DetectionLoss.__init__ = _init_with_focal

# Check 1: same values as Ultralytics' own FocalLoss.
pred, label = torch.randn(4, 50, 8), torch.rand(4, 50, 8)
ours = ElementwiseFocalLoss(FOCAL_GAMMA, FOCAL_ALPHA)(pred, label).mean(1).sum()
ref = uloss.FocalLoss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA)(pred, label)
assert torch.allclose(ours, ref, atol=1e-6), (ours, ref)

# Check 2: the YOLO26 criterion really uses Focal Loss in both heads.
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
m = YOLO(str(WEIGHTS_ROOT / 'best_ModelD2_lesion_ita_seed42.pt')).model
m.args = get_cfg()
crit = m.init_criterion()
heads = [crit.one2many, crit.one2one] if hasattr(crit, 'one2many') else [crit]
assert all(isinstance(h.bce, ElementwiseFocalLoss) for h in heads)
del m, crit
print(f'Focal Loss active (alpha={FOCAL_ALPHA}, gamma={FOCAL_GAMMA}) in {len(heads)} detection head(s).')
"""

TRAIN = r"""def checkpoint_state(checkpoint: Path) -> str:
    try:
        epoch = torch.load(checkpoint, map_location='cpu', weights_only=False).get('epoch', 0)
    except Exception:
        return 'corrupt'
    return 'finished' if epoch == -1 else 'resumable'


def train_stage2(seed: int) -> Path:
    name = f'ModelD_full_seed{seed}'
    destination = WEIGHTS_ROOT / f'best_ModelD_full_seed{seed}.pt'
    if destination.is_file():
        print(f'Seed {seed}: already finished -> {destination.name}')
        return destination
    run = RUNS / name
    last = run / 'weights' / 'last.pt'
    state = checkpoint_state(last) if last.is_file() else 'missing'
    if state == 'corrupt':
        print(f'{last} is unreadable; restarting seed {seed}.')
        shutil.rmtree(run)
        state = 'missing'
    if state == 'resumable':
        print(f'Seed {seed}: resuming from {last}')
        YOLO(str(last)).train(resume=True)
    elif state == 'missing':
        print(f'Seed {seed}: Stage 2 from best_ModelD2_lesion_ita_seed{seed}.pt '
              f'(freeze layers 0-{FREEZE - 1}, Focal Loss)')
        YOLO(str(WEIGHTS_ROOT / f'best_ModelD2_lesion_ita_seed{seed}.pt')).train(
            data=str(DATA_YAML), epochs=EPOCHS, patience=PATIENCE, imgsz=IMG_SIZE, batch=BATCH,
            seed=seed, deterministic=True, freeze=FREEZE, optimizer='AdamW', lr0=LR0, warmup_epochs=1,
            project=str(RUNS), name=name, exist_ok=True, plots=True, verbose=True,
        )
    best = run / 'weights' / 'best.pt'
    assert best.is_file(), f'Missing validation-selected weights: {best}'
    shutil.copy2(best, destination)
    return destination


for seed in SEEDS:
    train_stage2(seed)
print('Stage 2 finished for seeds', SEEDS)
"""

VALIDATE = r"""# Decide on VALIDATION: does Stage 2 + Focal Loss improve over Stage 1 (D2)?
import pandas as pd
rows = []
for seed in SEEDS:
    for label, weights in (('D2 (Stage 1 only)', f'best_ModelD2_lesion_ita_seed{seed}.pt'),
                           ('Full D (Stage 2 + Focal)', f'best_ModelD_full_seed{seed}.pt')):
        m = YOLO(str(WEIGHTS_ROOT / weights)).val(data=str(DATA_YAML), split='val', batch=BATCH, imgsz=IMG_SIZE,
                                                    workers=0, plots=False, verbose=False)
        rows.append({'model': label, 'seed': seed, 'mAP50': 100 * m.box.map50, 'mAP50_95': 100 * m.box.map,
                     'precision': 100 * m.box.mp, 'recall': 100 * m.box.mr})
val = pd.DataFrame(rows)
val.to_csv(HERE / 'full_D_validation.csv', index=False)
summary = val.groupby('model')[['mAP50', 'mAP50_95', 'precision', 'recall']].agg(['mean', 'std']).round(1)
print(summary.to_string())
"""

TEST = r"""# Final report on the TEST set: A, B, C and the full D, all SOP tables and statistical tests.
import importlib
os.environ['SOP_MODEL_D'] = 'full'
sys.path.insert(0, str(HERE))
import sop_analysis
importlib.reload(sop_analysis)
sop_analysis.main()
print((sop_analysis.OUT / 'SOP_RESULTS.md').read_text(encoding='utf-8'))
"""

HEADLESS = """## Run headless (recommended: less RAM, no VS Code)

In a terminal, from `ablation_training/final`:

```bash
setsid nohup ../../TOOL-26/.venv/bin/jupyter nbconvert --to notebook --execute train_model_D_full.ipynb \\
  --output train_model_D_full_done.ipynb --ExecutePreprocessor.timeout=-1 \\
  --ExecutePreprocessor.kernel_name=tool26 > train_full_D_log.txt 2>&1 &
```

Progress: `tail -f train_full_D_log.txt` or `runs/ablation_split/training/ModelD_full_seed*/results.csv`.
Results: `full_D_validation.csv` and `sop_results_full_D/SOP_RESULTS.md`.
"""


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.splitlines(keepends=True)}


cells = [
    md(INTRO),
    md("## Setup"), code(SETUP),
    md("## Focal Loss (α = 0.25, γ = 2.0)\nReplaces the class loss and checks that it is really used."), code(FOCAL),
    md("## Stage 2 training (frozen backbone + Focal Loss), seeds 42, 43, 44"), code(TRAIN),
    md("## Validation: Stage 1 (D2) vs full D\nThis is the decision table (the test set is not used here)."),
    code(VALIDATE),
    md("## Test set: SOP tables for A, B, C and the full D"), code(TEST),
    md(HEADLESS),
]
notebook = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "tool26", "display_name": "TOOL-26 (.venv)", "language": "python"},
                         "language_info": {"name": "python"}}}
out = HERE / "train_model_D_full.ipynb"
out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print("Wrote", out)
