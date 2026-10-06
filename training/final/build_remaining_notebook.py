import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

INTRO = """# Remaining ablation runs (manuscript Models A, B, C)

Run this **after** `train_model_D_full.ipynb` has finished (one training at a time on the 4 GB GPU).

| Order | Run | Why | Time |
|---|---|---|---|
| 1 | **Model C: raw images + Focal Loss, seed 42** | Needed for the SOP (A, B, D already have seed 42) | ~45 min |
| 2–3 | Model A (raw), seeds 43, 44 | 3 seeds for every model | ~45 min each |
| 4–5 | Model B (fixed L*-CLAHE β = 2.0), seeds 43, 44 | 3 seeds for every model | ~45 min each |
| 6–7 | Model C (Focal Loss), seeds 43, 44 | 3 seeds for every model | ~45 min each |

Every run uses the same settings as the existing models: YOLO26n, 300 epochs, patience 50, batch 8, 640, deterministic,
optimizer auto. Model C differs from Model A **only** in the loss (Focal Loss, α = 0.25, γ = 2.0).

If it stops, Run All again: finished runs are skipped and an interrupted one resumes.
After run 1 the SOP analysis can already be done while the rest keeps training.
"""

SETUP = r"""import os, shutil
from pathlib import Path
os.environ.setdefault('MPLBACKEND', 'Agg')

HERE = Path.cwd().resolve()
if not (HERE / 'sop_analysis.py').is_file():
    HERE = HERE / 'final'
assert (HERE / 'sop_analysis.py').is_file(), 'Open this notebook from ablation_training/final.'
ROOT = HERE.parent
DATASETS = ROOT / 'datasets' / 'ablation_yolov26'
WEIGHTS_ROOT = ROOT / 'weights' / 'ablation_split'
RUNS = ROOT / 'runs' / 'ablation_split' / 'training'
CHECKPOINT = ROOT / 'daniel' / 'yolo26n.pt'   # same pretrained start as every other model

EPOCHS, PATIENCE, BATCH, IMG_SIZE = 300, 50, 8, 640
FOCAL_ALPHA, FOCAL_GAMMA = 0.25, 2.0

# (run name, dataset folder, Focal Loss?, seed), in training order.
RUNS_TODO = [
    ('ModelC_focal', 'ModelA_raw', True, 42),
    ('ModelA_raw', 'ModelA_raw', False, 43),
    ('ModelA_raw', 'ModelA_raw', False, 44),
    ('ModelC_fixed_l_clahe', 'ModelC_fixed_l_clahe', False, 43),
    ('ModelC_fixed_l_clahe', 'ModelC_fixed_l_clahe', False, 44),
    ('ModelC_focal', 'ModelA_raw', True, 43),
    ('ModelC_focal', 'ModelA_raw', True, 44),
]

import torch
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT AVAILABLE')
assert torch.cuda.is_available(), 'No GPU: select the TOOL-26 (.venv) kernel.'
assert CHECKPOINT.is_file(), f'Missing {CHECKPOINT}'
for _, folder, _, _ in RUNS_TODO:
    assert (DATASETS / folder / 'data.yaml').is_file(), f'Missing dataset {folder}'
print('Ready:', len(RUNS_TODO), 'runs')
"""

FOCAL = r"""# Focal Loss for Model C only: switched on per run with USE_FOCAL.
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils import loss as uloss

USE_FOCAL = False


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

def _init_maybe_focal(self, *args, **kwargs):
    uloss.v8DetectionLoss._init_without_focal(self, *args, **kwargs)
    if USE_FOCAL:
        self.bce = ElementwiseFocalLoss(FOCAL_GAMMA, FOCAL_ALPHA)

uloss.v8DetectionLoss.__init__ = _init_maybe_focal

# Check: identical to Ultralytics' own FocalLoss.
pred, label = torch.randn(4, 50, 8), torch.rand(4, 50, 8)
ours = ElementwiseFocalLoss(FOCAL_GAMMA, FOCAL_ALPHA)(pred, label).mean(1).sum()
assert torch.allclose(ours, uloss.FocalLoss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA)(pred, label), atol=1e-6)
print(f'Focal Loss ready (alpha={FOCAL_ALPHA}, gamma={FOCAL_GAMMA}); used only for ModelC_focal.')
"""

TRAIN = r"""from ultralytics import YOLO


def checkpoint_state(checkpoint: Path) -> str:
    try:
        epoch = torch.load(checkpoint, map_location='cpu', weights_only=False).get('epoch', 0)
    except Exception:
        return 'corrupt'
    return 'finished' if epoch == -1 else 'resumable'


def train_one(name: str, folder: str, focal: bool, seed: int) -> Path:
    global USE_FOCAL
    destination = WEIGHTS_ROOT / f'best_{name}_seed{seed}.pt'
    if destination.is_file():
        print(f'{name} seed {seed}: already finished')
        return destination
    USE_FOCAL = focal   # read when the trainer builds the loss
    run = RUNS / f'{name}_seed{seed}'
    last = run / 'weights' / 'last.pt'
    state = checkpoint_state(last) if last.is_file() else 'missing'
    if state == 'corrupt':
        print(f'{last} is unreadable; restarting {name} seed {seed}.')
        shutil.rmtree(run)
        state = 'missing'
    print(f'\n=== {name} seed {seed} ({"Focal Loss" if focal else "standard loss"}, data {folder}) ===')
    if state == 'resumable':
        YOLO(str(last)).train(resume=True)
    else:
        YOLO(str(CHECKPOINT)).train(
            data=str(DATASETS / folder / 'data.yaml'), epochs=EPOCHS, patience=PATIENCE, imgsz=IMG_SIZE,
            batch=BATCH, seed=seed, deterministic=True, project=str(RUNS), name=f'{name}_seed{seed}',
            exist_ok=True, plots=True, verbose=True,
        )
    USE_FOCAL = False
    best = run / 'weights' / 'best.pt'
    assert best.is_file(), f'Missing validation-selected weights: {best}'
    shutil.copy2(best, destination)
    return destination


for run_args in RUNS_TODO:
    train_one(*run_args)
print('\nAll remaining runs finished.')
"""

HEADLESS = """## Run headless (recommended)

From `ablation_training/final`, after the full D notebook has finished:

```bash
setsid nohup ../../TOOL-26/.venv/bin/jupyter nbconvert --to notebook --execute train_remaining.ipynb \\
  --output train_remaining_done.ipynb --ExecutePreprocessor.timeout=-1 \\
  --ExecutePreprocessor.kernel_name=tool26 > train_remaining_log.txt 2>&1 &
```

Progress: `tail -f train_remaining_log.txt`. Weights appear in `weights/ablation_split/`.
"""


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.splitlines(keepends=True)}


cells = [md(INTRO), md("## Setup"), code(SETUP), md("## Focal Loss (Model C only)"), code(FOCAL),
         md("## Train (Model C seed 42 first)"), code(TRAIN), md(HEADLESS)]
notebook = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "tool26", "display_name": "TOOL-26 (.venv)", "language": "python"},
                         "language_info": {"name": "python"}}}
out = HERE / "train_remaining.ipynb"
out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print("Wrote", out)
