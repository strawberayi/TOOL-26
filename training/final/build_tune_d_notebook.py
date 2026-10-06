import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

INTRO = """# Model D: Stage 2 tuning (grid search on VALIDATION)

The paper allows a grid search for the Focal Loss parameters. Every candidate starts from the same Stage 1
weights (D2), trains Stage 2 with seed 42, and is scored on the **validation** set only. The winner is then trained
with seeds 43 and 44, and only then is the **test** set used (once) for the SOP tables.

| Candidate | Focal γ | Frozen layers | Status |
|---|---|---|---|
| g2.0_f11 (current full D) | 2.0 | 0–10 (whole backbone) | done |
| g1.0_f11 | 1.0 | 0–10 | this notebook |
| g1.5_f11 | 1.5 | 0–10 | this notebook |
| g2.5_f11 | 2.5 | 0–10 | this notebook |
| g2.0_f9 | 2.0 | 0–8 (SPPF + C2PSA trainable) | this notebook |

**Selection rule (fixed before training):** highest mean of validation mAP@50 and mAP@50-95 (the two SOP 1 metrics)
at each run's best epoch. α = 0.25, lr0 = 0.0002, 150 epochs, patience 50, batch 8, 640 for every candidate.

About 20–25 min per run: ~1.5 h for the 4 candidates, ~45 min for the winner's seeds 43 and 44.
If it stops, Run All again: finished runs are skipped and an interrupted one resumes.
"""

SETUP = r"""import os, sys, json, shutil, csv
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
CHOICE_FILE = HERE / 'd_tuning_choice.json'

FOCAL_ALPHA = 0.25
EPOCHS, PATIENCE, LR0, BATCH, IMG_SIZE = 150, 50, 0.0002, 8, 640
CANDIDATES = {
    'ModelD_full': (2.0, 11),
    'ModelD_g1.0_f11': (1.0, 11),
    'ModelD_g1.5_f11': (1.5, 11),
    'ModelD_g2.5_f11': (2.5, 11),
    'ModelD_g2.0_f9': (2.0, 9),
}

import torch
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT AVAILABLE')
assert torch.cuda.is_available(), 'No GPU: select the TOOL-26 (.venv) kernel.'
for seed in (42, 43, 44):
    assert (WEIGHTS_ROOT / f'best_ModelD2_lesion_ita_seed{seed}.pt').is_file(), f'Missing Stage 1 seed {seed}'
assert (RUNS / 'ModelD_full_seed42' / 'results.csv').is_file(), 'Missing the current full D run (seed 42)'
print('Ready.')
"""

FOCAL = r"""import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils import loss as uloss

FOCAL_GAMMA = 2.0


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

for g in (1.0, 1.5, 2.0, 2.5):
    pred, label = torch.randn(4, 50, 8), torch.rand(4, 50, 8)
    ours = ElementwiseFocalLoss(g, FOCAL_ALPHA)(pred, label).mean(1).sum()
    assert torch.allclose(ours, uloss.FocalLoss(gamma=g, alpha=FOCAL_ALPHA)(pred, label), atol=1e-6)
print('Focal Loss ready.')
"""

TRAIN = r"""from ultralytics import YOLO


def checkpoint_state(checkpoint: Path) -> str:
    try:
        epoch = torch.load(checkpoint, map_location='cpu', weights_only=False).get('epoch', 0)
    except Exception:
        return 'corrupt'
    return 'finished' if epoch == -1 else 'resumable'


def train_stage2(name: str, seed: int) -> Path:
    global FOCAL_GAMMA
    gamma, freeze = CANDIDATES[name]
    destination = WEIGHTS_ROOT / f'best_{name}_seed{seed}.pt'
    if destination.is_file():
        print(f'{name} seed {seed}: already finished')
        return destination
    FOCAL_GAMMA = gamma
    run = RUNS / f'{name}_seed{seed}'
    last = run / 'weights' / 'last.pt'
    state = checkpoint_state(last) if last.is_file() else 'missing'
    if state == 'corrupt':
        shutil.rmtree(run)
        state = 'missing'
    print(f'\n=== {name} seed {seed}: gamma {gamma}, freeze layers 0-{freeze - 1} ===')
    if state == 'resumable':
        YOLO(str(last)).train(resume=True)
    else:
        YOLO(str(WEIGHTS_ROOT / f'best_ModelD2_lesion_ita_seed{seed}.pt')).train(
            data=str(DATA_YAML), epochs=EPOCHS, patience=PATIENCE, imgsz=IMG_SIZE, batch=BATCH,
            seed=seed, deterministic=True, freeze=freeze, optimizer='AdamW', lr0=LR0, warmup_epochs=1,
            project=str(RUNS), name=f'{name}_seed{seed}', exist_ok=True, plots=True, verbose=True,
        )
    best = run / 'weights' / 'best.pt'
    assert best.is_file(), f'Missing validation-selected weights: {best}'
    shutil.copy2(best, destination)
    return destination


def best_validation(name: str, seed: int) -> dict:
    rows = list(csv.DictReader(open(RUNS / f'{name}_seed{seed}' / 'results.csv')))
    fit = lambda r: 0.9 * float(r['metrics/mAP50-95(B)']) + 0.1 * float(r['metrics/mAP50(B)'])
    b = max(rows, key=fit)
    m50, m5095 = 100 * float(b['metrics/mAP50(B)']), 100 * float(b['metrics/mAP50-95(B)'])
    return {'candidate': name, 'seed': seed, 'gamma': CANDIDATES[name][0], 'freeze': CANDIDATES[name][1],
            'best_epoch': int(b['epoch']), 'val_mAP50': m50, 'val_mAP50_95': m5095, 'score': (m50 + m5095) / 2}


for name in CANDIDATES:
    if name != 'ModelD_full':
        train_stage2(name, 42)
"""

SELECT = r"""import pandas as pd
table = pd.DataFrame([best_validation(n, 42) for n in CANDIDATES]).sort_values('score', ascending=False)
table.to_csv(HERE / 'd_tuning_validation.csv', index=False)
print(table.round(2).to_string(index=False))
winner = table.iloc[0]['candidate']
print(f'\nWinner on validation: {winner}')
"""

WINNER_SEEDS = r"""for seed in (43, 44):
    train_stage2(winner, seed)
CHOICE_FILE.write_text(json.dumps({'winner': winner, 'gamma': CANDIDATES[winner][0],
                                   'freeze': CANDIDATES[winner][1], 'selection': 'mean of val mAP50 and mAP50-95, seed 42'},
                                  indent=2), encoding='utf-8')
print('Saved', CHOICE_FILE)
"""

TEST = r"""import importlib
os.environ['SOP_MODELS'] = 'manuscript'
sys.path.insert(0, str(HERE))
import sop_analysis
importlib.reload(sop_analysis)
sop_analysis.main()
print((sop_analysis.OUT / 'SOP_RESULTS.md').read_text(encoding='utf-8'))
"""


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.splitlines(keepends=True)}


cells = [md(INTRO), md("## Setup"), code(SETUP), md("## Focal Loss with adjustable γ"), code(FOCAL),
         md("## Train the 4 candidates (seed 42)"), code(TRAIN),
         md("## Select on validation"), code(SELECT),
         md("## Winner: seeds 43 and 44"), code(WINNER_SEEDS),
         md("## Test set: SOP tables (A, B, C, chosen D)"), code(TEST)]
notebook = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "tool26", "display_name": "TOOL-26 (.venv)", "language": "python"},
                         "language_info": {"name": "python"}}}
out = HERE / "tune_model_D.ipynb"
out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print("Wrote", out)
