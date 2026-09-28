"""
Generate the split-training notebooks from notebooks/ablation_study_yolov26.ipynb,
so they use exactly the same preprocessing, training and evaluation code.

Called by build_package.py, which writes them into the separate
ablation_training/ folder.
"""

from __future__ import annotations

import json
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parent
MAIN = TRAINING_DIR.parent / "notebooks" / "ablation_study_yolov26.ipynb"
cells = [("".join(c["source"]), c["cell_type"]) for c in json.loads(MAIN.read_text())["cells"]]

TRAINERS = {
    "Izzy": ("ModelA_raw", "ModelB_rgb_clahe", "ModelC_fixed_l_clahe"),
    "Daniel": ("ModelC2_fixed_l_clahe_global", "ModelD_proposed"),
}
FILES = {"Izzy": "izzy_train_A_B_C.ipynb", "Daniel": "daniel_train_C2_D.ipynb"}
LABELS = {"Izzy": "A, B, C", "Daniel": "C′, D"}


def main_cell(marker: str) -> str:
    matches = [src for src, kind in cells if kind == "code" and marker in src]
    assert len(matches) == 1, marker
    return matches[0]


def replace_line(src: str, prefix: str, new_line: str) -> str:
    lines = src.splitlines()
    hits = [i for i, line in enumerate(lines) if line.startswith(prefix)]
    assert len(hits) == 1, prefix
    lines[hits[0]] = new_line
    return "\n".join(lines)


def config(train_models=None) -> str:
    src = main_cell("MODEL_CHECKPOINT = ")
    src = replace_line(src, "SEEDS = ", "SEEDS = (42,)                    # Same seed as the original protocol.")
    src = replace_line(src, "PRIMARY_SEED = ", "PRIMARY_SEED = 42")
    # This split round keeps its own runs and weights; the first run stays untouched.
    src = replace_line(src, "RUNS_ROOT = ", "RUNS_ROOT = PROJECT_ROOT / 'runs' / 'ablation_split'")
    src = replace_line(src, "WEIGHTS_ROOT = ", "WEIGHTS_ROOT = PROJECT_ROOT / 'weights' / 'ablation_split'")
    if train_models:
        src += f"\n\nTRAIN_MODELS = {train_models!r}  # the models this trainer trains"
    return src


SETUP = r"""
import sys
from pathlib import Path

HERE = Path.cwd().resolve()
TRAINING_DIR = next((d for d in (HERE, HERE / 'training', HERE.parent / 'training') if (d / 'split_utils.py').is_file()), None)
assert TRAINING_DIR, 'Open this notebook from its own folder inside ablation_training/.'
sys.path.insert(0, str(TRAINING_DIR))
import split_utils

print('Environment check:')
split_utils.check_environment()
print('\nInput data check (must match the reference):')
split_utils.compare('Inputs', split_utils.input_fingerprint(), split_utils.reference()['inputs'])
"""

ITA = r"""
# ITA per image comes from the shared ita_table.csv (in the data kit), so every
# machine uses exactly the same Model D clip limits.
ITA_CACHE = ABLATION_ROOT / 'ita_table.csv'
assert ITA_CACHE.is_file(), 'Missing datasets/ablation_yolov26/ita_table.csv: the ablation_training folder is incomplete.'
member1_config = json.loads(MEMBER1_CONFIG_JSON.read_text(encoding='utf-8'))
processor = MaskingITAProcessor(MaskingITAConfig(**member1_config['masking']))
ita_table = pd.read_csv(ITA_CACHE)
ITA_BY_ID = ita_table.set_index('image_id')
print(pd.crosstab(ita_table['split'], ita_table['bracket'].fillna('no ITA'), margins=True))
"""

DATASET_CHECK = r"""
print('Preprocessed dataset check (must be identical on every machine):')
reference = split_utils.reference()['datasets']
split_utils.compare('Preprocessed datasets', split_utils.dataset_fingerprint(TRAIN_MODELS),
                    {m: reference[m] for m in TRAIN_MODELS})
"""


def dataset_cell(only_trainer_models: bool) -> str:
    src = main_cell("def make_dataset(")
    if not only_trainer_models:
        return src
    head = src[: src.index("for model_name in MODELS:\n    print(f'Preparing")]
    return head + """for model_name in TRAIN_MODELS:
    print(f'Preparing {model_name}...')
    make_dataset(model_name)

if 'ModelD_proposed' in TRAIN_MODELS:
    audit_d = pd.read_csv(ABLATION_ROOT / 'ModelD_proposed' / 'preprocessing_audit.csv')
    print('\\nModel D beta usage per split:')
    print(pd.crosstab(audit_d['split'], audit_d['beta']))
"""


def yaml_cell(only_trainer_models: bool) -> str:
    src = main_cell("def write_data_yaml(")
    if only_trainer_models:
        assert src.count("in MODELS") == 2
        src = src.replace("in MODELS", "in TRAIN_MODELS")
    return src


def training_cell() -> str:
    src = main_cell("def train_one(")
    src = replace_line(src, "TRAIN_ONLY = ", "TRAIN_ONLY = TRAIN_MODELS")
    # Stable best_<model>.pt copies are made by the combine notebook.
    src = src[: src.index("# Stable names for the primary seed")].rstrip()
    return src.replace("print('Still to train:', missing or 'none')",
                       "print('Still to train here:', [m for m in missing if m.split(' seed')[0] in TRAIN_MODELS] or 'none')")


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")}


def trainer_notebook(trainer: str) -> list[dict]:
    models = TRAINERS[trainer]
    return [
        md(f"""
# Ablation training: {trainer} (Models {LABELS[trainer]})

Trains **Models {LABELS[trainer]}** with seed 42, using exactly the same data, code and settings as everyone else.
Then it zips the results into `results/results_{trainer.lower()}.zip`.

**Run All once.** If it stops (power loss, crash), just **Run All** again:
- Finished models are skipped.
- An interrupted model resumes from its last epoch.

Before you start, follow `README.md` in the `ablation_training` folder (install the exact versions).
Each model takes about **45–75 minutes** on an RTX 3050 laptop. Keep the laptop plugged in and awake.
"""),
        code(SETUP),
        code(config(models)),
        code(main_cell("IMAGE_SUFFIXES = ")),
        code(ITA),
        code(dataset_cell(True)),
        code(DATASET_CHECK),
        code(yaml_cell(True)),
        md(f"## Train Models {LABELS[trainer]}"),
        code(training_cell()),
        code(main_cell("# Training curves (primary seed)")),
        md("## Package the results"),
        code(f"""
package = split_utils.package_results({trainer!r}, SEEDS[0], TRAIN_MODELS, WEIGHTS_ROOT, RUNS_ROOT)
print('Send this file to Izzy, who puts it in results/incoming/:' if {trainer!r} != 'Izzy' else 'Keep it here; combine_and_evaluate.ipynb picks it up.')
print(package)
"""),
    ]


IMPORT = r"""
# Unpack results_izzy.zip (results/) and results_daniel.zip (results/incoming/),
# checking every machine used identical data.
metas = split_utils.import_packages(MODELS)
trained_by = pd.DataFrame([
    {'model': m, 'trainer': meta['trainer'], 'gpu': meta['gpu'], 'epochs': meta['epochs_trained'].get(m),
     'ultralytics': meta['versions']['ultralytics'], 'torch': meta['versions']['torch']}
    for meta in metas for m in meta['models']
]).set_index('model').loc[list(MODELS)]
RUNS_ROOT.mkdir(parents=True, exist_ok=True)
trained_by.to_csv(RUNS_ROOT / 'trained_by.csv')
trained_by
"""

COLLECT = r"""
# No training here: gather the validation-selected weights of every model.
best_weights = {
    (model_name, seed): WEIGHTS_ROOT / f'best_{model_name}_seed{seed}.pt'
    for model_name in MODELS for seed in SEEDS
    if (WEIGHTS_ROOT / f'best_{model_name}_seed{seed}.pt').is_file()
}
missing = [f'{m} seed {s}' for m in MODELS for s in SEEDS if (m, s) not in best_weights]
assert not missing, f'Missing weights (put results_daniel.zip in results/incoming/): {missing}'

def run_dir(model_name: str, seed: int) -> Path:
    return RUNS_ROOT / 'training' / f'{model_name}_seed{seed}'

# Stable names used by the app export: best_ModelA_raw.pt ... best_ModelD_proposed.pt
for model_name in MODELS:
    shutil.copy2(best_weights[(model_name, PRIMARY_SEED)], WEIGHTS_ROOT / f'best_{model_name}.pt')
print(f'{len(best_weights)} models ready:', sorted(m for m, _ in best_weights))
"""

APP = r"""
# Optional: put these models and their test results into the app.
UPDATE_APP = False  # set True to replace the app's models and Ablation Benchmark with this round

if UPDATE_APP:
    import os
    import subprocess
    TOOL26 = PROJECT_ROOT.parent / 'TOOL-26'  # the app project next to ablation_training/
    assert (TOOL26 / 'backend' / 'export_app_models.py').is_file(), f'TOOL-26 not found at {TOOL26}'
    env = dict(os.environ, ABLATION_RUNS=str(RUNS_ROOT), ABLATION_WEIGHTS=str(WEIGHTS_ROOT))
    for script in ('export_app_models.py', 'export_app_benchmark.py'):
        out = subprocess.run([sys.executable, str(TOOL26 / 'backend' / script)],
                             capture_output=True, text=True, cwd=TOOL26, env=env)
        print(script, 'OK' if out.returncode == 0 else 'FAILED', out.stdout[-400:], out.stderr[-400:])
    print('Now rebuild the APK: TOOL-26/phone-development/build-apk.sh')
else:
    print('App not changed (UPDATE_APP = False).')
"""


def combine_notebook() -> list[dict]:
    return [
        md("""
# Combine Izzy's and Daniel's models and evaluate

Run this **once, on Izzy's laptop**, after both training notebooks are finished.
Put Daniel's `results_daniel.zip` in `results/incoming/`.
Izzy's own `results_izzy.zip` is picked up from `results/` automatically.

This notebook does **not train**. It:
1. imports the weights of all five models,
2. checks that both machines used identical data,
3. evaluates A, B, C, C′, D on the held-out test set (on this one machine),
4. writes the per-skin-tone results, figures and visual proof,
5. optionally updates the app.

Results go to `ablation_training/runs/ablation_split/`. Nothing in `TOOL-26` changes unless you turn on `UPDATE_APP`.
"""),
        code(SETUP),
        code(config()),
        code(main_cell("IMAGE_SUFFIXES = ")),
        code(ITA),
        code(dataset_cell(False)),
        code(yaml_cell(False)),
        md("## Import both trainers' results"),
        code(IMPORT),
        code(COLLECT),
        code(main_cell("# Training curves (primary seed)")),
        md("## Final evaluation on the held-out test split"),
        code(main_cell("def evaluate(")),
        code(main_cell("# Per-skin-tone evaluation")),
        code(main_cell("# YOLO writes the confusion matrix")),
        md("## Visual proof: Model A vs Model D"),
        code(main_cell("def draw_ground_truth(")),
        md("## Update the app (optional)"),
        code(APP),
        md("""
## For the manuscript
- Report `mAP@0.5` and `mAP@0.5:0.95` from `runs/ablation_split/ablation_test_metrics.csv`.
- **Limitation:** Models A, B, C were trained on Izzy's laptop and C′, D on Daniel's (see `runs/ablation_split/trained_by.csv`). Different GPUs can shift results slightly, so small differences between the two groups should be interpreted with care.
- Everything else was identical and verified: package versions, seed 42, settings, and byte-identical input and preprocessed data (fingerprint checks above). All models were evaluated on the same machine.
"""),
    ]


def write(path: Path, notebook_cells: list[dict]) -> None:
    for i, cell in enumerate(notebook_cells):
        cell["id"] = f"cell-{i:02d}"
        lines = cell["source"].split("\n")
        cell["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    payload = {
        "cells": notebook_cells,
        "metadata": {
            "kernelspec": {"display_name": "TOOL-26 (.venv)", "language": "python", "name": "tool26"},
            "language_info": {"name": "python", "version": "3.14"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", path)
