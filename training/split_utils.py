from __future__ import annotations

import hashlib
import importlib
import json
import platform
import zipfile
from datetime import datetime
from pathlib import Path

SPLIT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SPLIT_DIR.parent
SHARE_DIR = PROJECT_ROOT / "results"
FINGERPRINTS = SPLIT_DIR / "fingerprints.json"

EXPECTED_VERSIONS = {
    "ultralytics": "8.4.163",
    "torch": "2.14",
    "cv2": "5.0.0",
    "sklearn": "1.9.1",
    "numpy": "2.5",
}

INPUT_FILES = [
    "backend/phase0_calibration.json",
    "backend/member1_phase0_config.json",
    "datasets/ablation_yolov26/ita_table.csv",
    "datasets/source_yolo/classes.json",
]


def versions() -> dict:
    found = {}
    for name in EXPECTED_VERSIONS:
        try:
            found[name] = importlib.import_module(name).__version__
        except Exception:
            found[name] = None
    found["python"] = platform.python_version()
    return found


def gpu_name() -> str:
    import torch

    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"


def check_environment(allow_mismatch: bool = False) -> dict:
    found = versions()
    bad = {k: (found[k], want) for k, want in EXPECTED_VERSIONS.items()
           if not (found[k] or "").startswith(want)}
    for name, want in EXPECTED_VERSIONS.items():
        mark = "OK " if name not in bad else "BAD"
        print(f"  [{mark}] {name:12s} have {found[name]}   need {want}")
    print(f"  GPU: {gpu_name()}   Python {found['python']}")
    if bad and not allow_mismatch:
        raise RuntimeError(
            "Package versions differ from the seed-42 run: "
            + ", ".join(f"{k} {h} (need {w})" for k, (h, w) in bad.items())
            + ". Install the exact versions (see README) so every model is trained the same way."
        )
    return found


def _hash_tree(root: Path, suffixes: set[str] | None = None) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if suffixes and path.suffix.lower() not in suffixes:
            continue
        digest.update(str(path.relative_to(root)).encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def input_fingerprint() -> dict:
    prints = {}
    for rel in INPUT_FILES:
        path = PROJECT_ROOT / rel
        prints[rel] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    prints["datasets/source_yolo (images+labels)"] = _hash_tree(
        PROJECT_ROOT / "datasets" / "source_yolo", {".png", ".txt"})
    return prints


def dataset_fingerprint(models) -> dict:
    root = PROJECT_ROOT / "datasets" / "ablation_yolov26"
    return {
        m: hashlib.sha256((_hash_tree(root / m / "images", {".png"}) + _hash_tree(root / m / "labels", {".txt"})).encode()).hexdigest()
        for m in models
    }


def reference() -> dict:
    return json.loads(FINGERPRINTS.read_text(encoding="utf-8"))


def compare(label: str, have: dict, want: dict, strict: bool = True) -> None:
    bad = [k for k in want if have.get(k) != want[k]]
    for key in want:
        print(f"  [{'OK ' if key not in bad else 'BAD'}] {key}")
    if bad and strict:
        raise RuntimeError(
            f"{label} differ from the reference machine: {bad}. "
            "Re-copy the data kit and pull the latest code (see README); do not train on different data."
        )


def package_results(trainer: str, seed: int, models, weights_root: Path, runs_root: Path) -> Path:
    SHARE_DIR.mkdir(parents=True, exist_ok=True)
    package = SHARE_DIR / f"results_{trainer.lower()}.zip"
    missing = [m for m in models if not (weights_root / f"best_{m}_seed{seed}.pt").is_file()]
    if missing:
        raise RuntimeError(f"Training not finished: {missing}")
    epochs = {}
    with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED) as zf:
        for m in models:
            weights = weights_root / f"best_{m}_seed{seed}.pt"
            zf.write(weights, weights.relative_to(PROJECT_ROOT))
            run = runs_root / "training" / f"{m}_seed{seed}"
            for path in run.rglob("*"):
                if path.is_file() and "weights" not in path.relative_to(run).parts:
                    zf.write(path, path.relative_to(PROJECT_ROOT))
            results = run / "results.csv"
            if results.is_file():
                epochs[m] = len(results.read_text().strip().splitlines()) - 1
        meta = {
            "trainer": trainer,
            "seed": seed,
            "models": list(models),
            "created": datetime.now().isoformat(timespec="minutes"),
            "gpu": gpu_name(),
            "versions": versions(),
            "epochs_trained": epochs,
            "input_fingerprint": input_fingerprint(),
            "dataset_fingerprint": dataset_fingerprint(models),
        }
        zf.writestr("training_meta.json", json.dumps(meta, indent=2))
    print(f"Package: {package}  ({package.stat().st_size / 1e6:.1f} MB)")
    return package


def import_packages(models, strict: bool = True) -> list[dict]:
    packages = sorted(set(SHARE_DIR.glob("results_*.zip")) | set((SHARE_DIR / "incoming").glob("results_*.zip")))
    ref = reference()
    metas = []
    for package in packages:
        with zipfile.ZipFile(package) as zf:
            meta = json.loads(zf.read("training_meta.json"))
            print(f"\n{package.name}: {meta['trainer']} trained {meta['models']} (seed {meta['seed']}) on {meta['gpu']}")
            compare("Inputs", meta["input_fingerprint"], ref["inputs"], strict)
            compare("Preprocessed datasets", meta["dataset_fingerprint"],
                    {m: ref["datasets"][m] for m in meta["models"]}, strict)
            for name in zf.namelist():
                if name != "training_meta.json":
                    zf.extract(name, PROJECT_ROOT)
        metas.append(meta)
    print("\nThis machine:")
    compare("Inputs", input_fingerprint(), ref["inputs"], strict)
    compare("Preprocessed datasets", dataset_fingerprint(models), ref["datasets"], strict)
    return metas


def write_reference(models) -> None:
    FINGERPRINTS.write_text(json.dumps({
        "created": datetime.now().isoformat(timespec="minutes"),
        "versions": versions(),
        "inputs": input_fingerprint(),
        "datasets": dataset_fingerprint(models),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {FINGERPRINTS}")
