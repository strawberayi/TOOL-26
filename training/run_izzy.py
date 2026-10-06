from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "run_izzy_log.txt"
VENV = ROOT / ".venv"
TRAIN_NOTEBOOK = ROOT / "izzy" / "izzy_train_A_B_C.ipynb"
COMBINE_NOTEBOOK = ROOT / "izzy" / "combine_and_evaluate.ipynb"
DANIEL_RESULTS = ROOT / "results" / "incoming" / "results_daniel.zip"

TORCH, TORCHVISION = "2.14.0", "0.29.0"
PINNED = {"ultralytics": "8.4.163", "opencv-python": "5.0.0.93", "scikit-learn": "1.9.1",
          "numpy": "2.5.3", "pillow": "12.3.0"}
EXTRA = ["pandas", "pyyaml", "matplotlib", "jinja2"]


class Tee:

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def banner(text: str) -> None:
    print("\n" + "=" * 70 + f"\n  {text}\n" + "=" * 70, flush=True)


def run(cmd, env=None):
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return result.returncode, result.stdout + result.stderr


def fail(message: str) -> None:
    print("\n*** STOPPED ***\n" + message + f"\n\nFull log: {LOG}", flush=True)
    sys.exit(1)


def installed_versions(python: str) -> dict:
    code, out = run([python, "-c", (
        "import json, importlib.metadata as m\n"
        "def v(n):\n"
        "    try: return m.version(n)\n"
        "    except Exception: return None\n"
        f"print(json.dumps({{n: v(n) for n in {list(PINNED) + EXTRA!r}}}))")])
    return json.loads(out.strip().splitlines()[-1]) if code == 0 else {}


def torch_status(python: str) -> str:
    code, out = run([python, "-c", "import torch; print(torch.__version__, torch.cuda.is_available())"])
    return out.strip().splitlines()[-1] if code == 0 and out.strip() else "not installed"


def pip_install(python: str, *args: str) -> None:
    env = dict(os.environ, UV_HTTP_TIMEOUT="600", UV_HTTP_RETRIES="5")
    if shutil.which("uv"):
        cmd = ["uv", "pip", "install", "--python", python, *args]
    else:
        if run([python, "-m", "pip", "--version"])[0] != 0:
            code, out = run([python, "-m", "ensurepip", "--upgrade"])
            if code != 0:
                raise PermissionError(out)
        cmd = [python, "-m", "pip", "install", "--timeout", "600", "--retries", "5", *args]
    print("Installing:", " ".join(a for a in args if not a.startswith("http")), "(this can take a while)", flush=True)
    out = ""
    for attempt in range(1, 4):
        code, out = run(cmd, env)
        if code == 0:
            return
        if "externally" in out.lower() or "permission denied" in out.lower():
            raise PermissionError(out[-800:])
        print(f"  attempt {attempt} failed" + ("; retrying..." if attempt < 3 else "."), flush=True)
    fail("Installing packages failed. Check the internet connection and run again.\n" + out[-1500:])


def ensure_packages(python: str, cuda_tag: str) -> None:
    status = torch_status(python)
    torch_ok = status.startswith(TORCH) and status.endswith("True")
    print(f"PyTorch: {status}" + ("  -> GPU ready" if torch_ok else "  -> installing the CUDA build"), flush=True)
    if not torch_ok:
        pip_install(python, f"torch=={TORCH}+{cuda_tag}", f"torchvision=={TORCHVISION}+{cuda_tag}",
                    "--index-url", f"https://download.pytorch.org/whl/{cuda_tag}")
        status = torch_status(python)
        if not (status.startswith(TORCH) and status.endswith("True")):
            fail(f"PyTorch was installed but cannot use the GPU ({status}). Update the NVIDIA driver and run again.")
    have = installed_versions(python)
    missing = [f"{n}=={v}" for n, v in PINNED.items() if have.get(n) != v]
    missing += [n for n in EXTRA if not have.get(n)]
    if missing:
        pip_install(python, *missing)
    print("All packages ready.", flush=True)


def setup() -> str:
    banner("Step 1/3: Checking Python, the NVIDIA driver and packages")
    if sys.version_info[:2] != (3, 14):
        print(f"Note: Python {sys.version.split()[0]}; Daniel used 3.14. Install Python 3.14 if packages fail.")
    if not DANIEL_RESULTS.is_file():
        fail(f"Daniel's results are missing: {DANIEL_RESULTS}\nPut results_daniel.zip there and run again.")

    smi = shutil.which("nvidia-smi")
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", run([smi])[1] if smi else "")
    if not match:
        fail("No NVIDIA driver found. Install it from https://www.nvidia.com/Download/index.aspx\n"
             "(or the NVIDIA App), restart the computer, then run again.")
    driver = (int(match.group(1)), int(match.group(2)))
    cuda_tag = "cu130" if driver >= (13, 0) else "cu126" if driver >= (12, 6) else None
    if cuda_tag is None:
        fail(f"The NVIDIA driver only supports CUDA {driver[0]}.{driver[1]}. Update the driver and run again.")
    print(f"NVIDIA driver OK, supports CUDA {driver[0]}.{driver[1]} -> PyTorch build {cuda_tag}", flush=True)

    python = sys.executable
    venv_python = VENV / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if venv_python.is_file():
        python = str(venv_python)
    try:
        ensure_packages(python, cuda_tag)
    except PermissionError:
        print(f"\nThis Python does not allow installing packages; creating {VENV} instead.", flush=True)
        code, out = run([sys.executable, "-m", "venv", str(VENV)])
        if code != 0:
            fail("Could not create .venv. On Ubuntu run: sudo apt install python3.14-venv\n" + out[-600:])
        python = str(venv_python)
        ensure_packages(python, cuda_tag)
    return python


def notebook_code(path: Path) -> list[str]:
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    code = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
    return [c for c in code if "Setup in one run" not in c and "GPU setup:" not in c]


def execute(path: Path, title: str) -> None:
    banner(title)
    os.chdir(path.parent)
    namespace = {"__name__": "__notebook__"}
    epochs = os.environ.get("ABLATION_SMOKE_EPOCHS")
    for index, source in enumerate(notebook_code(path), 1):
        if epochs:
            source = source.replace("EPOCHS = 300", f"EPOCHS = {epochs}")
        try:
            exec(compile(source, f"{path.name} [cell {index}]", "exec"), namespace)
        except SystemExit:
            raise
        except Exception as exc:
            import traceback
            traceback.print_exc()
            fail(f"{path.name}, cell {index} failed: {exc}\n"
                 "If a [BAD] check failed, send this log to Daniel. Otherwise run again.")


def keep_awake() -> None:
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def stage_two() -> None:
    keep_awake()
    os.environ.setdefault("MPLBACKEND", "Agg")
    start = time.time()
    execute(TRAIN_NOTEBOOK, "Step 2/3: Training Models A, B, C (about 2.5-3.5 hours)")
    execute(COMBINE_NOTEBOOK, "Step 3/3: Combining with Daniel's C' and D, evaluating all five")
    hours = (time.time() - start) / 3600
    banner(f"DONE in {hours:.1f} h")
    print(f"Results: {ROOT / 'runs' / 'ablation_split'}")
    print("  ablation_test_metrics.csv            mAP, precision, recall of A, B, C, C', D")
    print("  ablation_test_map50_95_by_bracket.csv per skin tone")
    print("  trained_by.csv                       who trained each model, on which GPU")
    print("  figures/, visual_proof/              confusion matrices, curves, pictures")
    print(f"\nSend the whole folder {ROOT / 'runs' / 'ablation_split'} back to Daniel (zip it).")


def main() -> None:
    log = LOG.open("a", encoding="utf-8")
    log.write(f"\n\n##### run started {time.strftime('%Y-%m-%d %H:%M:%S')} #####\n")
    sys.stdout = Tee(sys.__stdout__, log)
    sys.stderr = Tee(sys.__stderr__, log)

    if "--stage2" in sys.argv:
        stage_two()
        return
    python = setup()
    code = subprocess.call([python, str(Path(__file__).resolve()), "--stage2"])
    sys.exit(code)


if __name__ == "__main__":
    main()
