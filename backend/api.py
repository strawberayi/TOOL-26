import base64
import json
import os
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from clahe_calibration import apply_clahe, ita_to_beta
from masking_ita import MaskingITAConfig, MaskingITAProcessor, MaskingITAResult, ProcessingStatus

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

# Model D weights by default; override with IDENTISKIN_WEIGHTS=/path/to/best.pt
WEIGHTS = Path(os.environ.get(
    "IDENTISKIN_WEIGHTS",
    PROJECT_ROOT / "weights" / "ablation_yolov26" / "best_ModelD_proposed.pt",
))
# Inference runs on CPU by default so it never competes with GPU training.
DEVICE = os.environ.get("IDENTISKIN_DEVICE", "cpu")
# Phone photos are downscaled for speed; ITA is a mean colour and the
# detector sees 640 px anyway.
MAX_SIDE = int(os.environ.get("IDENTISKIN_MAX_SIDE", "1280"))
CONFIDENCE = float(os.environ.get("IDENTISKIN_CONF", "0.25"))
# ITA is a mean skin colour, so the K-means mask runs on a small copy.
ITA_MAX_SIDE = int(os.environ.get("IDENTISKIN_ITA_MAX_SIDE", "512"))

calibration = json.loads((BACKEND_DIR / "phase0_calibration.json").read_text(encoding="utf-8"))
member1_config = json.loads((BACKEND_DIR / "member1_phase0_config.json").read_text(encoding="utf-8"))
processor = MaskingITAProcessor(MaskingITAConfig(**member1_config["masking"]))
TILE_GRID = tuple(calibration["CLAHE"]["tile_grid_size"])

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ABLATION_WEIGHTS = PROJECT_ROOT / "weights" / "ablation_yolov26"
FIXED_BETA = 2.0  # Models B and C, as in the notebook.

# id, weights file, description. Preprocessing matches the ablation notebook.
COMPARE_MODELS = [
    ("A", "best_ModelA_raw.pt", "Raw image"),
    ("B", "best_ModelB_rgb_clahe.pt", f"RGB CLAHE, β={FIXED_BETA}"),
    ("C", "best_ModelC_fixed_l_clahe.pt", f"L*-CLAHE, β={FIXED_BETA}"),
    ("C′", "best_ModelC2_fixed_l_clahe_global.pt", f"L*-CLAHE, β={calibration['beta_global']} (no ITA)"),
    ("D", "best_ModelD_proposed.pt", "K-means + ITA → L*-CLAHE"),
]

_models = {}


def load_yolo(path: Path):
    if path not in _models:
        from ultralytics import YOLO
        _models[path] = YOLO(str(path))
    return _models[path]


def get_model():
    if not WEIGHTS.is_file():
        raise HTTPException(503, f"Model weights not found: {WEIGHTS}")
    return load_yolo(WEIGHTS)


def read_and_resize(contents: bytes):
    try:
        rgb = processor.load_image_bytes(contents)
    except Exception as exc:
        raise HTTPException(400, f"Invalid image: {exc}")
    height, width = rgb.shape[:2]
    scale = min(1.0, MAX_SIDE / max(height, width))
    if scale < 1.0:
        rgb = cv2.resize(rgb, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    return rgb


def compute_ita(rgb):
    # ITA is a mean skin colour, so the K-means mask runs on a small copy.
    height, width = rgb.shape[:2]
    scale = min(1.0, ITA_MAX_SIDE / max(height, width))
    small = rgb if scale == 1.0 else cv2.resize(
        rgb, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    return processor.process_image(small, filename="upload")


def model_d_beta(ita_result):
    if ita_result.ita is not None:
        return ita_to_beta(ita_result.ita, calibration["beta_high"], calibration["beta_mid"], calibration["beta_low"]), "ita_bracket"
    # Same rule as the notebook: no ITA -> ITA-agnostic beta_global.
    return float(calibration["beta_global"]), "beta_global_fallback"


def rgb_clahe(rgb, beta):
    clahe = cv2.createCLAHE(clipLimit=beta, tileGridSize=TILE_GRID)
    return cv2.merge([clahe.apply(channel) for channel in cv2.split(rgb)])


def to_jpeg_data_url(rgb, max_side: int = 800) -> str | None:
    h, w = rgb.shape[:2]
    if max(h, w) > max_side:
        k = max_side / max(h, w)
        rgb = cv2.resize(rgb, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode(".jpg", rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(jpeg.tobytes()).decode() if ok else None


def summarize(prediction, names):
    totals = {}
    for cls, conf in zip(prediction.boxes.cls.tolist(), prediction.boxes.conf.tolist()):
        totals[names[int(cls)]] = totals.get(names[int(cls)], 0.0) + conf
    top = max(totals, key=totals.get) if totals else None
    top_conf = max((c for k, c in zip(prediction.boxes.cls.tolist(), prediction.boxes.conf.tolist())
                    if names[int(k)] == top), default=None)
    return top, (round(float(top_conf), 4) if top_conf is not None else None)


@app.get("/health")
def health():
    return {"status": "ok", "weights": str(WEIGHTS), "weights_found": WEIGHTS.is_file(),
            "calibration_status": calibration.get("status")}


@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        image_rgb = processor.load_image_bytes(contents)
        result = processor.process_image(image_rgb, filename=file.filename or "upload")
    except Exception as exc:
        result = MaskingITAResult(
            filename=file.filename or "upload",
            status=ProcessingStatus.IMAGE_FAILED.value,
            failure_reason=str(exc),
            user_message="Invalid image; please retake image.",
        )
    return result.to_dict()


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """Model D pipeline: skin mask + ITA -> bracket beta -> L*-CLAHE -> YOLO."""
    rgb = read_and_resize(await file.read())
    ita_result = compute_ita(rgb)
    beta, beta_source = model_d_beta(ita_result)
    enhanced = apply_clahe(rgb, beta=beta, tile_grid_size=TILE_GRID)

    model = get_model()
    prediction = model.predict(enhanced[:, :, ::-1].copy(), conf=CONFIDENCE, device=DEVICE, verbose=False)[0]
    h, w = enhanced.shape[:2]
    detections = []
    for box, cls, conf in zip(prediction.boxes.xyxy.tolist(), prediction.boxes.cls.tolist(), prediction.boxes.conf.tolist()):
        detections.append({
            "label": model.names[int(cls)],
            "confidence": round(float(conf), 4),
            # Normalised to the image so the app can draw at any size.
            "box": [round(box[0] / w, 4), round(box[1] / h, 4), round(box[2] / w, 4), round(box[3] / h, 4)],
        })
    detections.sort(key=lambda d: d["confidence"], reverse=True)

    # Image-level suggestion: class with the highest summed confidence.
    totals = {}
    for d in detections:
        totals[d["label"]] = totals.get(d["label"], 0.0) + d["confidence"]
    top = max(totals, key=totals.get) if totals else None

    return {
        # The real L*-CLAHE output, so the app shows it instead of an imitation.
        "enhanced_image": to_jpeg_data_url(enhanced),
        "top_label": top,
        "top_confidence": max((d["confidence"] for d in detections if d["label"] == top), default=None),
        "detections": detections,
        "ita": ita_result.ita,
        "bracket": ita_result.bracket,
        "mask_status": ita_result.status,
        "beta": beta,
        "beta_source": beta_source,
        "weights": WEIGHTS.name,
        "calibration_status": calibration.get("status"),
    }


@app.post("/compare")
async def compare(file: UploadFile = File(...)):
    """Same photo through every ablation model, each with its own preprocessing."""
    rgb = read_and_resize(await file.read())
    ita_result = compute_ita(rgb)
    beta_d, beta_source = model_d_beta(ita_result)
    preprocess = {
        "A": (lambda: rgb, None),
        "B": (lambda: rgb_clahe(rgb, FIXED_BETA), FIXED_BETA),
        "C": (lambda: apply_clahe(rgb, beta=FIXED_BETA, tile_grid_size=TILE_GRID), FIXED_BETA),
        "C′": (lambda: apply_clahe(rgb, beta=float(calibration["beta_global"]), tile_grid_size=TILE_GRID),
               float(calibration["beta_global"])),
        "D": (lambda: apply_clahe(rgb, beta=beta_d, tile_grid_size=TILE_GRID), beta_d),
    }
    results = []
    for model_id, weights_name, description in COMPARE_MODELS:
        weights = ABLATION_WEIGHTS / weights_name
        if not weights.is_file():
            results.append({"id": model_id, "description": description, "available": False})
            continue
        make_input, beta = preprocess[model_id]
        image = make_input()
        model = load_yolo(weights)
        prediction = model.predict(image[:, :, ::-1].copy(), conf=CONFIDENCE, device=DEVICE, verbose=False)[0]
        top, top_conf = summarize(prediction, model.names)
        annotated = prediction.plot(line_width=2)[:, :, ::-1]  # plot() returns BGR
        results.append({
            "id": model_id, "description": description, "available": True, "beta": beta,
            "top_label": top, "top_confidence": top_conf, "boxes": len(prediction.boxes),
            "image": to_jpeg_data_url(annotated, max_side=640),
        })
    return {
        "ita": ita_result.ita, "bracket": ita_result.bracket, "mask_status": ita_result.status,
        "beta_d": beta_d, "beta_source": beta_source, "models": results,
    }
