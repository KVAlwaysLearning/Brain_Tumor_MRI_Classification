"""FastAPI server: /api/predict (best model), /api/explain (any of the 3 models) and the built React site."""
import base64
import os
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .inference import CLASSES, CONFIG, MODELS, Explainer, Predictor
from .preprocessing import input_warnings, load_image, to_model_input, to_tensor

MAX_BYTES = 10 * 1024 * 1024
STATIC = Path(os.environ.get("STATIC_DIR", Path(__file__).resolve().parents[2] / "frontend" / "dist"))

app = FastAPI(title="Brain Tumor MRI Classifier", docs_url="/api/docs", openapi_url="/api/openapi.json")
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
                   allow_methods=["*"], allow_headers=["*"])

predictor = Predictor()
explainer = Explainer()
predictor(np.zeros((1, 224, 224, 3), np.float32))          # warm-up so the first user request is fast


async def _read(file: UploadFile):
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image larger than 10 MB.")
    try:
        img = load_image(data)
    except ValueError as e:
        raise HTTPException(415, str(e))
    gray = to_model_input(img)
    return img, gray, to_tensor(gray)


@app.get("/api/health")
def health():
    return {"status": "ok", "best_model": CONFIG["best_model"], "classes": CLASSES}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    img, gray, x = await _read(file)
    t0 = time.perf_counter()
    probs = predictor(x)
    ms = (time.perf_counter() - t0) * 1000
    top = int(np.argmax(probs))
    warns = input_warnings(img, CONFIG["input_checks"], CONFIG["foreground_threshold"])
    if probs[top] < CONFIG["input_checks"]["min_confidence"]:
        warns.append(f"Low confidence ({probs[top]:.0%}, below {CONFIG['input_checks']['min_confidence']:.0%}) — "
                     "treat this prediction as uncertain.")
    return {"model": CONFIG["best_model"], "predicted": CLASSES[top], "predicted_index": top,
            "confidence": probs[top], "probabilities": dict(zip(CLASSES, map(float, probs))),
            "warnings": warns, "inference_ms": round(ms, 1),
            "original_size": list(img.size), "original_mode": img.mode}


@app.post("/api/explain")
async def explain(file: UploadFile = File(...), model: str = Query("mobilenetv2"), k: int = Query(8, ge=2, le=16)):
    if model not in MODELS:
        raise HTTPException(400, f"model must be one of {list(MODELS)}")
    _, gray, x = await _read(file)
    t0 = time.perf_counter()
    net = explainer(model, x, k)
    net["inference_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    net["model"], net["k"] = model, k
    net["input"] = {"gray": base64.b64encode(gray.tobytes()).decode(), "size": 224}
    return net


# ---------------------------------------------------------------- built frontend (single-page app)
if STATIC.exists():
    app.mount("/assets", StaticFiles(directory=STATIC / "assets"), name="assets")
    for sub in ("data", "img"):
        if (STATIC / sub).exists():
            app.mount(f"/{sub}", StaticFiles(directory=STATIC / sub), name=sub)

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        f = STATIC / path
        return FileResponse(f if path and f.is_file() else STATIC / "index.html")
