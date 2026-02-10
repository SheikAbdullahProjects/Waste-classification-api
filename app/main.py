from io import BytesIO
import os

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch

from .config import MODEL_PATH
from .model import load_model
from .predict import predict_image

app = FastAPI(title="Garbage Classification API")

allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "https://waste-classification-client.vercel.app")
allowed_origins_regex = os.getenv("ALLOWED_ORIGINS_REGEX")
allowed_origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=allowed_origins_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None
_class_names = None
_model_mtime = None


@app.on_event("startup")
def _load_model():
    global _model, _class_names, _model_mtime
    if not MODEL_PATH.exists():
        return
    _model, _class_names = load_model(MODEL_PATH, _device)
    _model_mtime = MODEL_PATH.stat().st_mtime


def _ensure_model_loaded():
    global _model, _class_names, _model_mtime
    if not MODEL_PATH.exists():
        return

    current_mtime = MODEL_PATH.stat().st_mtime
    if _model is None or _model_mtime != current_mtime:
        _model, _class_names = load_model(MODEL_PATH, _device)
        _model_mtime = current_mtime


@app.get("/health")
def health():
    _ensure_model_loaded()
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "classes": _class_names,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    _ensure_model_loaded()
    if _model is None or _class_names is None:
        raise HTTPException(status_code=400, detail="Model not found. Train the model first.")

    if file.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    label, score = predict_image(_model, _class_names, image, _device)
    return {"label": label, "confidence": score}
