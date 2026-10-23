"""
BPRAA — FastAPI application entry point.

Routes:
    GET  /api/health       — liveness / readiness check
    GET  /api/model-info   — static model metadata
    POST /api/predict      — image classification
    POST /api/explain      — Grad-CAM explanation

Run:
    cd backend
    python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000

Swagger UI:
    http://localhost:8000/docs
"""

from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

import model as model_module
from explainability import explain_image
from inference import predict_image
from schemas import (
    ErrorResponse,
    ExplanationResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

load_dotenv()

FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:5173")
MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "10"))
MAX_UPLOAD_BYTES: int = MAX_UPLOAD_MB * 1024 * 1024

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}

# ---------------------------------------------------------------------------
# Lifespan — load model once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the BPRAA model before the server starts accepting requests."""
    print("[BPRAA] Starting up — loading model...")
    loaded = model_module.load_model()
    model_module.set_model(loaded)
    print("[BPRAA] Model ready. Server accepting requests.")
    yield
    print("[BPRAA] Shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="BPRAA — Bloodstain Pattern Recognition and Analysis",
    description=(
        "AI-Assisted Bloodstain Pattern Analysis API.\n\n"
        "**Research prototype** — predictions are for research and educational "
        "purposes only and must not replace examination, interpretation, or "
        "conclusions by qualified forensic professionals."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Image validation helper
# ---------------------------------------------------------------------------

def _validate_and_open_image(upload: UploadFile) -> Image.Image:
    """
    Validate the uploaded file and return a PIL Image.

    Checks:
        - Content-Type header (must be an accepted image type)
        - File size (must be ≤ MAX_UPLOAD_MB)
        - Actual image data (PIL must be able to open it)

    Args:
        upload: FastAPI UploadFile from the multipart form.

    Returns:
        PIL.Image.Image

    Raises:
        HTTPException 400 for invalid / oversized files.
        HTTPException 422 if PIL cannot parse the image data.
    """
    # Check content-type header
    content_type = (upload.content_type or "").lower()
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type: '{content_type}'. "
                f"Accepted: {sorted(ALLOWED_CONTENT_TYPES)}"
            ),
        )

    # Read raw bytes
    raw = upload.file.read()

    # Check file size
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"File too large: {len(raw) / (1024*1024):.1f} MB. "
                f"Maximum allowed: {MAX_UPLOAD_MB} MB."
            ),
        )

    if len(raw) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # Verify actual image contents (do not trust filename or content-type alone)
    try:
        image = Image.open(io.BytesIO(raw))
        image.verify()          # Raise if corrupted
        # Re-open after verify (verify closes the image)
        image = Image.open(io.BytesIO(raw))
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Could not decode the uploaded file as an image.",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Image validation failed: {exc}",
        )

    return image


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get(
    "/api/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["System"],
)
def health_check() -> HealthResponse:
    """
    Return service health status and model readiness.

    Always returns HTTP 200. Check `model_loaded` to verify readiness.
    """
    try:
        model_module.get_model()
        loaded = True
    except RuntimeError:
        loaded = False

    return HealthResponse(
        status="healthy" if loaded else "degraded",
        model_loaded=loaded,
        device=str(model_module.DEVICE),
    )


@app.get(
    "/api/model-info",
    response_model=ModelInfoResponse,
    summary="Model information",
    tags=["System"],
)
def model_info() -> ModelInfoResponse:
    """Return static metadata about the loaded BPRAA model."""
    return ModelInfoResponse(
        model_name="MobileNetV3-Small",
        architecture="CNN",
        task="Binary image classification",
        classes=model_module.CLASS_NAMES,
        input_size="224x224",
        resize_size="256x256",
        device=str(model_module.DEVICE),
    )


@app.post(
    "/api/predict",
    response_model=PredictionResponse,
    summary="Classify a bloodstain image",
    tags=["Inference"],
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file or file type"},
        422: {"model": ErrorResponse, "description": "Unprocessable image"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
)
def predict(
    file: UploadFile = File(..., description="Bloodstain image (.jpg, .jpeg, .png, .webp)"),
) -> PredictionResponse:
    """
    Upload a bloodstain image and receive a real MobileNetV3-Small prediction.

    The model classifies whether the bloodstain was produced from a drop height
    of **25 cm** or **100 cm**.

    Returns softmax probabilities for both classes along with the predicted
    class and confidence score.
    """
    # Ensure model is ready
    try:
        model_module.get_model()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )

    image = _validate_and_open_image(file)

    try:
        result = predict_image(image)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {exc}",
        )

    return PredictionResponse(**result)


@app.post(
    "/api/explain",
    response_model=ExplanationResponse,
    summary="Grad-CAM explanation",
    tags=["Explainability"],
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file or file type"},
        422: {"model": ErrorResponse, "description": "Unprocessable image"},
        500: {"model": ErrorResponse, "description": "Grad-CAM computation failed"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
)
def explain(
    file: UploadFile = File(..., description="Bloodstain image (.jpg, .jpeg, .png, .webp)"),
) -> ExplanationResponse:
    """
    Upload a bloodstain image and receive a **real Grad-CAM** explanation.

    Grad-CAM (Gradient-weighted Class Activation Mapping) highlights the image
    regions that had the most influence on the model's prediction.

    The `gradcam_base64` field contains a base64-encoded PNG of the original
    image with the Grad-CAM heatmap blended on top.

    **Note:** This is an interpretability aid and must not be treated as
    forensic evidence.
    """
    try:
        model_module.get_model()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )

    image = _validate_and_open_image(file)

    try:
        result = explain_image(image)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Grad-CAM computation failed: {exc}",
        )

    return ExplanationResponse(**result)
