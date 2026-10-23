"""
BPRAA — Pydantic schemas for request/response validation and Swagger docs.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

class ProbabilitiesSchema(BaseModel):
    """Per-class softmax probabilities."""

    model_config = {"populate_by_name": True}

    field_25cm: float = Field(..., alias="25cm", description="Probability for 25cm class")
    field_100cm: float = Field(..., alias="100cm", description="Probability for 100cm class")


# ---------------------------------------------------------------------------
# /api/health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Health-check response."""

    status: str = Field(..., description="Service status — 'healthy' or 'degraded'")
    model_loaded: bool = Field(..., description="Whether the model is loaded and ready")
    device: str = Field(..., description="PyTorch compute device in use")


# ---------------------------------------------------------------------------
# /api/model-info
# ---------------------------------------------------------------------------

class ModelInfoResponse(BaseModel):
    """Static information about the loaded model."""

    model_name: str
    architecture: str
    task: str
    classes: list[str]
    input_size: str
    resize_size: str
    device: str


# ---------------------------------------------------------------------------
# /api/predict
# ---------------------------------------------------------------------------

class PredictionResponse(BaseModel):
    """Response from POST /api/predict."""

    prediction: str = Field(..., description="Predicted class: '25cm' or '100cm'")
    confidence: float = Field(..., description="Softmax probability of the predicted class")
    probabilities: dict[str, float] = Field(
        ..., description="Softmax probabilities for all classes"
    )
    model: str = Field(..., description="Model name")


# ---------------------------------------------------------------------------
# /api/explain
# ---------------------------------------------------------------------------

class ExplanationResponse(BaseModel):
    """Response from POST /api/explain."""

    prediction: str
    confidence: float
    probabilities: dict[str, float]
    gradcam_base64: str = Field(
        ..., description="Base64-encoded PNG of the Grad-CAM overlay"
    )
    target_layer: str = Field(..., description="Name of the Grad-CAM target layer")
    label: str = Field(default="Model Attention Visualization")
    disclaimer: str
    model: str


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    """Standard error envelope."""

    detail: str
