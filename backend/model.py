"""
BPRAA — Bloodstain Pattern Recognition and Analysis
Model loader for MobileNetV3-Small binary classifier (25cm vs 100cm).

The checkpoint is stored as an extracted PyTorch zip archive (folder format).
We reconstruct it in-memory and load it via torch.load without modifying
the original files on disk.
"""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_NAMES: list[str] = ["25cm", "100cm"]

DEVICE: torch.device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Checkpoint is the extracted folder produced by Kaggle/PyTorch 2.x
_CHECKPOINT_DIR = (
    Path(__file__).parent
    / "models"
    / "bpraa_mobilenetv3_final (1)"
    / "bpraa_mobilenetv3_final"
)


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

def build_architecture() -> nn.Module:
    """
    Instantiate MobileNetV3-Small with the BPRAA classifier head.

    Classifier: Dropout(0.5) → Linear(576, 2)
    Matches the architecture used during training on Kaggle.
    """
    model = mobilenet_v3_small(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(576, 2),
    )
    return model


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _load_checkpoint_from_folder(folder: Path) -> dict:
    """
    Reconstruct the PyTorch zip archive from an extracted folder and load it.

    PyTorch saves models as zip archives. Kaggle sometimes extracts them to
    a directory. We re-zip in memory so torch.load can deserialise the pickle
    without modifying any file on disk.

    Args:
        folder: Path to the extracted checkpoint directory.

    Returns:
        The checkpoint dict as produced by torch.save on the original model.

    Raises:
        FileNotFoundError: If the checkpoint folder does not exist.
        RuntimeError: If the zip cannot be loaded by PyTorch.
    """
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Checkpoint folder not found: {folder}\n"
            "Expected the extracted PyTorch archive at that path."
        )

    buf = io.BytesIO()
    parent = folder.parent

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for root, _dirs, files in os.walk(folder):
            for fname in files:
                filepath = Path(root) / fname
                arcname = filepath.relative_to(parent).as_posix()
                zf.write(filepath, arcname)

    buf.seek(0)

    try:
        checkpoint = torch.load(buf, map_location=DEVICE, weights_only=False)
    except Exception as exc:
        raise RuntimeError(
            f"torch.load failed to deserialise the checkpoint: {exc}"
        ) from exc

    return checkpoint


# ---------------------------------------------------------------------------
# Model loader (called once at startup)
# ---------------------------------------------------------------------------

def load_model() -> nn.Module:
    """
    Build the MobileNetV3-Small model and load the trained BPRAA weights.

    Supports checkpoints that store weights under the key 'state_dict'
    (as saved by the Kaggle training script) as well as bare state dicts.

    Returns:
        The model in eval mode on the correct device.

    Raises:
        FileNotFoundError: Checkpoint directory missing.
        RuntimeError: Weight loading failure.
        KeyError: Expected key not present in checkpoint.
    """
    print(f"[BPRAA] Loading checkpoint from: {_CHECKPOINT_DIR}")
    print(f"[BPRAA] Device: {DEVICE}")

    checkpoint = _load_checkpoint_from_folder(_CHECKPOINT_DIR)
    print(f"[BPRAA] Checkpoint keys: {list(checkpoint.keys())}")

    model = build_architecture()

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        # Assume the dict itself is the state dict
        state_dict = checkpoint
    else:
        raise RuntimeError(
            f"Unexpected checkpoint type: {type(checkpoint)}. "
            "Expected a dict with 'state_dict' or 'model_state_dict'."
        )

    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing:
        raise RuntimeError(f"Missing keys when loading state dict: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys when loading state dict: {unexpected}")

    model.to(DEVICE)
    model.eval()

    print("===========================================")
    print("  BPRAA MODEL LOADED SUCCESSFULLY")
    print(f"  Architecture : MobileNetV3-Small")
    print(f"  Classes      : {CLASS_NAMES}")
    print(f"  Device       : {DEVICE}")
    print("===========================================")

    return model


# ---------------------------------------------------------------------------
# Singleton — loaded once at import time by app.py startup event
# ---------------------------------------------------------------------------

# This is set by app.py during the lifespan startup event.
# Do not call load_model() here directly; let the FastAPI lifespan handle it.
_model: nn.Module | None = None


def get_model() -> nn.Module:
    """
    Return the loaded singleton model.

    Raises:
        RuntimeError: If the model has not been loaded yet.
    """
    if _model is None:
        raise RuntimeError(
            "Model has not been initialised. "
            "Ensure the FastAPI lifespan startup has completed."
        )
    return _model


def set_model(model: nn.Module) -> None:
    """Store the loaded model singleton (called from app.py lifespan)."""
    global _model
    _model = model