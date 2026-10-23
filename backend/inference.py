"""
BPRAA — Inference module.

Provides predict_image() which takes a PIL image and returns a structured
prediction dict using the loaded MobileNetV3-Small model.
All values originate from the actual model — no hardcoding.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import CLASS_NAMES, DEVICE, get_model

# ---------------------------------------------------------------------------
# Preprocessing transform — must match training preprocessing exactly
# ---------------------------------------------------------------------------

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_image(image: Image.Image) -> dict:
    """
    Run inference on a PIL image using the loaded BPRAA model.

    Preprocessing pipeline:
        image → RGB → Resize(256,256) → CenterCrop(224)
              → ToTensor → ImageNet Normalize
              → model forward pass → softmax → argmax

    Args:
        image: PIL.Image.Image — any mode, converted to RGB internally.

    Returns:
        dict with keys:
            prediction   : str  — "25cm" or "100cm"
            confidence   : float — probability of the predicted class
            probabilities: dict  — {class_name: probability} for all classes
            model        : str  — model name string
    """
    model = get_model()

    # Ensure RGB (handles RGBA, grayscale, palette, etc.)
    image_rgb = image.convert("RGB")

    # Preprocess
    tensor = INFERENCE_TRANSFORM(image_rgb).unsqueeze(0).to(DEVICE)

    # Forward pass — no gradient computation needed at inference
    with torch.no_grad():
        logits = model(tensor)                   # shape: (1, 2)
        probs = F.softmax(logits, dim=1)         # shape: (1, 2)

    probs_list = probs[0].cpu().tolist()         # [p_25cm, p_100cm]
    pred_idx = int(torch.argmax(probs[0]).item())
    prediction = CLASS_NAMES[pred_idx]
    confidence = round(probs_list[pred_idx], 6)

    probabilities = {
        cls: round(p, 6)
        for cls, p in zip(CLASS_NAMES, probs_list)
    }

    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probabilities,
        "model": "MobileNetV3-Small",
    }
