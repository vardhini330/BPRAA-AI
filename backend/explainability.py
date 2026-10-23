"""
BPRAA — Grad-CAM explainability module.

Implements real Grad-CAM (Gradient-weighted Class Activation Mapping) using
the actual model's activations and gradients. No placeholder images.

Target layer: features[12] — the last convolutional block before pooling in
MobileNetV3-Small. This block outputs a spatial feature map that Grad-CAM
uses to highlight discriminative image regions.

Reference:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
    Gradient-based Localization", ICCV 2017.
"""

from __future__ import annotations

import base64
import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from model import CLASS_NAMES, DEVICE, get_model

# ---------------------------------------------------------------------------
# Preprocessing — same as inference
# ---------------------------------------------------------------------------

_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# ---------------------------------------------------------------------------
# Grad-CAM implementation
# ---------------------------------------------------------------------------

class GradCAM:
    """
    Hooks into a target layer to capture forward activations and
    backward gradients for Grad-CAM computation.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self._model = model
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output) -> None:  # noqa: ANN001
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output) -> None:  # noqa: ANN001
        self._gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        """Clean up forward and backward hooks."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def compute(
        self,
        tensor: torch.Tensor,
        class_idx: int,
    ) -> np.ndarray:
        """
        Compute the Grad-CAM heatmap for a given class index.

        Args:
            tensor: Preprocessed input tensor, shape (1, C, H, W).
            class_idx: Index of the class to explain.

        Returns:
            2-D numpy array in [0, 1], shape (H_feat, W_feat).
        """
        self._model.zero_grad()

        # Forward pass (gradients enabled)
        logits = self._model(tensor)

        # Backward on the target class score
        target = logits[0, class_idx]
        target.backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError(
                "Grad-CAM hooks did not capture activations/gradients. "
                "The target layer may not be part of the forward path."
            )

        # Global-average-pool the gradients over spatial dimensions → (C,)
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of feature maps → (1, H, W)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalise to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam


# ---------------------------------------------------------------------------
# Heatmap overlay
# ---------------------------------------------------------------------------

def _apply_heatmap_overlay(
    original_image: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """
    Overlay the Grad-CAM heatmap on the original image.

    Args:
        original_image: The original PIL image (RGB, 224×224).
        cam: Normalised Grad-CAM array in [0, 1].
        alpha: Blend factor — higher means more heatmap.

    Returns:
        Blended PIL image (RGB).
    """
    # Resize CAM to match the image
    cam_resized = np.uint8(255 * cam)
    cam_pil = Image.fromarray(cam_resized).resize(
        original_image.size, resample=Image.BILINEAR
    )

    # Jet-like colourmap: blue → green → red
    cam_array = np.array(cam_pil, dtype=np.float32) / 255.0
    r = np.clip(1.5 - np.abs(cam_array * 4.0 - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(cam_array * 4.0 - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(cam_array * 4.0 - 1.0), 0, 1)

    heatmap = np.stack([r, g, b], axis=-1)
    heatmap_pil = Image.fromarray(np.uint8(heatmap * 255))

    # Blend
    original_array = np.array(original_image.convert("RGB"), dtype=np.float32)
    heatmap_array = np.array(heatmap_pil, dtype=np.float32)
    blended = (1 - alpha) * original_array + alpha * heatmap_array
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(blended)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def explain_image(image: Image.Image) -> dict:
    """
    Generate a real Grad-CAM explanation for the given image.

    The Grad-CAM is computed using actual model activations and gradients
    with respect to the predicted class. No placeholder or fake output.

    Args:
        image: PIL.Image.Image — input image (any mode).

    Returns:
        dict with keys:
            prediction      : str
            confidence      : float
            probabilities   : dict
            gradcam_base64  : str — base64-encoded PNG of the overlay
            target_layer    : str — name of the hooked layer
            label           : str — UI display label
            disclaimer      : str
    """
    model = get_model()

    # Identify the Grad-CAM target layer
    # MobileNetV3-Small: features[12] is the last conv block (Conv2d 1×1 → BN → Hardswish)
    # features[12][0] is the Conv2d that produces the final spatial feature map
    target_layer = model.features[12][0]
    target_layer_name = "features.12.0 (last conv block)"

    image_rgb = image.convert("RGB")

    # Preprocess
    tensor = _TRANSFORM(image_rgb).unsqueeze(0).to(DEVICE)

    # Set model to train-mode temporarily so gradients flow through BatchNorm
    model.train()
    grad_cam = GradCAM(model, target_layer)

    try:
        # Forward + backward
        logits = model(tensor)
        probs = F.softmax(logits.detach(), dim=1)
        probs_list = probs[0].cpu().tolist()
        pred_idx = int(torch.argmax(probs[0]).item())
        prediction = CLASS_NAMES[pred_idx]
        confidence = round(probs_list[pred_idx], 6)

        cam = grad_cam.compute(tensor, pred_idx)
    finally:
        grad_cam.remove_hooks()
        model.eval()  # Always restore eval mode

    # Build overlay on the 224×224 centre-cropped version
    crop_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
    ])
    display_image = crop_transform(image_rgb)

    overlay = _apply_heatmap_overlay(display_image, cam, alpha=0.45)

    # Encode overlay as base64 PNG
    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")

    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": {
            cls: round(p, 6)
            for cls, p in zip(CLASS_NAMES, probs_list)
        },
        "gradcam_base64": b64,
        "target_layer": target_layer_name,
        "label": "Model Attention Visualization",
        "disclaimer": (
            "Grad-CAM highlights image regions that contributed to the model "
            "prediction. It is an interpretability aid and not forensic evidence."
        ),
        "model": "MobileNetV3-Small",
    }
