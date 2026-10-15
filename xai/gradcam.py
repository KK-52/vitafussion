"""
VitaFusion AI — Grad-CAM Explainability
=========================================
Gradient-weighted Class Activation Mapping for retinal and tongue CNN encoders.
Produces spatially-localized risk heatmaps.

Reference: Selvaraju et al. (2017) "Grad-CAM: Visual Explanations from CNNs"
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, Optional, Tuple


def compute_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_class: Optional[int] = None,
    target_layer_name: str = "layer4",
) -> Tuple[np.ndarray, float]:
    """
    Compute Grad-CAM heatmap for a CNN model.

    Args:
        model:             PyTorch CNN model with registered hooks
        input_tensor:      [1, 3, H, W] input image tensor
        target_class:      Class index to explain. None = argmax
        target_layer_name: Name of target convolutional layer

    Returns:
        (heatmap [H, W] float32 0-1, confidence_score float)
    """
    model.eval()
    input_tensor.requires_grad_(True)

    # ── Forward pass ─────────────────────────────────────────────────────────
    output = model(input_tensor)

    # Get logits for target class
    if hasattr(output, "get"):
        logits_key = [k for k in output if "logits" in k][0]
        logits = output[logits_key]
    else:
        logits = output

    if target_class is None:
        target_class = int(logits.argmax(dim=-1).item())

    confidence = float(F.softmax(logits, dim=-1)[0, target_class].item())

    # ── Backward pass ─────────────────────────────────────────────────────────
    model.zero_grad()
    logits[0, target_class].backward(retain_graph=True)

    # ── Retrieve features and gradients ───────────────────────────────────────
    features, gradients = model.get_gradcam_data()

    if features is None or gradients is None:
        # Fallback: return uniform heatmap
        h, w = input_tensor.shape[2:]
        return np.ones((h, w), dtype=np.float32) * 0.5, confidence

    # ── Compute weighted activation map ───────────────────────────────────────
    # Global average pooling of gradients → weights
    weights = gradients.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]

    # Weighted combination of feature maps
    cam = (weights * features).sum(dim=1, keepdim=True)  # [1, 1, H', W']
    cam = F.relu(cam)                                     # Only positive influence

    # Normalize to [0, 1]
    cam = cam.squeeze().detach().cpu().numpy()
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    else:
        cam = np.zeros_like(cam)

    # Resize to input image size
    h, w = input_tensor.shape[2:]
    cam_resized = cv2.resize(cam, (w, h))

    return cam_resized.astype(np.float32), confidence


def simulate_gradcam(
    image_array: np.ndarray,
    risk_score: float,
    modality: str = "retinal",
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a realistic-looking Grad-CAM heatmap for demo mode.
    Uses anatomical priors to place attention in plausible regions.

    Args:
        image_array:  RGB image [H, W, 3] uint8
        risk_score:   0-1 risk level (higher = more intense heatmap)
        modality:     "retinal" or "tongue"
        seed:         Random seed

    Returns:
        heatmap [H, W] float32 in [0, 1]
    """
    h, w = image_array.shape[:2]
    rng  = np.random.default_rng(seed or int(image_array.mean() * 1000))

    # Base: Gaussian blobs at anatomically relevant positions
    heatmap = np.zeros((h, w), dtype=np.float32)

    if modality == "retinal":
        # Optic disc: center-right region (~60% x, 50% y)
        centers = [
            (int(w * 0.60), int(h * 0.50), int(min(h, w) * 0.15)),  # Optic disc
            (int(w * 0.40), int(h * 0.50), int(min(h, w) * 0.20)),  # Macula
            (int(w * 0.75), int(h * 0.35), int(min(h, w) * 0.08)),  # Vessel junction
            (int(w * 0.25), int(h * 0.65), int(min(h, w) * 0.07)),  # Lesion area
        ]
        intensities = [
            risk_score * 0.9,
            risk_score * 0.7,
            risk_score * 0.5 + rng.uniform(0, 0.2),
            risk_score * 0.4 + rng.uniform(0, 0.2),
        ]
    else:  # tongue
        # Central tongue body, tip, sides
        centers = [
            (int(w * 0.50), int(h * 0.55), int(min(h, w) * 0.25)),  # Central body
            (int(w * 0.50), int(h * 0.30), int(min(h, w) * 0.12)),  # Tip
            (int(w * 0.25), int(h * 0.55), int(min(h, w) * 0.10)),  # Left edge
            (int(w * 0.75), int(h * 0.55), int(min(h, w) * 0.10)),  # Right edge
        ]
        intensities = [
            risk_score * 0.85,
            risk_score * 0.6,
            risk_score * 0.45,
            risk_score * 0.45,
        ]

    # Draw Gaussian blobs
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    for (cx, cy, sigma), intensity in zip(centers, intensities):
        blob = np.exp(-((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2 * sigma ** 2))
        heatmap += blob * float(intensity)

    # Add small noise for realism
    heatmap += rng.uniform(0, 0.05, (h, w)).astype(np.float32)

    # Normalize
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    return np.clip(heatmap, 0, 1).astype(np.float32)


def overlay_heatmap(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Create Grad-CAM overlay on original image.

    Returns:
        RGB overlay [H, W, 3] uint8
    """
    h, w = image_rgb.shape[:2]

    # Resize heatmap to match image
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_color   = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_rgb     = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Blend
    overlay = (image_rgb.astype(np.float32) * (1 - alpha) +
               heatmap_rgb.astype(np.float32) * alpha)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def get_risk_regions(
    heatmap: np.ndarray,
    threshold: float = 0.6,
    modality: str = "retinal",
) -> Dict:
    """
    Identify and describe high-risk regions in a Grad-CAM heatmap.

    Returns:
        dict with high_risk_mask, coverage_pct, and region descriptions
    """
    high_risk_mask = (heatmap > threshold).astype(np.uint8) * 255
    coverage_pct   = float((heatmap > threshold).sum() / heatmap.size * 100)

    # Connected components
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(high_risk_mask)

    regions = []
    for i in range(1, n_labels):  # Skip background
        area   = stats[i, cv2.CC_STAT_AREA]
        cx, cy = centroids[i]
        if area > 50:  # Filter tiny blobs
            # Map pixel position to anatomical description
            label = _localize_region(cx, cy, heatmap.shape[1], heatmap.shape[0], modality)
            regions.append({"label": label, "area": area, "center": (cx, cy)})

    return {
        "high_risk_mask": high_risk_mask,
        "coverage_pct":   coverage_pct,
        "regions":        regions,
        "n_regions":      len(regions),
    }


def _localize_region(cx: float, cy: float, w: int, h: int, modality: str) -> str:
    """Map (cx, cy) pixel location to anatomical region name."""
    x_rel = cx / w
    y_rel = cy / h

    if modality == "retinal":
        if 0.5 < x_rel < 0.7 and 0.4 < y_rel < 0.6:
            return "Optic Disc"
        if 0.3 < x_rel < 0.55 and 0.4 < y_rel < 0.6:
            return "Macula / Fovea"
        if y_rel < 0.35:
            return "Superior Retina"
        if y_rel > 0.65:
            return "Inferior Retina"
        if x_rel < 0.35:
            return "Nasal Retina"
        return "Retinal Vessel Region"
    else:  # tongue
        if y_rel < 0.4:
            return "Tongue Tip"
        if y_rel > 0.7:
            return "Tongue Base/Root"
        if x_rel < 0.3:
            return "Left Lateral Edge"
        if x_rel > 0.7:
            return "Right Lateral Edge"
        return "Central Tongue Body"
