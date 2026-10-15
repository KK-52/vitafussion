"""
VitaFusion AI — Tongue Image Preprocessing
===========================================
Color normalization, ROI detection, and feature extraction for tongue images.
"""

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from typing import Dict, Optional, Tuple
import io


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

TONGUE_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def normalize_tongue_color(image_rgb: np.ndarray) -> np.ndarray:
    """
    Color normalization for tongue images.
    Corrects for varying lighting conditions using white-balance adjustment.
    """
    image_f = image_rgb.astype(np.float32)

    # Gray-world assumption white balance
    mean_r = image_f[:, :, 0].mean()
    mean_g = image_f[:, :, 1].mean()
    mean_b = image_f[:, :, 2].mean()
    mean_gray = (mean_r + mean_g + mean_b) / 3.0

    scale_r = mean_gray / (mean_r + 1e-8)
    scale_g = mean_gray / (mean_g + 1e-8)
    scale_b = mean_gray / (mean_b + 1e-8)

    normalized = image_f.copy()
    normalized[:, :, 0] = np.clip(image_f[:, :, 0] * scale_r, 0, 255)
    normalized[:, :, 1] = np.clip(image_f[:, :, 1] * scale_g, 0, 255)
    normalized[:, :, 2] = np.clip(image_f[:, :, 2] * scale_b, 0, 255)

    return normalized.astype(np.uint8)


def extract_tongue_roi(image_rgb: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Attempt to segment tongue region from image.
    Uses HSV color-based segmentation targeting pink/red tongue colors.

    Returns:
        (image_with_roi, mask) — mask is None if detection fails
    """
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Tongue color range in HSV (pink, red, light red)
    lower1 = np.array([0,   40,  60])
    upper1 = np.array([20,  255, 255])
    lower2 = np.array([160, 40,  60])
    upper2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask  = cv2.bitwise_or(mask1, mask2)

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    # Check if we got a meaningful region
    coverage = mask.sum() / (mask.shape[0] * mask.shape[1] * 255)
    if coverage < 0.05 or coverage > 0.9:
        return image_rgb, None  # Segmentation failed — use full image

    # Apply mask
    roi = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
    return roi, mask


def extract_tongue_color_features(image_rgb: np.ndarray) -> Dict:
    """
    Extract quantitative color features from tongue image.
    These features map to clinical indicators used in tongue diagnosis.

    Returns dict of color statistics.
    """
    # RGB statistics
    r = image_rgb[:, :, 0].astype(np.float32)
    g = image_rgb[:, :, 1].astype(np.float32)
    b = image_rgb[:, :, 2].astype(np.float32)

    # HSV statistics
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    h_ch = hsv[:, :, 0] / 180.0
    s_ch = hsv[:, :, 1] / 255.0
    v_ch = hsv[:, :, 2] / 255.0

    # Lab statistics (perceptual color)
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    # Tongue diagnosis indicators
    pallor_score     = 1 - (r.mean() / 255)    # Pale = low red
    redness_score    = r.mean() / 255           # Dark red = high red
    cyanosis_score   = b.mean() / (r.mean() + 1e-8)  # Blue > red
    yellowness_score = (r.mean() * 0.5 + g.mean() * 0.5 - b.mean()) / 255
    moisture_score   = v_ch.mean()              # Brightness proxy
    coating_score    = 1 - s_ch.mean()          # Low saturation = thick coating

    return {
        "r_mean": float(r.mean()),
        "g_mean": float(g.mean()),
        "b_mean": float(b.mean()),
        "hue_mean":    float(h_ch.mean()),
        "saturation":  float(s_ch.mean()),
        "brightness":  float(v_ch.mean()),
        "pallor_score":     float(np.clip(pallor_score, 0, 1)),
        "redness_score":    float(np.clip(redness_score, 0, 1)),
        "cyanosis_score":   float(np.clip(cyanosis_score, 0, 1)),
        "yellowness_score": float(np.clip(yellowness_score, 0, 1)),
        "moisture_score":   float(np.clip(moisture_score, 0, 1)),
        "coating_score":    float(np.clip(coating_score, 0, 1)),
    }


def preprocess_tongue_image(image_input, return_viz: bool = True) -> Dict:
    """
    Full preprocessing pipeline for tongue images.

    Args:
        image_input:  PIL Image, numpy array, or bytes
        return_viz:   Return visualization arrays

    Returns:
        dict with tensor, viz_array, color_features, roi_mask
    """
    # ── Load ─────────────────────────────────────────────────────────────────
    if isinstance(image_input, bytes):
        pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        pil_img = Image.fromarray(image_input.astype(np.uint8))
    else:
        pil_img = image_input.convert("RGB")

    img_rgb = np.array(pil_img)
    original_size = (pil_img.height, pil_img.width)

    # ── Color Normalization ──────────────────────────────────────────────────
    normalized = normalize_tongue_color(img_rgb)

    # ── ROI Extraction ───────────────────────────────────────────────────────
    roi, mask = extract_tongue_roi(normalized)

    # ── Color Features ────────────────────────────────────────────────────────
    color_features = extract_tongue_color_features(roi)

    # ── Model Tensor ──────────────────────────────────────────────────────────
    roi_pil = Image.fromarray(roi)
    tensor  = TONGUE_TRANSFORM(roi_pil).unsqueeze(0)   # [1, 3, 224, 224]

    # ── Visualization ─────────────────────────────────────────────────────────
    viz_array  = np.array(roi_pil.resize((224, 224)))
    orig_array = np.array(pil_img.resize((224, 224)))

    return {
        "tensor":         tensor,
        "viz_array":      viz_array,
        "orig_array":     orig_array,
        "roi_mask":       mask,
        "color_features": color_features,
        "original_size":  original_size,
        "normalized_rgb": normalized,
    }
