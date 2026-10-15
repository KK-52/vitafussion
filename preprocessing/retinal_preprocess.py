"""
VitaFusion AI — Retinal Image Preprocessing
============================================
CLAHE enhancement, vessel preprocessing, normalization for fundus images.
"""

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from typing import Tuple, Optional, Dict
import io


# ImageNet normalization stats (standard for pretrained ResNet/EfficientNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# VitaFusion retinal transform pipeline
RETINAL_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# For visualization (no normalization)
RETINAL_VIZ_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])


def apply_clahe(image_bgr: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to fundus image.
    Enhances retinal vessel visibility and optic disc contrast.

    Args:
        image_bgr:  BGR image [H, W, 3]
        clip_limit: CLAHE clip limit (higher = more contrast)
        tile_size:  Grid size for local histogram

    Returns:
        CLAHE-enhanced BGR image
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_enhanced = clahe.apply(l)

    lab_enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def remove_black_border(image_bgr: np.ndarray, threshold: int = 10) -> np.ndarray:
    """
    Remove black circular border common in fundus photographs.
    Crops to the bounding box of the non-black region.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Find bounding box of the fundus circle
    coords = cv2.findNonZero(mask)
    if coords is None:
        return image_bgr

    x, y, w, h = cv2.boundingRect(coords)
    cropped = image_bgr[y:y+h, x:x+w]

    # Make square
    min_dim = min(w, h)
    cropped = cv2.resize(cropped, (min_dim, min_dim))
    return cropped


def green_channel_enhance(image_bgr: np.ndarray) -> np.ndarray:
    """
    Extract and enhance the green channel (best contrast for retinal vessels).
    Returns 3-channel image with enhanced green channel.
    """
    enhanced = image_bgr.copy().astype(np.float32)
    green = enhanced[:, :, 1]

    # Normalize green channel
    green = (green - green.min()) / (green.max() - green.min() + 1e-8)

    # Replace all channels with normalized green (grayscale fundus)
    enhanced[:, :, 0] = enhanced[:, :, 1] = enhanced[:, :, 2] = (green * 255).astype(np.uint8)
    return enhanced.astype(np.uint8)


def preprocess_retinal_image(
    image_input,
    apply_clahe_flag: bool = True,
    remove_border: bool = True,
    return_viz: bool = True,
) -> Dict:
    """
    Full preprocessing pipeline for retinal fundus images.

    Args:
        image_input:      PIL Image, numpy array, or file bytes
        apply_clahe_flag: Whether to apply CLAHE enhancement
        remove_border:    Whether to crop black circular border
        return_viz:       Whether to return visualization arrays

    Returns:
        dict with:
            - tensor:         torch.Tensor [1, 3, 224, 224] model-ready
            - viz_array:      np.ndarray [224, 224, 3] for display (0-255)
            - clahe_array:    np.ndarray CLAHE-enhanced [224, 224, 3]
            - original_size:  tuple (H, W)
            - quality_score:  float 0-1 image quality estimate
    """
    # ── Load image ──────────────────────────────────────────────────────────
    if isinstance(image_input, bytes):
        pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        pil_img = Image.fromarray(image_input.astype(np.uint8))
    else:
        pil_img = image_input.convert("RGB")

    original_size = (pil_img.height, pil_img.width)

    # Convert to BGR for OpenCV processing
    img_rgb = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # ── Preprocessing steps ──────────────────────────────────────────────────
    if remove_border:
        img_bgr = remove_black_border(img_bgr)

    clahe_bgr = apply_clahe(img_bgr) if apply_clahe_flag else img_bgr.copy()

    # Convert back to RGB PIL for transforms
    clahe_rgb = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2RGB)
    clahe_pil = Image.fromarray(clahe_rgb)

    # Model-ready tensor
    tensor = RETINAL_TRANSFORM(clahe_pil).unsqueeze(0)  # [1, 3, 224, 224]

    # Visualization arrays
    viz_array   = np.array(clahe_pil.resize((224, 224)))
    orig_array  = np.array(Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).resize((224, 224)))

    # Image quality estimate
    quality = _estimate_image_quality(img_bgr)

    return {
        "tensor":        tensor,
        "viz_array":     viz_array,
        "orig_array":    orig_array,
        "clahe_array":   viz_array,
        "original_size": original_size,
        "quality_score": quality,
    }


def _estimate_image_quality(image_bgr: np.ndarray) -> float:
    """Estimate fundus image quality score (0=poor, 1=excellent)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Laplacian variance (sharpness)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_score = float(np.clip(sharpness / 500, 0, 1))

    # Brightness (avoid over/under-exposed)
    brightness = gray.mean() / 255.0
    brightness_score = 1 - abs(brightness - 0.45) * 2

    # Combined
    quality = float(np.clip(sharpness_score * 0.6 + brightness_score * 0.4, 0, 1))
    return quality


def generate_gradcam_overlay(
    original_img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on original image.

    Args:
        original_img: RGB [H, W, 3] uint8
        heatmap:      Grayscale [H, W] float32, range [0, 1]
        alpha:        Overlay transparency
        colormap:     OpenCV colormap for heatmap
    Returns:
        RGB overlay image [H, W, 3]
    """
    h, w = original_img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_color   = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_rgb     = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = (original_img.astype(np.float32) * (1 - alpha) +
               heatmap_rgb.astype(np.float32) * alpha)
    return np.clip(overlay, 0, 255).astype(np.uint8)
