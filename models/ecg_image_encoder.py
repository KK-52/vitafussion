"""
VitaFusion AI — ECG Image Encoder
====================================
EfficientNet-B0 based classifier for ECG image inputs.
Trained on the erhmrai/ecg-image-data Kaggle dataset.

4 Classes:
  0 → Normal
  1 → Abnormal Heartbeat
  2 → History of Myocardial Infarction
  3 → Myocardial Infarction

Maps to VitaFusion risk scores and health categories.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Optional

# ── Class definitions ─────────────────────────────────────────────────────────

# 6-class MIT-BIH image dataset (erhmrai/ecg-image-data)
# Folder → VitaFusion index mapping (alphabetical sort: F=0,M=1,N=2,Q=3,S=4,V=5 → remapped)
ECG_IMAGE_CLASSES = [
    "Normal",                        # 0  N
    "Myocardial Infarction",          # 1  M
    "Supraventricular Tachycardia",  # 2  S
    "Ventricular PVC",               # 3  V
    "Fusion Beat",                   # 4  F
    "Unknown / Pacemaker",           # 5  Q
]

# Risk contributions per class for fusion model
ECG_IMAGE_RISK_MAP = {
    0: {"cardiovascular": 0.08, "arrhythmia": 0.06, "label": "Normal",                       "vitafusion_category": "Good"},
    1: {"cardiovascular": 0.92, "arrhythmia": 0.70, "label": "Myocardial Infarction",         "vitafusion_category": "Immediate"},
    2: {"cardiovascular": 0.50, "arrhythmia": 0.72, "label": "Supraventricular Tachycardia",  "vitafusion_category": "Medium"},
    3: {"cardiovascular": 0.60, "arrhythmia": 0.75, "label": "Ventricular PVC",               "vitafusion_category": "Medium"},
    4: {"cardiovascular": 0.40, "arrhythmia": 0.55, "label": "Fusion Beat",                   "vitafusion_category": "Medium"},
    5: {"cardiovascular": 0.35, "arrhythmia": 0.45, "label": "Unknown / Pacemaker",           "vitafusion_category": "Medium"},
}

# Number of classes for model head
N_ECG_CLASSES = 6

# Pretrained weights path (saved after training)
WEIGHTS_PATH = Path(__file__).parent / "pretrained" / "ecg_image_encoder.pth"


# ══════════════════════════════════════════════════════════════════════════════
#  Model Architecture: EfficientNet-B0 Backbone + Disease Heads
# ══════════════════════════════════════════════════════════════════════════════

class ECGImageEncoder(nn.Module):
    """
    EfficientNet-B0 backbone fine-tuned for 6-class ECG image classification.
    Trained on the MIT-BIH image dataset (erhmrai/ecg-image-data).

    Input:  RGB image tensor [B, 3, 224, 224]  (normalized)
    Output: class logits [B, 6] + 256-dim embedding
    """

    def __init__(self, n_classes: int = N_ECG_CLASSES, embedding_dim: int = 256, pretrained: bool = True):
        super().__init__()
        try:
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = efficientnet_b0(weights=weights)
        except ImportError:
            from torchvision.models import efficientnet_b0
            backbone = efficientnet_b0(pretrained=pretrained)

        # Use all EfficientNet layers except the final classifier
        self.features = backbone.features        # CNN backbone
        self.avgpool  = backbone.avgpool         # Adaptive avg pool → [B, 1280, 1, 1]

        # ── Embedding projection ───────────────────────────────────────────
        self.embedding_head = nn.Sequential(
            nn.Linear(1280, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        # ── Classification head ────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(embedding_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """x: [B, 3, 224, 224]"""
        feat = self.features(x)          # [B, 1280, 7, 7]
        feat = self.avgpool(feat)        # [B, 1280, 1, 1]
        feat = feat.flatten(1)           # [B, 1280]
        emb  = self.embedding_head(feat) # [B, 256]
        logits = self.classifier(emb)    # [B, 4]
        return {"embedding": emb, "logits": logits, "probs": F.softmax(logits, dim=-1)}


# ══════════════════════════════════════════════════════════════════════════════
#  Preprocessing
# ══════════════════════════════════════════════════════════════════════════════

# ImageNet normalization (EfficientNet trained with these stats)
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def preprocess_ecg_image(image_input) -> torch.Tensor:
    """
    Preprocess an ECG image for inference.

    Args:
        image_input: bytes | str (path) | PIL.Image | np.ndarray

    Returns:
        Tensor [1, 3, 224, 224]
    """
    import torch
    from PIL import Image
    import torchvision.transforms as T

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=_MEAN, std=_STD),
    ])

    if isinstance(image_input, bytes):
        import io
        pil = Image.open(io.BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input, (str, Path)):
        pil = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        pil = Image.fromarray(image_input.astype("uint8")).convert("RGB")
    else:
        pil = image_input.convert("RGB")  # assume PIL

    return transform(pil).unsqueeze(0)   # [1, 3, 224, 224]


# ══════════════════════════════════════════════════════════════════════════════
#  Inference Wrapper (matches VitaFusion encoder interface)
# ══════════════════════════════════════════════════════════════════════════════

class ECGImageEncoderSimulated:
    """
    Inference wrapper for ECG image classification.

    If trained weights exist at models/pretrained/ecg_image_encoder.pth,
    runs the real EfficientNet model.  Otherwise, falls back to
    heuristic simulation based on image statistics.
    """

    def __init__(self, embedding_dim: int = 256):
        self.embedding_dim = embedding_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = None
        self._load_model()

    def _load_model(self):
        if WEIGHTS_PATH.exists():
            try:
                self.model = ECGImageEncoder(n_classes=N_ECG_CLASSES, embedding_dim=self.embedding_dim, pretrained=False)
                state = torch.load(WEIGHTS_PATH, map_location=self.device)
                self.model.load_state_dict(state)
                self.model.to(self.device).eval()
                print(f"[ECGImageEncoder] Loaded trained weights from {WEIGHTS_PATH}")
            except Exception as e:
                print(f"[ECGImageEncoder] Failed to load weights: {e}. Using simulation.")
                self.model = None
        else:
            print(f"[ECGImageEncoder] No weights at {WEIGHTS_PATH}. Using simulation mode.")

    def predict(self, image_input, clinical_risk_hint: float = 0.3) -> Dict:
        """
        Args:
            image_input: bytes | PIL.Image | np.ndarray | path
            clinical_risk_hint: float (0-1) from clinical model

        Returns dict compatible with ECGEncoderSimulated output.
        """
        if self.model is not None:
            return self._predict_real(image_input, clinical_risk_hint)
        return self._predict_simulated(image_input, clinical_risk_hint)

    def _predict_real(self, image_input, clinical_risk_hint: float) -> Dict:
        """Run actual EfficientNet inference."""
        tensor = preprocess_ecg_image(image_input).to(self.device)
        with torch.no_grad():
            out = self.model(tensor)

        probs  = out["probs"].cpu().numpy()[0]       # [6]
        emb    = out["embedding"].cpu().numpy()[0]   # [256]
        cls_idx = int(probs.argmax())

        return self._build_result(cls_idx, probs, emb, clinical_risk_hint)

    def _predict_simulated(self, image_input, clinical_risk_hint: float) -> Dict:
        """Heuristic simulation based on image brightness/texture."""
        try:
            from PIL import Image
            import io
            if isinstance(image_input, bytes):
                pil = Image.open(io.BytesIO(image_input)).convert("RGB")
            elif isinstance(image_input, np.ndarray):
                pil = Image.fromarray(image_input.astype("uint8"))
            else:
                pil = image_input

            arr        = np.array(pil.resize((64, 64))).astype(float)
            seed       = int(abs(arr.mean() * 1000)) % 2**31
            rng        = np.random.default_rng(seed)
            brightness = arr.mean() / 255.0
            contrast   = arr.std()  / 128.0

            # 6-class heuristic dirichlet
            base_normal = max(brightness * (1 - clinical_risk_hint * 0.5), 0.05)
            raw = rng.dirichlet([
                base_normal,                           # N  Normal
                max(clinical_risk_hint * 0.8,  0.05), # M  MI
                max(contrast * clinical_risk_hint * 1.2, 0.08),  # S SVT
                max(contrast * clinical_risk_hint,    0.08),     # V PVC
                max(contrast * 0.3,                   0.03),     # F Fusion
                max(clinical_risk_hint * 0.3,          0.05),    # Q Unknown
            ])
            cls_idx = int(raw.argmax())
            emb     = rng.normal(0, 1, self.embedding_dim).astype(np.float32)
        except Exception:
            rng     = np.random.default_rng(42)
            raw     = rng.dirichlet([0.7, 0.05, 0.08, 0.08, 0.04, 0.05])
            cls_idx = 0
            emb     = rng.normal(0, 1, self.embedding_dim).astype(np.float32)

        return self._build_result(cls_idx, raw, emb, clinical_risk_hint)

    def _build_result(self, cls_idx: int, probs: np.ndarray, emb: np.ndarray,
                      clinical_risk_hint: float) -> Dict:
        """Construct the result dict from 6-class classification output."""
        risk_map   = ECG_IMAGE_RISK_MAP[cls_idx]
        class_name = ECG_IMAGE_CLASSES[cls_idx]

        # Map 6-class to 8-class arrhythmia classes used by ECGEncoderSimulated
        ARRHYTHMIA_MAP = {
            0: (0, "Normal Sinus Rhythm"),           # N  Normal
            1: (2, "ST-Elevation (STEMI)"),           # M  MI → STEMI
            2: (1, "Atrial Fibrillation"),            # S  SVT → AFib (closest)
            3: (5, "Premature Ventricular Contraction"),  # V  PVC
            4: (0, "Normal Sinus Rhythm"),            # F  Fusion → Normal-like
            5: (0, "Normal Sinus Rhythm"),            # Q  Unknown → default Normal
        }
        arr_idx, arr_class = ARRHYTHMIA_MAP[cls_idx]

        # Build 8-class arrhythmia probability vector
        arr_probs_8 = np.ones(8) * 0.02
        arr_probs_8[arr_idx] = float(probs[cls_idx]) * 0.8 + 0.1
        arr_probs_8 /= arr_probs_8.sum()

        cardio_contrib = float(np.clip(risk_map["cardiovascular"] * 0.7 + clinical_risk_hint * 0.3, 0, 1))
        arr_contrib    = float(np.clip(risk_map["arrhythmia"]     * 0.7 + clinical_risk_hint * 0.3, 0, 1))

        rng = np.random.default_rng(int(abs(emb[0]) * 1e4) % 2**31)
        # Cardiac metrics by class
        METRICS = {
            0: (72, 45, 155, 88,  415),   # Normal
            1: (88, 18, 180, 120, 460),   # MI
            2: (105, 55, 145, 92, 435),   # SVT — fast
            3: (80, 40, 160, 100, 440),   # PVC
            4: (76, 35, 162, 95,  430),   # Fusion
            5: (78, 38, 165, 98,  438),   # Unknown
        }
        hr, hrv, pr, qrs, qtc = METRICS[cls_idx]

        return {
            "embedding":              emb,
            "ecg_image_class":        class_name,
            "ecg_image_class_idx":    cls_idx,
            "ecg_image_probs":        probs.tolist(),
            "vitafusion_category":    risk_map["vitafusion_category"],
            # Arrhythmia fields (compatible with ECGEncoderSimulated)
            "arrhythmia_probs":       arr_probs_8.tolist(),
            "arrhythmia_class":       arr_class,
            "arrhythmia_idx":         arr_idx,
            "arrhythmia_contribution":arr_contrib,
            "cardiovascular_contribution": cardio_contrib,
            # Cardiac metrics (estimated per class)
            "heart_rate":   float(hr  + rng.uniform(-5, 5)),
            "hrv_ms":       float(hrv + rng.uniform(-5, 5)),
            "pr_interval":  float(pr  + rng.uniform(-10, 10)),
            "qrs_duration": float(qrs + rng.uniform(-5, 5)),
            "qtc_interval": float(qtc + rng.uniform(-10, 10)),
            "afib_score":   float(probs[2] * 0.8) if len(probs) > 2 else 0.0,   # S class
            "stemi_score":  float(probs[1] * 0.9) if len(probs) > 1 else 0.0,   # M class
            # Placeholders for fusion compat
            "r_peaks":    [],
            "signal_norm":[],
            "sample_rate": 125.0,
            "lead_names": ["Image-derived"],
            "quality":     0.85,
            "hrv_features":{},
        }
