"""
VitaFusion AI — Tongue Image Encoder
======================================
EfficientNet-B0 backbone for tongue image analysis.
Predicts: 21 TCM tongue categories from the TMC dataset.

Reference: Tan & Le (2019) "EfficientNet: Rethinking Model Scaling for CNNs"
Dataset:   21-category Tongue Dataset (TMC) — Body × Coating combinations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from typing import Tuple, Optional, Dict

from config import TMC_21_CLASSES, TMC_N_CLASSES, TMC_TO_VITAFUSION_GROUP

TONGUE_CLASS_NAMES = TMC_21_CLASSES
N_TMC_CLASSES       = TMC_N_CLASSES  # 21


class TongueEncoder(nn.Module):
    """
    EfficientNet-B0 based encoder for tongue images.

    Classifies into 21 TCM tongue categories (TMC dataset):
      - Body colour × Coating type combinations
      - Normal/Pale/Red/DarkRed/Purple × ThinWhite/ThickWhite/Yellow/Black/Peeled

    Architecture:
        - EfficientNet-B0 backbone (pretrained on ImageNet)
        - 21-class classifier head (CrossEntropyLoss)
        - 512-dim embedding
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        num_tongue_classes: int = N_TMC_CLASSES,
        pretrained: bool = True,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # ── Backbone ───────────────────────────────────────────────────────
        backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # Keep backbone without classifier
        self.backbone    = backbone.features
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)

        backbone_out = 1280  # EfficientNet-B0 last feature channels

        # ── Embedding Head ─────────────────────────────────────────────────
        self.embedding_head = nn.Sequential(
            nn.Linear(backbone_out, 768),
            nn.BatchNorm1d(768),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(768, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        # ── Tongue Classification Head ─────────────────────────────────────
        # Classes: Normal, Pale, Red, Purple/Blue, Yellow-coat, Cracked
        self.tongue_classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_tongue_classes),
        )

        # ── Specific Feature Heads ─────────────────────────────────────────
        self.color_head   = nn.Linear(embedding_dim, 4)   # RGBA-derived color score
        self.coating_head = nn.Linear(embedding_dim, 1)   # coating thickness (0-1)
        self.moisture_head= nn.Linear(embedding_dim, 1)   # moisture level (0-1)
        self.shape_head   = nn.Linear(embedding_dim, 1)   # swelling score (0-1)

        # Grad-CAM hooks
        self._features = None
        self._gradients = None
        self._register_hooks()

    def _register_hooks(self):
        """Register hooks on last EfficientNet block for Grad-CAM."""
        def forward_hook(module, input, output):
            self._features = output

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0]

        self.backbone[-1].register_forward_hook(forward_hook)
        self.backbone[-1].register_full_backward_hook(backward_hook)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Tongue image tensor [B, 3, 224, 224]

        Returns:
            dict with:
                - embedding:         [B, 512]
                - tongue_logits:     [B, 21]    21-class TMC logits
                - tongue_probs:      [B, 21]    Softmax probabilities
                - color_scores:      [B, 4]
                - coating_score:     [B, 1]
                - moisture_score:    [B, 1]
                - shape_score:       [B, 1]
        """
        feat = self.backbone(x)             # [B, 1280, H', W']
        feat = self.adaptive_pool(feat)     # [B, 1280, 1, 1]
        feat = feat.flatten(1)             # [B, 1280]

        emb  = self.embedding_head(feat)    # [B, 512]

        tongue_logits = self.tongue_classifier(emb)
        tongue_probs  = F.softmax(tongue_logits, dim=-1)

        return {
            "embedding":      emb,
            "tongue_logits":  tongue_logits,
            "tongue_probs":   tongue_probs,
            "color_scores":   torch.sigmoid(self.color_head(emb)),
            "coating_score":  torch.sigmoid(self.coating_head(emb)),
            "moisture_score": torch.sigmoid(self.moisture_head(emb)),
            "shape_score":    torch.sigmoid(self.shape_head(emb)),
        }

    def get_gradcam_data(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self._features, self._gradients


class TongueEncoderSimulated:
    """
    Simulation wrapper for TongueEncoder.
    Maps image color statistics to TMC 21-class predictions and VitaFusion groups.
    """

    TONGUE_CLASSES = TMC_21_CLASSES          # 21 classes
    VITAFUSION_GROUPS = [
        "Normal", "Pale", "Red/Dark", "Purple/Blue", "Yellow Coat", "Cracked/Dry"
    ]
    # VitaFusion 6-group disease risk map (same as before)
    GROUP_DISEASE_MAP = {
        "Normal":       {"cardiovascular": 0.1,  "diabetes": 0.1,  "hypertension": 0.1},
        "Pale":         {"cardiovascular": 0.4,  "diabetes": 0.2,  "hypertension": 0.15},
        "Red/Dark":     {"cardiovascular": 0.55, "diabetes": 0.3,  "hypertension": 0.5},
        "Purple/Blue":  {"cardiovascular": 0.7,  "diabetes": 0.25, "hypertension": 0.45},
        "Yellow Coat":  {"cardiovascular": 0.3,  "diabetes": 0.55, "hypertension": 0.3},
        "Cracked/Dry":  {"cardiovascular": 0.25, "diabetes": 0.65, "hypertension": 0.25},
    }

    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim

    def predict(self, image_array: np.ndarray, clinical_risk_hint: float = 0.3) -> Dict:
        """
        Generate 21-class TMC tongue predictions from image color statistics.
        """
        rng = np.random.default_rng(int(image_array.mean() * 999))

        r_mean = float(image_array[:, :, 0].mean())
        g_mean = float(image_array[:, :, 1].mean())
        b_mean = float(image_array[:, :, 2].mean())
        brightness = (r_mean + g_mean + b_mean) / 3.0

        # Heuristic body colour determination
        if brightness > 0.75:
            body = "Normal"
        elif r_mean < 0.35 and brightness < 0.45:
            body = "Pale"
        elif r_mean > 0.55 and g_mean < 0.35:
            body = rng.choice(["Red", "DarkRed"])
        elif b_mean > 0.45:
            body = "Purple"
        else:
            body = rng.choice(["Normal", "Pale", "Red"])

        # Heuristic coating determination
        if g_mean > r_mean * 0.9 and brightness > 0.5:
            coating = "Yellow"
        elif brightness < 0.3:
            coating = "Black"
        elif brightness > 0.8:
            coating = "Peeled"
        else:
            coating = rng.choice(["ThinWhite", "ThickWhite"])

        # Build predicted class name
        tmc_class = f"{body}_{coating}"
        if tmc_class not in TMC_21_CLASSES:
            # fallback to nearest
            tmc_class = next(
                (c for c in TMC_21_CLASSES if c.startswith(body)), "Normal_ThinWhite"
            )

        tmc_idx = TMC_21_CLASSES.index(tmc_class)
        vitafusion_group = TMC_TO_VITAFUSION_GROUP.get(tmc_class, "Normal")
        disease_risks    = self.GROUP_DISEASE_MAP.get(vitafusion_group, self.GROUP_DISEASE_MAP["Normal"])

        # Build 21-class probability vector
        probs = rng.dirichlet(np.ones(21) * 0.1)
        probs[tmc_idx] = 0.55 + rng.uniform(0, 0.25)
        probs /= probs.sum()

        embedding = rng.normal(0, 1, self.embedding_dim).astype(np.float32)
        coating_score = float(np.clip(rng.uniform(0.1, 0.8), 0, 1))
        moisture      = float(np.clip(1 - coating_score * 0.7 + rng.uniform(-0.1, 0.1), 0, 1))

        return {
            "embedding":       embedding,
            "tongue_probs":    probs,               # 21-class vector
            "tongue_class":    tmc_class,            # TMC canonical name
            "tongue_idx":      tmc_idx,
            "vitafusion_group":vitafusion_group,     # 6-class display group
            "coating_score":   coating_score,
            "moisture_score":  moisture,
            "shape_score":     float(np.clip(disease_risks["hypertension"] + rng.uniform(-0.05, 0.05), 0, 1)),
            "cardiovascular_contribution": float(np.clip(disease_risks["cardiovascular"] + rng.uniform(-0.05, 0.1) + clinical_risk_hint * 0.2, 0, 1)),
            "diabetes_contribution":       float(np.clip(disease_risks["diabetes"] + rng.uniform(-0.05, 0.1) + clinical_risk_hint * 0.2, 0, 1)),
            "hypertension_contribution":   float(np.clip(disease_risks["hypertension"] + rng.uniform(-0.05, 0.1) + clinical_risk_hint * 0.2, 0, 1)),
        }
