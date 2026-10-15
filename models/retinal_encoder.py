"""
VitaFusion AI — Retinal (Fundus) Image Encoder
================================================
ResNet-50 backbone fine-tuned for fundus image feature extraction.
Predicts: 8 ocular disease labels from ODIR-5K (multi-label).

Reference: He et al. (2016) "Deep Residual Learning for Image Recognition"
Dataset:   ODIR-5K — Ocular Disease Intelligent Recognition (ICDM 2019)
           https://odir5k.grand-challenge.org/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from typing import Tuple, Optional, Dict

from config import ODIR_CLASSES, ODIR_TO_DISEASE

# ODIR-5K disease classes (multi-label, 8 classes)
ODIR_CLASS_NAMES = ODIR_CLASSES
N_ODIR_CLASSES   = len(ODIR_CLASS_NAMES)  # 8


class RetinalEncoder(nn.Module):
    """
    ResNet-50 based encoder for retinal fundus images.

    Architecture:
        - ResNet-50 backbone (pretrained on ImageNet)
        - ODIR-5K multi-label disease head (8 classes, Sigmoid)
        - Feature embedding head (512-dim)
        - Auxiliary outputs: vessel density, optic disc health, macular status

    Training loss: Binary Cross-Entropy (multi-label, one BCE per disease class)
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        num_odir_classes: int = N_ODIR_CLASSES,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.embedding_dim   = embedding_dim
        self.num_odir_classes= num_odir_classes

        # ── Backbone ───────────────────────────────────────────────────────
        backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        )
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        backbone_out = 2048

        # ── Embedding Head ─────────────────────────────────────────────────
        self.embedding_head = nn.Sequential(
            nn.Linear(backbone_out, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        # ── ODIR Multi-Label Head (8 diseases, Sigmoid) ────────────────────
        self.odir_classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_odir_classes),
        )

        # ── Auxiliary Heads ────────────────────────────────────────────────
        self.vessel_head = nn.Linear(embedding_dim, 1)
        self.optic_head  = nn.Linear(embedding_dim, 1)
        self.macula_head = nn.Linear(embedding_dim, 1)

        self._features = None
        self._gradients = None
        self._register_hooks()

    def _register_hooks(self):
        """Register forward/backward hooks on layer4 for Grad-CAM."""
        def forward_hook(module, input, output):
            self._features = output

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0]

        # Hook on the last ResNet block
        target_layer = list(self.backbone.children())[-2][-1]
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Retinal image tensor [B, 3, 224, 224]

        Returns:
            dict with keys:
                - embedding:      [B, 512]   feature embedding
                - odir_logits:    [B, 8]     ODIR disease logits
                - odir_probs:     [B, 8]     ODIR disease probabilities (Sigmoid, multi-label)
                - vessel_score:   [B, 1]
                - optic_score:    [B, 1]
                - macula_score:   [B, 1]
        """
        feat = self.backbone(x).flatten(1)       # [B, 2048]
        emb  = self.embedding_head(feat)          # [B, 512]

        odir_logits = self.odir_classifier(emb)
        odir_probs  = torch.sigmoid(odir_logits)  # multi-label

        return {
            "embedding":    emb,
            "odir_logits":  odir_logits,
            "odir_probs":   odir_probs,
            "vessel_score": torch.sigmoid(self.vessel_head(emb)),
            "optic_score":  torch.sigmoid(self.optic_head(emb)),
            "macula_score": torch.sigmoid(self.macula_head(emb)),
        }

    def get_gradcam_data(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return stored features and gradients for Grad-CAM computation."""
        return self._features, self._gradients


class RetinalEncoderSimulated:
    """
    Simulation wrapper for RetinalEncoder.
    Produces ODIR-style 8-class multi-label predictions from image statistics.
    Used in demo mode.
    """

    CLASS_NAMES = ODIR_CLASS_NAMES  # 8 ODIR labels

    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim

    def predict(self, image_array: np.ndarray, clinical_risk_hint: float = 0.3) -> Dict:
        """
        Generate plausible ODIR multi-label predictions from image statistics.
        """
        brightness = float(image_array.mean())
        contrast   = float(image_array.std())
        red_ratio  = float(image_array[:, :, 0].mean() / (brightness + 1e-6))
        green_ratio= float(image_array[:, :, 1].mean() / (brightness + 1e-6))

        rng = np.random.default_rng(int(brightness * 1000))
        base_risk = clinical_risk_hint * 0.5 + rng.uniform(0.0, 0.3)

        # Heuristic per-class probabilities (Sigmoid-style, independent)
        odir_probs = np.array([
            max(0.05, 1 - base_risk - rng.uniform(0, 0.2)),      # N Normal
            min(0.95, base_risk * 0.9 + rng.uniform(-0.1, 0.2)), # D Diabetes
            min(0.95, base_risk * 0.5 + rng.uniform(-0.1, 0.15)),# G Glaucoma
            min(0.95, base_risk * 0.4 + rng.uniform(-0.1, 0.15)),# C Cataract
            min(0.95, base_risk * 0.35+ rng.uniform(-0.1, 0.15)),# A AMD
            min(0.95, base_risk * 0.7 + rng.uniform(-0.1, 0.15)),# H Hypertension
            min(0.95, base_risk * 0.3 + rng.uniform(-0.1, 0.15)),# M Myopia
            min(0.95, base_risk * 0.2 + rng.uniform(-0.1, 0.15)),# O Other
        ], dtype=np.float32)
        odir_probs = np.clip(odir_probs, 0, 1)

        # Primary label = highest-prob positive class
        primary_idx   = int(np.argmax(odir_probs))
        primary_label = self.CLASS_NAMES[primary_idx]

        embedding = rng.normal(0, 1, self.embedding_dim).astype(np.float32)

        return {
            "embedding":      embedding,
            "odir_probs":     odir_probs,
            "odir_labels":    self.CLASS_NAMES,
            "primary_label":  primary_label,
            "primary_idx":    primary_idx,
            # Legacy fields kept for backward-compat with fusion model
            "dr_probs":       odir_probs[:5],
            "dr_grade":       int(odir_probs[1] > 0.5),
            "dr_grade_label": "Diabetes" if odir_probs[1] > 0.5 else "Normal",
            "vessel_score":   float(np.clip(contrast * 2.5 + rng.uniform(-0.1, 0.2), 0, 1)),
            "optic_score":    float(np.clip(1 - base_risk * 0.6, 0, 1)),
            "macula_score":   float(np.clip(odir_probs[4], 0, 1)),
            "cardiovascular_contribution": float(np.clip(odir_probs[1] * 0.4 + odir_probs[5] * 0.6, 0, 1)),
            "diabetes_contribution":       float(np.clip(odir_probs[1] * 0.9, 0, 1)),
            "hypertension_contribution":   float(np.clip(odir_probs[5] * 0.9, 0, 1)),
        }
