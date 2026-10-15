"""
VitaFusion AI — Multimodal Fusion Model
==========================================
Cross-modal attention fusion of retinal, tongue, ECG, and clinical embeddings.
Produces final multi-disease risk predictions.

Architecture:
    4 modality embeddings
        → Modality-specific projection
        → Cross-modal attention (each modality attends to all others)
        → Gated fusion
        → Shared representation
        → 5 disease prediction heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, List


class ModalityProjector(nn.Module):
    """Projects each modality embedding to a unified dimension."""

    def __init__(self, in_dim: int, unified_dim: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, unified_dim),
            nn.LayerNorm(unified_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class CrossModalAttention(nn.Module):
    """
    Multi-head cross-modal attention.
    Allows each modality to attend to contextual information from others.
    """

    def __init__(self, dim: int = 256, n_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            batch_first=True,
            dropout=0.1,
        )
        self.norm = nn.LayerNorm(dim)
        self.ffn  = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query:   [B, 1, D] — attending modality
            context: [B, N, D] — all modalities (keys/values)
        """
        attended, _ = self.attention(query, context, context)
        x = self.norm(query + attended)
        x = self.norm2(x + self.ffn(x))
        return x.squeeze(1)  # [B, D]


class GatingNetwork(nn.Module):
    """
    Learns modality importance weights.
    Accounts for missing modalities via mask.
    """

    def __init__(self, n_modalities: int = 4, dim: int = 256):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(n_modalities * dim, 256),
            nn.GELU(),
            nn.Linear(256, n_modalities),
            nn.Softmax(dim=-1),
        )

    def forward(self, embeddings: List[torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: list of [B, D] tensors, one per modality
            mask:       [B, n_modalities] binary mask (1=present, 0=missing)
        Returns:
            weights: [B, n_modalities]
        """
        concat = torch.cat(embeddings, dim=-1)  # [B, N*D]
        weights = self.gate(concat)              # [B, N]
        # Mask out missing modalities and renormalize
        weights = weights * mask
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        return weights


class DiseaseHead(nn.Module):
    """Prediction head for a single disease."""

    def __init__(self, in_dim: int = 256, name: str = "disease"):
        super().__init__()
        self.name = name
        self.head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(x))


class VitaFusionModel(nn.Module):
    """
    Full multimodal fusion model.

    Handles:
        - Missing modalities (graceful degradation)
        - Cross-modal attention between all 4 sources
        - Gated fusion weighted by modality availability
        - 5 disease prediction heads
    """

    MODALITIES = ["retinal", "tongue", "ecg", "clinical"]
    DISEASES   = ["cardiovascular", "diabetes", "hypertension", "arrhythmia", "metabolic"]

    def __init__(
        self,
        retinal_dim:  int = 512,
        tongue_dim:   int = 512,
        ecg_dim:      int = 256,
        clinical_dim: int = 256,
        unified_dim:  int = 256,
        n_heads:      int = 8,
    ):
        super().__init__()
        self.unified_dim = unified_dim
        self.n_modalities = 4

        # ── Modality Projectors ────────────────────────────────────────────
        self.projectors = nn.ModuleDict({
            "retinal":  ModalityProjector(retinal_dim,  unified_dim),
            "tongue":   ModalityProjector(tongue_dim,   unified_dim),
            "ecg":      ModalityProjector(ecg_dim,      unified_dim),
            "clinical": ModalityProjector(clinical_dim, unified_dim),
        })

        # ── Cross-modal Attention ──────────────────────────────────────────
        self.cross_attention = nn.ModuleDict({
            mod: CrossModalAttention(unified_dim, n_heads)
            for mod in self.MODALITIES
        })

        # ── Gating Network ─────────────────────────────────────────────────
        self.gating = GatingNetwork(self.n_modalities, unified_dim)

        # ── Fusion MLP ─────────────────────────────────────────────────────
        self.fusion_mlp = nn.Sequential(
            nn.Linear(unified_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )

        # ── Disease Heads ──────────────────────────────────────────────────
        self.disease_heads = nn.ModuleDict({
            disease: DiseaseHead(256, disease)
            for disease in self.DISEASES
        })

        # Confidence head (epistemic uncertainty estimate)
        self.confidence_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, len(self.DISEASES)),
            nn.Sigmoid(),
        )

    def forward(
        self,
        retinal_emb:  Optional[torch.Tensor] = None,
        tongue_emb:   Optional[torch.Tensor] = None,
        ecg_emb:      Optional[torch.Tensor] = None,
        clinical_emb: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args: Optional modality embeddings. Missing modalities → zeros + mask
        """
        device = (retinal_emb or tongue_emb or ecg_emb or clinical_emb).device
        B = (retinal_emb or tongue_emb or ecg_emb or clinical_emb).shape[0]

        # ── Project & Mask ─────────────────────────────────────────────────
        projected, mask = {}, []
        for i, (name, emb) in enumerate(zip(
            self.MODALITIES,
            [retinal_emb, tongue_emb, ecg_emb, clinical_emb]
        )):
            if emb is not None:
                projected[name] = self.projectors[name](emb)
                mask.append(torch.ones(B, 1, device=device))
            else:
                projected[name] = torch.zeros(B, self.unified_dim, device=device)
                mask.append(torch.zeros(B, 1, device=device))

        mask_tensor = torch.cat(mask, dim=-1)  # [B, 4]

        # ── Cross-modal Attention ──────────────────────────────────────────
        all_embs = torch.stack(list(projected.values()), dim=1)  # [B, 4, D]

        attended = {}
        for name in self.MODALITIES:
            q = projected[name].unsqueeze(1)               # [B, 1, D]
            attended[name] = self.cross_attention[name](q, all_embs)  # [B, D]

        # ── Gated Fusion ───────────────────────────────────────────────────
        attended_list = [attended[name] for name in self.MODALITIES]
        weights = self.gating(attended_list, mask_tensor)  # [B, 4]

        fused = sum(
            w.unsqueeze(-1) * emb
            for w, emb in zip(weights.unbind(dim=1), attended_list)
        )  # [B, D]

        # ── Fusion MLP ─────────────────────────────────────────────────────
        shared = self.fusion_mlp(fused)  # [B, 256]

        # ── Predictions ────────────────────────────────────────────────────
        risks = {
            disease: self.disease_heads[disease](shared)  # [B, 1]
            for disease in self.DISEASES
        }
        confidences = self.confidence_head(shared)  # [B, 5]

        return {
            **{f"{k}_risk": v for k, v in risks.items()},
            "modality_weights": weights,         # [B, 4]
            "confidence":       confidences,     # [B, 5]
            "shared_embedding": shared,          # [B, 256]
        }


class VitaFusionSimulated:
    """
    Simulation wrapper: fuses per-modality predictions into final risk scores.
    Uses evidence-based weighting of each modality's contribution.
    """

    # Modality contribution weights per disease (evidence-based)
    DISEASE_MODALITY_WEIGHTS = {
        "cardiovascular": {"retinal": 0.30, "tongue": 0.15, "ecg": 0.30, "clinical": 0.25},
        "diabetes":       {"retinal": 0.35, "tongue": 0.25, "ecg": 0.05, "clinical": 0.35},
        "hypertension":   {"retinal": 0.20, "tongue": 0.20, "ecg": 0.25, "clinical": 0.35},
        "arrhythmia":     {"retinal": 0.05, "tongue": 0.05, "ecg": 0.70, "clinical": 0.20},
        "metabolic":      {"retinal": 0.20, "tongue": 0.20, "ecg": 0.10, "clinical": 0.50},
    }

    def fuse(
        self,
        retinal_preds:  Optional[Dict] = None,
        tongue_preds:   Optional[Dict] = None,
        ecg_preds:      Optional[Dict] = None,
        clinical_preds: Optional[Dict] = None,
    ) -> Dict:
        """
        Fuse modality predictions into unified disease risk scores (0-100%).
        """
        rng = np.random.default_rng(99)
        results = {}
        modality_weights_used = {}

        for disease, weights in self.DISEASE_MODALITY_WEIGHTS.items():
            score = 0.0
            total_weight = 0.0

            # Retinal contribution
            key = f"{disease}_contribution" if disease != "arrhythmia" else "cardiovascular_contribution"
            if retinal_preds:
                rc = retinal_preds.get(key, retinal_preds.get("cardiovascular_contribution", 0.2))
                score += weights["retinal"] * rc
                total_weight += weights["retinal"]

            # Tongue contribution
            if tongue_preds:
                tc_key = f"{disease}_contribution"
                tc = tongue_preds.get(tc_key, tongue_preds.get("cardiovascular_contribution", 0.2))
                score += weights["tongue"] * tc
                total_weight += weights["tongue"]

            # ECG contribution
            if ecg_preds:
                ec_key = f"{disease}_contribution" if disease in ["cardiovascular", "arrhythmia"] else "cardiovascular_contribution"
                ec = ecg_preds.get(ec_key, 0.15)
                if disease == "arrhythmia":
                    ec = ecg_preds.get("arrhythmia_contribution", 0.2)
                score += weights["ecg"] * ec
                total_weight += weights["ecg"]

            # Clinical contribution
            if clinical_preds:
                cc = clinical_preds.get(f"{disease}_risk", 0.2)
                score += weights["clinical"] * cc
                total_weight += weights["clinical"]

            # Normalize by available modalities
            if total_weight > 0:
                score /= total_weight

            # Add small jitter for realism
            score = float(np.clip(score + rng.uniform(-0.02, 0.02), 0, 1))
            results[disease] = score * 100  # Convert to percentage

            # Track modality weights
            modality_weights_used[disease] = {
                k: v for k, v in weights.items()
                if (k == "retinal"  and retinal_preds) or
                   (k == "tongue"   and tongue_preds) or
                   (k == "ecg"      and ecg_preds) or
                   (k == "clinical" and clinical_preds)
            }

        # Compute confidence based on number of modalities available
        n_modalities = sum([
            retinal_preds is not None,
            tongue_preds  is not None,
            ecg_preds     is not None,
            clinical_preds is not None,
        ])
        base_confidence = 0.5 + (n_modalities / 4) * 0.4
        confidences = {
            disease: float(np.clip(base_confidence + rng.uniform(-0.05, 0.05), 0, 1))
            for disease in self.DISEASE_MODALITY_WEIGHTS
        }

        return {
            "risk_scores":        results,          # {disease: 0-100}
            "confidence_scores":  confidences,      # {disease: 0-1}
            "modality_weights":   modality_weights_used,
            "n_modalities_used":  n_modalities,
        }
