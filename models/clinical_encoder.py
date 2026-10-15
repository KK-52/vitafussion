"""
VitaFusion AI — Clinical Data Encoder
========================================
MLP encoder for structured clinical/tabular data.

Encodes: demographics, vitals, lifestyle, medical history, lab values
Output:  256-dim embedding + per-disease risk logits

Datasets supported:
  - Full 22-feature VitaFusion clinical schema (default)
  - Pima Diabetes 8-feature schema (PIMA_FEATURE_DIM=8)

Architecture:
    Input → Feature Embedding → 3-Layer MLP with residual connections
    → LayerNorm → 256-dim embedding → Multi-task heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.preprocessing import StandardScaler

from config import PIMA_FEATURES, PIMA_FEATURE_DIM

# Pima 8-feature names in column order
PIMA_FEATURE_NAMES = PIMA_FEATURES


class ClinicalEncoder(nn.Module):
    """
    3-Layer MLP encoder for clinical tabular data.
    Handles mixed types: continuous, binary, ordinal.
    """

    def __init__(
        self,
        input_dim: int = 22,
        embedding_dim: int = 256,
        hidden_dims: List[int] = [256, 512, 384],
        dropout: float = 0.3,
        num_diseases: int = 5,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # ── Feature Embedding Layer ────────────────────────────────────────
        # Each feature gets its own learned embedding projection
        self.feature_embedding = nn.Linear(input_dim, hidden_dims[0])
        self.input_norm = nn.LayerNorm(hidden_dims[0])

        # ── MLP Layers with Residual ───────────────────────────────────────
        layers = []
        in_dim = hidden_dims[0]
        for out_dim in hidden_dims[1:]:
            layers.append(nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ))
            in_dim = out_dim

        self.mlp_layers = nn.ModuleList(layers)

        # Residual projection if dims differ
        self.res_proj = nn.ModuleList([
            nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            if hidden_dims[i] != hidden_dims[i + 1] else nn.Identity()
            for i in range(len(hidden_dims) - 1)
        ])

        # ── Embedding Head ─────────────────────────────────────────────────
        self.embedding_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        # ── Multi-task Disease Heads ───────────────────────────────────────
        self.disease_heads = nn.ModuleDict({
            "cardiovascular": nn.Linear(embedding_dim, 1),
            "diabetes":       nn.Linear(embedding_dim, 1),
            "hypertension":   nn.Linear(embedding_dim, 1),
            "arrhythmia":     nn.Linear(embedding_dim, 1),
            "metabolic":      nn.Linear(embedding_dim, 1),
        })

        # ── Risk Factor Importance Head ────────────────────────────────────
        # Produces attention weights over input features (used for SHAP)
        self.feature_attention = nn.Sequential(
            nn.Linear(hidden_dims[-1], input_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Clinical feature tensor [B, 22]

        Returns:
            dict with embedding and per-disease risk probabilities
        """
        h = self.feature_embedding(x)  # [B, 256]
        h = self.input_norm(h)

        prev_h = h
        for layer, res in zip(self.mlp_layers, self.res_proj):
            h_new = layer(prev_h)
            h_new = h_new + res(prev_h)  # Residual
            prev_h = h_new

        attn = self.feature_attention(prev_h)   # [B, input_dim]
        emb  = self.embedding_head(prev_h)       # [B, 256]

        # Per-disease risk scores
        risks = {}
        for disease, head in self.disease_heads.items():
            risks[f"{disease}_risk"] = torch.sigmoid(head(emb))  # [B, 1]

        return {
            "embedding":        emb,
            "feature_attention": attn,
            **risks,
        }


class ClinicalEncoderSimulated:
    """
    Physics/medicine-based clinical risk calculator.
    Uses established clinical risk scores (Framingham, ADA, JNC) as basis.
    """

    def __init__(self, embedding_dim: int = 256):
        self.embedding_dim = embedding_dim

    def predict(self, features: Dict) -> Dict:
        """
        Args:
            features: dict of clinical values matching CLINICAL_FEATURES schema
        """
        rng = np.random.default_rng(42)

        # ── Extract values with safe defaults ────────────────────────────
        age    = float(features.get("age", 45))
        gender = features.get("gender", "Male")
        bmi    = float(features.get("bmi", 25))
        systolic  = float(features.get("systolic_bp", 120))
        diastolic = float(features.get("diastolic_bp", 80))
        hr        = float(features.get("heart_rate", 72))
        spo2      = float(features.get("oxygen_saturation", 98))

        smoking  = features.get("smoking", "Never")
        alcohol  = features.get("alcohol", "None")
        activity = features.get("physical_activity", "Moderate")
        sleep    = float(features.get("sleep_hours", 7))

        dm_hist  = bool(features.get("diabetes_history", False))
        htn_hist = bool(features.get("hypertension_history", False))
        fhx_hd   = bool(features.get("family_heart_disease", False))

        glucose  = float(features.get("fasting_glucose", 90))
        hba1c    = float(features.get("hba1c", 5.5))
        chol     = float(features.get("total_cholesterol", 180))
        hdl      = float(features.get("hdl_cholesterol", 50))
        trig     = float(features.get("triglycerides", 150))

        # ── Framingham-inspired Cardiovascular Risk ───────────────────────
        cv_score = 0.0
        cv_score += self._age_score(age, gender)
        cv_score += 0.12 if smoking == "Current" else (0.05 if smoking == "Former" else 0)
        cv_score += 0.08 if systolic > 140 else (0.04 if systolic > 130 else 0)
        cv_score += 0.06 if chol > 240 else (0.02 if chol > 200 else 0)
        cv_score += -0.04 if hdl > 60 else (0.03 if hdl < 40 else 0)
        cv_score += 0.08 if fhx_hd else 0
        cv_score += 0.05 if activity == "Sedentary" else (-0.03 if activity == "Active" else 0)
        cv_score += 0.03 if bmi > 30 else 0
        cv_score = float(np.clip(cv_score + rng.uniform(-0.03, 0.03), 0, 1))

        # ── ADA-inspired Diabetes Risk ─────────────────────────────────────
        dm_score = 0.0
        dm_score += 0.15 if dm_hist else 0
        dm_score += self._bmi_score(bmi) * 0.25
        dm_score += (glucose - 70) / 430 * 0.4
        dm_score += (hba1c - 4.0) / 11.0 * 0.3
        dm_score += 0.05 if age > 45 else 0
        dm_score += 0.05 if activity == "Sedentary" else (-0.03 if activity == "Active" else 0)
        dm_score = float(np.clip(dm_score + rng.uniform(-0.03, 0.03), 0, 1))

        # ── JNC-inspired Hypertension Risk ────────────────────────────────
        htn_score = 0.0
        htn_score += 0.2 if htn_hist else 0
        htn_score += (systolic - 100) / 120 * 0.35
        htn_score += (diastolic - 60) / 80 * 0.2
        htn_score += 0.06 if fhx_hd else 0
        htn_score += 0.04 if smoking == "Current" else 0
        htn_score += self._bmi_score(bmi) * 0.15
        htn_score += 0.04 if alcohol == "Heavy" else 0
        htn_score = float(np.clip(htn_score + rng.uniform(-0.03, 0.03), 0, 1))

        # ── Arrhythmia Risk ───────────────────────────────────────────────
        arr_score = 0.0
        arr_score += self._age_score(age, gender) * 0.3
        arr_score += 0.1 if hr > 100 or hr < 55 else 0
        arr_score += 0.08 if spo2 < 95 else 0
        arr_score += 0.06 if smoking == "Current" else 0
        arr_score += 0.05 if htn_hist else 0
        arr_score = float(np.clip(arr_score + rng.uniform(-0.03, 0.03), 0, 1))

        # ── Metabolic Syndrome Risk (ATP III criteria) ────────────────────
        ms_score = 0.0
        ms_score += 0.2 if bmi > 30 else (0.1 if bmi > 25 else 0)
        ms_score += 0.15 if trig > 150 else 0
        ms_score += 0.1  if hdl < 40 else 0
        ms_score += (systolic - 100) / 120 * 0.15
        ms_score += (glucose - 70) / 430 * 0.2
        ms_score = float(np.clip(ms_score + rng.uniform(-0.03, 0.03), 0, 1))

        embedding = rng.normal(0, 1, self.embedding_dim).astype(np.float32)

        # SHAP-like feature importances
        feature_importances = self._compute_feature_importances(features, cv_score, dm_score, htn_score)

        return {
            "embedding":               embedding,
            "cardiovascular_risk":     cv_score,
            "diabetes_risk":           dm_score,
            "hypertension_risk":       htn_score,
            "arrhythmia_risk":         arr_score,
            "metabolic_risk":          ms_score,
            "feature_importances":     feature_importances,
        }

    def _age_score(self, age: float, gender: str) -> float:
        """Age-adjusted risk (Framingham-style)."""
        base = (age - 18) / 82  # 0 at 18, 1 at 100
        multiplier = 1.0 if gender == "Male" else 0.85
        return float(base * multiplier * 0.25)

    def _bmi_score(self, bmi: float) -> float:
        if bmi < 18.5: return 0.1
        if bmi < 25.0: return 0.0
        if bmi < 30.0: return 0.15
        if bmi < 35.0: return 0.30
        return 0.45

    def _compute_feature_importances(self, features: Dict, cv: float, dm: float, htn: float) -> Dict:
        """Compute relative feature importance for SHAP-like visualization."""
        importances = {
            "Age":                  self._age_score(float(features.get("age", 45)), features.get("gender", "Male")) / 0.25,
            "BMI":                  self._bmi_score(float(features.get("bmi", 25))),
            "Systolic BP":          (float(features.get("systolic_bp", 120)) - 100) / 120,
            "Fasting Glucose":      (float(features.get("fasting_glucose", 90)) - 70) / 430,
            "HbA1c":               (float(features.get("hba1c", 5.5)) - 4) / 11,
            "Total Cholesterol":    (float(features.get("total_cholesterol", 180)) - 100) / 300,
            "Triglycerides":        (float(features.get("triglycerides", 150)) - 50) / 950,
            "HDL (protective)":     -(float(features.get("hdl_cholesterol", 50)) - 20) / 80 * 0.5,
            "Smoking":             0.12 if features.get("smoking") == "Current" else (0.05 if features.get("smoking") == "Former" else 0),
            "Physical Activity":   -0.03 if features.get("physical_activity") == "Active" else (0.05 if features.get("physical_activity") == "Sedentary" else 0),
            "Family Hx Heart Dis": 0.08 if features.get("family_heart_disease") else 0,
            "Diabetes History":    0.15 if features.get("diabetes_history") else 0,
            "Hypertension History":0.2  if features.get("hypertension_history") else 0,
            "Alcohol":             0.04 if features.get("alcohol") == "Heavy" else 0,
            "Sleep Hours":         (7 - float(features.get("sleep_hours", 7))) * 0.02,
            "Heart Rate":          abs(float(features.get("heart_rate", 72)) - 70) / 130 * 0.1,
        }
        return importances


# ══════════════════════════════════════════════════════════════════════════════
#  Pima Diabetes Clinical Encoder
# ══════════════════════════════════════════════════════════════════════════════

class PimaClinicalEncoder(ClinicalEncoder):
    """
    ClinicalEncoder subclass tuned for the 8-feature Pima Diabetes dataset.

    Input:  [B, 8]  (Pregnancies, Glucose, BP, SkinThickness,
                     Insulin, BMI, DiabetesPedigreeFunction, Age)
    Output: 256-dim embedding + binary diabetes logit

    Inherits the full MLP+residual architecture; only input_dim changes.
    The diabetes head uses BCEWithLogitsLoss for binary classification.
    """

    FEATURE_NAMES = PIMA_FEATURE_NAMES

    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dims: List[int] = [128, 256, 384],
        dropout: float = 0.3,
    ):
        super().__init__(
            input_dim    = PIMA_FEATURE_DIM,   # 8
            embedding_dim= embedding_dim,
            hidden_dims  = hidden_dims,
            dropout      = dropout,
            num_diseases = 1,   # diabetes only
        )
        # Replace the 5-disease heads with a single diabetes binary head
        self.disease_heads = nn.ModuleDict({
            "diabetes": nn.Linear(embedding_dim, 1),
        })
        # Dedicated Pima feature attention (8 features)
        self.feature_attention = nn.Sequential(
            nn.Linear(hidden_dims[-1], PIMA_FEATURE_DIM),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Pima feature tensor [B, 8]

        Returns:
            dict with:
                - embedding:         [B, 256]
                - diabetes_logit:    [B, 1]   (raw, use BCEWithLogitsLoss)
                - diabetes_prob:     [B, 1]   (Sigmoid)
                - feature_attention: [B, 8]   per-feature importance
        """
        out = super().forward(x)
        logit = out["embedding"] @ self.disease_heads["diabetes"].weight.T \
                + self.disease_heads["diabetes"].bias
        return {
            **out,
            "diabetes_logit": logit,
            "diabetes_prob":  torch.sigmoid(logit),
        }
