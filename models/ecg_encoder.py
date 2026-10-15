"""
VitaFusion AI — ECG Signal Encoder
=====================================
1D-CNN + Bidirectional LSTM for ECG signal processing and arrhythmia detection.

Architecture:
    Multi-scale 1D-CNN → BiLSTM → Attention → Prediction heads

Reference:
    - Hannun et al. (2019) "Cardiologist-level arrhythmia detection with a CNN"
    - PTB-XL dataset: Wagner et al. (2020)

Dataset: PTB-XL (physionet.org/content/ptb-xl), MIMIC-IV ECG
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, List


# ── Sub-modules ─────────────────────────────────────────────────────────────

class MultiScaleConvBlock(nn.Module):
    """
    Parallel 1D convolutions at 3 time scales to capture:
      - Fine features (kernel=3):  P/QRS/T wave shapes
      - Mid features  (kernel=7):  rhythmic patterns
      - Coarse features(kernel=15): baseline trends
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        branch_out = out_channels // 3

        self.branch_fine   = self._make_branch(in_channels, branch_out, kernel=3)
        self.branch_mid    = self._make_branch(in_channels, branch_out, kernel=7)
        self.branch_coarse = self._make_branch(in_channels, out_channels - 2 * branch_out, kernel=15)

        self.norm = nn.BatchNorm1d(out_channels)
        self.act  = nn.GELU()

    def _make_branch(self, in_ch, out_ch, kernel):
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        out = torch.cat([
            self.branch_fine(x),
            self.branch_mid(x),
            self.branch_coarse(x),
        ], dim=1)
        return self.act(self.norm(out))


class ResidualConvBlock1D(nn.Module):
    """Residual block for 1D ECG signals."""

    def __init__(self, channels: int, kernel: int = 7, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel, padding=kernel // 2),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel, padding=kernel // 2),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class TemporalAttention(nn.Module):
    """Self-attention over time steps to weight important ECG segments."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, H]"""
        scores = self.attention(x)          # [B, T, 1]
        weights = F.softmax(scores, dim=1)  # [B, T, 1]
        return (x * weights).sum(dim=1)    # [B, H]


class ECGEncoder(nn.Module):
    """
    Full ECG encoder: Multi-scale 1D-CNN → Residual blocks → BiLSTM → Attention.

    Input:  ECG signal [B, leads, time_steps]  e.g. [B, 12, 2500] for 10s@250Hz
    Output: 256-dim embedding + arrhythmia classification
    """

    ARRHYTHMIA_CLASSES = [
        "Normal Sinus Rhythm",
        "Atrial Fibrillation",
        "ST-Elevation (STEMI)",
        "Left Bundle Branch Block",
        "Right Bundle Branch Block",
        "Premature Ventricular Contraction",
        "Sinus Tachycardia",
        "Sinus Bradycardia",
    ]

    def __init__(
        self,
        n_leads: int = 12,
        n_timesteps: int = 2500,
        embedding_dim: int = 256,
        n_arrhythmia_classes: int = 8,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # ── CNN Stem ───────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, 64, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )

        # ── Multi-scale blocks ─────────────────────────────────────────────
        self.ms_block1 = MultiScaleConvBlock(64,  128)
        self.ms_block2 = MultiScaleConvBlock(128, 256)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # ── Residual Blocks ────────────────────────────────────────────────
        self.res_blocks = nn.Sequential(
            ResidualConvBlock1D(256, kernel=7),
            ResidualConvBlock1D(256, kernel=7),
            ResidualConvBlock1D(256, kernel=5),
        )

        # ── BiLSTM ─────────────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if lstm_layers > 1 else 0,
        )

        lstm_out = lstm_hidden * 2  # bidirectional

        # ── Temporal Attention ─────────────────────────────────────────────
        self.attention = TemporalAttention(lstm_out)

        # ── Embedding Head ─────────────────────────────────────────────────
        self.embedding_head = nn.Sequential(
            nn.Linear(lstm_out, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        # ── Prediction Heads ───────────────────────────────────────────────
        self.arrhythmia_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_arrhythmia_classes),
        )
        self.hr_head     = nn.Linear(embedding_dim, 1)   # estimated heart rate
        self.pr_head     = nn.Linear(embedding_dim, 1)   # PR interval (ms)
        self.qrs_head    = nn.Linear(embedding_dim, 1)   # QRS duration (ms)
        self.qt_head     = nn.Linear(embedding_dim, 1)   # QTc interval (ms)

        # Attribution storage
        self._conv_features = None
        self.res_blocks[-1].block[0].register_forward_hook(
            lambda m, i, o: setattr(self, '_conv_features', o)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, leads, timesteps]  e.g. [B, 12, 2500]
        """
        # CNN
        h = self.stem(x)             # [B, 64, T/2]
        h = self.ms_block1(h)        # [B, 128, T/2]
        h = self.pool(h)             # [B, 128, T/4]
        h = self.ms_block2(h)        # [B, 256, T/4]
        h = self.pool(h)             # [B, 256, T/8]
        h = self.res_blocks(h)       # [B, 256, T/8]

        # LSTM: need [B, T, C]
        h = h.permute(0, 2, 1)      # [B, T/8, 256]
        h, _ = self.lstm(h)          # [B, T/8, lstm_out]

        # Attention pooling
        h = self.attention(h)        # [B, lstm_out]

        # Embedding
        emb = self.embedding_head(h) # [B, 256]

        # Predictions
        arr_logits = self.arrhythmia_head(emb)
        arr_probs  = F.softmax(arr_logits, dim=-1)

        return {
            "embedding":        emb,
            "arrhythmia_logits": arr_logits,
            "arrhythmia_probs":  arr_probs,
            "heart_rate":       F.relu(self.hr_head(emb)) * 200 + 30,  # 30-230 bpm
            "pr_interval":      F.relu(self.pr_head(emb)) * 200 + 100,  # ms
            "qrs_duration":     F.relu(self.qrs_head(emb)) * 100 + 60,
            "qtc_interval":     F.relu(self.qt_head(emb)) * 200 + 300,
        }


class ECGEncoderSimulated:
    """
    Simulation wrapper for ECGEncoder.
    Computes real signal features (RR intervals, amplitude, HRV) then maps to risks.
    """

    ARRHYTHMIA_CLASSES = ECGEncoder.ARRHYTHMIA_CLASSES

    def __init__(self, embedding_dim: int = 256, sample_rate: int = 250):
        self.embedding_dim = embedding_dim
        self.sample_rate = sample_rate

    def predict(self, ecg_signal: np.ndarray, clinical_risk_hint: float = 0.3) -> Dict:
        """
        Args:
            ecg_signal: [leads, timesteps] or [timesteps] (single lead)
        """
        if ecg_signal.ndim == 1:
            ecg_signal = ecg_signal[np.newaxis, :]  # [1, T]

        lead_ii = ecg_signal[1] if ecg_signal.shape[0] > 1 else ecg_signal[0]

        # ── Real signal feature computation ──────────────────────────────
        hr, hrv, rr_std = self._compute_hr_hrv(lead_ii)
        amplitude = float(np.percentile(lead_ii, 95) - np.percentile(lead_ii, 5))
        noise_lvl = float(np.std(np.diff(np.diff(lead_ii))))

        rng = np.random.default_rng(int(abs(lead_ii.mean()) * 1e6) % 2**31)

        # Risk heuristics based on signal features
        afib_score = float(np.clip(hrv / 100 * 0.4 + rng.uniform(0, 0.2) + clinical_risk_hint * 0.3, 0, 1))
        stemi_score= float(np.clip((amplitude - 0.5) * 0.5 + rng.uniform(0, 0.15) + clinical_risk_hint * 0.25, 0, 1))
        arr_idx    = 0  # Normal by default

        if afib_score > 0.5:
            arr_idx = 1  # AFib
        elif stemi_score > 0.4:
            arr_idx = 2  # STEMI
        elif hr > 100:
            arr_idx = 6  # Tachycardia
        elif hr < 55:
            arr_idx = 7  # Bradycardia

        # Probability vector
        arr_probs = rng.dirichlet(np.ones(8) * 0.1)
        arr_probs[arr_idx] = 0.5 + rng.uniform(0, 0.3)
        arr_probs /= arr_probs.sum()

        # ECG metrics
        pr_interval  = float(np.clip(rng.normal(160, 20), 120, 220))
        qrs_duration = float(np.clip(rng.normal(90, 12), 60, 160))
        qtc_interval = float(np.clip(rng.normal(420, 30), 350, 520))

        embedding = rng.normal(0, 1, self.embedding_dim).astype(np.float32)

        return {
            "embedding":              embedding,
            "arrhythmia_probs":       arr_probs,
            "arrhythmia_class":       self.ARRHYTHMIA_CLASSES[arr_idx],
            "arrhythmia_idx":         arr_idx,
            "heart_rate":             float(np.clip(hr, 40, 200)),
            "hrv_ms":                 float(np.clip(hrv, 5, 150)),
            "pr_interval":            pr_interval,
            "qrs_duration":           qrs_duration,
            "qtc_interval":           qtc_interval,
            "afib_score":             afib_score,
            "stemi_score":            stemi_score,
            "arrhythmia_contribution":float(np.clip(arr_probs[1:].sum() * 0.7 + clinical_risk_hint * 0.3, 0, 1)),
            "cardiovascular_contribution": float(np.clip(stemi_score * 0.6 + afib_score * 0.3 + clinical_risk_hint * 0.2, 0, 1)),
        }

    def _compute_hr_hrv(self, lead: np.ndarray) -> tuple:
        """Estimate heart rate and HRV from ECG lead via peak detection."""
        try:
            from scipy.signal import find_peaks
            # Normalize
            lead_norm = (lead - lead.mean()) / (lead.std() + 1e-8)
            # Find R-peaks
            min_dist = int(self.sample_rate * 0.4)  # 0.4s minimum RR
            peaks, _ = find_peaks(lead_norm, height=0.5, distance=min_dist)

            if len(peaks) < 2:
                return 75.0, 35.0, 30.0

            rr_intervals = np.diff(peaks) / self.sample_rate * 1000  # ms
            hr  = float(60000 / rr_intervals.mean())
            hrv = float(np.std(rr_intervals))
            return hr, hrv, float(rr_intervals.std())
        except Exception:
            return 75.0, 35.0, 30.0
