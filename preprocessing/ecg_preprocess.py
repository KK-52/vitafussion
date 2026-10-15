"""
VitaFusion AI — ECG Signal Preprocessing
==========================================
Bandpass filtering, baseline wander removal, R-peak detection,
segmentation, and normalization for ECG signals.

Supports: CSV, WFDB (.dat/.hea), and raw numpy arrays
"""

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.signal import butter, filtfilt, find_peaks
from typing import Dict, List, Optional, Tuple, Union
import io


# ── Constants ────────────────────────────────────────────────────────────────
DEFAULT_FS    = 250    # Hz
ECG_DURATION  = 10    # seconds
N_LEADS       = 12
LEAD_NAMES    = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


# ── Filters ──────────────────────────────────────────────────────────────────

def bandpass_filter(signal: np.ndarray, fs: float = DEFAULT_FS,
                    lowcut: float = 0.5, highcut: float = 40.0, order: int = 4) -> np.ndarray:
    """
    Butterworth bandpass filter.
    Removes baseline wander (<0.5 Hz) and EMG noise (>40 Hz).
    """
    nyquist = fs / 2.0
    low  = lowcut  / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    if signal.ndim == 1:
        return filtfilt(b, a, signal)
    return np.apply_along_axis(lambda x: filtfilt(b, a, x), axis=-1, arr=signal)


def remove_baseline_wander(signal: np.ndarray, fs: float = DEFAULT_FS,
                             cutoff: float = 0.5, order: int = 3) -> np.ndarray:
    """High-pass filter to remove baseline wander."""
    nyquist = fs / 2.0
    b, a = butter(order, cutoff / nyquist, btype='high')
    if signal.ndim == 1:
        return filtfilt(b, a, signal)
    return np.apply_along_axis(lambda x: filtfilt(b, a, x), axis=-1, arr=signal)


def notch_filter(signal: np.ndarray, fs: float = DEFAULT_FS,
                 freq: float = 50.0, Q: float = 30.0) -> np.ndarray:
    """Notch filter for powerline interference (50 Hz or 60 Hz)."""
    b, a = scipy_signal.iirnotch(freq / (fs / 2), Q)
    if signal.ndim == 1:
        return filtfilt(b, a, signal)
    return np.apply_along_axis(lambda x: filtfilt(b, a, x), axis=-1, arr=signal)


# ── R-Peak Detection (Pan-Tompkins inspired) ─────────────────────────────────

def detect_r_peaks(lead_signal: np.ndarray, fs: float = DEFAULT_FS) -> np.ndarray:
    """
    Detect R-peaks using differential + moving average threshold.
    Simplified Pan-Tompkins algorithm.

    Returns: array of R-peak sample indices
    """
    # Differentiate + square
    diff    = np.diff(lead_signal)
    squared = diff ** 2

    # Moving average (150ms window)
    window = int(0.15 * fs)
    if window < 1:
        window = 1
    ma = np.convolve(squared, np.ones(window) / window, mode='same')

    # Dynamic threshold
    threshold  = 0.5 * ma.max()
    min_dist   = int(0.3 * fs)  # Minimum 300ms between beats (200 bpm max)

    peaks, props = find_peaks(ma, height=threshold, distance=min_dist)
    return peaks


def compute_hrv_features(r_peaks: np.ndarray, fs: float = DEFAULT_FS) -> Dict:
    """
    Compute Heart Rate Variability (HRV) time-domain features.
    """
    if len(r_peaks) < 2:
        return {"hr_bpm": 75.0, "sdnn_ms": 0.0, "rmssd_ms": 0.0, "pnn50": 0.0}

    rr_intervals_ms = np.diff(r_peaks) / fs * 1000  # ms

    hr_bpm = 60000.0 / rr_intervals_ms.mean()
    sdnn   = float(rr_intervals_ms.std())            # Standard deviation of NN intervals
    rmssd  = float(np.sqrt(np.mean(np.diff(rr_intervals_ms) ** 2)))  # Root mean square successive diff
    pnn50  = float(np.sum(np.abs(np.diff(rr_intervals_ms)) > 50) / len(rr_intervals_ms) * 100)

    return {
        "hr_bpm":  float(np.clip(hr_bpm, 20, 300)),
        "sdnn_ms": float(np.clip(sdnn, 0, 500)),
        "rmssd_ms":float(np.clip(rmssd, 0, 500)),
        "pnn50":   float(np.clip(pnn50, 0, 100)),
        "rr_mean_ms":  float(rr_intervals_ms.mean()),
        "rr_std_ms":   float(rr_intervals_ms.std()),
        "n_beats":     len(r_peaks),
    }


# ── Signal Normalization ──────────────────────────────────────────────────────

def normalize_ecg(signal: np.ndarray) -> np.ndarray:
    """
    Per-channel z-score normalization.
    signal: [..., timesteps]
    """
    if signal.ndim == 1:
        std = signal.std()
        return (signal - signal.mean()) / (std + 1e-8)
    means = signal.mean(axis=-1, keepdims=True)
    stds  = signal.std(axis=-1, keepdims=True)
    return (signal - means) / (stds + 1e-8)


# ── Loaders ──────────────────────────────────────────────────────────────────

def load_ecg_from_csv(
    file_input: Union[str, bytes, io.BytesIO],
    fs_column: Optional[str] = None,
    lead_columns: Optional[List[str]] = None,
    sep: str = ",",
) -> Tuple[np.ndarray, float]:
    """
    Load ECG from CSV file.

    Expected format:
        - One column per lead (or single-lead)
        - Rows = time samples
        - Optional header row with lead names

    Returns:
        (signal [leads, timesteps], sample_rate)
    """
    if isinstance(file_input, bytes):
        df = pd.read_csv(io.BytesIO(file_input), sep=sep)
    elif isinstance(file_input, str):
        df = pd.read_csv(file_input, sep=sep)
    else:
        df = pd.read_csv(file_input, sep=sep)

    # Try to identify lead columns
    if lead_columns:
        ecg_df = df[lead_columns]
    else:
        # Auto-detect: use columns containing lead names or all numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Filter out time/sample columns
        ecg_cols = [c for c in numeric_cols
                    if not any(kw in c.lower() for kw in ["time", "sample", "index", "t_"])]
        ecg_df = df[ecg_cols] if ecg_cols else df[numeric_cols]

    ecg_array = ecg_df.values.T.astype(np.float32)  # [leads, timesteps]
    return ecg_array, DEFAULT_FS


def load_ecg_from_wfdb(record_path: str) -> Tuple[np.ndarray, float]:
    """Load ECG from PhysioNet WFDB format (.dat + .hea)."""
    try:
        import wfdb
        record = wfdb.rdrecord(record_path)
        signal = record.p_signal.T.astype(np.float32)  # [leads, timesteps]
        fs     = float(record.fs)
        return signal, fs
    except ImportError:
        raise ImportError("Install wfdb: pip install wfdb")


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def preprocess_ecg(
    signal: np.ndarray,
    fs: float = DEFAULT_FS,
    target_fs: float = DEFAULT_FS,
    target_duration: float = ECG_DURATION,
    target_leads: int = N_LEADS,
    apply_notch: bool = True,
    notch_freq: float = 50.0,
) -> Dict:
    """
    Full ECG preprocessing pipeline.

    Args:
        signal:           [leads, timesteps] or [timesteps]
        fs:               Input sample rate (Hz)
        target_fs:        Output sample rate
        target_duration:  Output duration (seconds)
        target_leads:     Expected number of leads
        apply_notch:      Apply powerline notch filter
        notch_freq:       50 or 60 Hz

    Returns dict with:
        - tensor:       torch.Tensor [1, leads, timesteps] model-ready
        - signal_norm:  np.ndarray [leads, timesteps] normalized signal
        - signal_raw:   np.ndarray original (after resample)
        - r_peaks:      R-peak indices (lead II or first lead)
        - hrv_features: dict of HRV metrics
        - quality:      float signal quality score
    """
    import torch

    # ── Ensure 2D [leads, timesteps] ─────────────────────────────────────────
    if signal.ndim == 1:
        signal = signal[np.newaxis, :]

    n_leads_in, n_timesteps = signal.shape

    # ── Resample if needed ───────────────────────────────────────────────────
    if fs != target_fs:
        new_len = int(n_timesteps * target_fs / fs)
        signal = scipy_signal.resample(signal, new_len, axis=-1)
        fs = target_fs

    # ── Crop/Pad to target duration ───────────────────────────────────────────
    target_samples = int(target_duration * target_fs)
    if n_timesteps > target_samples:
        signal = signal[:, :target_samples]
    elif n_timesteps < target_samples:
        pad = target_samples - signal.shape[1]
        signal = np.pad(signal, ((0, 0), (0, pad)), mode='edge')

    # ── Pad/Trim leads ────────────────────────────────────────────────────────
    if n_leads_in < target_leads:
        # Repeat available leads (simulate missing leads)
        repeats = (target_leads // n_leads_in) + 1
        signal = np.tile(signal, (repeats, 1))[:target_leads, :]
    elif n_leads_in > target_leads:
        signal = signal[:target_leads, :]

    # ── Bandpass filter ───────────────────────────────────────────────────────
    signal_filt = bandpass_filter(signal, fs)

    # ── Notch filter ──────────────────────────────────────────────────────────
    if apply_notch:
        signal_filt = notch_filter(signal_filt, fs, notch_freq)

    # ── Normalize ─────────────────────────────────────────────────────────────
    signal_norm = normalize_ecg(signal_filt)

    # ── R-peak detection (use Lead II = index 1, or first lead) ──────────────
    ref_lead = signal_norm[min(1, signal_norm.shape[0] - 1)]
    r_peaks  = detect_r_peaks(ref_lead, fs)

    # ── HRV Features ──────────────────────────────────────────────────────────
    hrv_features = compute_hrv_features(r_peaks, fs)

    # ── Quality score ─────────────────────────────────────────────────────────
    quality = _compute_ecg_quality(signal_norm, r_peaks, fs)

    # ── Model tensor ──────────────────────────────────────────────────────────
    tensor = torch.tensor(signal_norm, dtype=torch.float32).unsqueeze(0)  # [1, L, T]

    return {
        "tensor":       tensor,
        "signal_norm":  signal_norm,
        "signal_raw":   signal,
        "r_peaks":      r_peaks,
        "hrv_features": hrv_features,
        "sample_rate":  fs,
        "n_leads":      target_leads,
        "n_timesteps":  target_samples,
        "lead_names":   LEAD_NAMES[:target_leads],
        "quality":      quality,
    }


def _compute_ecg_quality(signal: np.ndarray, r_peaks: np.ndarray, fs: float) -> float:
    """Estimate ECG signal quality (0=noise, 1=clean)."""
    # Flatline detection
    if signal.std() < 0.01:
        return 0.0

    # SNR estimate
    noise = signal - scipy_signal.savgol_filter(signal, min(51, signal.shape[1] // 10 * 2 + 1), 3)
    snr = float(signal.var() / (noise.var() + 1e-8))
    snr_score = float(np.clip(snr / 20, 0, 1))

    # Beat regularity
    if len(r_peaks) >= 3:
        rr = np.diff(r_peaks)
        regularity = 1 - min(1, rr.std() / (rr.mean() + 1e-8))
    else:
        regularity = 0.3

    quality = float(np.clip(snr_score * 0.5 + regularity * 0.5, 0, 1))
    return quality
