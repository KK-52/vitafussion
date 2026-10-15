"""
VitaFusion AI — ECG Attribution (Integrated Gradients)
=========================================================
Highlights which ECG signal segments are most important for arrhythmia prediction.
Uses saliency-based attribution in simulation mode.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional, Tuple
import io


# ── Simulated ECG Attribution ────────────────────────────────────────────────

def compute_ecg_attribution_simulated(
    ecg_signal: np.ndarray,
    r_peaks: np.ndarray,
    arrhythmia_class: str,
    arrhythmia_prob: float,
    fs: float = 250.0,
) -> np.ndarray:
    """
    Generate realistic ECG attribution map in simulation mode.

    Different arrhythmia types have characteristic regions of importance:
    - AFib: irregular RR intervals → attribution on QRS complexes
    - STEMI: ST segment elevation → attribution on ST regions
    - LBBB: wide QRS → QRS complex attribution
    - Normal: minimal attribution

    Args:
        ecg_signal: [timesteps] (single lead)
        r_peaks:    R-peak indices
        arrhythmia_class: Name of arrhythmia
        arrhythmia_prob:  Confidence (0-1)
        fs:         Sample rate

    Returns:
        attribution [timesteps] float32 in [0, 1]
    """
    n = len(ecg_signal)
    attribution = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(int(abs(ecg_signal.mean()) * 1e5) % 2**31)

    # QRS complex width (samples)
    qrs_width = int(0.08 * fs)  # ~80ms
    st_width  = int(0.12 * fs)  # ~120ms after R peak (ST segment)
    p_width   = int(0.06 * fs)  # ~60ms before R peak (P wave)

    for r_idx in r_peaks:
        intensity_scale = float(arrhythmia_prob) * rng.uniform(0.7, 1.0)

        if "Fibrillation" in arrhythmia_class or "AFib" in arrhythmia_class:
            # AFib: irregular QRS + absence of P waves
            qrs_lo = max(0, r_idx - qrs_width // 2)
            qrs_hi = min(n, r_idx + qrs_width // 2)
            attribution[qrs_lo:qrs_hi] = intensity_scale * np.random.uniform(0.6, 1.0)
            # Irregular RR = high attribution on intervals between beats
            if r_idx + int(0.2 * fs) < n:
                loc = r_idx + int(rng.uniform(0.1, 0.2) * fs)
                attribution[loc:loc + int(0.05 * fs)] = intensity_scale * 0.5

        elif "STEMI" in arrhythmia_class or "ST-Elevation" in arrhythmia_class:
            # STEMI: ST segment elevation after R peak
            st_lo = min(n, r_idx + qrs_width // 2)
            st_hi = min(n, st_lo + st_width)
            attribution[st_lo:st_hi] = intensity_scale * np.linspace(1.0, 0.3, st_hi - st_lo)

        elif "Bundle Branch" in arrhythmia_class:
            # LBBB/RBBB: wide QRS complex
            qrs_lo = max(0, r_idx - qrs_width)
            qrs_hi = min(n, r_idx + qrs_width)  # Wider
            attribution[qrs_lo:qrs_hi] = intensity_scale

        elif "Tachycardia" in arrhythmia_class or "Bradycardia" in arrhythmia_class:
            # Sinus abnormalities: P wave + RR interval
            p_lo = max(0, r_idx - int(0.16 * fs))
            p_hi = max(0, r_idx - int(0.1 * fs))
            attribution[p_lo:p_hi] = intensity_scale * 0.7

        elif "Ventricular" in arrhythmia_class:
            # PVC: broad, bizarre QRS
            qrs_lo = max(0, r_idx - int(1.5 * qrs_width))
            qrs_hi = min(n, r_idx + int(1.5 * qrs_width))
            attribution[qrs_lo:qrs_hi] = intensity_scale

        else:
            # Normal: very low attribution
            qrs_lo = max(0, r_idx - qrs_width // 2)
            qrs_hi = min(n, r_idx + qrs_width // 2)
            attribution[qrs_lo:qrs_hi] = 0.15 * rng.uniform(0.5, 1.0)

    # Smooth
    from scipy.ndimage import gaussian_filter1d
    attribution = gaussian_filter1d(attribution, sigma=5)

    # Normalize
    if attribution.max() > 0:
        attribution = attribution / attribution.max()

    return attribution.astype(np.float32)


# ── ECG Visualization ─────────────────────────────────────────────────────────

def plot_ecg_with_attribution(
    ecg_signal: np.ndarray,
    attribution: np.ndarray,
    r_peaks: np.ndarray,
    lead_name: str = "II",
    fs: float = 250.0,
    n_seconds: float = 5.0,
    arrhythmia_class: str = "Normal Sinus Rhythm",
    figsize: Tuple[int, int] = (14, 5),
) -> bytes:
    """
    Plot ECG signal with attribution overlay and R-peak markers.

    Returns PNG bytes.
    """
    n_samples = min(len(ecg_signal), int(n_seconds * fs))
    t = np.linspace(0, n_samples / fs, n_samples)

    sig  = ecg_signal[:n_samples]
    attr = attribution[:n_samples]
    sig_norm = (sig - sig.mean()) / (sig.std() + 1e-8)

    fig, (ax_ecg, ax_attr) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                           facecolor="#111827",
                                           gridspec_kw={"height_ratios": [3, 1]})
    fig.subplots_adjust(hspace=0.05)

    # ── ECG trace ─────────────────────────────────────────────────────────────
    ax_ecg.set_facecolor("#111827")

    # Background shading by attribution intensity
    for i in range(len(t) - 1):
        intensity = attr[i]
        if intensity > 0.15:
            color = plt.cm.YlOrRd(intensity)
            ax_ecg.axvspan(t[i], t[i + 1], alpha=intensity * 0.4,
                           color=color, linewidth=0)

    # ECG line
    ax_ecg.plot(t, sig_norm, color="#00D4FF", linewidth=1.2, alpha=0.95)

    # R-peak markers
    r_in_window = r_peaks[r_peaks < n_samples]
    if len(r_in_window) > 0:
        ax_ecg.scatter(t[r_in_window], sig_norm[r_in_window],
                       color="#EF4444", s=60, zorder=5, marker="v", label="R-peak")

    ax_ecg.set_ylabel(f"Lead {lead_name} (normalized)", color="#9CA3AF", fontsize=10)
    ax_ecg.set_title(
        f"ECG Analysis — {arrhythmia_class}  |  Lead {lead_name}",
        color="#E2E8F0", fontsize=12, fontweight="bold"
    )
    ax_ecg.legend(fontsize=9, facecolor="#1F2937", labelcolor="#E2E8F0",
                  edgecolor="#374151")
    ax_ecg.tick_params(colors="#9CA3AF")
    for spine in ax_ecg.spines.values():
        spine.set_edgecolor("#374151")
    ax_ecg.grid(True, color="#1F2937", linewidth=0.5, alpha=0.7)

    # ── Attribution bar ────────────────────────────────────────────────────────
    ax_attr.set_facecolor("#111827")
    ax_attr.fill_between(t, attr, alpha=0.8,
                          color="#F59E0B")
    ax_attr.plot(t, attr, color="#F59E0B", linewidth=0.8)
    ax_attr.set_ylabel("Attribution", color="#9CA3AF", fontsize=9)
    ax_attr.set_xlabel("Time (seconds)", color="#9CA3AF", fontsize=10)
    ax_attr.set_ylim(0, 1.05)
    ax_attr.tick_params(colors="#9CA3AF")
    for spine in ax_attr.spines.values():
        spine.set_edgecolor("#374151")
    ax_attr.grid(True, color="#1F2937", linewidth=0.5)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="#111827", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def plot_ecg_12_lead(
    ecg_signal: np.ndarray,
    lead_names: List[str],
    r_peaks: Optional[np.ndarray] = None,
    fs: float = 250.0,
    n_seconds: float = 2.5,
    figsize: Tuple[int, int] = (16, 10),
) -> bytes:
    """
    Plot 12-lead ECG in standard clinical layout (3 rows × 4 leads).
    Returns PNG bytes.
    """
    n_samples = min(ecg_signal.shape[1], int(n_seconds * fs))
    t = np.linspace(0, n_samples / fs, n_samples)

    # Standard 12-lead layout
    n_rows, n_cols = 3, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, facecolor="#111827")
    fig.suptitle("12-Lead ECG", color="#E2E8F0", fontsize=14, fontweight="bold", y=0.98)

    for idx, (ax, lead_name) in enumerate(zip(axes.flat, lead_names[:12])):
        ax.set_facecolor("#0D1B2A")
        lead_sig = ecg_signal[idx, :n_samples]
        lead_norm = (lead_sig - lead_sig.mean()) / (lead_sig.std() + 1e-8)

        ax.plot(t, lead_norm, color="#00D4FF", linewidth=0.9, alpha=0.9)

        # Grid (ECG paper style)
        ax.grid(True, color="#1A3A4A", linewidth=0.4, alpha=0.8)
        ax.set_title(f"Lead {lead_name}", color="#9CA3AF", fontsize=9, pad=3)
        ax.set_xlim(0, t[-1])
        ax.tick_params(colors="#374151", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1A3A4A")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor="#111827", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.read()
