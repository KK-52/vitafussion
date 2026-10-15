"""
VitaFusion AI — Synthetic Data Generator
==========================================
Generates realistic synthetic patient data for demo mode.
All values are correlated to produce physiologically consistent profiles.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFilter
import io


# ── Patient Profile Generator ─────────────────────────────────────────────────

# Pre-defined demo patient archetypes
DEMO_PATIENTS = [
    {
        "id": "P001",
        "name": "Rajesh Kumar",
        "age": 58,
        "gender": "Male",
        "profile": "high_cv_risk",
        "description": "Middle-aged male with hypertension and family history of heart disease",
    },
    {
        "id": "P002",
        "name": "Priya Sharma",
        "age": 45,
        "gender": "Female",
        "profile": "diabetes_risk",
        "description": "Pre-diabetic female with elevated BMI and sedentary lifestyle",
    },
    {
        "id": "P003",
        "name": "Mohammed Al-Hassan",
        "age": 62,
        "gender": "Male",
        "profile": "arrhythmia",
        "description": "Older male with known atrial fibrillation and metabolic syndrome",
    },
    {
        "id": "P004",
        "name": "Sarah Chen",
        "age": 34,
        "gender": "Female",
        "profile": "low_risk",
        "description": "Young healthy female with excellent lifestyle habits",
    },
    {
        "id": "P005",
        "name": "David Okafor",
        "age": 52,
        "gender": "Male",
        "profile": "metabolic_syndrome",
        "description": "Male with metabolic syndrome, obesity, and dyslipidemia",
    },
]


PROFILE_FEATURES = {
    "high_cv_risk": {
        "age": 58, "gender": "Male", "height_cm": 172, "weight_kg": 88,
        "systolic_bp": 148, "diastolic_bp": 95, "heart_rate": 78, "oxygen_saturation": 97,
        "smoking": "Former", "alcohol": "Moderate", "physical_activity": "Light", "sleep_hours": 6.5,
        "diabetes_history": False, "hypertension_history": True, "family_heart_disease": True,
        "fasting_glucose": 108, "hba1c": 5.9, "total_cholesterol": 238,
        "hdl_cholesterol": 38, "triglycerides": 185, "creatinine": 1.1,
    },
    "diabetes_risk": {
        "age": 45, "gender": "Female", "height_cm": 158, "weight_kg": 82,
        "systolic_bp": 128, "diastolic_bp": 82, "heart_rate": 76, "oxygen_saturation": 98,
        "smoking": "Never", "alcohol": "None", "physical_activity": "Sedentary", "sleep_hours": 7,
        "diabetes_history": True, "hypertension_history": False, "family_heart_disease": False,
        "fasting_glucose": 118, "hba1c": 6.2, "total_cholesterol": 195,
        "hdl_cholesterol": 48, "triglycerides": 162, "creatinine": 0.85,
    },
    "arrhythmia": {
        "age": 62, "gender": "Male", "height_cm": 175, "weight_kg": 92,
        "systolic_bp": 138, "diastolic_bp": 88, "heart_rate": 95, "oxygen_saturation": 96,
        "smoking": "Current", "alcohol": "Moderate", "physical_activity": "Sedentary", "sleep_hours": 5.5,
        "diabetes_history": False, "hypertension_history": True, "family_heart_disease": True,
        "fasting_glucose": 105, "hba1c": 5.8, "total_cholesterol": 212,
        "hdl_cholesterol": 42, "triglycerides": 195, "creatinine": 1.2,
    },
    "low_risk": {
        "age": 34, "gender": "Female", "height_cm": 163, "weight_kg": 58,
        "systolic_bp": 112, "diastolic_bp": 72, "heart_rate": 64, "oxygen_saturation": 99,
        "smoking": "Never", "alcohol": "None", "physical_activity": "Active", "sleep_hours": 8,
        "diabetes_history": False, "hypertension_history": False, "family_heart_disease": False,
        "fasting_glucose": 85, "hba1c": 5.1, "total_cholesterol": 162,
        "hdl_cholesterol": 68, "triglycerides": 88, "creatinine": 0.72,
    },
    "metabolic_syndrome": {
        "age": 52, "gender": "Male", "height_cm": 178, "weight_kg": 105,
        "systolic_bp": 142, "diastolic_bp": 92, "heart_rate": 82, "oxygen_saturation": 97,
        "smoking": "Former", "alcohol": "Heavy", "physical_activity": "Sedentary", "sleep_hours": 6,
        "diabetes_history": True, "hypertension_history": True, "family_heart_disease": True,
        "fasting_glucose": 132, "hba1c": 6.8, "total_cholesterol": 248,
        "hdl_cholesterol": 35, "triglycerides": 285, "creatinine": 1.15,
    },
}


def get_demo_patient(patient_id: str) -> Dict:
    """Get pre-defined demo patient with clinical features."""
    patient_meta = next((p for p in DEMO_PATIENTS if p["id"] == patient_id), DEMO_PATIENTS[0])
    profile = patient_meta["profile"]
    features = PROFILE_FEATURES[profile].copy()
    # Compute BMI
    h = features["height_cm"] / 100
    features["bmi"] = round(features["weight_kg"] / h**2, 1)
    return {**patient_meta, "clinical_features": features}


def get_all_demo_patients() -> List[Dict]:
    return DEMO_PATIENTS


# ── Synthetic ECG Generator ────────────────────────────────────────────────────

def generate_synthetic_ecg(
    ecg_type: str = "normal",
    n_leads: int = 12,
    fs: float = 250.0,
    duration: float = 10.0,
    heart_rate: float = 75.0,
    noise_level: float = 0.05,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate synthetic ECG signal for demo purposes.

    Types: normal | afib | stemi | bradycardia | tachycardia | lbbb | pvc

    Returns: np.ndarray [n_leads, timesteps]
    """
    rng = np.random.default_rng(seed or 42)
    n_samples = int(duration * fs)
    t = np.linspace(0, duration, n_samples)

    ecg_signal = np.zeros((n_leads, n_samples), dtype=np.float32)

    # Lead II as reference
    rr_interval = 60.0 / heart_rate  # seconds

    if ecg_type == "normal":
        lead_ii = _generate_sinus_rhythm(t, rr_interval, rng, noise_level)
    elif ecg_type == "afib":
        lead_ii = _generate_afib(t, rr_interval, rng, noise_level)
    elif ecg_type == "stemi":
        lead_ii = _generate_stemi(t, rr_interval, rng, noise_level)
    elif ecg_type == "bradycardia":
        lead_ii = _generate_sinus_rhythm(t, 60.0 / 45.0, rng, noise_level)
    elif ecg_type == "tachycardia":
        lead_ii = _generate_sinus_rhythm(t, 60.0 / 115.0, rng, noise_level)
    elif ecg_type == "pvc":
        lead_ii = _generate_pvc(t, rr_interval, rng, noise_level)
    else:
        lead_ii = _generate_sinus_rhythm(t, rr_interval, rng, noise_level)

    # Fill all leads with correlated variants
    lead_amplitudes = [0.8, 1.0, 0.4, -0.7, 0.2, 0.6, -0.5, 0.9, 1.1, 1.2, 1.0, 0.7]
    for i in range(n_leads):
        amp   = lead_amplitudes[i % len(lead_amplitudes)]
        noise = rng.normal(0, noise_level, n_samples).astype(np.float32)
        ecg_signal[i] = lead_ii * amp + noise

    return ecg_signal


def _generate_sinus_rhythm(t, rr_interval, rng, noise_level):
    """Synthesize normal sinus rhythm ECG."""
    n = len(t)
    fs = 1.0 / (t[1] - t[0])
    signal = np.zeros(n, dtype=np.float32)

    # Generate beat positions
    beat_times = np.arange(0, t[-1], rr_interval)

    for bt in beat_times:
        idx = int(bt * fs)
        if idx >= n:
            break

        # P wave
        p_idx = max(0, idx - int(0.16 * fs))
        _add_gaussian(signal, p_idx, int(0.04 * fs), 0.15, n)

        # QRS complex
        _add_gaussian(signal, idx, int(0.02 * fs), 1.0, n)  # R peak
        _add_gaussian(signal, idx - int(0.02 * fs), int(0.01 * fs), -0.2, n)  # Q
        _add_gaussian(signal, idx + int(0.02 * fs), int(0.01 * fs), -0.3, n)  # S

        # T wave
        t_idx = idx + int(0.2 * fs)
        _add_gaussian(signal, t_idx, int(0.06 * fs), 0.35, n)

    signal += rng.normal(0, noise_level, n).astype(np.float32)
    return signal


def _generate_afib(t, rr_interval, rng, noise_level):
    """Synthesize atrial fibrillation ECG (irregular RR, no P waves, fine baseline flutter)."""
    n = len(t)
    fs = 1.0 / (t[1] - t[0])
    signal = np.zeros(n, dtype=np.float32)

    # Irregular RR intervals
    rr_mean = rr_interval
    current_time = rng.uniform(0, rr_mean * 0.3)
    while current_time < t[-1]:
        rr = rng.uniform(0.5 * rr_mean, 1.8 * rr_mean)  # Irregular
        idx = int(current_time * fs)
        if idx < n:
            _add_gaussian(signal, idx, int(0.02 * fs), rng.uniform(0.6, 1.2), n)
            _add_gaussian(signal, idx - int(0.015 * fs), int(0.01 * fs), -0.15, n)
            _add_gaussian(signal, idx + int(0.015 * fs), int(0.01 * fs), -0.25, n)
            t_idx = idx + int(0.18 * fs)
            _add_gaussian(signal, t_idx, int(0.06 * fs), 0.3, n)
        current_time += rr

    # Fibrillatory baseline
    flutter_freq = rng.uniform(350, 450)
    signal += 0.08 * np.sin(2 * np.pi * flutter_freq / fs * np.arange(n)).astype(np.float32)
    signal += rng.normal(0, noise_level * 1.5, n).astype(np.float32)
    return signal


def _generate_stemi(t, rr_interval, rng, noise_level):
    """Synthesize STEMI ECG with ST elevation."""
    signal = _generate_sinus_rhythm(t, rr_interval, rng, noise_level)
    n = len(t)
    fs = 1.0 / (t[1] - t[0])

    beat_times = np.arange(0, t[-1], rr_interval)
    for bt in beat_times:
        idx = int(bt * fs)
        st_start = idx + int(0.04 * fs)
        st_end   = idx + int(0.18 * fs)
        if st_end < n:
            st_elevation = rng.uniform(0.2, 0.4)
            signal[st_start:st_end] += st_elevation
    return signal


def _generate_pvc(t, rr_interval, rng, noise_level):
    """Synthesize PVC — premature ventricular contractions."""
    signal = _generate_sinus_rhythm(t, rr_interval, rng, noise_level)
    n = len(t)
    fs = 1.0 / (t[1] - t[0])

    # Add PVC beats every ~4-6 beats
    beat_times = np.arange(0, t[-1], rr_interval)
    for i, bt in enumerate(beat_times):
        if i % 5 == 2:  # Every 5th beat
            idx = int((bt - 0.2 * rr_interval) * fs)
            if idx > 0 and idx < n:
                _add_gaussian(signal, idx, int(0.04 * fs), -2.0, n)  # Wide negative QRS
                _add_gaussian(signal, idx + int(0.05 * fs), int(0.04 * fs), 1.5, n)
    return signal


def _add_gaussian(signal, center, sigma, amplitude, n):
    """Add a Gaussian waveform to signal array."""
    lo = max(0, center - 3 * sigma)
    hi = min(n, center + 3 * sigma)
    x  = np.arange(lo, hi)
    signal[lo:hi] += amplitude * np.exp(-0.5 * ((x - center) / (sigma + 1e-8)) ** 2).astype(np.float32)


# ── Synthetic Image Generator ─────────────────────────────────────────────────

def generate_synthetic_retinal_image(
    dr_grade: int = 0,
    size: int = 224,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a synthetic fundus-like image for demo.
    Returns RGB numpy array [H, W, 3].
    """
    rng = np.random.default_rng(seed or dr_grade * 7)
    img = np.zeros((size, size, 3), dtype=np.uint8)

    # Dark red/maroon background (retinal tissue)
    base_r = rng.integers(80, 120)
    base_g = rng.integers(25, 50)
    img[:, :, 0] = base_r
    img[:, :, 1] = base_g
    img[:, :, 2] = rng.integers(10, 30)

    # Circular crop mask
    center = (size // 2, size // 2)
    Y, X = np.ogrid[:size, :size]
    dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = dist > (size * 0.47)
    img[mask] = 0

    # Optic disc (bright circular region)
    oc_x, oc_y = int(size * 0.62), int(size * 0.50)
    oc_r = int(size * 0.08)
    Y2, X2 = np.ogrid[:size, :size]
    od_mask = (X2 - oc_x)**2 + (Y2 - oc_y)**2 < oc_r**2
    img[od_mask] = [255, 240, 200]

    # Blood vessels (lines)
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    for _ in range(rng.integers(8, 15)):
        x0, y0 = oc_x + rng.integers(-10, 10), oc_y + rng.integers(-10, 10)
        angle  = rng.uniform(0, 2 * np.pi)
        length = rng.integers(40, 80)
        x1 = int(x0 + np.cos(angle) * length)
        y1 = int(y0 + np.sin(angle) * length)
        width = rng.integers(1, 3)
        color = (rng.integers(150, 200), rng.integers(20, 50), rng.integers(20, 50))
        draw.line([(x0, y0), (x1, y1)], fill=color, width=width)

    # DR-grade severity: add lesions
    if dr_grade >= 1:
        for _ in range(dr_grade * 5):
            lx = rng.integers(40, size - 40)
            ly = rng.integers(40, size - 40)
            lr = rng.integers(2, 6)
            draw.ellipse([(lx-lr, ly-lr), (lx+lr, ly+lr)],
                         fill=(rng.integers(180, 255), rng.integers(0, 40), 0))

    img_result = np.array(pil)
    pil_blur = Image.fromarray(img_result).filter(ImageFilter.GaussianBlur(radius=1))
    return np.array(pil_blur)


def generate_synthetic_tongue_image(
    tongue_class: str = "Normal",
    size: int = 224,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a synthetic tongue-like image for demo.
    Returns RGB numpy array [H, W, 3].
    """
    rng = np.random.default_rng(seed or hash(tongue_class) % 100)

    COLOR_MAP = {
        "Normal":        (205, 120, 120),
        "Pale":          (230, 185, 185),
        "Red/Dark":      (180, 50, 50),
        "Purple/Blue":   (120, 80, 160),
        "Yellow Coat":   (210, 185, 100),
        "Cracked/Dry":   (195, 140, 120),
    }
    base_color = COLOR_MAP.get(tongue_class, (205, 120, 120))
    img = np.zeros((size, size, 3), dtype=np.uint8)

    # Dark background
    img[:] = [20, 15, 15]

    # Tongue shape (oval)
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    margin_x, margin_y = int(size * 0.15), int(size * 0.1)
    draw.ellipse(
        [(margin_x, margin_y), (size - margin_x, size - margin_y * 0.5)],
        fill=base_color,
    )

    # Texture
    arr = np.array(pil).astype(np.float32)
    texture = rng.normal(0, 8, (size, size, 3)).astype(np.float32)
    arr = np.clip(arr + texture, 0, 255).astype(np.uint8)

    if tongue_class == "Cracked/Dry":
        pil2 = Image.fromarray(arr)
        draw2 = ImageDraw.Draw(pil2)
        for _ in range(rng.integers(5, 12)):
            x0, y0 = rng.integers(margin_x, size-margin_x), rng.integers(margin_y, size-margin_y)
            draw2.line([(x0, y0), (x0+rng.integers(10,30), y0+rng.integers(-5,5))],
                       fill=(160, 100, 90), width=1)
        arr = np.array(pil2)

    pil_final = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius=2))
    return np.array(pil_final)
