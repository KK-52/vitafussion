"""
VitaFusion AI — Clinical Data Preprocessing
============================================
Imputation, scaling, encoding, and BMI calculation for clinical features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
import torch


# Feature groups
CONTINUOUS_FEATURES = [
    "age", "height_cm", "weight_kg", "bmi",
    "systolic_bp", "diastolic_bp", "heart_rate", "oxygen_saturation",
    "sleep_hours", "fasting_glucose", "hba1c",
    "total_cholesterol", "hdl_cholesterol", "triglycerides", "creatinine",
]
CATEGORICAL_FEATURES = ["gender", "smoking", "alcohol", "physical_activity"]
BOOLEAN_FEATURES = ["diabetes_history", "hypertension_history", "family_heart_disease"]

# Feature defaults (population medians)
FEATURE_DEFAULTS = {
    "age": 45.0,
    "height_cm": 165.0,
    "weight_kg": 70.0,
    "bmi": 25.7,
    "systolic_bp": 120.0,
    "diastolic_bp": 78.0,
    "heart_rate": 72.0,
    "oxygen_saturation": 98.0,
    "sleep_hours": 7.0,
    "fasting_glucose": 95.0,
    "hba1c": 5.5,
    "total_cholesterol": 185.0,
    "hdl_cholesterol": 50.0,
    "triglycerides": 135.0,
    "creatinine": 0.95,
    "gender": "Male",
    "smoking": "Never",
    "alcohol": "None",
    "physical_activity": "Moderate",
    "diabetes_history": False,
    "hypertension_history": False,
    "family_heart_disease": False,
}

# Categorical encodings
ENCODINGS = {
    "gender":            {"Male": 0, "Female": 1, "Other": 0.5},
    "smoking":           {"Never": 0, "Former": 1, "Current": 2},
    "alcohol":           {"None": 0, "Moderate": 1, "Heavy": 2},
    "physical_activity": {"Sedentary": 0, "Light": 1, "Moderate": 2, "Active": 3},
}

# Normalization parameters (population-level stats)
FEATURE_MEANS = {
    "age": 45.0, "height_cm": 165.0, "weight_kg": 70.0, "bmi": 25.5,
    "systolic_bp": 122.0, "diastolic_bp": 79.0, "heart_rate": 74.0, "oxygen_saturation": 97.8,
    "sleep_hours": 7.1, "fasting_glucose": 98.0, "hba1c": 5.6,
    "total_cholesterol": 188.0, "hdl_cholesterol": 52.0, "triglycerides": 138.0, "creatinine": 0.95,
}
FEATURE_STDS = {
    "age": 16.0, "height_cm": 10.0, "weight_kg": 15.0, "bmi": 5.0,
    "systolic_bp": 18.0, "diastolic_bp": 10.0, "heart_rate": 12.0, "oxygen_saturation": 1.5,
    "sleep_hours": 1.4, "fasting_glucose": 25.0, "hba1c": 0.8,
    "total_cholesterol": 38.0, "hdl_cholesterol": 14.0, "triglycerides": 75.0, "creatinine": 0.25,
}


def compute_bmi(height_cm: float, weight_kg: float) -> float:
    """Compute Body Mass Index."""
    height_m = height_cm / 100.0
    return float(weight_kg / (height_m ** 2))


def classify_bmi(bmi: float) -> str:
    """WHO BMI classification."""
    if bmi < 18.5: return "Underweight"
    if bmi < 25.0: return "Normal"
    if bmi < 30.0: return "Overweight"
    if bmi < 35.0: return "Obese Class I"
    if bmi < 40.0: return "Obese Class II"
    return "Obese Class III"


def classify_bp(systolic: float, diastolic: float) -> str:
    """ACC/AHA 2017 Blood Pressure classification."""
    if systolic < 120 and diastolic < 80:
        return "Normal"
    if systolic < 130 and diastolic < 80:
        return "Elevated"
    if systolic < 140 or diastolic < 90:
        return "Stage 1 Hypertension"
    return "Stage 2 Hypertension"


def classify_glucose(glucose: float, hba1c: float) -> str:
    """ADA glucose classification."""
    if glucose >= 126 or hba1c >= 6.5:
        return "Diabetes"
    if glucose >= 100 or hba1c >= 5.7:
        return "Prediabetes"
    return "Normal"


def preprocess_clinical_features(features: Dict) -> Dict:
    """
    Preprocess clinical features into model-ready tensor and metadata.

    Steps:
        1. Fill missing values with population medians
        2. Compute derived features (BMI, classifications)
        3. Encode categoricals
        4. Z-score normalize continuous features
        5. Build feature vector

    Returns:
        dict with:
            - tensor:           torch.Tensor [1, 22]
            - feature_vector:   np.ndarray [22,]
            - filled_features:  dict with imputed values
            - derived:          dict with BMI class, BP class, etc.
            - feature_names:    list of feature names
    """
    # ── Step 1: Fill missing values ──────────────────────────────────────────
    filled = {}
    for key, default in FEATURE_DEFAULTS.items():
        val = features.get(key, None)
        filled[key] = val if val is not None and val != "" else default

    # ── Step 2: Compute derived features ─────────────────────────────────────
    # Auto-compute BMI if not provided
    if not features.get("bmi"):
        filled["bmi"] = compute_bmi(float(filled["height_cm"]), float(filled["weight_kg"]))

    derived = {
        "bmi_value":    filled["bmi"],
        "bmi_class":    classify_bmi(float(filled["bmi"])),
        "bp_class":     classify_bp(float(filled["systolic_bp"]), float(filled["diastolic_bp"])),
        "glucose_class":classify_glucose(float(filled["fasting_glucose"]), float(filled["hba1c"])),
        "pulse_pressure":float(filled["systolic_bp"]) - float(filled["diastolic_bp"]),
    }

    # ── Step 3: Encode categoricals ───────────────────────────────────────────
    encoded = {}
    for feat, enc_map in ENCODINGS.items():
        val = str(filled.get(feat, list(enc_map.keys())[0]))
        encoded[feat] = enc_map.get(val, 0)

    # ── Step 4: Normalize continuous features ─────────────────────────────────
    normalized = {}
    for feat in CONTINUOUS_FEATURES:
        val   = float(filled.get(feat, FEATURE_DEFAULTS[feat]))
        mean  = FEATURE_MEANS.get(feat, val)
        std   = FEATURE_STDS.get(feat, 1.0)
        normalized[feat] = (val - mean) / (std + 1e-8)

    # ── Step 5: Build feature vector ──────────────────────────────────────────
    feature_names = (
        CONTINUOUS_FEATURES +
        CATEGORICAL_FEATURES +
        BOOLEAN_FEATURES
    )

    vector = np.array(
        [normalized[f] for f in CONTINUOUS_FEATURES] +
        [encoded[f] / max(ENCODINGS[f].values()) for f in CATEGORICAL_FEATURES] +
        [float(bool(filled.get(f, False))) for f in BOOLEAN_FEATURES],
        dtype=np.float32,
    )

    tensor = torch.tensor(vector, dtype=torch.float32).unsqueeze(0)  # [1, 22]

    return {
        "tensor":          tensor,
        "feature_vector":  vector,
        "filled_features": filled,
        "derived":         derived,
        "feature_names":   feature_names,
        "n_features":      len(vector),
    }


def validate_clinical_features(features: Dict) -> Tuple[bool, List[str]]:
    """
    Validate clinical input values against physiologically plausible ranges.

    Returns:
        (is_valid, list_of_warnings)
    """
    from config import CLINICAL_FEATURES

    warnings = []
    for key, schema in CLINICAL_FEATURES.items():
        val = features.get(key)
        if val is None:
            continue

        if schema["type"] == "numeric":
            lo, hi = schema["range"]
            if not (lo <= float(val) <= hi):
                warnings.append(
                    f"⚠️ {key}: {val} is outside expected range [{lo}, {hi}] {schema.get('unit', '')}"
                )
        elif schema["type"] == "categorical":
            if str(val) not in schema["values"]:
                warnings.append(f"⚠️ {key}: '{val}' not in {schema['values']}")

    return (len(warnings) == 0), warnings
