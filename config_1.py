"""
VitaFusion AI — Central Configuration
=====================================
All global settings, thresholds, and constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Paths ─────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
SAMPLE_DIR = DATA_DIR / "sample"
MODELS_DIR = ROOT_DIR / "models" / "pretrained"
PTBXL_DIR  = Path(os.getenv("PTBXL_DIR", str(DATA_DIR / "ptbxl")))
ODIR_DIR   = Path(os.getenv("ODIR_DIR",  str(DATA_DIR / "odir")))
TMC_DIR    = Path(os.getenv("TMC_DIR",   str(DATA_DIR / "tmc")))
PIMA_PATH  = Path(os.getenv("PIMA_PATH", str(DATA_DIR / "pima" / "diabetes.csv")))

# ─── App Settings ──────────────────────────────────────────────────────────
APP_NAME = "VitaFusion AI"
APP_VERSION = "1.0.0"
APP_TAGLINE = "Multimodal Healthcare Intelligence Platform"
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ─── LLM Settings ──────────────────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # openai | gemini | ollama | groq
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ─── ODIR-5K Dataset Settings ───────────────────────────────────────────────
ODIR_CLASSES = ["Normal", "Diabetes", "Glaucoma", "Cataract",
                "AMD", "Hypertension", "Myopia", "Other"]
ODIR_N_CLASSES  = 8
ODIR_N_PATIENTS = 5000
# ODIR label → VitaFusion disease (for fusion weighting)
ODIR_TO_DISEASE = {
    "Normal":       None,
    "Diabetes":     "diabetes",
    "Glaucoma":     "cardiovascular",
    "Cataract":     "metabolic",
    "AMD":          "metabolic",
    "Hypertension": "hypertension",
    "Myopia":       None,
    "Other":        "cardiovascular",
}

# ─── TMC 21-Category Tongue Dataset Settings ────────────────────────────────
TMC_21_CLASSES = [
    # Body Normal
    "Normal_ThinWhite",   "Normal_ThickWhite",  "Normal_Yellow",
    "Normal_Black",       "Normal_Peeled",
    # Body Pale
    "Pale_ThinWhite",     "Pale_ThickWhite",    "Pale_Yellow",
    "Pale_Black",         "Pale_Peeled",
    # Body Red
    "Red_ThinWhite",      "Red_ThickWhite",     "Red_Yellow",
    "Red_Black",          "Red_Peeled",
    # Body DarkRed
    "DarkRed_ThinWhite",  "DarkRed_ThickWhite", "DarkRed_Yellow",
    # Body Purple
    "Purple_ThinWhite",   "Purple_ThickWhite",  "Purple_Yellow",
]
TMC_N_CLASSES = 21
# Map 21 TMC classes → 6 VitaFusion display groups
TMC_TO_VITAFUSION_GROUP = {
    **{c: "Normal"      for c in TMC_21_CLASSES if c.startswith("Normal")},
    **{c: "Pale"        for c in TMC_21_CLASSES if c.startswith("Pale")},
    **{c: "Red/Dark"    for c in TMC_21_CLASSES if c.startswith("Red") or c.startswith("DarkRed")},
    **{c: "Purple/Blue" for c in TMC_21_CLASSES if c.startswith("Purple")},
}

# ─── Shenzhen Tongue Dataset Settings (YOLO-format, 20 classes) ─────────────
# Class indices match the shezhenv3-txt.yaml nc:20 definition
SHENZHEN_20_CLASSES = [
    "jiankangshe",    # 0  healthy tongue
    "botaishe",       # 1  thin-coat tongue
    "hongshe",        # 2  red tongue
    "zishe",          # 3  purple tongue
    "pangdashe",      # 4  swollen/puffy tongue
    "shoushe",        # 5  thin/lean tongue
    "hongdianshe",    # 6  red-dots tongue
    "liewenshe",      # 7  cracked tongue
    "chihenshe",      # 8  teeth-mark tongue
    "baitaishe",      # 9  white-coat tongue
    "huangtaishe",    # 10 yellow-coat tongue
    "heitaishe",      # 11 black-coat tongue
    "huataishe",      # 12 slippery/greasy coat
    "shenquao",       # 13 kidney-deficiency (concave)
    "shenqutu",       # 14 kidney-deficiency (convex)
    "gandanao",       # 15 liver-gallbladder (concave)
    "gandantu",       # 16 liver-gallbladder (convex)
    "piweiao",        # 17 spleen-stomach (concave)
    "xinfeitu",       # 18 heart-lung (convex)
    "xinfeiao",       # 19 heart-lung (concave)
]
SHENZHEN_N_CLASSES = 20

# Map 20 Shenzhen classes → 6 VitaFusion display groups
SHENZHEN_TO_VITAFUSION = {
    # Normal / Healthy
    "jiankangshe":  "Normal",
    "botaishe":     "Normal",
    "baitaishe":    "Normal",
    "huataishe":    "Normal",
    # Pale / Deficient
    "pangdashe":    "Pale",
    "shoushe":      "Pale",
    "chihenshe":    "Pale",
    # Red / Heat
    "hongshe":      "Red/Dark",
    "hongdianshe":  "Red/Dark",
    "shenqutu":     "Red/Dark",
    "gandantu":     "Red/Dark",
    "xinfeitu":     "Red/Dark",
    # Purple / Stagnation
    "zishe":        "Purple/Blue",
    "shenquao":     "Purple/Blue",
    "gandanao":     "Purple/Blue",
    "piweiao":      "Purple/Blue",
    "xinfeiao":     "Purple/Blue",
    # Yellow Coat / Damp-Heat
    "huangtaishe":  "Yellow Coat",
    "heitaishe":    "Yellow Coat",
    # Cracked / Dry / Yin-deficiency
    "liewenshe":    "Cracked/Dry",
}

# ─── Pima Diabetes Dataset Settings ──────────────────────────────────────────
PIMA_FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
PIMA_FEATURE_DIM = 8
PIMA_N_RECORDS   = 768
# Zero-valued fields that need imputation (physiologically impossible to be 0)
PIMA_ZERO_IMPUTE_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# ─── PTB-XL Dataset Settings ───────────────────────────────────────────────
PTBXL_SAMPLING_RATE   = int(os.getenv("PTBXL_SAMPLING_RATE", "100"))   # 100 or 500 Hz
PTBXL_TARGET_FS       = 250    # Resample to match VitaFusion pipeline
PTBXL_N_RECORDS       = 21799
PTBXL_N_PATIENTS      = 18869
PTBXL_N_SCP_CODES     = 71

# SCP-ECG code → VitaFusion arrhythmia class index
# (0=Normal, 1=AFib, 2=STEMI, 3=LBBB, 4=RBBB, 5=PVC, 6=Tachy, 7=Brady)
SCP_TO_VITAFUSION = {
    # Normal / Sinus
    "NORM":   0,  "SR":    0,
    # Atrial Fibrillation / Flutter
    "AFIB":   1,  "AFLT":  1,
    # STEMI / MI
    "AMI":    2,  "IMI":   2,  "LMI":   2,  "ALMI":  2,
    "ASMI":   2,  "ILMI":  2,  "IPMI":  2,  "IPLMI": 2,
    "INJAL":  2,  "INJIN": 2,  "INJLA": 2,
    # LBBB
    "LBBB":   3,  "CLBBB": 3,
    # RBBB
    "RBBB":   4,  "CRBBB": 4,  "IRBBB": 4,  "LPFB":  4,
    # PVC / Ectopics
    "PVC":    5,  "TRIGU": 5,  "BIGU":  5,  "SVTAC": 5,
    # Tachycardia
    "STACH":  6,  "PSVT":  6,  "SVARR": 6,
    # Bradycardia / Pacing
    "SBRAD":  7,  "PACE":  7,  "SARRH": 7,
}

# Diagnostic superclass mapping (for 5-class diagnostic mode)
SCP_SUPERCLASS = {
    "NORM": "NORM",
    # Myocardial Infarction
    "AMI":"MI", "IMI":"MI", "LMI":"MI", "ALMI":"MI", "ASMI":"MI",
    "ILMI":"MI", "IPMI":"MI", "IPLMI":"MI", "INJAL":"MI", "INJIN":"MI", "INJLA":"MI",
    # ST/T Change
    "NST_":"STTC", "DIG":"STTC", "LNGQT":"STTC", "ISCA":"STTC", "ISCI":"STTC",
    "ISC_":"STTC", "INVT":"STTC", "ANEUR":"STTC", "EL":"STTC", "VCLVH":"STTC",
    # Conduction Disturbance
    "LBBB":"CD", "RBBB":"CD", "CLBBB":"CD", "CRBBB":"CD", "IRBBB":"CD",
    "LPFB":"CD", "LAFB":"CD", "WPW":"CD", "IVCD":"CD", "AVB":"CD",
    "1AVB":"CD", "2AVB":"CD", "3AVB":"CD",
    # Hypertrophy
    "LVH":"HYP", "RVH":"HYP", "LVOLT":"HYP", "HVOLT":"HYP", "LAO/LAE":"HYP",
    "RAO/RAE":"HYP", "SEHYP":"HYP",
}

# ─── Model Architecture ─────────────────────────────────────────────────────
RETINAL_IMG_SIZE = (224, 224)
TONGUE_IMG_SIZE = (224, 224)
ECG_SAMPLE_RATE = 250          # Hz
ECG_DURATION = 10              # seconds
ECG_LEADS = 12                 # number of leads
CLINICAL_FEATURE_DIM = 22      # number of clinical input features
EMBEDDING_DIM = 512            # unified embedding dimension

# ─── Disease Risk Thresholds ────────────────────────────────────────────────
RISK_THRESHOLDS = {
    "low":      (0,  30),
    "moderate": (30, 60),
    "high":     (60, 80),
    "critical": (80, 100),
}

RISK_COLORS = {
    "low":      "#22C55E",  # green
    "moderate": "#F59E0B",  # amber
    "high":     "#EF4444",  # red
    "critical": "#7C3AED",  # purple
}

# ─── Disease Labels ──────────────────────────────────────────────────────────
DISEASES = {
    "cardiovascular": {
        "label": "Cardiovascular Risk",
        "icon":  "🫀",
        "color": "#EF4444",
        "description": "Risk of coronary artery disease, heart attack, or heart failure",
    },
    "diabetes": {
        "label": "Diabetes Risk",
        "icon":  "🩸",
        "color": "#F59E0B",
        "description": "Risk of Type 2 Diabetes Mellitus based on metabolic indicators",
    },
    "hypertension": {
        "label": "Hypertension Risk",
        "icon":  "💊",
        "color": "#8B5CF6",
        "description": "Risk of high blood pressure and associated complications",
    },
    "arrhythmia": {
        "label": "Arrhythmia Risk",
        "icon":  "📊",
        "color": "#00D4FF",
        "description": "Risk of irregular heart rhythm detected from ECG analysis",
    },
    "metabolic": {
        "label": "Metabolic Syndrome",
        "icon":  "⚗️",
        "color": "#10B981",
        "description": "Cluster of conditions increasing risk of heart disease and diabetes",
    },
}

# ─── Clinical Feature Schema ─────────────────────────────────────────────────
CLINICAL_FEATURES = {
    # Demographics
    "age":              {"type": "numeric", "range": (18, 100),   "unit": "years"},
    "gender":           {"type": "categorical", "values": ["Male", "Female", "Other"]},
    "height_cm":        {"type": "numeric", "range": (100, 220),  "unit": "cm"},
    "weight_kg":        {"type": "numeric", "range": (30, 200),   "unit": "kg"},
    "bmi":              {"type": "numeric", "range": (10, 60),    "unit": "kg/m²"},
    # Vitals
    "systolic_bp":      {"type": "numeric", "range": (70, 220),   "unit": "mmHg"},
    "diastolic_bp":     {"type": "numeric", "range": (40, 140),   "unit": "mmHg"},
    "heart_rate":       {"type": "numeric", "range": (30, 200),   "unit": "bpm"},
    "oxygen_saturation":{"type": "numeric", "range": (70, 100),   "unit": "%"},
    # Lifestyle
    "smoking":          {"type": "categorical", "values": ["Never", "Former", "Current"]},
    "alcohol":          {"type": "categorical", "values": ["None", "Moderate", "Heavy"]},
    "physical_activity":{"type": "categorical", "values": ["Sedentary", "Light", "Moderate", "Active"]},
    "sleep_hours":      {"type": "numeric", "range": (3, 12),     "unit": "hours/day"},
    # Medical History
    "diabetes_history": {"type": "boolean"},
    "hypertension_history": {"type": "boolean"},
    "family_heart_disease": {"type": "boolean"},
    # Lab Values (optional)
    "fasting_glucose":  {"type": "numeric", "range": (50, 500),   "unit": "mg/dL"},
    "hba1c":            {"type": "numeric", "range": (4, 15),     "unit": "%"},
    "total_cholesterol":{"type": "numeric", "range": (100, 400),  "unit": "mg/dL"},
    "hdl_cholesterol":  {"type": "numeric", "range": (20, 100),   "unit": "mg/dL"},
    "triglycerides":    {"type": "numeric", "range": (50, 1000),  "unit": "mg/dL"},
    "creatinine":       {"type": "numeric", "range": (0.3, 10),   "unit": "mg/dL"},
}

# ─── XAI Settings ────────────────────────────────────────────────────────────
GRADCAM_LAYER = "layer4"       # ResNet target layer
SHAP_N_SAMPLES = 100           # background samples for SHAP
SHAP_MAX_DISPLAY = 15          # top features to display

# ─── Color Palette ───────────────────────────────────────────────────────────
COLORS = {
    "primary":    "#00D4FF",
    "secondary":  "#7C3AED",
    "accent":     "#F59E0B",
    "success":    "#22C55E",
    "danger":     "#EF4444",
    "bg":         "#0A0E1A",
    "bg_card":    "#111827",
    "bg_panel":   "#1F2937",
    "text":       "#E2E8F0",
    "text_muted": "#9CA3AF",
    "border":     "#374151",
}
