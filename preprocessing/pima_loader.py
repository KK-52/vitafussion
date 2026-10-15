"""
VitaFusion AI — Pima Indians Diabetes Dataset Loader
=====================================================
Loads the Pima Indians Diabetes Database (UCI / Kaggle).
https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

Dataset summary:
  - 768 female patients of Pima Indian heritage, aged >= 21
  - 8 clinical features + binary diabetes outcome
  - Known issue: physiologically impossible zeros in some columns → imputed

Features:
  0  Pregnancies              — number of times pregnant
  1  Glucose                  — plasma glucose concentration (mg/dL)
  2  BloodPressure            — diastolic blood pressure (mm Hg)
  3  SkinThickness            — triceps skinfold thickness (mm)
  4  Insulin                  — 2-hour serum insulin (mu U/ml)
  5  BMI                      — body mass index (kg/m²)
  6  DiabetesPedigreeFunction — diabetes pedigree function (genetic risk)
  7  Age                      — age (years)

Target:
  Outcome  — 0 = No diabetes, 1 = Diabetes

Expected file:
  <pima_path>  or  data/pima/diabetes.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from config import PIMA_FEATURES, PIMA_FEATURE_DIM, PIMA_ZERO_IMPUTE_COLS

# ── Constants ─────────────────────────────────────────────────────────────────
FEATURE_NAMES = PIMA_FEATURES   # 8 feature names in column order
N_FEATURES    = PIMA_FEATURE_DIM
TARGET_COL    = "Outcome"

# Clinical reference ranges for normalisation display
FEATURE_RANGES = {
    "Pregnancies":              (0, 17),
    "Glucose":                  (44, 199),
    "BloodPressure":            (24, 122),
    "SkinThickness":            (7, 99),
    "Insulin":                  (14, 846),
    "BMI":                      (18.2, 67.1),
    "DiabetesPedigreeFunction": (0.078, 2.42),
    "Age":                      (21, 81),
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_pima_csv(pima_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load Pima diabetes CSV. Handles both standard Kaggle header and headerless formats.

    Returns DataFrame with standard column names + 'Outcome' target.
    """
    pima_path = Path(pima_path)
    if not pima_path.exists():
        raise FileNotFoundError(
            f"Pima CSV not found: {pima_path}\n"
            f"Download from: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database\n"
            f"Save as: data/pima/diabetes.csv  (or set PIMA_PATH in .env)"
        )

    df = pd.read_csv(pima_path)

    # If no header (raw UCI format) — add column names
    if df.columns[0].isdigit() or df.shape[1] == 9 and TARGET_COL not in df.columns:
        df.columns = FEATURE_NAMES + [TARGET_COL]

    # Rename common alternatives
    rename_map = {
        "outcome": TARGET_COL,
        "diabetes": TARGET_COL,
        "label": TARGET_COL,
        "class": TARGET_COL,
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    return df


def impute_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace biologically impossible zeros with column median.
    Affects: Glucose, BloodPressure, SkinThickness, Insulin, BMI.
    """
    df = df.copy()
    for col in PIMA_ZERO_IMPUTE_COLS:
        if col in df.columns:
            median = df[col].replace(0, np.nan).median()
            df[col] = df[col].replace(0, median)
    return df


def preprocess_pima(
    df: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
    fit_scaler: bool = True,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Full preprocessing pipeline:
      1. Zero imputation
      2. StandardScaler normalisation (fit on train, transform on val/test)

    Returns:
        X: np.ndarray [N, 8]  normalised features
        y: np.ndarray [N]     binary labels
        scaler: fitted StandardScaler
    """
    df = impute_zeros(df)

    X = df[FEATURE_NAMES].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.int64)

    if scaler is None:
        scaler = StandardScaler()
    if fit_scaler:
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    return X, y, scaler


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

if TORCH_AVAILABLE:
    class PimaDataset(Dataset):
        """
        PyTorch Dataset for Pima Diabetes.

        Returns:
            features: torch.Tensor [8] normalised clinical features
            label:    torch.LongTensor  0 (no diabetes) or 1 (diabetes)
            raw:      dict of original (un-normalised) feature values
        """

        def __init__(self, X: np.ndarray, y: np.ndarray, raw_df: pd.DataFrame):
            self.X      = torch.tensor(X, dtype=torch.float32)
            self.y      = torch.tensor(y, dtype=torch.long)
            self.raw_df = raw_df.reset_index(drop=True)

        def __len__(self) -> int:
            return len(self.X)

        def __getitem__(self, idx: int) -> Dict:
            raw_row = self.raw_df.iloc[idx]
            return {
                "features": self.X[idx],
                "label":    self.y[idx],
                "raw": {col: float(raw_row.get(col, 0)) for col in FEATURE_NAMES},
            }


    class PimaDataModule:
        """
        Train/Val/Test DataLoaders for Pima Diabetes.
        Stratified 80/10/10 split preserving class balance.
        """

        def __init__(
            self,
            pima_path: Union[str, Path],
            batch_size: int = 64,
            num_workers: int = 0,
        ):
            self.pima_path   = Path(pima_path)
            self.batch_size  = batch_size
            self.num_workers = num_workers
            self.scaler      = None
            self._ready      = False

        def setup(self):
            df     = load_pima_csv(self.pima_path)
            df_imp = impute_zeros(df)

            # Stratified split
            y = df_imp[TARGET_COL].values
            rng = np.random.default_rng(42)
            pos_idx = np.where(y == 1)[0]
            neg_idx = np.where(y == 0)[0]

            def split_idx(idx):
                idx = rng.permutation(idx)
                t   = int(len(idx) * 0.8)
                v   = int(len(idx) * 0.9)
                return idx[:t], idx[t:v], idx[v:]

            p_tr, p_v, p_te = split_idx(pos_idx)
            n_tr, n_v, n_te = split_idx(neg_idx)
            tr = np.concatenate([p_tr, n_tr])
            vl = np.concatenate([p_v, n_v])
            te = np.concatenate([p_te, n_te])

            # Fit scaler on train only
            X_all = df_imp[FEATURE_NAMES].values.astype(np.float32)
            y_all = df_imp[TARGET_COL].values.astype(np.int64)
            self.scaler = StandardScaler()
            self.scaler.fit(X_all[tr])

            X_tr = self.scaler.transform(X_all[tr])
            X_vl = self.scaler.transform(X_all[vl])
            X_te = self.scaler.transform(X_all[te])

            self.train_dataset = PimaDataset(X_tr, y_all[tr], df_imp.iloc[tr])
            self.val_dataset   = PimaDataset(X_vl, y_all[vl], df_imp.iloc[vl])
            self.test_dataset  = PimaDataset(X_te, y_all[te], df_imp.iloc[te])
            self._ready        = True

        def train_dataloader(self):
            return DataLoader(self.train_dataset, batch_size=self.batch_size,
                              shuffle=True,  num_workers=self.num_workers)

        def val_dataloader(self):
            return DataLoader(self.val_dataset,   batch_size=self.batch_size,
                              shuffle=False, num_workers=self.num_workers)

        def test_dataloader(self):
            return DataLoader(self.test_dataset,  batch_size=self.batch_size,
                              shuffle=False, num_workers=self.num_workers)

        def summary(self) -> Dict:
            if not self._ready:
                raise RuntimeError("Call setup() first.")
            return {
                "train": len(self.train_dataset),
                "val":   len(self.val_dataset),
                "test":  len(self.test_dataset),
                "n_features": N_FEATURES,
                "feature_names": FEATURE_NAMES,
            }


# ── Dataset Statistics ─────────────────────────────────────────────────────────

def compute_pima_stats(pima_path: Union[str, Path]) -> Dict:
    """
    Compute Pima dataset statistics for the explorer page.
    Returns JSON-serializable dict.
    """
    try:
        df = load_pima_csv(pima_path)
    except FileNotFoundError as e:
        return {"available": False, "error": str(e)}

    df_imp = impute_zeros(df)
    n      = len(df_imp)

    # Class distribution
    outcome_counts = df_imp[TARGET_COL].value_counts().to_dict()
    class_dist = {
        "No Diabetes": int(outcome_counts.get(0, 0)),
        "Diabetes":    int(outcome_counts.get(1, 0)),
    }
    prevalence = round(class_dist["Diabetes"] / n * 100, 1)

    # Per-feature stats
    feature_stats = {}
    for feat in FEATURE_NAMES:
        if feat in df_imp.columns:
            col = df_imp[feat]
            feature_stats[feat] = {
                "mean":   round(float(col.mean()), 2),
                "std":    round(float(col.std()),  2),
                "min":    round(float(col.min()),  2),
                "max":    round(float(col.max()),  2),
                "median": round(float(col.median()),2),
                # Mean stratified by outcome
                "mean_no_dm": round(float(df_imp[df_imp[TARGET_COL]==0][feat].mean()), 2),
                "mean_dm":    round(float(df_imp[df_imp[TARGET_COL]==1][feat].mean()), 2),
            }

    # Age distribution
    age_hist = {}
    if "Age" in df_imp.columns:
        bins   = [20, 30, 40, 50, 60, 70, 82]
        labels = ["20-29","30-39","40-49","50-59","60-69","70+"]
        buckets = pd.cut(df_imp["Age"], bins=bins, labels=labels, right=False)
        age_hist = {str(k): int(v) for k, v in buckets.value_counts().sort_index().items()}

    # Correlation with outcome
    correlations = {}
    for feat in FEATURE_NAMES:
        if feat in df_imp.columns:
            corr = float(df_imp[feat].corr(df_imp[TARGET_COL]))
            correlations[feat] = round(corr, 3)

    return {
        "available":        True,
        "n_records":        n,
        "n_features":       N_FEATURES,
        "feature_names":    FEATURE_NAMES,
        "target_col":       TARGET_COL,
        "class_distribution": class_dist,
        "prevalence_pct":   prevalence,
        "feature_stats":    feature_stats,
        "age_distribution": age_hist,
        "feature_outcome_correlation": correlations,
        "imputed_columns":  PIMA_ZERO_IMPUTE_COLS,
        "task":             "binary classification (diabetes prediction)",
        "reference":        "Smith et al. (1988). UCI ML Repository — Pima Indians Diabetes Database",
    }


# ── Inference helper ──────────────────────────────────────────────────────────

def pima_features_from_clinical(clinical_dict: Dict) -> np.ndarray:
    """
    Extract and map VitaFusion clinical feature dict → Pima 8-feature vector.
    Used when running inference with clinical form data.

    Returns: np.ndarray [8]  (un-normalised, ready for scaler.transform)
    """
    mapping = {
        "Pregnancies":              float(clinical_dict.get("pregnancies", 0)),
        "Glucose":                  float(clinical_dict.get("fasting_glucose", 100)),
        "BloodPressure":            float(clinical_dict.get("diastolic_bp", 80)),
        "SkinThickness":            float(clinical_dict.get("skin_thickness", 20)),
        "Insulin":                  float(clinical_dict.get("insulin", 80)),
        "BMI":                      float(clinical_dict.get("bmi", 25)),
        "DiabetesPedigreeFunction": float(clinical_dict.get("diabetes_pedigree", 0.3)),
        "Age":                      float(clinical_dict.get("age", 35)),
    }
    return np.array([mapping[f] for f in FEATURE_NAMES], dtype=np.float32)
