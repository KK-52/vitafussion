"""
VitaFusion AI — PTB-XL ECG Dataset Loader
==========================================
Loads the PTB-XL dataset (Wagner et al., 2020) from PhysioNet.
https://physionet.org/content/ptb-xl/1.0.3/

Dataset summary:
  - 21,799 clinical 12-lead ECGs from 18,869 patients
  - 10 seconds @ 500 Hz (also downsampled 100 Hz version)
  - 71 SCP-ECG statements: diagnostic, form, rhythm
  - Recommended splits: fold 9 = validation, fold 10 = test

Usage:
    from preprocessing.ptbxl_loader import PTBXLDataset, PTBXLDataModule, compute_dataset_stats

    dm = PTBXLDataModule(ptbxl_dir="data/ptbxl")
    dm.setup()
    train_loader = dm.train_dataloader()

Reference:
    Wagner, P. et al. (2020) PTB-XL, a large publicly available
    electrocardiography dataset. Scientific Data 7, 154.
"""

import os
import ast
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# ── Optional heavy imports (graceful fallback) ────────────────────────────────
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import wfdb
    WFDB_AVAILABLE = True
except ImportError:
    WFDB_AVAILABLE = False

from scipy.signal import resample as scipy_resample

from config import SCP_TO_VITAFUSION, SCP_SUPERCLASS

# ── Constants ─────────────────────────────────────────────────────────────────
LEAD_NAMES   = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
N_LEADS      = 12
TARGET_FS    = 250       # VitaFusion internal sample rate
DURATION_SEC = 10        # PTB-XL records are 10 seconds
TARGET_LEN   = TARGET_FS * DURATION_SEC   # 2500 samples

VITAFUSION_CLASSES = [
    "Normal Sinus Rhythm",
    "Atrial Fibrillation",
    "ST-Elevation (STEMI)",
    "Left Bundle Branch Block",
    "Right Bundle Branch Block",
    "Premature Ventricular Contraction",
    "Sinus Tachycardia",
    "Sinus Bradycardia",
]

DIAGNOSTIC_SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


# ── Low-level readers ─────────────────────────────────────────────────────────

def load_raw_wfdb(record_path: str, sampling_rate: int = 100) -> Tuple[np.ndarray, int]:
    """
    Load a single PTB-XL ECG record via WFDB.

    Args:
        record_path: Path without extension (e.g. 'data/ptbxl/records100/00000/00001_lr')
        sampling_rate: 100 (low-res) or 500 (high-res)

    Returns:
        signal [n_leads, n_samples] float32, actual_fs
    """
    if not WFDB_AVAILABLE:
        raise ImportError(
            "wfdb is required to read PTB-XL records. "
            "Install with: pip install wfdb"
        )
    record = wfdb.rdrecord(record_path)
    signal = record.p_signal.T.astype(np.float32)   # [leads, timesteps]
    fs     = int(record.fs)
    return signal, fs


def resample_to_vitafusion(signal: np.ndarray, source_fs: int) -> np.ndarray:
    """Resample [leads, timesteps] from source_fs to TARGET_FS (250 Hz)."""
    if source_fs == TARGET_FS:
        return signal
    new_len = int(signal.shape[1] * TARGET_FS / source_fs)
    return scipy_resample(signal, new_len, axis=1).astype(np.float32)


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Per-lead z-score normalization."""
    means = signal.mean(axis=1, keepdims=True)
    stds  = signal.std(axis=1, keepdims=True)
    return ((signal - means) / (stds + 1e-8)).astype(np.float32)


def pad_or_crop(signal: np.ndarray, target_len: int = TARGET_LEN) -> np.ndarray:
    """Ensure exact [leads, target_len] shape."""
    n_leads, n_samples = signal.shape
    if n_samples >= target_len:
        return signal[:, :target_len]
    pad = target_len - n_samples
    return np.pad(signal, ((0, 0), (0, pad)), mode="edge")


# ── Label parsing ─────────────────────────────────────────────────────────────

def parse_scp_labels(scp_str: str) -> Dict[str, float]:
    """
    Parse the scp_codes column from ptbxl_database.csv.
    Values are dicts like {"NORM": 100.0, "SR": 0.0}
    Returns: dict of code → confidence (0–100)
    """
    try:
        if isinstance(scp_str, dict):
            return scp_str
        return ast.literal_eval(scp_str)
    except Exception:
        return {}


def scp_to_vitafusion_label(scp_codes: Dict[str, float]) -> int:
    """
    Map SCP code dict to VitaFusion 8-class arrhythmia index.
    Selects the highest-confidence mapped code.
    Returns 0 (Normal) if no match found.
    """
    best_idx  = 0
    best_conf = -1.0
    for code, conf in scp_codes.items():
        code_upper = code.upper()
        if code_upper in SCP_TO_VITAFUSION:
            if conf > best_conf:
                best_conf = conf
                best_idx  = SCP_TO_VITAFUSION[code_upper]
    return best_idx


def scp_to_superclass_label(scp_codes: Dict[str, float]) -> str:
    """Map SCP codes to one of 5 diagnostic superclasses (NORM/MI/STTC/CD/HYP)."""
    best_cls  = "NORM"
    best_conf = -1.0
    for code, conf in scp_codes.items():
        sc = SCP_SUPERCLASS.get(code.upper())
        if sc and conf > best_conf:
            best_conf = conf
            best_cls  = sc
    return best_cls


def scp_to_multilabel_vector(
    scp_codes: Dict[str, float],
    scp_statements: pd.DataFrame,
    threshold: float = 50.0,
) -> np.ndarray:
    """
    Build a multi-hot vector over all 71 SCP classes.
    Includes codes with confidence >= threshold.

    Args:
        scp_codes: parsed dict from record
        scp_statements: loaded from scp_statements.csv (index = code)
        threshold: minimum confidence to include label (0–100)

    Returns: np.ndarray [n_scp_codes] binary
    """
    codes_list = list(scp_statements.index)
    vec = np.zeros(len(codes_list), dtype=np.float32)
    for code, conf in scp_codes.items():
        code_upper = code.upper()
        if code_upper in codes_list and conf >= threshold:
            vec[codes_list.index(code_upper)] = 1.0
    return vec


# ── Metadata loader ───────────────────────────────────────────────────────────

def load_ptbxl_metadata(ptbxl_dir: Union[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and parse ptbxl_database.csv + scp_statements.csv.

    Returns:
        metadata_df: one row per record with parsed scp_codes
        scp_df:      SCP statement definitions
    """
    ptbxl_dir = Path(ptbxl_dir)
    db_path  = ptbxl_dir / "ptbxl_database.csv"
    scp_path = ptbxl_dir / "scp_statements.csv"

    if not db_path.exists():
        raise FileNotFoundError(
            f"PTB-XL database not found at {db_path}.\n"
            f"Download from: https://physionet.org/content/ptb-xl/1.0.3/\n"
            f"Set PTBXL_DIR in .env to the extracted directory."
        )

    df = pd.read_csv(db_path, index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(parse_scp_labels)
    df["vitafusion_label"] = df["scp_codes"].apply(scp_to_vitafusion_label)
    df["diagnostic_superclass"] = df["scp_codes"].apply(scp_to_superclass_label)

    scp_df = pd.read_csv(scp_path, index_col=0) if scp_path.exists() else pd.DataFrame()

    return df, scp_df


def get_split_indices(
    metadata_df: pd.DataFrame,
    val_fold: int = 9,
    test_fold: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset using recommended strat_fold column.
      - test  = fold 10
      - val   = fold 9
      - train = folds 1-8
    """
    test_df  = metadata_df[metadata_df["strat_fold"] == test_fold]
    val_df   = metadata_df[metadata_df["strat_fold"] == val_fold]
    train_df = metadata_df[~metadata_df["strat_fold"].isin([val_fold, test_fold])]
    return train_df, val_df, test_df


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

if TORCH_AVAILABLE:
    class PTBXLDataset(Dataset):
        """
        PyTorch Dataset for PTB-XL.

        Returns dict with:
            - signal:     torch.Tensor [12, 2500] float32
            - label:      torch.LongTensor scalar  (VitaFusion 8-class)
            - ecg_id:     int
            - metadata:   dict (age, sex, heart_rate, etc.)
        """

        def __init__(
            self,
            metadata_df: pd.DataFrame,
            ptbxl_dir: Union[str, Path],
            sampling_rate: int = 100,
            augment: bool = False,
            label_mode: str = "arrhythmia",  # "arrhythmia" | "superclass"
        ):
            self.df           = metadata_df.reset_index()
            self.ptbxl_dir    = Path(ptbxl_dir)
            self.sampling_rate= sampling_rate
            self.augment      = augment
            self.label_mode   = label_mode
            self.fs_folder    = "records100" if sampling_rate == 100 else "records500"

        def __len__(self) -> int:
            return len(self.df)

        def __getitem__(self, idx: int) -> Dict:
            row    = self.df.iloc[idx]
            ecg_id = int(row["ecg_id"])

            # Build record path
            filename_col = "filename_lr" if self.sampling_rate == 100 else "filename_hr"
            rel_path = str(row[filename_col]).replace("\\", "/")
            record_path = str(self.ptbxl_dir / rel_path)

            # Load + preprocess
            try:
                signal, fs = load_raw_wfdb(record_path, self.sampling_rate)
            except Exception:
                signal = np.zeros((N_LEADS, TARGET_LEN), dtype=np.float32)
                fs     = self.sampling_rate

            signal = resample_to_vitafusion(signal, fs)
            signal = pad_or_crop(signal, TARGET_LEN)
            signal = normalize_signal(signal)

            # Augmentation
            if self.augment:
                signal = self._augment(signal)

            # Label
            if self.label_mode == "superclass":
                sc = row.get("diagnostic_superclass", "NORM")
                label = DIAGNOSTIC_SUPERCLASSES.index(sc) if sc in DIAGNOSTIC_SUPERCLASSES else 0
            else:
                label = int(row.get("vitafusion_label", 0))

            tensor = torch.tensor(signal, dtype=torch.float32)

            meta = {
                "ecg_id":     ecg_id,
                "age":        float(row.get("age", 0)),
                "sex":        int(row.get("sex", 0)),   # 0=male, 1=female
                "heart_rate": float(row.get("heart_rate", 0)) if "heart_rate" in row else 0.0,
                "strat_fold": int(row.get("strat_fold", 1)),
            }

            return {
                "signal":   tensor,
                "label":    torch.tensor(label, dtype=torch.long),
                "ecg_id":   ecg_id,
                "metadata": meta,
            }

        def _augment(self, signal: np.ndarray) -> np.ndarray:
            """Simple augmentations: Gaussian noise + random amplitude scaling."""
            rng = np.random.default_rng()
            # Additive noise
            noise_scale = rng.uniform(0.005, 0.02)
            signal = signal + rng.normal(0, noise_scale, signal.shape).astype(np.float32)
            # Amplitude scaling
            scale = rng.uniform(0.9, 1.1)
            signal = (signal * scale).astype(np.float32)
            return signal


    class PTBXLDataModule:
        """
        Convenience class to create train/val/test DataLoaders for PTB-XL.

        Example:
            dm = PTBXLDataModule("data/ptbxl", batch_size=32)
            dm.setup()
            for batch in dm.train_dataloader():
                signals = batch["signal"]   # [B, 12, 2500]
                labels  = batch["label"]    # [B]
        """

        def __init__(
            self,
            ptbxl_dir: Union[str, Path],
            sampling_rate: int = 100,
            batch_size: int = 32,
            num_workers: int = 0,
            label_mode: str = "arrhythmia",
            val_fold: int = 9,
            test_fold: int = 10,
        ):
            self.ptbxl_dir     = Path(ptbxl_dir)
            self.sampling_rate = sampling_rate
            self.batch_size    = batch_size
            self.num_workers   = num_workers
            self.label_mode    = label_mode
            self.val_fold      = val_fold
            self.test_fold     = test_fold
            self._setup_done   = False

        def setup(self):
            metadata_df, self.scp_df = load_ptbxl_metadata(self.ptbxl_dir)
            train_df, val_df, test_df = get_split_indices(
                metadata_df, self.val_fold, self.test_fold
            )
            self.train_dataset = PTBXLDataset(
                train_df, self.ptbxl_dir, self.sampling_rate,
                augment=True, label_mode=self.label_mode
            )
            self.val_dataset   = PTBXLDataset(
                val_df,   self.ptbxl_dir, self.sampling_rate,
                augment=False, label_mode=self.label_mode
            )
            self.test_dataset  = PTBXLDataset(
                test_df,  self.ptbxl_dir, self.sampling_rate,
                augment=False, label_mode=self.label_mode
            )
            self._setup_done = True

        def train_dataloader(self) -> "DataLoader":
            return DataLoader(
                self.train_dataset, batch_size=self.batch_size,
                shuffle=True, num_workers=self.num_workers,
                pin_memory=True, drop_last=True,
            )

        def val_dataloader(self) -> "DataLoader":
            return DataLoader(
                self.val_dataset, batch_size=self.batch_size,
                shuffle=False, num_workers=self.num_workers,
            )

        def test_dataloader(self) -> "DataLoader":
            return DataLoader(
                self.test_dataset, batch_size=self.batch_size,
                shuffle=False, num_workers=self.num_workers,
            )

        def summary(self) -> Dict:
            if not self._setup_done:
                raise RuntimeError("Call setup() first.")
            return {
                "train_size": len(self.train_dataset),
                "val_size":   len(self.val_dataset),
                "test_size":  len(self.test_dataset),
                "total":      len(self.train_dataset) + len(self.val_dataset) + len(self.test_dataset),
                "batch_size": self.batch_size,
                "label_mode": self.label_mode,
            }


# ── Dataset Statistics ─────────────────────────────────────────────────────────

def compute_dataset_stats(
    ptbxl_dir: Union[str, Path],
) -> Dict:
    """
    Compute rich statistics over the full PTB-XL dataset.
    Used by the Flask /api/ptbxl/stats endpoint.

    Returns a JSON-serializable dict covering:
      - record / patient counts
      - age / sex distribution
      - SCP label frequency (top 20)
      - VitaFusion class distribution
      - Diagnostic superclass distribution
      - Recommended split sizes
    """
    try:
        df, scp_df = load_ptbxl_metadata(ptbxl_dir)
    except FileNotFoundError as e:
        return {"available": False, "error": str(e)}

    # Basic counts
    n_records  = len(df)
    n_patients = df["patient_id"].nunique() if "patient_id" in df.columns else n_records

    # Age distribution (bucket into decades)
    age_col = "age" if "age" in df.columns else None
    age_hist = {}
    if age_col:
        df_age = df[age_col].dropna()
        bins = [0, 20, 30, 40, 50, 60, 70, 80, 90, 120]
        labels = ["<20", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90+"]
        age_buckets = pd.cut(df_age, bins=bins, labels=labels, right=False)
        age_hist = age_buckets.value_counts().sort_index().to_dict()
        age_hist = {str(k): int(v) for k, v in age_hist.items()}

    # Sex distribution
    sex_dist = {}
    if "sex" in df.columns:
        sex_counts = df["sex"].value_counts()
        sex_dist = {
            "Male":   int(sex_counts.get(0, 0)),
            "Female": int(sex_counts.get(1, 0)),
        }

    # SCP code frequency (top 30)
    scp_counts: Dict[str, int] = {}
    for scp_dict in df["scp_codes"]:
        for code in scp_dict:
            scp_counts[code.upper()] = scp_counts.get(code.upper(), 0) + 1
    top_scp = dict(sorted(scp_counts.items(), key=lambda x: x[1], reverse=True)[:30])

    # VitaFusion 8-class distribution
    vita_counts = df["vitafusion_label"].value_counts().to_dict()
    vita_dist = {
        VITAFUSION_CLASSES[i]: int(vita_counts.get(i, 0))
        for i in range(len(VITAFUSION_CLASSES))
    }

    # Diagnostic superclass distribution
    sc_counts = df["diagnostic_superclass"].value_counts().to_dict()
    sc_dist   = {str(k): int(v) for k, v in sc_counts.items()}

    # Split sizes
    train_df, val_df, test_df = get_split_indices(df)
    splits = {
        "train": len(train_df),
        "val":   len(val_df),
        "test":  len(test_df),
    }

    # Heart rate stats (if column present)
    hr_stats = {}
    if "heart_rate" in df.columns:
        hr = df["heart_rate"].dropna()
        hr_stats = {
            "mean": round(float(hr.mean()), 1),
            "std":  round(float(hr.std()),  1),
            "min":  round(float(hr.min()),  1),
            "max":  round(float(hr.max()),  1),
            "median": round(float(hr.median()), 1),
        }

    # Devices / recorders (if present)
    device_dist = {}
    if "device" in df.columns:
        device_dist = {str(k): int(v) for k, v in df["device"].value_counts().head(10).items()}

    return {
        "available":             True,
        "n_records":             n_records,
        "n_patients":            n_patients,
        "n_scp_codes":           len(scp_counts),
        "n_leads":               N_LEADS,
        "duration_sec":          DURATION_SEC,
        "sampling_rates":        [100, 500],
        "age_distribution":      age_hist,
        "sex_distribution":      sex_dist,
        "top_scp_codes":         top_scp,
        "vitafusion_class_dist": vita_dist,
        "superclass_dist":       sc_dist,
        "splits":                splits,
        "heart_rate_stats":      hr_stats,
        "device_distribution":   device_dist,
        "lead_names":            LEAD_NAMES,
        "reference": "Wagner et al. (2020) PTB-XL. Scientific Data 7, 154.",
    }


def load_sample_ecg(
    ptbxl_dir: Union[str, Path],
    ecg_id: Optional[int] = None,
    sampling_rate: int = 100,
) -> Dict:
    """
    Load a single ECG record from PTB-XL for visualization.

    Args:
        ptbxl_dir:    Root PTB-XL directory
        ecg_id:       Specific record ID (1-21799). Random if None.
        sampling_rate: 100 or 500 Hz

    Returns dict with signal, labels, and metadata for the explorer page.
    """
    df, scp_df = load_ptbxl_metadata(ptbxl_dir)

    if ecg_id is not None:
        if ecg_id not in df.index:
            raise ValueError(f"ECG ID {ecg_id} not found in PTB-XL database.")
        row = df.loc[ecg_id]
    else:
        row = df.sample(1).iloc[0]
        ecg_id = int(row.name) if hasattr(row, 'name') else 1

    # Load waveform
    filename_col = "filename_lr" if sampling_rate == 100 else "filename_hr"
    rel_path = str(row[filename_col]).replace("\\", "/")
    record_path = str(Path(ptbxl_dir) / rel_path)

    try:
        signal, fs = load_raw_wfdb(record_path, sampling_rate)
        signal_rs  = resample_to_vitafusion(signal, fs)
        signal_norm= normalize_signal(pad_or_crop(signal_rs))
    except Exception as e:
        signal_norm = np.zeros((N_LEADS, TARGET_LEN), dtype=np.float32)
        fs = sampling_rate

    scp_codes = row["scp_codes"] if isinstance(row["scp_codes"], dict) else {}

    return {
        "ecg_id":               int(ecg_id),
        "signal":               signal_norm.tolist(),        # [12, 2500]
        "lead_names":           LEAD_NAMES,
        "sample_rate":          TARGET_FS,
        "duration_sec":         DURATION_SEC,
        "vitafusion_label":     int(row.get("vitafusion_label", 0)),
        "vitafusion_class":     VITAFUSION_CLASSES[int(row.get("vitafusion_label", 0))],
        "diagnostic_superclass":str(row.get("diagnostic_superclass", "NORM")),
        "scp_codes":            {k: float(v) for k, v in scp_codes.items()},
        "metadata": {
            "age":          float(row.get("age", 0)),
            "sex":          "Female" if int(row.get("sex", 0)) == 1 else "Male",
            "heart_rate":   float(row.get("heart_rate", 0)) if "heart_rate" in row.index else None,
            "strat_fold":   int(row.get("strat_fold", 1)),
            "validated_by": str(row.get("validated_by", "")),
            "second_opinion": str(row.get("second_opinion_by", "")),
            "report":       str(row.get("report", "")),
        },
    }
