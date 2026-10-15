"""
VitaFusion AI — ODIR-5K Ocular Disease Dataset Loader
======================================================
Loads the ODIR-5K (Ocular Disease Intelligent Recognition) dataset.
https://odir5k.grand-challenge.org/

Dataset summary:
  - 5,000 patients (10,000 fundus images — left + right eye)
  - 8 disease categories: N, D, G, C, A, H, M, O
  - Multi-label: a patient can have multiple conditions
  - Annotations in Excel (ODIR-5K_Training_Annotations.xlsx)

Folder structure expected:
    <odir_dir>/
        ODIR-5K_Training_Annotations.xlsx   (or .csv)
        ODIR-5K_Training_Dataset/
            <patient_id>_left.jpg
            <patient_id>_right.jpg
        ODIR-5K_Testing_Images/             (optional)

Reference:
    Peking University — ICDM 2019 Smart Health Challenge
"""

import os
import io
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as T
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from config import ODIR_CLASSES, ODIR_TO_DISEASE

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE    = 224
LABEL_COLS  = ["N", "D", "G", "C", "A", "H", "M", "O"]   # 8 disease columns
N_CLASSES   = 8
CLASS_NAMES = ODIR_CLASSES  # ["Normal","Diabetes","Glaucoma","Cataract","AMD","Hypertension","Myopia","Other"]

# ── Annotation loader ─────────────────────────────────────────────────────────

def load_odir_annotations(odir_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Load ODIR annotation file (Excel or CSV fallback).
    Expected columns: ID, Patient Age, Patient Sex, Left-Fundus, Right-Fundus,
                      Left-Diagnostic Keywords, Right-Diagnostic Keywords,
                      N, D, G, C, A, H, M, O   (binary label columns)

    Returns DataFrame with one row per patient.
    """
    odir_dir = Path(odir_dir)

    # Try Excel first, then CSV
    for fname in [
        "ODIR-5K_Training_Annotations.xlsx",
        "ODIR-5K_Training_Annotations.csv",
        "annotations.xlsx",
        "annotations.csv",
        "labels.csv",
    ]:
        fpath = odir_dir / fname
        if fpath.exists():
            if fpath.suffix == ".xlsx":
                df = pd.read_excel(fpath)
            else:
                df = pd.read_csv(fpath)
            return _normalize_odir_df(df)

    raise FileNotFoundError(
        f"ODIR annotation file not found in {odir_dir}.\n"
        f"Expected: ODIR-5K_Training_Annotations.xlsx\n"
        f"Download from: https://odir5k.grand-challenge.org/\n"
        f"Set ODIR_DIR in .env to the extracted directory."
    )


def _normalize_odir_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and ensure label columns exist."""
    # Standardize column names
    df.columns = [c.strip() for c in df.columns]

    # If label columns not present, try to parse from diagnostic keywords
    if not all(col in df.columns for col in LABEL_COLS):
        df = _parse_labels_from_keywords(df)

    # Ensure numeric types for label columns
    for col in LABEL_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.float32)

    # Standardize ID column
    if "ID" not in df.columns and df.columns[0] not in LABEL_COLS:
        df = df.rename(columns={df.columns[0]: "ID"})

    return df


def _parse_labels_from_keywords(df: pd.DataFrame) -> pd.DataFrame:
    """Fallback: infer binary labels from diagnostic keyword columns."""
    keyword_map = {
        "N": ["normal fundus"],
        "D": ["diabetic retinopathy", "proliferative diabetic"],
        "G": ["glaucoma", "glaucomatous"],
        "C": ["cataract", "lens"],
        "A": ["macular degeneration", "amd", "age-related"],
        "H": ["hypertensive", "hypertension"],
        "M": ["myopia", "myopic"],
        "O": [],  # anything else
    }
    kw_cols = [c for c in df.columns if "keyword" in c.lower() or "diagnos" in c.lower()]

    for code, keywords in keyword_map.items():
        if code not in df.columns:
            if keywords and kw_cols:
                combined = df[kw_cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
                df[code] = combined.apply(lambda x: 1.0 if any(k in x for k in keywords) else 0.0)
            else:
                df[code] = 0.0

    # "Other" = has no other label
    other_mask = df[["N", "D", "G", "C", "A", "H", "M"]].sum(axis=1) == 0
    df["O"] = other_mask.astype(np.float32)
    return df


def find_image_path(odir_dir: Path, patient_id, eye: str = "left") -> Optional[Path]:
    """
    Locate a fundus image for a given patient + eye side.
    Tries multiple common naming conventions.
    """
    candidates = [
        odir_dir / "ODIR-5K_Training_Dataset" / f"{patient_id}_{eye}.jpg",
        odir_dir / "ODIR-5K_Training_Dataset" / f"{patient_id}_{eye}.png",
        odir_dir / "training_images" / f"{patient_id}_{eye}.jpg",
        odir_dir / "images" / f"{patient_id}_{eye}.jpg",
        odir_dir / f"{patient_id}_{eye}.jpg",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# ── Image transforms ──────────────────────────────────────────────────────────

def get_transforms(split: str = "train") -> "T.Compose":
    """ImageNet-normalised transforms for ODIR fundus images."""
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if split == "train":
        return T.Compose([
            T.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            T.RandomCrop(IMG_SIZE),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            T.RandomRotation(15),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

if TORCH_AVAILABLE:
    class ODIRDataset(Dataset):
        """
        PyTorch Dataset for ODIR-5K.

        Each record is one eye image (left or right).
        Returns:
            image:    torch.Tensor [3, 224, 224]
            label:    torch.Tensor [8] float  multi-hot disease vector
            patient_id: int
            eye:      str "left" or "right"
        """

        def __init__(
            self,
            df: pd.DataFrame,
            odir_dir: Union[str, Path],
            split: str = "train",
            eyes: List[str] = ["left", "right"],
            return_both_eyes: bool = False,
        ):
            self.df              = df.reset_index(drop=True)
            self.odir_dir        = Path(odir_dir)
            self.split           = split
            self.eyes            = eyes
            self.return_both     = return_both_eyes
            self.transform       = get_transforms(split)
            self._build_index()

        def _build_index(self):
            """Build flat list of (patient_idx, eye) pairs."""
            self.index = []
            for i in range(len(self.df)):
                for eye in self.eyes:
                    img_path = self._get_img_path(i, eye)
                    if img_path is not None:
                        self.index.append((i, eye, img_path))

        def _get_img_path(self, df_idx: int, eye: str) -> Optional[Path]:
            row = self.df.iloc[df_idx]
            pid = row.get("ID", df_idx)

            # Try filename columns first
            col_map = {"left": "Left-Fundus", "right": "Right-Fundus"}
            fname_col = col_map.get(eye)
            if fname_col and fname_col in self.df.columns:
                fname = str(row[fname_col]).strip()
                for folder in ["ODIR-5K_Training_Dataset", "training_images", "images", ""]:
                    p = self.odir_dir / folder / fname if folder else self.odir_dir / fname
                    if p.exists():
                        return p

            return find_image_path(self.odir_dir, pid, eye)

        def __len__(self) -> int:
            return len(self.index)

        def __getitem__(self, idx: int) -> Dict:
            df_idx, eye, img_path = self.index[idx]
            row = self.df.iloc[df_idx]

            # Load image
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = self.transform(img)
            except Exception:
                img_tensor = torch.zeros(3, IMG_SIZE, IMG_SIZE)

            # Multi-hot label
            label = torch.tensor(
                [float(row.get(col, 0)) for col in LABEL_COLS],
                dtype=torch.float32,
            )

            return {
                "image":      img_tensor,
                "label":      label,
                "patient_id": int(row.get("ID", df_idx)),
                "eye":        eye,
                "age":        float(row.get("Patient Age", 0)),
                "sex":        str(row.get("Patient Sex", "")),
            }


    class ODIRDataModule:
        """
        Train/Val/Test DataLoaders for ODIR-5K.
        Splits patients 80/10/10 (stratified on most common label).
        """

        def __init__(
            self,
            odir_dir: Union[str, Path],
            batch_size: int = 32,
            num_workers: int = 0,
        ):
            self.odir_dir    = Path(odir_dir)
            self.batch_size  = batch_size
            self.num_workers = num_workers
            self._ready      = False

        def setup(self):
            df = load_odir_annotations(self.odir_dir)
            n  = len(df)
            idx = np.random.default_rng(42).permutation(n)
            t_end = int(n * 0.8)
            v_end = int(n * 0.9)

            self.train_dataset = ODIRDataset(df.iloc[idx[:t_end]],  self.odir_dir, "train")
            self.val_dataset   = ODIRDataset(df.iloc[idx[t_end:v_end]], self.odir_dir, "val")
            self.test_dataset  = ODIRDataset(df.iloc[idx[v_end:]], self.odir_dir, "test")
            self._ready = True

        def train_dataloader(self):
            return DataLoader(self.train_dataset, batch_size=self.batch_size,
                              shuffle=True,  num_workers=self.num_workers, pin_memory=True)

        def val_dataloader(self):
            return DataLoader(self.val_dataset,   batch_size=self.batch_size,
                              shuffle=False, num_workers=self.num_workers)

        def test_dataloader(self):
            return DataLoader(self.test_dataset,  batch_size=self.batch_size,
                              shuffle=False, num_workers=self.num_workers)


# ── Dataset Statistics ─────────────────────────────────────────────────────────

def compute_odir_stats(odir_dir: Union[str, Path]) -> Dict:
    """
    Compute ODIR-5K dataset statistics for the explorer page.
    Returns JSON-serializable dict.
    """
    try:
        df = load_odir_annotations(odir_dir)
    except FileNotFoundError as e:
        return {"available": False, "error": str(e)}

    n_patients = len(df)

    # Per-class counts
    class_counts = {}
    for col, name in zip(LABEL_COLS, CLASS_NAMES):
        if col in df.columns:
            class_counts[name] = int(df[col].sum())

    # Multi-label stats
    if all(c in df.columns for c in LABEL_COLS):
        label_mat   = df[LABEL_COLS].values
        n_multilabel = int((label_mat.sum(axis=1) > 1).sum())
        avg_labels   = float(label_mat.sum(axis=1).mean())
    else:
        n_multilabel = 0
        avg_labels   = 1.0

    # Age distribution
    age_hist = {}
    if "Patient Age" in df.columns:
        ages = df["Patient Age"].dropna()
        bins   = [0, 20, 30, 40, 50, 60, 70, 80, 120]
        labels = ["<20","20-29","30-39","40-49","50-59","60-69","70-79","80+"]
        buckets = pd.cut(ages, bins=bins, labels=labels, right=False)
        age_hist = {str(k): int(v) for k, v in buckets.value_counts().sort_index().items()}

    # Sex distribution
    sex_dist = {}
    if "Patient Sex" in df.columns:
        sex_dist = {str(k): int(v) for k, v in df["Patient Sex"].value_counts().items()}

    return {
        "available":      True,
        "n_patients":     n_patients,
        "n_images":       n_patients * 2,
        "n_classes":      N_CLASSES,
        "class_names":    CLASS_NAMES,
        "class_counts":   class_counts,
        "n_multilabel":   n_multilabel,
        "avg_labels_per_patient": round(avg_labels, 2),
        "age_distribution": age_hist,
        "sex_distribution": sex_dist,
        "label_columns":  LABEL_COLS,
        "task":           "multi-label classification",
        "reference":      "ICDM 2019 Smart Health Challenge — Peking University",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Folder-Based Eye Dataset  (class-subfolder layout — E:\eye\datasets\)
# ══════════════════════════════════════════════════════════════════════════════
#
#  Expected structure:
#     <eye_dir>/
#         normal/           image001.png  image002.png  ...
#         diabetes/         ...
#         glaucoma/         ...
#         cataract/         ...
#         ageDegeneration/  ...
#         hypertension/     ...
#         myopia/           ...
#         others/           ...
#
#  Each image is treated as single-label (the folder is its class).
# ─────────────────────────────────────────────────────────────────────────────

# Folder name (lowercase key) → (ODIR label code, class index 0-7)
_FOLDER_TO_ODIR = {
    "normal":          ("N", 0),
    "diabetes":        ("D", 1),
    "glaucoma":        ("G", 2),
    "cataract":        ("C", 3),
    "agedegeneration": ("A", 4),   # actual folder: ageDegeneration
    "amd":             ("A", 4),
    "hypertension":    ("H", 5),
    "myopia":          ("M", 6),
    "others":          ("O", 7),
    "other":           ("O", 7),
}

_FOLDER_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def scan_eye_folder_structure(eye_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Scan a class-subfolder eye dataset and build a flat DataFrame.

    Args:
        eye_dir: Root directory containing one subfolder per disease class.

    Returns:
        DataFrame with columns:
            image_path, folder_name, label_code, class_idx, class_name
    """
    eye_dir = Path(eye_dir)
    if not eye_dir.exists():
        raise FileNotFoundError(
            f"Eye dataset directory not found: {eye_dir}\n"
            f"Set ODIR_DIR in .env to the directory containing class subfolders."
        )

    records = []
    for folder in sorted(eye_dir.iterdir()):
        if not folder.is_dir():
            continue
        key = folder.name.lower()
        if key not in _FOLDER_TO_ODIR:
            continue  # skip unrecognised folders
        label_code, class_idx = _FOLDER_TO_ODIR[key]
        class_name = CLASS_NAMES[class_idx]

        for img_file in folder.iterdir():
            if img_file.suffix.lower() in _FOLDER_IMG_EXTS:
                records.append({
                    "image_path": str(img_file),
                    "folder_name": folder.name,
                    "label_code":  label_code,
                    "class_idx":   class_idx,
                    "class_name":  class_name,
                })

    if not records:
        raise FileNotFoundError(
            f"No images found in recognised class subfolders under {eye_dir}.\n"
            f"Expected subfolders: normal/, diabetes/, glaucoma/, cataract/, "
            f"ageDegeneration/, hypertension/, myopia/, others/"
        )

    return pd.DataFrame(records)


if TORCH_AVAILABLE:
    class EyeFolderDataset(Dataset):
        """
        Single-label PyTorch Dataset for class-subfolder eye images.

        Returns:
            image:      torch.Tensor [3, 224, 224]
            label:      torch.LongTensor  scalar (0-7)
            class_name: str
            image_path: str
        """

        def __init__(self, df: pd.DataFrame, split: str = "train"):
            self.df        = df.reset_index(drop=True)
            self.split     = split
            self.transform = get_transforms(split)

        def __len__(self) -> int:
            return len(self.df)

        def __getitem__(self, idx: int) -> Dict:
            row = self.df.iloc[idx]
            try:
                img = Image.open(row["image_path"]).convert("RGB")
                img_tensor = self.transform(img)
            except Exception:
                img_tensor = torch.zeros(3, IMG_SIZE, IMG_SIZE)

            return {
                "image":      img_tensor,
                "label":      torch.tensor(int(row["class_idx"]), dtype=torch.long),
                "class_name": row["class_name"],
                "image_path": row["image_path"],
            }


    class EyeFolderDataModule:
        """
        Train / Val / Test DataLoaders for the class-subfolder eye dataset.
        Stratified 80 / 10 / 10 split per class.
        """

        def __init__(
            self,
            eye_dir: Union[str, Path],
            batch_size: int = 32,
            num_workers: int = 0,
        ):
            self.eye_dir     = Path(eye_dir)
            self.batch_size  = batch_size
            self.num_workers = num_workers
            self._ready      = False

        def setup(self):
            df  = scan_eye_folder_structure(self.eye_dir)
            rng = np.random.default_rng(42)

            train_rows, val_rows, test_rows = [], [], []
            for cls_idx in sorted(df["class_idx"].unique()):
                cls_df = df[df["class_idx"] == cls_idx]
                perm   = rng.permutation(len(cls_df))
                t_end  = int(len(perm) * 0.8)
                v_end  = int(len(perm) * 0.9)
                train_rows.append(cls_df.iloc[perm[:t_end]])
                val_rows.append(cls_df.iloc[perm[t_end:v_end]])
                test_rows.append(cls_df.iloc[perm[v_end:]])

            self.train_dataset = EyeFolderDataset(pd.concat(train_rows), "train")
            self.val_dataset   = EyeFolderDataset(pd.concat(val_rows),   "val")
            self.test_dataset  = EyeFolderDataset(pd.concat(test_rows),  "test")
            self._full_df      = df
            self._ready        = True

        def train_dataloader(self):
            from torch.utils.data import DataLoader as _DL
            return _DL(self.train_dataset, batch_size=self.batch_size,
                       shuffle=True,  num_workers=self.num_workers, pin_memory=True)

        def val_dataloader(self):
            from torch.utils.data import DataLoader as _DL
            return _DL(self.val_dataset, batch_size=self.batch_size,
                       shuffle=False, num_workers=self.num_workers)

        def test_dataloader(self):
            from torch.utils.data import DataLoader as _DL
            return _DL(self.test_dataset, batch_size=self.batch_size,
                       shuffle=False, num_workers=self.num_workers)

        def summary(self) -> Dict:
            if not self._ready:
                raise RuntimeError("Call setup() first.")
            return {
                "train": len(self.train_dataset),
                "val":   len(self.val_dataset),
                "test":  len(self.test_dataset),
                "total": len(self._full_df),
                "n_classes": N_CLASSES,
            }


def compute_eye_folder_stats(eye_dir: Union[str, Path]) -> Dict:
    """
    Compute class-subfolder eye dataset statistics for the explorer page.
    Returns JSON-serializable dict.
    """
    try:
        df = scan_eye_folder_structure(eye_dir)
    except FileNotFoundError as e:
        return {"available": False, "error": str(e)}

    n_images = len(df)
    class_counts = {
        CLASS_NAMES[i]: int((df["class_idx"] == i).sum())
        for i in range(N_CLASSES)
    }

    return {
        "available":    True,
        "n_images":     n_images,
        "n_classes":    N_CLASSES,
        "class_names":  CLASS_NAMES,
        "class_counts": class_counts,
        "label_format": "single-label (folder-based)",
        "task":         "multi-class classification (8 classes)",
        "note":         "Each image belongs to exactly one class (its parent folder).",
    }
