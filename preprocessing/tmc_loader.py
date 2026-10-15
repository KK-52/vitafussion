"""
VitaFusion AI — TMC 21-Category Tongue Dataset Loader
======================================================
Loads the 21-category Tongue dataset (TMC).

Expected folder structure:
    <tmc_dir>/
        Normal_ThinWhite/
            img001.jpg
            img002.jpg
            ...
        Pale_ThinWhite/
            ...
        Red_Yellow/
            ...
        (21 class folders total)

If your dataset uses different folder names, set a custom mapping
via TMC_FOLDER_REMAP in your .env or pass folder_remap= to TMCDataset.

Reference:
    21-category tongue image classification dataset (TCM diagnostics).
    Body colour (5) × Coating (5) → 21 clinically significant combinations.
"""

import os
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

from config import TMC_21_CLASSES, TMC_N_CLASSES, TMC_TO_VITAFUSION_GROUP

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE    = 224
N_CLASSES   = TMC_N_CLASSES   # 21
CLASS_NAMES = TMC_21_CLASSES

# Maps the 21 fine-grained classes to the 6 VitaFusion display groups
VITAFUSION_6_GROUPS = ["Normal", "Pale", "Red/Dark", "Purple/Blue", "Yellow Coat", "Cracked/Dry"]

# Body-colour prefix → VitaFusion 6-group (used for simulated predictions)
BODY_TO_GROUP = {
    "Normal":  "Normal",
    "Pale":    "Pale",
    "Red":     "Red/Dark",
    "DarkRed": "Red/Dark",
    "Purple":  "Purple/Blue",
}
COATING_TO_GROUP = {
    "Yellow":  "Yellow Coat",
    "Black":   "Yellow Coat",   # dark coating similar clinical meaning
    "Peeled":  "Cracked/Dry",
}


def parse_class_name(cls: str) -> Tuple[str, str]:
    """Split 'Body_Coating' into (body, coating). e.g. 'Red_Yellow' → ('Red','Yellow')."""
    parts = cls.split("_", 1)
    return (parts[0], parts[1]) if len(parts) == 2 else (parts[0], "ThinWhite")


def class_to_vitafusion_group(cls: str) -> str:
    """Map a TMC 21-class name to one of 6 VitaFusion display groups."""
    if cls in TMC_TO_VITAFUSION_GROUP:
        return TMC_TO_VITAFUSION_GROUP[cls]
    body, coating = parse_class_name(cls)
    if coating in COATING_TO_GROUP:
        return COATING_TO_GROUP[coating]
    return BODY_TO_GROUP.get(body, "Normal")


# ── Metadata builder ──────────────────────────────────────────────────────────

def scan_tmc_directory(
    tmc_dir: Union[str, Path],
    folder_remap: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Scan TMC directory and build a flat DataFrame of (image_path, class_idx, class_name).

    Args:
        tmc_dir:      Root directory containing class subfolders
        folder_remap: Optional dict mapping actual folder names → TMC_21_CLASSES names
                      e.g. {"normal_thin": "Normal_ThinWhite"}

    Returns: DataFrame with columns [image_path, class_name, class_idx, body, coating, group]
    """
    tmc_dir = Path(tmc_dir)
    if not tmc_dir.exists():
        raise FileNotFoundError(
            f"TMC dataset directory not found: {tmc_dir}\n"
            f"Expected 21 class subfolders inside.\n"
            f"Set TMC_DIR in .env to the correct path."
        )

    records = []
    found_classes = set()

    for folder in sorted(tmc_dir.iterdir()):
        if not folder.is_dir():
            continue
        folder_name = folder.name

        # Map folder name to canonical class name
        if folder_remap and folder_name in folder_remap:
            class_name = folder_remap[folder_name]
        elif folder_name in TMC_21_CLASSES:
            class_name = folder_name
        else:
            # Fuzzy match: try case-insensitive
            match = next(
                (c for c in TMC_21_CLASSES if c.lower() == folder_name.lower()), None
            )
            class_name = match if match else folder_name  # keep as-is, will be idx=-1

        class_idx  = TMC_21_CLASSES.index(class_name) if class_name in TMC_21_CLASSES else -1
        body, coat = parse_class_name(class_name)
        group      = class_to_vitafusion_group(class_name)
        found_classes.add(class_name)

        for img_file in folder.iterdir():
            if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                records.append({
                    "image_path": str(img_file),
                    "class_name": class_name,
                    "class_idx":  class_idx,
                    "body":       body,
                    "coating":    coat,
                    "group":      group,
                })

    if not records:
        raise FileNotFoundError(
            f"No images found in {tmc_dir}.\n"
            f"Expected subfolders named like: Normal_ThinWhite/, Pale_Yellow/, etc."
        )

    df = pd.DataFrame(records)
    return df


# ── Image transforms ──────────────────────────────────────────────────────────

def get_transforms(split: str = "train") -> "T.Compose":
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if split == "train":
        return T.Compose([
            T.Resize((IMG_SIZE + 24, IMG_SIZE + 24)),
            T.RandomCrop(IMG_SIZE),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.3, hue=0.05),
            T.RandomRotation(10),
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
    class TMCDataset(Dataset):
        """
        PyTorch Dataset for the 21-category TMC tongue dataset.

        Returns:
            image:      torch.Tensor [3, 224, 224]
            label:      torch.LongTensor scalar (0-20)
            class_name: str
            group:      str (VitaFusion 6-class display group)
        """

        def __init__(
            self,
            df: pd.DataFrame,
            split: str = "train",
            folder_remap: Optional[Dict[str, str]] = None,
        ):
            # Filter out unknown classes (class_idx == -1)
            self.df        = df[df["class_idx"] >= 0].reset_index(drop=True)
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
                "group":      row["group"],
                "body":       row["body"],
                "coating":    row["coating"],
            }


    class TMCDataModule:
        """
        Train/Val/Test DataLoaders for the TMC 21-class tongue dataset.
        Performs stratified 80/10/10 split by class.
        """

        def __init__(
            self,
            tmc_dir: Union[str, Path],
            batch_size: int = 32,
            num_workers: int = 0,
            folder_remap: Optional[Dict[str, str]] = None,
        ):
            self.tmc_dir     = Path(tmc_dir)
            self.batch_size  = batch_size
            self.num_workers = num_workers
            self.folder_remap= folder_remap
            self._ready      = False

        def setup(self):
            df = scan_tmc_directory(self.tmc_dir, self.folder_remap)

            # Stratified split per class
            train_idx, val_idx, test_idx = [], [], []
            rng = np.random.default_rng(42)
            for cls_idx in df["class_idx"].unique():
                cls_df = df[df["class_idx"] == cls_idx]
                idx    = rng.permutation(len(cls_df))
                t_end  = int(len(idx) * 0.8)
                v_end  = int(len(idx) * 0.9)
                train_idx.extend(cls_df.index[idx[:t_end]].tolist())
                val_idx.extend(cls_df.index[idx[t_end:v_end]].tolist())
                test_idx.extend(cls_df.index[idx[v_end:]].tolist())

            self.train_dataset = TMCDataset(df.loc[train_idx], "train")
            self.val_dataset   = TMCDataset(df.loc[val_idx],   "val")
            self.test_dataset  = TMCDataset(df.loc[test_idx],  "test")
            self._full_df      = df
            self._ready        = True

        def train_dataloader(self):
            return DataLoader(self.train_dataset, batch_size=self.batch_size,
                              shuffle=True,  num_workers=self.num_workers, pin_memory=True, drop_last=True)

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
                "total": len(self._full_df),
                "n_classes": N_CLASSES,
            }


# ── Dataset Statistics ─────────────────────────────────────────────────────────

def compute_tmc_stats(tmc_dir: Union[str, Path]) -> Dict:
    """
    Compute TMC dataset statistics for the explorer page.
    Returns JSON-serializable dict.
    """
    try:
        df = scan_tmc_directory(tmc_dir)
    except FileNotFoundError as e:
        return {"available": False, "error": str(e)}

    n_images = len(df)
    class_counts = df["class_name"].value_counts().to_dict()
    group_counts  = df["group"].value_counts().to_dict()
    body_counts   = df["body"].value_counts().to_dict()
    coating_counts= df["coating"].value_counts().to_dict()

    found_classes = sorted(df["class_name"].unique().tolist())
    missing_classes = [c for c in TMC_21_CLASSES if c not in found_classes]

    return {
        "available":       True,
        "n_images":        n_images,
        "n_classes":       N_CLASSES,
        "class_names":     TMC_21_CLASSES,
        "found_classes":   found_classes,
        "missing_classes": missing_classes,
        "class_counts":    {str(k): int(v) for k, v in class_counts.items()},
        "group_counts":    {str(k): int(v) for k, v in group_counts.items()},
        "body_counts":     {str(k): int(v) for k, v in body_counts.items()},
        "coating_counts":  {str(k): int(v) for k, v in coating_counts.items()},
        "vitafusion_groups": VITAFUSION_6_GROUPS,
        "task":            "21-class single-label classification",
        "reference":       "21-category TCM Tongue Dataset (TMC)",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Shenzhen Tongue Dataset  (YOLO-format, 20 classes)
# ══════════════════════════════════════════════════════════════════════════════
#
#  Expected structure:
#     <shenzhen_dir>/
#         train/
#             images/   A (1000).jpg  A (1001).jpg  ...
#             labels/   A (1000).txt  A (1001).txt  ...
#         val/
#             images/   ...
#             labels/   ...
#         test/
#             images/   ...
#             labels/   ...
#         shezhenv3-txt.yaml   (optional, for reference)
#
#  Each .txt label file contains one or more YOLO rows:
#     <class_idx> <cx> <cy> <w> <h>
#  The dominant (most-frequent) class per image is used for classification.
# ─────────────────────────────────────────────────────────────────────────────

from config import SHENZHEN_20_CLASSES, SHENZHEN_TO_VITAFUSION, SHENZHEN_N_CLASSES

_SHENZHEN_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _dominant_class(label_path: Path, n_classes: int) -> int:
    """
    Parse a YOLO label file and return the most-frequent class index.
    Returns 0 (jiankangshe) if the file is missing or empty.
    """
    if not label_path.exists():
        return 0
    try:
        counts = [0] * n_classes
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cls = int(line.split()[0])
                if 0 <= cls < n_classes:
                    counts[cls] += 1
        return int(np.argmax(counts)) if any(c > 0 for c in counts) else 0
    except Exception:
        return 0


def scan_shenzhen_yolo(
    shenzhen_dir: Union[str, Path],
    splits: List[str] = ["train", "val", "test"],
) -> pd.DataFrame:
    """
    Scan the Shenzhen YOLO-format tongue dataset and build a flat DataFrame.

    Args:
        shenzhen_dir: Root directory containing train/, val/, test/ split folders.
        splits:       Which splits to include.

    Returns:
        DataFrame with columns:
            image_path, label_path, class_idx, class_name, group, split
    """
    shenzhen_dir = Path(shenzhen_dir)
    if not shenzhen_dir.exists():
        raise FileNotFoundError(
            f"Shenzhen tongue dataset not found: {shenzhen_dir}\n"
            f"Set TMC_DIR in .env to the directory containing train/val/test split folders."
        )

    records = []
    for split in splits:
        img_dir   = shenzhen_dir / split / "images"
        label_dir = shenzhen_dir / split / "labels"
        if not img_dir.exists():
            continue  # split not present — skip gracefully

        for img_file in sorted(img_dir.iterdir()):
            if img_file.suffix.lower() not in _SHENZHEN_IMG_EXTS:
                continue

            label_file = label_dir / (img_file.stem + ".txt")
            cls_idx    = _dominant_class(label_file, SHENZHEN_N_CLASSES)
            cls_name   = SHENZHEN_20_CLASSES[cls_idx]
            group      = SHENZHEN_TO_VITAFUSION.get(cls_name, "Normal")

            records.append({
                "image_path": str(img_file),
                "label_path": str(label_file),
                "class_idx":  cls_idx,
                "class_name": cls_name,
                "group":      group,
                "split":      split,
            })

    if not records:
        raise FileNotFoundError(
            f"No images found in {shenzhen_dir}/{{train,val,test}}/images/.\n"
            f"Check that the YOLO split directories exist and contain .jpg/.png files."
        )

    return pd.DataFrame(records)


if TORCH_AVAILABLE:
    class ShenzhenTongueDataset(Dataset):
        """
        PyTorch Dataset for the Shenzhen YOLO-format tongue dataset.

        Returns:
            image:      torch.Tensor [3, 224, 224]
            label:      torch.LongTensor  scalar (0-19)
            class_name: str
            group:      str (VitaFusion 6-class display group)
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
                "group":      row["group"],
                "image_path": row["image_path"],
            }


    class ShenzhenDataModule:
        """
        DataLoaders for the Shenzhen YOLO tongue dataset.
        Uses the pre-existing train / val / test directory splits.
        """

        def __init__(
            self,
            shenzhen_dir: Union[str, Path],
            batch_size: int = 32,
            num_workers: int = 0,
        ):
            self.shenzhen_dir = Path(shenzhen_dir)
            self.batch_size   = batch_size
            self.num_workers  = num_workers
            self._ready       = False

        def setup(self):
            all_df  = scan_shenzhen_yolo(self.shenzhen_dir)

            train_df = all_df[all_df["split"] == "train"].reset_index(drop=True)
            val_df   = all_df[all_df["split"] == "val"].reset_index(drop=True)
            test_df  = all_df[all_df["split"] == "test"].reset_index(drop=True)

            # If no explicit val/test splits exist, fall back to 80/10/10 random split
            if len(val_df) == 0 and len(test_df) == 0:
                rng    = np.random.default_rng(42)
                perm   = rng.permutation(len(all_df))
                t_end  = int(len(perm) * 0.8)
                v_end  = int(len(perm) * 0.9)
                train_df = all_df.iloc[perm[:t_end]]
                val_df   = all_df.iloc[perm[t_end:v_end]]
                test_df  = all_df.iloc[perm[v_end:]]

            self.train_dataset = ShenzhenTongueDataset(train_df, "train")
            self.val_dataset   = ShenzhenTongueDataset(val_df,   "val")
            self.test_dataset  = ShenzhenTongueDataset(test_df,  "test")
            self._full_df      = all_df
            self._ready        = True

        def train_dataloader(self):
            from torch.utils.data import DataLoader as _DL
            return _DL(self.train_dataset, batch_size=self.batch_size,
                       shuffle=True,  num_workers=self.num_workers,
                       pin_memory=True)

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
                "train":     len(self.train_dataset),
                "val":       len(self.val_dataset),
                "test":      len(self.test_dataset),
                "total":     len(self._full_df),
                "n_classes": SHENZHEN_N_CLASSES,
            }


def compute_shenzhen_stats(shenzhen_dir: Union[str, Path]) -> Dict:
    """
    Compute Shenzhen tongue dataset statistics for the explorer page.
    Returns JSON-serializable dict.
    """
    try:
        df = scan_shenzhen_yolo(shenzhen_dir)
    except FileNotFoundError as e:
        return {"available": False, "error": str(e)}

    n_images     = len(df)
    class_counts = {SHENZHEN_20_CLASSES[i]: int((df["class_idx"] == i).sum())
                    for i in range(SHENZHEN_N_CLASSES)}
    group_counts = df["group"].value_counts().to_dict()
    split_counts = df["split"].value_counts().to_dict()

    return {
        "available":      True,
        "n_images":       n_images,
        "n_classes":      SHENZHEN_N_CLASSES,
        "class_names":    SHENZHEN_20_CLASSES,
        "class_counts":   {str(k): int(v) for k, v in class_counts.items()},
        "group_counts":   {str(k): int(v) for k, v in group_counts.items()},
        "split_counts":   {str(k): int(v) for k, v in split_counts.items()},
        "vitafusion_groups": VITAFUSION_6_GROUPS,
        "task":           "20-class single-label classification (Shenzhen TCM)",
        "label_format":   "YOLO txt (dominant class per image)",
        "reference":      "Shenzhen TCM Tongue Dataset v3 (shezhenv3-txt)",
    }
