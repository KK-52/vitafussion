"""
VitaFusion — Merge ECG Datasets
=================================
Merges E:\\dataset1\\ecg_data (4-class clinical) into
       E:\\ecgdataset2\\ECG_Image_data (6-class MIT-BIH)

Mapping (Dataset1 folder → Dataset2 class folder):
  normal_ecg_images              → train/N  (Normal)
  myocardial_infarction...       → train/M  (Myocardial Infarction)
  abnormal_heartbeat_ecg_images  → train/S  (SVT / Arrhythmia, closest match)
  post_mi_history_ecg_images     → train/Q  (Unknown / History MI)

Files are COPIED (not moved) with a prefix to avoid name collisions.
Run this once before training.

Usage:
  python merge_ecg_datasets.py
"""

import shutil
import sys
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
DATASET1    = Path(r"E:\dataset1\ecg_data")
DATASET2    = Path(r"E:\ecgdataset2\ECG_Image_data")

# ── Mapping: Dataset1 subfolder → Dataset2 class letter ───────────────────────
# Dataset2 train folders: N, M, S, V, F, Q
MERGE_MAP = {
    "normal_ecg_images":                "N",   # Normal  → N
    "myocardial_infarction_ecg_images": "M",   # MI      → M
    "abnormal_heartbeat_ecg_images":    "S",   # Abnormal→ S (Supraventricular / arrhythmia)
    "post_mi_history_ecg_images":       "Q",   # Post MI → Q (closest: unknown/other)
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def merge(dry_run: bool = False):
    total_copied = 0
    total_skipped = 0

    print(f"\n{'='*60}")
    print(f"  VitaFusion — ECG Dataset Merger")
    print(f"  Source : {DATASET1}")
    print(f"  Target : {DATASET2}")
    print(f"  Dry run: {dry_run}")
    print(f"{'='*60}\n")

    for src_folder, dst_class in MERGE_MAP.items():
        src_dir = DATASET1 / src_folder
        if not src_dir.exists():
            print(f"  [SKIP] Source not found: {src_dir}")
            continue

        images = [f for f in src_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS]
        print(f"  [{src_folder}] -> [{dst_class}]  ({len(images)} images)")

        # Copy into BOTH train and test splits
        # Dataset1 has no train/test split → put 80% in train, 20% in test
        split_idx  = int(len(images) * 0.8)
        train_imgs = images[:split_idx]
        test_imgs  = images[split_idx:]

        for split, split_images in [("train", train_imgs), ("test", test_imgs)]:
            dst_dir = DATASET2 / split / dst_class
            if not dry_run:
                dst_dir.mkdir(parents=True, exist_ok=True)

            for img_path in split_images:
                # Prefix with "ds1_" to avoid name collision with existing files
                dst_file = dst_dir / f"ds1_{img_path.name}"
                if dst_file.exists():
                    total_skipped += 1
                    continue
                if not dry_run:
                    shutil.copy2(img_path, dst_file)
                total_copied += 1

        print(f"    Train: {len(train_imgs)} | Test: {len(test_imgs)}")

    print(f"\n  Done!")
    print(f"  Copied  : {total_copied} images")
    print(f"  Skipped : {total_skipped} (already existed)")
    if dry_run:
        print(f"\n  [DRY RUN] No files were actually copied.")
        print(f"  Run without --dry to apply.")
    print()

    # Print updated class counts
    print("  Updated class distribution (train):")
    train_dir = DATASET2 / "train"
    for cls in sorted(["N", "M", "S", "V", "F", "Q"]):
        cls_dir = train_dir / cls
        if cls_dir.exists():
            count = len([f for f in cls_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS])
            bar   = "|" * min(int(count / 2000), 40)
            print(f"    [{cls}]  {count:6d}  {bar}")
    print()


if __name__ == "__main__":
    dry_run = "--dry" in sys.argv
    merge(dry_run=dry_run)
