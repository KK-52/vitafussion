"""
VitaFusion AI — Retinal Encoder Training Script (ODIR-5K)
==========================================================
Trains the RetinalEncoder (ResNet-50) on the ODIR-5K ocular disease dataset.

Task:    Multi-label classification (8 disease classes)
Loss:    Binary Cross-Entropy with Logits (one per disease)
Metrics: Per-class AUROC, Macro-AUROC, Hamming accuracy

Usage:
    # Quick dry-run (validates pipeline):
    python train_retinal.py --quick

    # Full training:
    python train_retinal.py --odir_dir data/odir --epochs 30 --batch_size 32

    # Fine-tune from pretrained checkpoint:
    python train_retinal.py --odir_dir data/odir --resume models/pretrained/retinal_encoder_odir_best.pt
"""

import argparse
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from sklearn.metrics import roc_auc_score
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from models.retinal_encoder import RetinalEncoder, ODIR_CLASS_NAMES, N_ODIR_CLASSES
from preprocessing.odir_loader import (
    ODIRDataModule, compute_odir_stats,
    EyeFolderDataModule, compute_eye_folder_stats,
    scan_eye_folder_structure, _FOLDER_TO_ODIR,
)
import config


# ── Argument parser ────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train RetinalEncoder on ODIR-5K or folder-based eye dataset")
    p.add_argument("--odir_dir",    type=str, default=str(config.ODIR_DIR))
    p.add_argument("--dataset_format", type=str, default="auto",
                   choices=["auto", "odir", "folder"],
                   help="'folder' = class-subfolder layout (E:/eye/datasets); "
                        "'odir'   = ODIR-5K with annotation Excel; "
                        "'auto'   = detect by checking for class subfolders (default)")
    p.add_argument("--epochs",      type=int, default=30)
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--freeze_backbone", action="store_true",
                   help="Freeze ResNet-50 backbone; train heads only")
    p.add_argument("--resume",      type=str, default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--save_dir",    type=str, default="models/pretrained")
    p.add_argument("--quick",       action="store_true",
                   help="Dry-run: 2 epochs, 20 samples — verifies pipeline")
    return p.parse_args()


def _detect_format(odir_dir: Path) -> str:
    """Auto-detect dataset format: return 'folder' if class subfolders exist, else 'odir'."""
    for subfolder in _FOLDER_TO_ODIR.keys():
        for child in odir_dir.iterdir() if odir_dir.exists() else []:
            if child.is_dir() and child.name.lower() == subfolder:
                return "folder"
    return "odir"


# ── Helpers ────────────────────────────────────────────────────────────────────
def compute_pos_weights(dm: ODIRDataModule, device: torch.device) -> torch.Tensor:
    """Compute per-class positive weights for BCEWithLogitsLoss."""
    from preprocessing.odir_loader import LABEL_COLS
    df = dm.train_dataset.df
    n  = len(df)
    weights = []
    for col in LABEL_COLS:
        if col in df.columns:
            pos = df[col].sum()
            neg = n - pos
            weights.append(neg / max(pos, 1))
        else:
            weights.append(1.0)
    return torch.tensor(weights, dtype=torch.float32).to(device)


def macro_auroc(probs_all: np.ndarray, labels_all: np.ndarray) -> float:
    """Compute macro-averaged AUROC across all 8 classes."""
    if not SKLEARN_OK:
        return float("nan")
    aucs = []
    for i in range(probs_all.shape[1]):
        if labels_all[:, i].sum() > 0:
            try:
                aucs.append(roc_auc_score(labels_all[:, i], probs_all[:, i]))
            except Exception:
                pass
    return float(np.mean(aucs)) if aucs else float("nan")


# ── Training loop ──────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, quick=False, use_folder=False):
    model.train()
    total_loss = 0.0
    correct, total_samples = 0, 0
    for step, batch in enumerate(loader):
        if quick and step >= 5:
            break
        imgs   = batch["image"].to(device)
        labels = batch["label"].to(device)   # [B] long (folder) or [B,8] float (odir)

        out  = model(imgs)
        if use_folder:
            # Single-label: use odir_logits [B, 8] with CrossEntropyLoss + long labels
            loss   = criterion(out["odir_logits"], labels)
            preds  = out["odir_logits"].argmax(dim=1)
            correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
        else:
            loss = criterion(out["odir_logits"], labels.float())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        if step % 20 == 0:
            if use_folder:
                acc = correct / max(total_samples, 1) * 100
                print(f"  step {step:4d} | loss {loss.item():.4f} | acc {acc:.1f}%")
            else:
                print(f"  step {step:4d} | loss {loss.item():.4f}")

    return total_loss / max(step + 1, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, quick=False, use_folder=False):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []
    correct, total_samples = 0, 0

    for step, batch in enumerate(loader):
        if quick and step >= 5:
            break
        imgs   = batch["image"].to(device)
        labels = batch["label"].to(device)

        out    = model(imgs)
        if use_folder:
            loss = criterion(out["odir_logits"], labels)
            preds = out["odir_logits"].argmax(dim=1)
            correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
        else:
            loss = criterion(out["odir_logits"], labels.float())
        total_loss += loss.item()

        all_probs.append(out["odir_probs"].cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    if use_folder:
        acc = correct / max(total_samples, 1)
        return total_loss / max(step + 1, 1), acc

    probs_arr  = np.concatenate(all_probs,  axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)
    auroc = macro_auroc(probs_arr, labels_arr)
    return total_loss / max(step + 1, 1), auroc


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Determine dataset format ───────────────────────────────────────────────
    odir_path = Path(args.odir_dir)
    fmt = args.dataset_format
    if fmt == "auto":
        fmt = _detect_format(odir_path)

    print(f"\n{'='*60}")
    print(f"  VitaFusion — Retinal Encoder Training")
    print(f"{'='*60}")
    print(f"  Device        : {device}")
    print(f"  Dataset dir   : {args.odir_dir}")
    print(f"  Dataset format: {fmt}")
    print(f"  Epochs        : {args.epochs}")
    print(f"  Batch size    : {args.batch_size}")
    print(f"  Quick mode    : {args.quick}")
    print(f"{'='*60}\n")

    # ── Dataset ────────────────────────────────────────────────────────────────
    use_folder = (fmt == "folder")

    if use_folder:
        print("[1/4] Loading folder-based eye dataset ...")
        stats = compute_eye_folder_stats(args.odir_dir)
        if not stats.get("available"):
            print(f"✗ Dataset not found: {stats.get('error')}")
            sys.exit(1)
        print(f"  Images   : {stats['n_images']:,}")
        for cls, cnt in stats.get("class_counts", {}).items():
            print(f"  {cls:<20}: {cnt:,}")
        dm = EyeFolderDataModule(args.odir_dir, batch_size=args.batch_size,
                                 num_workers=args.num_workers)
    else:
        print("[1/4] Loading ODIR-5K ...")
        stats = compute_odir_stats(args.odir_dir)
        if not stats.get("available"):
            print(f"✗ Dataset not found: {stats.get('error')}")
            print("  Download from: https://odir5k.grand-challenge.org/")
            sys.exit(1)
        print(f"  Patients : {stats['n_patients']:,}")
        print(f"  Images   : {stats['n_images']:,}")
        for cls, cnt in stats.get("class_counts", {}).items():
            print(f"  {cls:<15}: {cnt:,}")
        dm = ODIRDataModule(args.odir_dir, batch_size=args.batch_size,
                            num_workers=args.num_workers)

    dm.setup()

    if args.quick:
        print("\n  [QUICK MODE] Limiting to 20 samples per split.\n")
        from torch.utils.data import Subset
        dm.train_dataset = Subset(dm.train_dataset, range(min(20, len(dm.train_dataset))))
        dm.val_dataset   = Subset(dm.val_dataset,   range(min(20, len(dm.val_dataset))))

    # ── Model ──────────────────────────────────────────────────────────────────
    print("[2/4] Building RetinalEncoder ...")
    model = RetinalEncoder(
        embedding_dim    = 512,
        num_odir_classes = N_ODIR_CLASSES,
        pretrained       = True,
        freeze_backbone  = args.freeze_backbone,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params:,}")

    if args.resume and Path(args.resume).exists():
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"], strict=False)
        print(f"  Resumed from: {args.resume}")

    # ── Loss & optimiser ───────────────────────────────────────────────────────
    print("[3/4] Setting up training ...")
    if use_folder:
        # Single-label: CrossEntropyLoss
        criterion = nn.CrossEntropyLoss()
    else:
        # Multi-label: BCEWithLogitsLoss with class-balance weights
        pos_weights = compute_pos_weights(dm, device)
        criterion   = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer   = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    epochs      = 2 if args.quick else args.epochs
    scheduler   = CosineAnnealingLR(optimizer, T_max=epochs)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Training ───────────────────────────────────────────────────────────────
    print(f"[4/4] Training for {epochs} epoch(s) ...\n")
    best_metric = 0.0
    history     = []
    metric_name = "accuracy" if use_folder else "macro_auroc"
    ckpt_stem   = "retinal_encoder_folder" if use_folder else "retinal_encoder_odir"

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        train_loss = train_one_epoch(model, dm.train_dataloader(), optimizer,
                                     criterion, device, args.quick,
                                     use_folder=use_folder)
        val_loss, val_metric = evaluate(model, dm.val_dataloader(), criterion,
                                        device, args.quick,
                                        use_folder=use_folder)
        scheduler.step()

        row = dict(epoch=epoch, train_loss=round(train_loss, 4),
                   val_loss=round(val_loss, 4))
        row[f"val_{metric_name}"] = round(float(val_metric), 4)
        history.append(row)
        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_{metric_name}={val_metric:.4f}\n")

        # Save latest
        ckpt = {"epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "history": history,
                "dataset_format": fmt}
        torch.save(ckpt, save_dir / f"{ckpt_stem}_latest.pt")

        # Save best
        if val_metric > best_metric:
            best_metric = val_metric
            torch.save(ckpt, save_dir / f"{ckpt_stem}_best.pt")
            print(f"  ✓ New best {metric_name}: {best_metric:.4f}\n")

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best {metric_name:<12}: {best_metric:.4f}")
    print(f"  Checkpoint saved : {save_dir}/{ckpt_stem}_best.pt")
    print(f"{'='*60}\n")



if __name__ == "__main__":
    main()
