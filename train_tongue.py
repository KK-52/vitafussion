"""
VitaFusion AI — Tongue Encoder Training Script (TMC 21-class)
==============================================================
Trains the TongueEncoder (EfficientNet-B0) on the 21-category TMC tongue dataset.

Task:    Single-label 21-class classification
Loss:    CrossEntropyLoss (class-weighted)
Metrics: Top-1 accuracy, Macro-F1

Usage:
    # Quick dry-run:
    python train_tongue.py --quick

    # Full training:
    python train_tongue.py --tmc_dir data/tmc --epochs 40 --batch_size 32

    # With custom folder mapping (if your folders have different names):
    python train_tongue.py --tmc_dir data/tmc --epochs 40
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from sklearn.metrics import f1_score
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from models.tongue_encoder import TongueEncoder, N_TMC_CLASSES, TONGUE_CLASS_NAMES
from preprocessing.tmc_loader import (
    TMCDataModule, compute_tmc_stats,
    ShenzhenDataModule, compute_shenzhen_stats,
)
from config import SHENZHEN_N_CLASSES
import config


def parse_args():
    p = argparse.ArgumentParser(description="Train TongueEncoder on TMC 21-class or Shenzhen 20-class dataset")
    p.add_argument("--tmc_dir",     type=str, default=str(config.TMC_DIR))
    p.add_argument("--dataset_format", type=str, default="shenzhen",
                   choices=["tmc", "shenzhen"],
                   help="'shenzhen' = YOLO-format Shenzhen 20-class (default); "
                        "'tmc'      = class-subfolder TMC 21-class")
    p.add_argument("--epochs",      type=int, default=40)
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--resume",      type=str, default=None)
    p.add_argument("--save_dir",    type=str, default="models/pretrained")
    p.add_argument("--quick",       action="store_true",
                   help="Dry-run: 2 epochs, 25 samples")
    return p.parse_args()


def compute_class_weights(dm, n_classes: int,
                          device: torch.device) -> torch.Tensor:
    """Inverse-frequency class weights for CrossEntropyLoss."""
    ds = dm.train_dataset
    # Handle Subset wrapping (created by --quick mode)
    if hasattr(ds, "dataset"):
        df = ds.dataset.df
        labels = df["class_idx"].iloc[list(ds.indices)].values
    else:
        labels = ds.df["class_idx"].values
    counts  = np.bincount(labels, minlength=n_classes).astype(float)
    counts  = np.maximum(counts, 1)
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32).to(device)


def train_one_epoch(model, loader, optimizer, criterion, device, quick=False):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    step = -1  # guard: handles empty DataLoader (e.g. quick subset < batch_size)
    for step, batch in enumerate(loader):
        if quick and step >= 5:
            break
        imgs   = batch["image"].to(device)
        labels = batch["label"].to(device)

        out    = model(imgs)
        loss   = criterion(out["tongue_logits"], labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        preds   = out["tongue_logits"].argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        total_loss += loss.item()

        if step % 20 == 0:
            acc = correct / max(total, 1) * 100
            print(f"  step {step:4d} | loss {loss.item():.4f} | acc {acc:.1f}%")

    return total_loss / max(step + 1, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, quick=False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    step = -1  # guard: handles empty DataLoader

    for step, batch in enumerate(loader):
        if quick and step >= 5:
            break
        imgs   = batch["image"].to(device)
        labels = batch["label"].to(device)

        out   = model(imgs)
        loss  = criterion(out["tongue_logits"], labels)
        preds = out["tongue_logits"].argmax(dim=1)

        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        total_loss += loss.item()

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = correct / max(total, 1)
    macro_f1 = float("nan")
    if SKLEARN_OK and all_labels:
        try:
            macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        except Exception:
            pass
    return total_loss / max(step + 1, 1), acc, macro_f1


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fmt    = args.dataset_format

    print(f"\n{'='*60}")
    print(f"  VitaFusion — Tongue Encoder Training")
    print(f"{'='*60}")
    print(f"  Device         : {device}")
    print(f"  Dataset dir    : {args.tmc_dir}")
    print(f"  Dataset format : {fmt}")
    print(f"  Epochs         : {args.epochs}")
    print(f"  Quick mode     : {args.quick}")
    print(f"{'='*60}\n")

    # Dataset
    use_shenzhen = (fmt == "shenzhen")
    print("[1/4] Loading dataset ...")
    if use_shenzhen:
        stats = compute_shenzhen_stats(args.tmc_dir)
        if not stats.get("available"):
            print(f"✗ Dataset not found: {stats.get('error')}")
            sys.exit(1)
        n_classes = SHENZHEN_N_CLASSES  # 20
        print(f"  Format : Shenzhen YOLO (20 classes)")
        print(f"  Images : {stats['n_images']:,}")
        for split, cnt in stats.get("split_counts", {}).items():
            print(f"  {split:<8}: {cnt:,}")
        dm = ShenzhenDataModule(args.tmc_dir, batch_size=args.batch_size,
                                num_workers=args.num_workers)
    else:
        stats = compute_tmc_stats(args.tmc_dir)
        if not stats.get("available"):
            print(f"✗ Dataset not found: {stats.get('error')}")
            print("  Expected: 21 class subfolders inside TMC_DIR")
            sys.exit(1)
        n_classes = N_TMC_CLASSES  # 21
        print(f"  Images found : {stats['n_images']:,}")
        print(f"  Classes found: {len(stats['found_classes'])}/21")
        if stats["missing_classes"]:
            print(f"  Missing      : {stats['missing_classes']}")
        dm = TMCDataModule(args.tmc_dir, batch_size=args.batch_size,
                           num_workers=args.num_workers)
    dm.setup()

    if args.quick:
        print("\n  [QUICK MODE] Limiting to 25 samples.\n")
        from torch.utils.data import Subset
        dm.train_dataset = Subset(dm.train_dataset, range(min(25, len(dm.train_dataset))))
        dm.val_dataset   = Subset(dm.val_dataset,   range(min(25, len(dm.val_dataset))))

    # Model
    print("[2/4] Building TongueEncoder ...")
    model = TongueEncoder(
        embedding_dim      = 512,
        num_tongue_classes = n_classes,
        pretrained         = True,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params : {n_params:,}")
    print(f"  Output classes   : {n_classes}")

    if args.resume and Path(args.resume).exists():
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"], strict=False)
        print(f"  Resumed from: {args.resume}")

    # Loss & optimizer
    print("[3/4] Setting up training ...")
    class_weights = compute_class_weights(dm, n_classes, device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)
    optimizer     = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    epochs        = 2 if args.quick else args.epochs
    scheduler     = CosineAnnealingLR(optimizer, T_max=epochs)

    ckpt_stem = "tongue_encoder_shenzhen" if use_shenzhen else "tongue_encoder_tmc"

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print(f"[4/4] Training for {epochs} epoch(s) ...\n")
    best_f1  = 0.0
    history  = []

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        tr_loss, tr_acc = train_one_epoch(model, dm.train_dataloader(),
                                           optimizer, criterion, device, args.quick)
        vl_loss, vl_acc, vl_f1 = evaluate(model, dm.val_dataloader(),
                                            criterion, device, args.quick)
        scheduler.step()

        row = dict(epoch=epoch, train_loss=round(tr_loss, 4), train_acc=round(tr_acc, 4),
                   val_loss=round(vl_loss, 4), val_acc=round(vl_acc, 4), val_macro_f1=round(vl_f1, 4))
        history.append(row)
        print(f"  train_loss={tr_loss:.4f} train_acc={tr_acc*100:.1f}%  "
              f"val_loss={vl_loss:.4f} val_acc={vl_acc*100:.1f}%  "
              f"val_f1={vl_f1:.4f}\n")

        ckpt = {"epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "history": history,
                "dataset_format": fmt, "n_classes": n_classes}
        torch.save(ckpt, save_dir / f"{ckpt_stem}_latest.pt")

        if vl_f1 > best_f1:
            best_f1 = vl_f1
            torch.save(ckpt, save_dir / f"{ckpt_stem}_best.pt")
            print(f"  ✓ New best Macro-F1: {best_f1:.4f}\n")

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best Macro-F1 : {best_f1:.4f}")
    print(f"  Checkpoint    : {save_dir}/{ckpt_stem}_best.pt")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
