"""
VitaFusion AI — ECG Encoder Training on PTB-XL
================================================
Trains the ECGEncoder model (1D-CNN + BiLSTM) on the PTB-XL dataset.

Usage:
    # Full training
    python train_ecg.py --ptbxl_dir data/ptbxl --epochs 50 --batch_size 64

    # Quick dry-run (5% data, 2 epochs — verify pipeline works)
    python train_ecg.py --quick

    # Resume from checkpoint
    python train_ecg.py --resume models/pretrained/ecg_encoder_ptbxl.pt

Reference:
    Wagner et al. (2020) PTB-XL. Scientific Data 7, 154.
    Hannun et al. (2019) Cardiologist-level arrhythmia detection with CNN.
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    class tqdm:  # Minimal fallback
        def __init__(self, iterable=None, **kwargs): self._it = iterable or []
        def __iter__(self): return iter(self._it)
        def set_postfix(self, **kw): pass
        def close(self): pass

from models.ecg_encoder import ECGEncoder
from preprocessing.ptbxl_loader import PTBXLDataModule, VITAFUSION_CLASSES
import config

# ── Metrics helpers ───────────────────────────────────────────────────────────

def compute_metrics(
    all_preds: list, all_labels: list, all_probs: list, n_classes: int
) -> dict:
    """Compute accuracy, per-class F1, and macro-AUROC."""
    preds  = np.array(all_preds)
    labels = np.array(all_labels)
    probs  = np.array(all_probs)   # [N, n_classes]

    accuracy = float((preds == labels).mean())

    # Per-class F1
    f1_scores = []
    for c in range(n_classes):
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_scores.append(float(f1))
    macro_f1 = float(np.mean(f1_scores))

    # Macro AUROC (one-vs-rest)
    try:
        from sklearn.metrics import roc_auc_score
        label_onehot = np.eye(n_classes)[labels]
        auroc = float(roc_auc_score(label_onehot, probs, average="macro",
                                     multi_class="ovr"))
    except Exception:
        auroc = 0.0

    return {
        "accuracy":  accuracy,
        "macro_f1":  macro_f1,
        "auroc":     auroc,
        "per_class_f1": {VITAFUSION_CLASSES[i]: f1_scores[i] for i in range(n_classes)},
    }


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model: ECGEncoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    n_classes: int,
) -> dict:
    model.train()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    bar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False) if HAS_TQDM else loader
    for batch in bar:
        signals = batch["signal"].to(device)   # [B, 12, 2500]
        labels  = batch["label"].to(device)    # [B]

        optimizer.zero_grad()
        out  = model(signals)
        loss = criterion(out["arrhythmia_logits"], labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        probs = F.softmax(out["arrhythmia_logits"], dim=-1).detach().cpu().numpy()
        preds = probs.argmax(axis=1)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        all_probs.extend(probs.tolist())

        if HAS_TQDM:
            bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(len(all_labels), 1)
    metrics  = compute_metrics(all_preds, all_labels, all_probs, n_classes)
    metrics["loss"] = avg_loss
    return metrics


@torch.no_grad()
def evaluate(
    model: ECGEncoder,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    split: str,
    n_classes: int,
) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    bar = tqdm(loader, desc=f"  [{split}]", leave=False) if HAS_TQDM else loader
    for batch in bar:
        signals = batch["signal"].to(device)
        labels  = batch["label"].to(device)

        out  = model(signals)
        loss = criterion(out["arrhythmia_logits"], labels)
        total_loss += loss.item() * len(labels)

        probs = F.softmax(out["arrhythmia_logits"], dim=-1).cpu().numpy()
        preds = probs.argmax(axis=1)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        all_probs.extend(probs.tolist())

    avg_loss = total_loss / max(len(all_labels), 1)
    metrics  = compute_metrics(all_preds, all_labels, all_probs, n_classes)
    metrics["loss"] = avg_loss
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train ECGEncoder on PTB-XL")
    p.add_argument("--ptbxl_dir",   type=str, default=str(config.PTBXL_DIR),
                   help="Path to extracted PTB-XL dataset directory")
    p.add_argument("--sampling_rate", type=int, default=100, choices=[100, 500],
                   help="PTB-XL sampling rate to load (100 or 500 Hz)")
    p.add_argument("--epochs",      type=int, default=50)
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader workers (0 = main process, safe on Windows)")
    p.add_argument("--label_mode",  type=str, default="arrhythmia",
                   choices=["arrhythmia", "superclass"])
    p.add_argument("--save_dir",    type=str, default="models/pretrained")
    p.add_argument("--resume",      type=str, default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--quick",       action="store_true",
                   help="Dry-run: 5%% of data, 2 epochs")
    p.add_argument("--no_cuda",     action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Device ──────────────────────────────────────────────────────────────
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device   = torch.device("cuda" if use_cuda else "cpu")
    print(f"\n{'='*60}")
    print(f"  VitaFusion AI — ECG Training on PTB-XL")
    print(f"{'='*60}")
    print(f"  Device       : {device}")
    print(f"  PTB-XL dir   : {args.ptbxl_dir}")
    print(f"  Sampling rate: {args.sampling_rate} Hz  →  resampled to 250 Hz")
    print(f"  Label mode   : {args.label_mode}")
    print(f"  Epochs       : {2 if args.quick else args.epochs}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Quick mode   : {args.quick}")
    print(f"{'='*60}\n")

    # ── Data ────────────────────────────────────────────────────────────────
    dm = PTBXLDataModule(
        ptbxl_dir     = args.ptbxl_dir,
        sampling_rate = args.sampling_rate,
        batch_size    = args.batch_size,
        num_workers   = args.num_workers,
        label_mode    = args.label_mode,
    )
    print("Loading PTB-XL metadata...", flush=True)
    dm.setup()
    summary = dm.summary()
    print(f"  Train: {summary['train_size']} | Val: {summary['val_size']} | Test: {summary['test_size']}")

    # Quick mode: subsample 5%
    if args.quick:
        def subset(ds, frac=0.05):
            n = max(1, int(len(ds) * frac))
            indices = np.random.choice(len(ds), n, replace=False).tolist()
            return Subset(ds, indices)
        dm.train_dataset = subset(dm.train_dataset)
        dm.val_dataset   = subset(dm.val_dataset)
        args.epochs = 2
        print(f"  [Quick] Subsampled → {len(dm.train_dataset)} train / {len(dm.val_dataset)} val")

    train_loader = dm.train_dataloader()
    val_loader   = dm.val_dataloader()

    # ── Model ────────────────────────────────────────────────────────────────
    n_classes = 5 if args.label_mode == "superclass" else 8
    model = ECGEncoder(
        n_leads            = 12,
        n_timesteps        = 2500,
        embedding_dim      = 256,
        n_arrhythmia_classes = n_classes,
        lstm_hidden        = 256,
        lstm_layers        = 2,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model params : {total_params:,}")

    # ── Optimizer & Scheduler ────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # ── Class weights (handle imbalance) ─────────────────────────────────────
    label_counts = np.bincount(
        [int(dm.train_dataset.dataset.df.iloc[i]["vitafusion_label"]) if hasattr(dm.train_dataset, 'dataset')
         else int(dm.train_dataset.df.iloc[i]["vitafusion_label"])
         for i in range(min(1000, len(dm.train_dataset)))],
        minlength=n_classes,
    ).astype(np.float32)
    label_counts = np.clip(label_counts, 1, None)
    class_weights = torch.tensor(1.0 / label_counts, dtype=torch.float32).to(device)
    class_weights = class_weights / class_weights.sum() * n_classes
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch = 1
    best_auroc  = 0.0
    history     = []
    save_dir    = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path   = save_dir / "ecg_encoder_ptbxl.pt"
    best_path   = save_dir / "ecg_encoder_ptbxl_best.pt"

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 1) + 1
        best_auroc  = ckpt.get("best_auroc", 0.0)
        history     = ckpt.get("history", [])
        print(f"  Resumed from epoch {start_epoch - 1} (best AUROC: {best_auroc:.4f})")

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n  Starting training...\n  {'─'*58}")
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_m = train_one_epoch(model, train_loader, optimizer, criterion,
                                   device, epoch, n_classes)
        val_m   = evaluate(model, val_loader, criterion, device, "val", n_classes)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"  Ep {epoch:>3}/{args.epochs} | "
            f"loss {train_m['loss']:.4f} / {val_m['loss']:.4f} | "
            f"acc {train_m['accuracy']:.3f} / {val_m['accuracy']:.3f} | "
            f"AUROC {val_m['auroc']:.4f} | "
            f"F1 {val_m['macro_f1']:.4f} | "
            f"{elapsed:.1f}s"
        )

        epoch_log = {
            "epoch":      epoch,
            "train_loss": train_m["loss"],
            "val_loss":   val_m["loss"],
            "train_acc":  train_m["accuracy"],
            "val_acc":    val_m["accuracy"],
            "val_auroc":  val_m["auroc"],
            "val_f1":     val_m["macro_f1"],
        }
        history.append(epoch_log)

        # Save latest checkpoint
        torch.save({
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_auroc":      best_auroc,
            "history":         history,
            "args":            vars(args),
            "label_mode":      args.label_mode,
            "n_classes":       n_classes,
            "class_names":     VITAFUSION_CLASSES[:n_classes],
        }, ckpt_path)

        # Save best checkpoint
        if val_m["auroc"] > best_auroc:
            best_auroc = val_m["auroc"]
            torch.save(model.state_dict(), best_path)
            print(f"  ★ New best AUROC: {best_auroc:.4f} — saved to {best_path}")

    # ── Final test evaluation ─────────────────────────────────────────────────
    print(f"\n  {'─'*58}")
    print("  Loading best model for final test evaluation...")
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))

    test_loader = dm.test_dataloader()
    test_m = evaluate(model, test_loader, criterion, device, "test", n_classes)

    print(f"\n  ── Final Test Results ──")
    print(f"  Accuracy : {test_m['accuracy']:.4f}")
    print(f"  Macro F1 : {test_m['macro_f1']:.4f}")
    print(f"  AUROC    : {test_m['auroc']:.4f}")
    print(f"\n  Per-class F1:")
    for cls, f1 in test_m["per_class_f1"].items():
        print(f"    {cls:<35} {f1:.4f}")

    # Save history + results
    results_path = save_dir / "ecg_training_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "history": history,
            "test_metrics": test_m,
            "args": vars(args),
        }, f, indent=2)
    print(f"\n  Results saved to {results_path}")
    print(f"  Best model:   {best_path}")
    print(f"  Last checkpoint: {ckpt_path}")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
