"""
VitaFusion AI — Clinical Encoder Training Script (Pima Diabetes)
================================================================
Trains the PimaClinicalEncoder on the Pima Indians Diabetes Dataset (UCI).

Task:    Binary classification (diabetes: 0 / 1)
Loss:    BCEWithLogitsLoss
Metrics: AUROC, Accuracy, Sensitivity (Recall), Specificity, F1

Usage:
    # Quick dry-run (validates pipeline — no dataset needed if CSV present):
    python train_clinical.py --quick

    # Full training (default):
    python train_clinical.py --pima_path data/pima/diabetes.csv --epochs 100

    # With custom hyperparameters:
    python train_clinical.py --pima_path data/pima/diabetes.csv \\
        --epochs 150 --lr 5e-4 --batch_size 128
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
    from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from models.clinical_encoder import PimaClinicalEncoder, PIMA_FEATURE_NAMES
from preprocessing.pima_loader import PimaDataModule, compute_pima_stats
import config


def parse_args():
    p = argparse.ArgumentParser(description="Train ClinicalEncoder on Pima Diabetes")
    p.add_argument("--pima_path",   type=str, default=str(config.PIMA_PATH))
    p.add_argument("--epochs",      type=int, default=100)
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--dropout",     type=float, default=0.3)
    p.add_argument("--resume",      type=str, default=None)
    p.add_argument("--save_dir",    type=str, default="models/pretrained")
    p.add_argument("--quick",       action="store_true",
                   help="Dry-run: 5 epochs, 50 samples")
    return p.parse_args()


def compute_pos_weight(dm: PimaDataModule, device: torch.device) -> torch.Tensor:
    """BCEWithLogitsLoss positive weight = neg/pos for class balance."""
    ds = dm.train_dataset
    # Handle Subset wrapping (created by --quick mode)
    if hasattr(ds, "dataset"):
        all_y  = ds.dataset.y.numpy()
        labels = all_y[list(ds.indices)]
    else:
        labels = ds.y.numpy()
    pos = labels.sum()
    neg = len(labels) - pos
    w   = neg / max(pos, 1)
    print(f"  Class balance -- Pos: {int(pos)}, Neg: {int(neg)}, pos_weight: {w:.2f}")
    return torch.tensor([w], dtype=torch.float32).to(device)


def train_one_epoch(model, loader, optimizer, criterion, device, quick=False):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for step, batch in enumerate(loader):
        if quick and step >= 5:
            break
        feats  = batch["features"].to(device)
        labels = batch["label"].float().unsqueeze(1).to(device)

        out    = model(feats)
        logit  = out["diabetes_logit"]
        loss   = criterion(logit, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        preds   = (torch.sigmoid(logit) > 0.5).float()
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        total_loss += loss.item()

        if step % 10 == 0:
            acc = correct / max(total, 1) * 100
            print(f"  step {step:4d} | loss {loss.item():.4f} | acc {acc:.1f}%")

    return total_loss / max(step + 1, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, quick=False):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    for step, batch in enumerate(loader):
        if quick and step >= 5:
            break
        feats  = batch["features"].to(device)
        labels = batch["label"].float().unsqueeze(1).to(device)

        out    = model(feats)
        logit  = out["diabetes_logit"]
        loss   = criterion(logit, labels)

        total_loss += loss.item()
        all_probs.extend(torch.sigmoid(logit).cpu().numpy().flatten())
        all_labels.extend(labels.cpu().numpy().flatten())

    probs  = np.array(all_probs)
    labels = np.array(all_labels).astype(int)
    preds  = (probs > 0.5).astype(int)

    acc     = (preds == labels).mean()
    auroc   = float("nan")
    sens, spec, f1 = float("nan"), float("nan"), float("nan")

    if SKLEARN_OK and len(np.unique(labels)) > 1:
        try:
            auroc = roc_auc_score(labels, probs)
            tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
            sens  = tp / max(tp + fn, 1)
            spec  = tn / max(tn + fp, 1)
            f1    = f1_score(labels, preds, zero_division=0)
        except Exception:
            pass

    return total_loss / max(step + 1, 1), acc, auroc, sens, spec, f1


def print_feature_importance(model, device):
    """Print feature attention weights from a random forward pass."""
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, len(PIMA_FEATURE_NAMES), device=device)
        out   = model(dummy)
        attn  = out.get("feature_attention", None)
        if attn is not None:
            attn = attn.squeeze().cpu().numpy()
            print("\n  Feature Attention Weights:")
            for feat, w in sorted(zip(PIMA_FEATURE_NAMES, attn),
                                   key=lambda x: -x[1]):
                bar = "#" * int(w * 40)   # ASCII-safe (no Unicode block chars)
                print(f"    {feat:<30} {bar} {w:.4f}")


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  VitaFusion — Clinical Encoder Training (Pima Diabetes)")
    print(f"{'='*60}")
    print(f"  Device      : {device}")
    print(f"  Pima CSV    : {args.pima_path}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Quick mode  : {args.quick}")
    print(f"  Features    : {PIMA_FEATURE_NAMES}")
    print(f"{'='*60}\n")

    # Dataset
    print("[1/4] Loading Pima Diabetes dataset ...")
    stats = compute_pima_stats(args.pima_path)
    if not stats.get("available"):
        print(f"✗ Dataset not found: {stats.get('error')}")
        print("  Download: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database")
        print("  Save as: data/pima/diabetes.csv")
        sys.exit(1)

    print(f"  Records     : {stats['n_records']}")
    print(f"  Prevalence  : {stats['prevalence_pct']}% diabetes")
    print(f"  Imputed     : {stats['imputed_columns']}")
    print(f"\n  Feature-Outcome Correlations:")
    for feat, corr in sorted(stats["feature_outcome_correlation"].items(),
                              key=lambda x: -abs(x[1])):
        bar = "+" if corr > 0 else "-"
        print(f"    {feat:<30} {bar} {abs(corr):.3f}")

    dm = PimaDataModule(args.pima_path, batch_size=args.batch_size)
    dm.setup()
    s  = dm.summary()
    print(f"\n  Split: train={s['train']}, val={s['val']}, test={s['test']}")

    # Compute pos_weight from full training set (before any quick-mode subsampling)
    print("[3/4] Setting up training ...")
    pos_weight = compute_pos_weight(dm, device)

    if args.quick:
        print("\n  [QUICK MODE] Limiting to 50 samples.\n")
        from torch.utils.data import Subset
        dm.train_dataset = Subset(dm.train_dataset, range(min(50, len(dm.train_dataset))))
        dm.val_dataset   = Subset(dm.val_dataset,   range(min(50, len(dm.val_dataset))))

    # Model
    print("[2/4] Building PimaClinicalEncoder ...")
    model = PimaClinicalEncoder(
        embedding_dim = 256,
        hidden_dims   = [128, 256, 384],
        dropout       = args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params : {n_params:,}")
    print(f"  Input features   : {len(PIMA_FEATURE_NAMES)} (Pima 8-feature schema)")

    if args.resume and Path(args.resume).exists():
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"], strict=False)
        print(f"  Resumed from: {args.resume}")

    # Loss & optimizer (pos_weight already computed above)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    epochs     = 5 if args.quick else args.epochs
    scheduler  = CosineAnnealingLR(optimizer, T_max=epochs)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print(f"[4/4] Training for {epochs} epoch(s) ...\n")
    best_auroc = 0.0
    history    = []

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        tr_loss, tr_acc = train_one_epoch(model, dm.train_dataloader(),
                                           optimizer, criterion, device, args.quick)
        vl_loss, vl_acc, vl_auroc, vl_sens, vl_spec, vl_f1 = evaluate(
            model, dm.val_dataloader(), criterion, device, args.quick
        )
        scheduler.step()

        row = dict(epoch=epoch,
                   train_loss=round(tr_loss, 4), train_acc=round(tr_acc, 4),
                   val_loss=round(vl_loss, 4),   val_acc=round(vl_acc, 4),
                   val_auroc=round(vl_auroc, 4), val_sens=round(vl_sens, 4),
                   val_spec=round(vl_spec, 4),   val_f1=round(vl_f1, 4))
        history.append(row)

        print(f"  train_loss={tr_loss:.4f} train_acc={tr_acc*100:.1f}%  "
              f"val_auroc={vl_auroc:.4f} val_acc={vl_acc*100:.1f}%  "
              f"sens={vl_sens:.3f} spec={vl_spec:.3f} f1={vl_f1:.4f}\n")

        ckpt = {"epoch": epoch, "model": model.state_dict(),
                "scaler": dm.scaler, "optimizer": optimizer.state_dict(),
                "history": history, "feature_names": PIMA_FEATURE_NAMES}
        torch.save(ckpt, save_dir / "clinical_encoder_pima_latest.pt")

        if vl_auroc > best_auroc:
            best_auroc = vl_auroc
            torch.save(ckpt, save_dir / "clinical_encoder_pima_best.pt")
            print(f"  [BEST] New best AUROC: {best_auroc:.4f}\n")

    # Print final feature attention
    print_feature_importance(model, device)

    # Test set evaluation
    print("\n[Test Set Evaluation]")
    te_loss, te_acc, te_auroc, te_sens, te_spec, te_f1 = evaluate(
        model, dm.test_dataloader(), criterion, device
    )
    print(f"  Test AUROC    : {te_auroc:.4f}")
    print(f"  Test Accuracy : {te_acc*100:.1f}%")
    print(f"  Sensitivity   : {te_sens:.3f}")
    print(f"  Specificity   : {te_spec:.3f}")
    print(f"  F1 Score      : {te_f1:.4f}")

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best Val AUROC : {best_auroc:.4f}")
    print(f"  Checkpoint     : {save_dir}/clinical_encoder_pima_best.pt")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
