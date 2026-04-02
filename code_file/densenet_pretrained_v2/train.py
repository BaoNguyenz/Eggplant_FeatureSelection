"""
Training Script for DenseNet121 Fine-tuning Pipeline
======================================================
Huấn luyện 2 giai đoạn:
  - Stage 1 (Warmup):   Freeze backbone, train classifier head (epoch 1→warmup_epochs).
  - Stage 2 (Fine-tune): Unfreeze toàn bộ, Discriminative LR + CosineAnnealingLR.

Xử lý mất cân bằng kép:
  - WeightedRandomSampler (trong dataset.py).
  - CrossEntropyLoss(weight=class_weights).

Multi-GPU: DataParallel.

Usage:
  python train.py
  python train.py --epochs 40 --lr 1e-3 --batch_size 32
  python train.py --gpu_ids 0 1
"""

import argparse
import os
import gc
import time
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm

from dataset import create_dataloaders
from model import (
    create_densenet121,
    freeze_backbone,
    unfreeze_backbone,
    get_discriminative_params,
    wrap_data_parallel,
)
from utils import (
    seed_everything,
    AverageMeter,
    EarlyStopping,
    FocalLoss,
    plot_training_curves,
    save_history,
    save_hyperparams,
)


# ---------------------------------------------------------------------------
# 1. Argparse
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="DenseNet121 Fine-tuning for Eggplant Leaf Disease Classification"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=r"D:\Eggplant_leaf_v2\Fold_1",
        help="Path to Fold directory (contains train_val/ and test/).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"D:\Eggplant_leaf_v2\code_file\densenet_pretrained_v2\checkpoints\output",
        help="Directory to save model weights, plots, and logs.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3, help="LR for classifier head.")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Stage 1: freeze backbone epochs.")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--gpu_ids",
        nargs="+",
        type=int,
        default=None,
        help="GPU IDs for DataParallel. None → auto detect.",
    )
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument(
        "--patience", type=int, default=7,
        help="Early stopping patience (epochs without improvement). 0 = disabled.",
    )
    parser.add_argument(
        "--es_min_delta", type=float, default=1e-4,
        help="Minimum improvement delta for early stopping.",
    )
    parser.add_argument(
        "--loss", type=str, default="ce", choices=["ce", "focal"],
        help="Loss function: 'ce' (CrossEntropyLoss) or 'focal' (FocalLoss).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 2. Train / Val cho 1 epoch
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, scheduler=None):
    """
    Train model trên 1 epoch.

    Returns:
        tuple: (avg_loss, accuracy, macro_f1)
    """
    model.train()
    loss_meter = AverageMeter("loss")
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc="  [Train]", leave=False, ncols=120)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # Metrics
        loss_meter.update(loss.item(), images.size(0))
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        # Giải phóng tham chiếu tensor — ngăn rò rỉ RAM
        del images, labels, outputs, loss, preds

        # Cập nhật progress bar
        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return loss_meter.avg, acc, f1


@torch.no_grad()
def validate(model, loader, criterion, device):
    """
    Validate model trên 1 epoch.

    Returns:
        tuple: (avg_loss, accuracy, macro_f1)
    """
    model.eval()
    loss_meter = AverageMeter("loss")
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc="  [Val]  ", leave=False, ncols=120)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss_meter.update(loss.item(), images.size(0))
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        # Giải phóng tham chiếu tensor
        del images, labels, outputs, loss, preds

        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return loss_meter.avg, acc, f1


# ---------------------------------------------------------------------------
# 3. Main Training Loop
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    seed_everything(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    save_hyperparams(args, args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[TRAIN] Device: {device}")

    # -----------------------------------------------------------------------
    # A. Data
    # -----------------------------------------------------------------------
    data = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    train_loader = data["train_loader"]
    val_loader = data["val_loader"]
    class_names = data["class_names"]
    num_classes = data["num_classes"]
    class_weights = data["class_weights"].to(device)

    # -----------------------------------------------------------------------
    # B. Model
    # -----------------------------------------------------------------------
    model = create_densenet121(num_classes=num_classes, dropout=args.dropout)
    model = wrap_data_parallel(model, gpu_ids=args.gpu_ids)

    # -----------------------------------------------------------------------
    # C. Loss Function
    # -----------------------------------------------------------------------
    if args.loss == "focal":
        criterion = FocalLoss(weight=class_weights, gamma=2.0)
        print(f"[TRAIN] FocalLoss with class weights, gamma={criterion.gamma}")
        print(f"[TRAIN] ★ Để tuning gamma, sửa trực tiếp trong utils.py (class FocalLoss) hoặc dòng trên.")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        print(f"[TRAIN] CrossEntropyLoss with class weights + label_smoothing=0.1")
    print(f"[TRAIN] Class weights: {class_weights.cpu().tolist()}")

    # -----------------------------------------------------------------------
    # D. Training History
    # -----------------------------------------------------------------------
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "train_f1": [], "val_f1": [],
    }

    best_f1 = 0.0
    best_epoch = 0
    best_model_path = os.path.join(args.output_dir, "best_model.pth")

    # -----------------------------------------------------------------------
    # E. Early Stopping (chỉ active sau warmup)
    # -----------------------------------------------------------------------
    early_stopper = None
    if args.patience > 0:
        early_stopper = EarlyStopping(
            patience=args.patience,
            min_delta=args.es_min_delta,
            mode="max",  # Theo dõi Val F1 (higher is better)
            warmup_epochs=args.warmup_epochs,
        )

    # -----------------------------------------------------------------------
    # F. 2-Stage Training
    # -----------------------------------------------------------------------
    total_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # ===================================================================
        # STAGE TRANSITION
        # ===================================================================
        if epoch == 1:
            # --- Stage 1: Warmup — Freeze backbone ---
            print(f"\n{'='*60}")
            print(f"[STAGE 1] WARMUP — Freeze backbone, train classifier only")
            print(f"{'='*60}")
            freeze_backbone(model)

            # Optimizer: chỉ train params có requires_grad (classifier)
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr,
                weight_decay=1e-2,
            )
            scheduler = None  # Không dùng scheduler ở Stage 1

        elif epoch == args.warmup_epochs + 1:
            # --- Stage 2: Fine-tune — Unfreeze toàn bộ ---
            print(f"\n{'='*60}")
            print(f"[STAGE 2] FINE-TUNE — Unfreeze all, Discriminative LR")
            print(f"{'='*60}")
            unfreeze_backbone(model)

            # Optimizer: Discriminative LR
            print(f"\n[OPT] Discriminative LR groups:")
            param_groups = get_discriminative_params(model, base_lr=args.lr)
            optimizer = torch.optim.AdamW(param_groups)

            # Scheduler: OneCycleLR — step per batch (gọi bên trong train_one_epoch)
            fine_tune_epochs = args.epochs - args.warmup_epochs
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=[g["lr"] for g in param_groups],
                steps_per_epoch=len(train_loader),
                epochs=fine_tune_epochs,
                pct_start=0.3,
                div_factor=10.0,
                final_div_factor=1e4,
            )
            print(f"[OPT] OneCycleLR: steps/epoch={len(train_loader)}, epochs={fine_tune_epochs}, pct_start=0.3")

        # ===================================================================
        # TRAIN + VALIDATE
        # ===================================================================
        # Truyền scheduler vào train_one_epoch (chỉ active ở Stage 2)
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scheduler=scheduler if epoch > args.warmup_epochs else None,
        )
        val_loss, val_acc, val_f1 = validate(
            model, val_loader, criterion, device
        )

        # Lấy LR để log (OneCycleLR đã step per-batch bên trong train_one_epoch)
        if scheduler is not None and epoch > args.warmup_epochs:
            current_lr = scheduler.get_last_lr()[-1]  # LR nhóm classifier
        else:
            current_lr = args.lr

        # Record history
        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_acc"].append(round(val_acc, 4))
        history["train_f1"].append(round(train_f1, 4))
        history["val_f1"].append(round(val_f1, 4))

        # Stage label
        stage = "WM" if epoch <= args.warmup_epochs else "FT"
        elapsed = time.time() - epoch_start

        print(
            f"[{stage}] Epoch {epoch:>3}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} | "
            f"LR: {current_lr:.2e} | {elapsed:.1f}s"
        )

        # ===================================================================
        # SAVE BEST MODEL (theo Val F1)
        # ===================================================================
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch

            # Lưu state_dict gốc (unwrap DataParallel nếu có)
            base_model = model.module if hasattr(model, "module") else model
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": base_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "class_names": class_names,
                "num_classes": num_classes,
            }
            torch.save(checkpoint, best_model_path)
            print(f"  ★ New best model saved! Val F1={val_f1:.4f} @ epoch {epoch}")

        # ===================================================================
        # EARLY STOPPING CHECK (chỉ active sau warmup)
        # ===================================================================
        if early_stopper is not None and early_stopper(epoch, val_f1):
            print(f"\n[TRAIN] Early stopping at epoch {epoch}. Best F1={best_f1:.4f} @ epoch {best_epoch}")
            break

        # ===================================================================
        # MEMORY CLEANUP — Giải phóng RAM/VRAM sau mỗi epoch
        # ===================================================================
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # F. Training Complete
    # -----------------------------------------------------------------------
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"[DONE] Training completed in {total_time/60:.1f} minutes")
    print(f"[DONE] Best Val F1: {best_f1:.4f} @ epoch {best_epoch}")
    print(f"[DONE] Best model: {best_model_path}")
    print(f"{'='*60}")

    # Lưu history và vẽ curves
    save_history(history, args.output_dir)
    plot_training_curves(history, args.output_dir)

    print(f"\n[NEXT] Run evaluation:")
    print(f"  python evaluate.py --data_dir \"{args.data_dir}\" --output_dir \"{args.output_dir}\"")


if __name__ == "__main__":
    main()
