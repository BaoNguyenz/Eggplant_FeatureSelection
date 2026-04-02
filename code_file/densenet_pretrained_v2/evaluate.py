"""
Evaluation Script for DenseNet121 Pipeline
=============================================
- Load best model checkpoint (.pth).
- Chạy inference trên tập Test.
- Xuất: Classification Report, Confusion Matrix, ROC/AUC curve.

Usage:
  python evaluate.py
  python evaluate.py --data_dir "path/to/Fold_1" --output_dir "path/to/output"
  python evaluate.py --checkpoint "path/to/best_model.pth"
"""

import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from dataset import create_dataloaders
from model import create_densenet121
from utils import (
    plot_confusion_matrix,
    plot_roc_auc,
    save_classification_report,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate DenseNet121 on Test set"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=r"D:\Eggplant_leaf_v2\Fold_1",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"D:\Eggplant_leaf_v2\code_file\densenet_pretrained_v2",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to .pth checkpoint. Default: output_dir/best_model.pth",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.4)
    return parser.parse_args()


@torch.no_grad()
def evaluate_model(model, loader, device):
    """
    Chạy inference trên toàn bộ DataLoader.

    Returns:
        tuple: (all_labels, all_preds, all_probs)
            - all_labels: np.array, ground truth labels.
            - all_preds: np.array, predicted class indices.
            - all_probs: np.array, softmax probabilities (N, num_classes).
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    for images, labels in tqdm(loader, desc="[EVAL] Predicting"):
        images = images.to(device, non_blocking=True)
        outputs = model(images)

        probs = F.softmax(outputs, dim=1).cpu().numpy()
        preds = outputs.argmax(dim=1).cpu().numpy()

        all_labels.extend(labels.numpy())
        all_preds.extend(preds)
        all_probs.extend(probs)

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EVAL] Device: {device}")

    # -----------------------------------------------------------------------
    # 1. Load Data (chỉ cần test_loader)
    # -----------------------------------------------------------------------
    data = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    test_loader = data["test_loader"]
    class_names = data["class_names"]
    num_classes = data["num_classes"]

    # -----------------------------------------------------------------------
    # 2. Load Model
    # -----------------------------------------------------------------------
    checkpoint_path = args.checkpoint or os.path.join(
        args.output_dir, "best_model.pth"
    )
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint không tồn tại: {checkpoint_path}\n"
            f"Hãy chạy train.py trước!"
        )

    model = create_densenet121(num_classes=num_classes, dropout=args.dropout)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    print(f"[EVAL] Loaded checkpoint: {checkpoint_path}")
    print(f"[EVAL] Checkpoint epoch: {checkpoint.get('epoch', '?')}")
    print(f"[EVAL] Checkpoint Val F1: {checkpoint.get('val_f1', '?'):.4f}")

    # -----------------------------------------------------------------------
    # 3. Evaluate on Test set
    # -----------------------------------------------------------------------
    all_labels, all_preds, all_probs = evaluate_model(
        model, test_loader, device
    )

    # -----------------------------------------------------------------------
    # 4. Generate Reports & Plots
    # -----------------------------------------------------------------------
    eval_dir = os.path.join(args.output_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    # Classification Report
    report = save_classification_report(
        all_labels, all_preds, class_names, eval_dir
    )

    # Confusion Matrix
    plot_confusion_matrix(
        all_labels, all_preds, class_names, eval_dir, normalize=True
    )

    # ROC / AUC
    roc_auc = plot_roc_auc(
        all_labels, all_probs, class_names, eval_dir
    )

    # -----------------------------------------------------------------------
    # 5. Summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("[EVAL] Evaluation Summary")
    print(f"{'='*60}")
    print(f"  Test Accuracy : {report['accuracy']:.4f}")
    print(f"  Macro F1      : {report['macro avg']['f1-score']:.4f}")
    print(f"  Weighted F1   : {report['weighted avg']['f1-score']:.4f}")
    print(f"  Macro AUC     : {roc_auc['macro']:.4f}")
    print(f"\n[EVAL] All outputs saved to: {eval_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
