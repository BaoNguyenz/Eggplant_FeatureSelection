"""
Utility Module for DenseNet121 Training Pipeline
==================================================
- Hàm vẽ Loss / Accuracy / F1 curves.
- Confusion Matrix.
- ROC/AUC curve (One-vs-Rest).
- Classification Report.
- AverageMeter, EarlyStopping helpers.
- Seed everything for reproducibility.
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend cho server
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    f1_score,
)
from itertools import cycle


# ---------------------------------------------------------------------------
# 1. Reproducibility
# ---------------------------------------------------------------------------
def seed_everything(seed: int = 42) -> None:
    """Đặt seed cho tất cả random generators để tái hiện kết quả."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[UTIL] Random seed set to {seed}")


# ---------------------------------------------------------------------------
# 2. AverageMeter — theo dõi running average
# ---------------------------------------------------------------------------
class AverageMeter:
    """Theo dõi giá trị trung bình và tổng cho một metric."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ---------------------------------------------------------------------------
# 3. EarlyStopping
# ---------------------------------------------------------------------------
class EarlyStopping:
    """
    Early Stopping để dừng huấn luyện khi Val metric không cải thiện.

    Chỉ active sau giai đoạn Warmup (Stage 1) để tránh dừng sớm
    khi backbone vẫn đang frozen.

    Bất kỳ sự cải thiện nào (dù rất nhỏ) cũng được tính là tiến bộ.

    Args:
        patience (int): Số epoch chờ không cải thiện trước khi dừng.
        mode (str): 'max' (F1, Accuracy) hoặc 'min' (Loss).
        warmup_epochs (int): Số epoch warmup — bỏ qua early stopping trong giai đoạn này.
    """

    def __init__(
        self,
        patience: int = 7,
        mode: str = "max",
        warmup_epochs: int = 5,
    ):
        self.patience = patience
        self.mode = mode
        self.warmup_epochs = warmup_epochs

        self.counter = 0
        self.best_score = None
        self.early_stop = False

        # Hàm so sánh dựa trên mode (không dùng min_delta)
        if mode == "max":
            self._is_improvement = lambda curr, best: curr > best
        else:
            self._is_improvement = lambda curr, best: curr < best

        print(
            f"[UTIL] EarlyStopping: patience={patience}, "
            f"mode={mode}, active_after_epoch={warmup_epochs}"
        )

    def __call__(self, epoch: int, metric_value: float) -> bool:
        """
        Kiểm tra có nên dừng huấn luyện không.

        Args:
            epoch: Epoch hiện tại (1-indexed).
            metric_value: Giá trị metric cần theo dõi (VD: val_f1).

        Returns:
            bool: True nếu nên dừng (early stop triggered).
        """
        # Bỏ qua trong giai đoạn Warmup
        if epoch <= self.warmup_epochs:
            return False

        if self.best_score is None:
            self.best_score = metric_value
            return False

        if self._is_improvement(metric_value, self.best_score):
            self.best_score = metric_value
            self.counter = 0
        else:
            self.counter += 1
            print(
                f"  [ES] No improvement for {self.counter}/{self.patience} epochs "
                f"(best={self.best_score:.4f}, current={metric_value:.4f})"
            )
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"  [ES] ⚠ Early stopping triggered at epoch {epoch}!")
                return True

        return False


# ---------------------------------------------------------------------------
# 4. Focal Loss — Xử lý class imbalance nâng cao
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """
    Focal Loss cho bài toán multi-class classification.

    Công thức:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Trong đó:
        - p_t: xác suất dự đoán của class đúng (sau softmax).
        - gamma: focusing parameter — giảm trọng số mẫu dễ, tập trung mẫu khó.
          + gamma = 0 → tương đương CrossEntropyLoss.
          + gamma = 2.0 → giá trị khuyến nghị (paper gốc).
          ★ TUNING: Thay đổi gamma tại đây hoặc khi khởi tạo.
        - alpha (weight): trọng số từng class để xử lý imbalance.

    Args:
        weight (Tensor | None): Trọng số cho từng class, shape (num_classes,).
                                Dùng class_weights từ dataset.py.
        gamma (float): Focusing parameter. Mặc định = 2.0.
        reduction (str): 'mean' | 'sum' | 'none'. Mặc định = 'mean'.

    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    """

    def __init__(self, weight=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma  # ★ TUNING POINT: thay đổi giá trị gamma tại đây
        self.reduction = reduction

        # Lưu class weights dưới dạng buffer (tự chuyển device theo model)
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits từ model, shape (N, C).
            targets: Ground truth labels, shape (N,).

        Returns:
            Focal loss scalar (nếu reduction='mean' hoặc 'sum').
        """
        # Tính log-softmax để có log(p) ổn định số học
        log_p = F.log_softmax(inputs, dim=1)  # (N, C)
        # Lấy log(p_t) cho class đúng
        log_pt = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)  # (N,)
        # p_t = exp(log(p_t))
        pt = log_pt.exp()  # (N,)

        # Focal modulating factor: (1 - p_t)^gamma
        focal_weight = (1.0 - pt) ** self.gamma  # (N,)

        # Loss = -focal_weight * log(p_t)
        loss = -focal_weight * log_pt  # (N,)

        # Áp dụng class weights (alpha_t)
        if self.weight is not None:
            alpha_t = self.weight.gather(0, targets)  # (N,)
            loss = loss * alpha_t

        # Reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

    def extra_repr(self):
        return f"gamma={self.gamma}, reduction='{self.reduction}'"


# ---------------------------------------------------------------------------
# 5. Vẽ Training Curves (Loss / Accuracy / F1)
# ---------------------------------------------------------------------------
def plot_training_curves(history: dict, output_dir: str) -> None:
    """
    Vẽ và lưu biểu đồ Loss, Accuracy, F1 theo epoch.

    Args:
        history: Dict có keys: train_loss, val_loss, train_acc, val_acc,
                 train_f1, val_f1. Mỗi key chứa list giá trị theo epoch.
        output_dir: Thư mục lưu ảnh.
    """
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Loss ---
    axes[0].plot(epochs, history["train_loss"], "b-o", markersize=3, label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], "r-o", markersize=3, label="Val Loss")
    axes[0].set_title("Loss Curve", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Accuracy ---
    axes[1].plot(epochs, history["train_acc"], "b-o", markersize=3, label="Train Acc")
    axes[1].plot(epochs, history["val_acc"], "r-o", markersize=3, label="Val Acc")
    axes[1].set_title("Accuracy Curve", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # --- F1 Score ---
    axes[2].plot(epochs, history["train_f1"], "b-o", markersize=3, label="Train F1")
    axes[2].plot(epochs, history["val_f1"], "r-o", markersize=3, label="Val F1")
    axes[2].set_title("F1 Score Curve (Macro)", fontsize=14, fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1 Score")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Training curves saved → {save_path}")


# ---------------------------------------------------------------------------
# 4. Confusion Matrix
# ---------------------------------------------------------------------------
def plot_confusion_matrix(
    y_true, y_pred, class_names, output_dir, normalize=True
) -> None:
    """
    Vẽ và lưu Confusion Matrix.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: Danh sách tên class.
        output_dir: Thư mục lưu ảnh.
        normalize: Nếu True, hiển thị tỉ lệ % thay vì count.
    """
    os.makedirs(output_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm_display = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
        title = "Confusion Matrix (Normalized)"
    else:
        cm_display = cm
        fmt = "d"
        title = "Confusion Matrix"

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
        linecolor="gray",
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Confusion Matrix saved → {save_path}")


# ---------------------------------------------------------------------------
# 5. ROC/AUC Curve (One-vs-Rest)
# ---------------------------------------------------------------------------
def plot_roc_auc(y_true, y_probs, class_names, output_dir) -> None:
    """
    Vẽ ROC/AUC curve cho từng class (One-vs-Rest) + Macro-average.

    Args:
        y_true: Ground truth labels (1D array, integer-encoded).
        y_probs: Predicted probabilities, shape (N, num_classes).
        class_names: Danh sách tên class.
        output_dir: Thư mục lưu ảnh.
    """
    os.makedirs(output_dir, exist_ok=True)

    num_classes = len(class_names)

    # Chuyển y_true thành one-hot
    y_true_arr = np.array(y_true)
    y_true_onehot = np.zeros((len(y_true_arr), num_classes))
    y_true_onehot[np.arange(len(y_true_arr)), y_true_arr] = 1

    y_probs_arr = np.array(y_probs)

    # Tính FPR, TPR, AUC cho từng class
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_probs_arr[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Macro-average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= num_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = cycle(plt.cm.Set2.colors)

    for i, color in zip(range(num_classes), colors):
        ax.plot(
            fpr[i], tpr[i], color=color, lw=1.5,
            label=f"{class_names[i]} (AUC={roc_auc[i]:.3f})"
        )

    ax.plot(
        fpr["macro"], tpr["macro"], color="navy", lw=2.5, linestyle="--",
        label=f"Macro-Avg (AUC={roc_auc['macro']:.3f})"
    )

    ax.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC / AUC Curve (One-vs-Rest)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "roc_auc_curve.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] ROC/AUC curve saved → {save_path}")

    return roc_auc


# ---------------------------------------------------------------------------
# 6. Classification Report (text + JSON)
# ---------------------------------------------------------------------------
def save_classification_report(
    y_true, y_pred, class_names, output_dir
) -> dict:
    """
    In Classification Report ra console và lưu file JSON.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: Tên các class.
        output_dir: Thư mục lưu.

    Returns:
        dict: Classification report dạng dict.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Text report
    report_text = classification_report(
        y_true, y_pred, target_names=class_names, digits=4
    )
    print(f"\n{'='*60}")
    print("Classification Report")
    print("=" * 60)
    print(report_text)

    # Dict report
    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, output_dict=True
    )

    # Lưu JSON
    json_path = os.path.join(output_dir, "classification_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)
    print(f"[REPORT] Classification report saved → {json_path}")

    # Lưu text
    txt_path = os.path.join(output_dir, "classification_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_dict


# ---------------------------------------------------------------------------
# 7. Lưu History dạng JSON
# ---------------------------------------------------------------------------
def save_history(history: dict, output_dir: str) -> None:
    """Lưu training history ra file JSON."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "training_history.json")
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[UTIL] Training history saved → {path}")


# ---------------------------------------------------------------------------
# 8. Lưu Hyperparameters
# ---------------------------------------------------------------------------
def save_hyperparams(args, output_dir: str) -> None:
    """Lưu toàn bộ hyperparameters (argparse) ra file JSON."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "hyperparams.json")
    with open(path, "w") as f:
        json.dump(vars(args), f, indent=2, default=str)
    print(f"[UTIL] Hyperparams saved → {path}")
