"""
utils.py — Tiện ích dùng chung cho pipeline Feature Selection.

Chức năng:
  1. Đọc file CSV và tách X (features), y (label).
  2. Đánh giá mô hình với 4 metrics: Accuracy, Precision, Recall, F1-Score.
  3. Tạo và lưu Confusion Matrix.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


# ---------------------------------------------------------------------------
# 1. Đọc dữ liệu
# ---------------------------------------------------------------------------
def load_data(csv_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Đọc file CSV, tách và trả về (X, y).

    Parameters
    ----------
    csv_path : str
        Đường dẫn tới file CSV chứa 1024 features + cột ``label``.

    Returns
    -------
    X : pd.DataFrame
        Ma trận đặc trưng (tất cả cột trừ ``label``).
    y : pd.Series
        Vector nhãn (cột ``label``).
    """
    df: pd.DataFrame = pd.read_csv(csv_path)

    if "label" not in df.columns:
        raise ValueError(
            f"Cột 'label' không tồn tại trong file: {csv_path}. "
            f"Các cột hiện có: {list(df.columns)}"
        )

    y: pd.Series = df["label"]
    X: pd.DataFrame = df.drop(columns=["label"])

    print(f"[INFO] Đã đọc dữ liệu: {X.shape[0]} mẫu, {X.shape[1]} features.")
    return X, y


# ---------------------------------------------------------------------------
# 2. Đánh giá mô hình
# ---------------------------------------------------------------------------
def evaluate_model(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    model_name: str = "Model",
) -> dict[str, float]:
    """Tính 4 metrics và in kết quả ra terminal.

    Parameters
    ----------
    y_true : array-like
        Nhãn thực tế.
    y_pred : array-like
        Nhãn dự đoán.
    model_name : str
        Tên mô hình (dùng để in tiêu đề).

    Returns
    -------
    dict[str, float]
        Dictionary chứa 4 metrics.
    """
    # average='weighted' phù hợp cho bài toán đa lớp không cân bằng
    # Nhân 100 và làm tròn 2 chữ số thập phân (hiển thị dạng phần trăm)
    metrics: dict[str, float] = {
        "Accuracy": round(accuracy_score(y_true, y_pred) * 100, 2),
        "Precision": round(precision_score(y_true, y_pred, average="weighted", zero_division=0) * 100, 2),
        "Recall": round(recall_score(y_true, y_pred, average="weighted", zero_division=0) * 100, 2),
        "F1-Score": round(f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100, 2),
    }

    print(f"\n{'=' * 55}")
    print(f"  {model_name}")
    print(f"{'=' * 55}")
    for name, value in metrics.items():
        print(f"  {name:<12}: {value:.2f}%")
    print(f"{'=' * 55}")

    return metrics


# ---------------------------------------------------------------------------
# 3. Confusion Matrix — Lưu dưới dạng ảnh heatmap
# ---------------------------------------------------------------------------
def save_confusion_matrix(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    class_names: list[str],
    model_name: str,
    output_dir: str,
    dpi: int = 200,
) -> str:
    """Tạo Confusion Matrix heatmap và lưu dưới dạng ảnh PNG.

    Parameters
    ----------
    y_true : array-like
        Nhãn thực tế.
    y_pred : array-like
        Nhãn dự đoán.
    class_names : list[str]
        Tên các lớp (theo thứ tự encode).
    model_name : str
        Tên mô hình (dùng đặt tên file).
    output_dir : str
        Thư mục lưu file ảnh.
    dpi : int
        Độ phân giải ảnh.

    Returns
    -------
    str
        Đường dẫn file ảnh đã lưu.
    """
    cm: np.ndarray = confusion_matrix(y_true, y_pred)

    os.makedirs(output_dir, exist_ok=True)

    # Kích thước tỉ lệ với số lớp
    n_classes: int = len(class_names)
    fig_size: float = max(6, n_classes * 1.2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    # Vẽ heatmap với seaborn
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
        annot_kws={"size": 11},
    )

    ax.set_xlabel("Predicted Label", fontsize=13, fontweight="bold")
    ax.set_ylabel("Actual Label", fontsize=13, fontweight="bold")
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=15, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()

    # Tên file an toàn: thay khoảng trắng bằng underscore
    safe_name: str = model_name.replace(" ", "_").lower()
    filepath: str = os.path.join(output_dir, f"cm_{safe_name}.png")
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"\n--- Confusion Matrix: {model_name} ---")
    # In bảng CM ra terminal
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.index.name = "Actual \\ Predicted"
    print(cm_df.to_string())
    print(f"[INFO] Đã lưu CM plot → {filepath}")

    return filepath
