"""
step1_feature_importance.py — Chấm điểm quan trọng cho 1024 features bằng
Random Forest (Embedded Method) và xuất kết quả.

Output:
  1. File CSV chứa điểm quan trọng (giảm dần).
  2. Biểu đồ cột (bar chart) độ phân giải cao cho tất cả 1024 features.

Usage:
  python step1_feature_importance.py \
      --data_dir  data/densenet121_features.csv \
      --output_csv importance_scores.csv \
      --output_plot feature_importance.png
"""

from __future__ import annotations

import argparse
import time

import matplotlib
# Sử dụng Agg backend để lưu ảnh mà không cần GUI (pop-up)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from utils import load_data


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------
def compute_feature_importance(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 300,
    random_state: int = 42,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Huấn luyện Random Forest và trích xuất điểm quan trọng.

    Parameters
    ----------
    X : pd.DataFrame
        Ma trận đặc trưng.
    y : pd.Series
        Vector nhãn.
    n_estimators : int
        Số lượng cây trong rừng.
    random_state : int
        Seed tái lập kết quả.
    n_jobs : int
        Số CPU cores sử dụng (-1 = tất cả).

    Returns
    -------
    pd.DataFrame
        DataFrame với 2 cột ``feature`` và ``importance``, sắp xếp giảm dần.
    """
    print(f"[INFO] Đang huấn luyện Random Forest với {n_estimators} cây ...")
    start: float = time.time()

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,           # Tận dụng toàn bộ CPU cores
        class_weight="balanced", # Cân bằng lớp tự động
    )
    rf.fit(X, y)

    elapsed: float = time.time() - start
    print(f"[INFO] Hoàn thành trong {elapsed:.2f}s.")

    importance_df = pd.DataFrame(
        {"feature": X.columns, "importance": rf.feature_importances_}
    ).sort_values(by="importance", ascending=False).reset_index(drop=True)

    return importance_df


def plot_importance(
    importance_df: pd.DataFrame,
    output_path: str,
    dpi: int = 300,
) -> None:
    """Vẽ biểu đồ cột cho TẤT CẢ features và lưu ra file ảnh.

    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame đã sắp xếp giảm dần theo ``importance``.
    output_path : str
        Đường dẫn lưu file ảnh (png).
    dpi : int
        Độ phân giải ảnh.
    """
    n_features: int = len(importance_df)

    # Kích thước lớn để hiển thị đầy đủ 1024 features
    fig, ax = plt.subplots(figsize=(max(60, n_features * 0.06), 12))

    ax.bar(
        range(n_features),
        importance_df["importance"].values,
        color="#2196F3",
        edgecolor="none",
        width=0.8,
    )

    ax.set_xlabel("Feature Index (sorted by importance)", fontsize=14)
    ax.set_ylabel("Importance Score", fontsize=14)
    ax.set_title(
        f"Random Forest Feature Importance — All {n_features} Features",
        fontsize=18,
        fontweight="bold",
    )

    # Chỉ hiển thị 1 tick mỗi 50 features để tránh đè chữ
    tick_step: int = max(1, n_features // 20)
    ax.set_xticks(range(0, n_features, tick_step))
    ax.set_xticklabels(
        [importance_df["feature"].iloc[i] for i in range(0, n_features, tick_step)],
        rotation=90,
        fontsize=7,
    )
    ax.set_xlim(-0.5, n_features - 0.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Đã lưu biểu đồ → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 1: Tính điểm Feature Importance bằng Random Forest."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=r"E:\PROJECTWORSHOP\Eggplant_regconition\code_file\A_feature_selection\csv_data\features_trainval_fold1.csv",
        help="Đường dẫn tới file CSV chứa 1024 features + cột 'label'.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=r"E:\PROJECTWORSHOP\Eggplant_regconition\code_file\A_feature_selection\output_csv\importance_scores.csv",
        help="Đường dẫn lưu file CSV điểm quan trọng (mặc định: importance_scores.csv).",
    )
    parser.add_argument(
        "--output_plot",
        type=str,
        default=r"E:\PROJECTWORSHOP\Eggplant_regconition\code_file\A_feature_selection\output_plot\feature_importance.png",
        help="Đường dẫn lưu biểu đồ (mặc định: feature_importance.png).",
    )
    return parser.parse_args()


def main() -> None:
    args: argparse.Namespace = parse_args()

    # 1. Đọc dữ liệu
    X, y = load_data(args.data_dir)

    # 2. Tính điểm quan trọng
    importance_df: pd.DataFrame = compute_feature_importance(X, y)

    # 3. Lưu ra CSV
    importance_df.to_csv(args.output_csv, index=False)
    print(f"[INFO] Đã lưu điểm quan trọng → {args.output_csv}")

    # 4. Vẽ biểu đồ
    plot_importance(importance_df, args.output_plot)

    # 5. Hiển thị top 20
    print("\n[INFO] Top 20 features quan trọng nhất:")
    print(importance_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
