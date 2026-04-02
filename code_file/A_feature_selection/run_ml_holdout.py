"""
run_ml_holdout.py — Đánh giá Hold-out thuần túy (Train trên TrainVal, Test trên Test ẩn).

Không sử dụng Cross-Validation. Chỉ train 1 lần duy nhất trên toàn bộ TrainVal,
sau đó predict trên Test set để lấy kết quả cuối cùng.

Usage:
  python run_ml_holdout.py --top_k 256
  python run_ml_holdout.py --top_k 512 --scale false
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from utils import load_data, evaluate_model, save_confusion_matrix

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. Chọn top-K features
# ---------------------------------------------------------------------------
def select_top_k_features(
    X: pd.DataFrame,
    importance_file: str,
    top_k: int,
) -> pd.DataFrame:
    """Cắt dữ liệu chỉ giữ lại top-K features quan trọng nhất."""
    importance_df: pd.DataFrame = pd.read_csv(importance_file)
    top_features: list[str] = importance_df["feature"].head(top_k).tolist()

    missing: set[str] = set(top_features) - set(X.columns)
    if missing:
        raise ValueError(f"Các feature sau không tìm thấy trong dữ liệu: {missing}")

    X_selected: pd.DataFrame = X[top_features]
    print(f"[INFO] Đã chọn top {top_k} features từ '{importance_file}'.")
    return X_selected


# ---------------------------------------------------------------------------
# 2. Định nghĩa 8 mô hình
# ---------------------------------------------------------------------------
def build_models(random_state: int = 42) -> list[tuple[str, Any]]:
    """Tạo danh sách 8 mô hình với cấu hình tối ưu (bao gồm GPU)."""
    models: list[tuple[str, Any]] = [
        (
            "KNN",
            KNeighborsClassifier(
                n_neighbors=5, weights="distance", n_jobs=-1,
            ),
        ),
        (
            "SVM",
            SVC(
                kernel="rbf", C=1.0, gamma="scale", random_state=random_state,
            ),
        ),
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=300, random_state=random_state,
                n_jobs=-1, class_weight="balanced",
            ),
        ),
        (
            "Logistic Regression",
            LogisticRegression(
                max_iter=2000, random_state=random_state,
                n_jobs=-1, class_weight="balanced",
                solver="lbfgs", multi_class="multinomial",
            ),
        ),
        (
            "Extra Trees",
            ExtraTreesClassifier(
                n_estimators=300, random_state=random_state,
                n_jobs=-1, class_weight="balanced",
            ),
        ),
        (
            "XGBoost",
            XGBClassifier(
                n_estimators=300, learning_rate=0.1, max_depth=6,
                device="cuda", eval_metric="mlogloss",
                random_state=random_state, verbosity=0,
            ),
        ),
        (
            "CatBoost",
            CatBoostClassifier(
                iterations=300, learning_rate=0.1, depth=6,
                task_type="GPU", devices="0",
                random_seed=random_state, verbose=0,
                auto_class_weights="Balanced",
            ),
        ),
        (
            "LightGBM",
            LGBMClassifier(
                n_estimators=300, learning_rate=0.1, max_depth=6,
                random_state=random_state, class_weight="balanced", verbose=-1,
            ),
        ),
    ]
    return models


# ---------------------------------------------------------------------------
# 3. Đánh giá Hold-out thuần túy
# ---------------------------------------------------------------------------
def evaluate_pure_holdout(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    models: list[tuple[str, Any]],
    class_names: list[str],
    output_dir: str,
) -> pd.DataFrame:
    """Train trên toàn bộ TrainVal, Evaluate trên Test ẩn."""

    cm_dir: str = os.path.join(output_dir, "confusion_matrices")
    results: list[dict[str, Any]] = []

    for name, model in models:
        print(f"\n>>> Đang huấn luyện: {name} ...")
        start: float = time.time()

        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        except Exception as e:
            # Fallback CPU nếu GPU lỗi
            print(f"[WARNING] {name} GPU failed: {e}")
            print(f"[INFO] Đang thử lại {name} trên CPU ...")
            from sklearn.base import clone
            model_cpu = clone(model)
            if hasattr(model_cpu, "device"):
                model_cpu.set_params(device="cpu")
            if hasattr(model_cpu, "task_type"):
                model_cpu.set_params(task_type="CPU")
            model_cpu.fit(X_train, y_train)
            y_pred = model_cpu.predict(X_test)

        elapsed: float = time.time() - start

        metrics: dict[str, float] = evaluate_model(y_test, y_pred, name)
        metrics["Model"] = name
        metrics["Time (s)"] = round(elapsed, 2)
        results.append(metrics)

        # Lưu Confusion Matrix
        save_confusion_matrix(y_test, y_pred, class_names, name, cm_dir)

    return pd.DataFrame(results)[
        ["Model", "Precision", "Recall", "F1-Score", "Accuracy", "Time (s)"]
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hold-out thuần túy: Train trên TrainVal, Test trên Test ẩn."
    )
    parser.add_argument(
        "--data_trainval", type=str,
        default=r"E:\PROJECTWORSHOP\Eggplant_regconition\code_file\A_feature_selection\csv_data\features_trainval_fold1.csv",
        help="CSV features của tập TrainVal.",
    )
    parser.add_argument(
        "--data_test", type=str,
        default=r"E:\PROJECTWORSHOP\Eggplant_regconition\code_file\A_feature_selection\csv_data\features_test_fold1.csv",
        help="CSV features của tập Test ẩn.",
    )
    parser.add_argument(
        "--importance_file", type=str,
        default=r"E:\PROJECTWORSHOP\Eggplant_regconition\code_file\A_feature_selection\output_csv\importance_scores.csv",
        help="Đường dẫn tới file importance_scores.csv.",
    )
    parser.add_argument(
        "--top_k", type=int, required=True,
        help="Số lượng features muốn giữ lại.",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=r"E:\PROJECTWORSHOP\Eggplant_regconition\code_file\A_feature_selection\output_csv",
        help="Thư mục lưu kết quả CSV và confusion matrices.",
    )
    parser.add_argument(
        "--scale", type=str, choices=["true", "false"], default="true",
        help="Chuẩn hóa features bằng StandardScaler (default: true).",
    )
    return parser.parse_args()


def main() -> None:
    args: argparse.Namespace = parse_args()

    # 1. Đọc 2 tập dữ liệu
    print(f"\n{'=' * 65}")
    print("  HOLD-OUT EVALUATION (TrainVal → Test ẩn)")
    print(f"{'=' * 65}")

    X_trainval, y_trainval = load_data(args.data_trainval)
    X_test, y_test = load_data(args.data_test)

    # 2. Encode nhãn
    le = LabelEncoder()
    y_tv_encoded: np.ndarray = le.fit_transform(y_trainval)
    y_test_encoded: np.ndarray = le.transform(y_test)
    class_names: list[str] = le.classes_.tolist()
    print(f"[INFO] Classes: {class_names}")

    # 3. Chọn top-K features cho CẢ 2 tập
    X_tv_selected: pd.DataFrame = select_top_k_features(
        X_trainval, args.importance_file, args.top_k
    )
    X_test_selected: pd.DataFrame = select_top_k_features(
        X_test, args.importance_file, args.top_k
    )

    # 4. Tạo 8 mô hình
    models: list = build_models()

    # 4.5. Wrap models trong Pipeline với StandardScaler nếu --scale true
    use_scale: bool = args.scale.lower() == "true"
    if use_scale:
        print("[INFO] StandardScaler ENABLED — features sẽ được chuẩn hóa (mean=0, std=1).")
        models = [
            (name, Pipeline([("scaler", StandardScaler()), ("clf", model)]))
            for name, model in models
        ]
    else:
        print("[INFO] StandardScaler DISABLED — dữ liệu giữ nguyên scale.")

    # 5. Huấn luyện & Đánh giá Hold-out
    print(f"\n{'#' * 60}")
    print(f"  HOLD-OUT | Top {args.top_k} Features | Scale={use_scale}")
    print(f"  Train: {X_tv_selected.shape[0]} samples | Test: {X_test_selected.shape[0]} samples")
    print(f"{'#' * 60}")

    os.makedirs(args.output_dir, exist_ok=True)

    results_df: pd.DataFrame = evaluate_pure_holdout(
        X_tv_selected, y_tv_encoded,
        X_test_selected, y_test_encoded,
        models, class_names, args.output_dir,
    )

    # 6. Lưu kết quả metrics ra CSV
    results_csv: str = os.path.join(
        args.output_dir, f"results_holdout_top{args.top_k}.csv"
    )
    results_df.to_csv(results_csv, index=False)
    print(f"\n[INFO] Đã lưu bảng kết quả → {results_csv}")

    # 7. Tổng kết
    print(f"\n\n{'*' * 65}")
    print("  BẢNG TỔNG KẾT: HOLD-OUT (TEST ẨN)")
    print(f"{'*' * 65}")
    print(results_df.to_string(index=False))
    print(f"{'*' * 65}\n")


if __name__ == "__main__":
    main()
