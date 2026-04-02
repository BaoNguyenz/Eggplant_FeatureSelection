"""
step2_model_training.py — Huấn luyện và đánh giá 8 mô hình ML trên top-K
features đã chọn từ bước Feature Importance.

Mô hình:
  1. KNN
  2. SVM
  3. Random Forest
  4. Logistic Regression
  5. Extra Trees
  6. XGBoost          (GPU — tree_method='gpu_hist')
  7. CatBoost         (GPU — task_type='GPU')
  8. LightGBM         (GPU — device='gpu')

Usage:
  python step2_model_training.py \
      --data_dir       data/densenet121_features.csv \
      --importance_file importance_scores.csv \
      --top_k           256 \
      --eval_method     holdout

  python step2_model_training.py \
      --data_dir       data/densenet121_features.csv \
      --importance_file importance_scores.csv \
      --top_k           256 \
      --eval_method     cv
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    cross_val_predict,
    train_test_split,
)
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
    """Cắt dữ liệu chỉ giữ lại top-K features quan trọng nhất.

    Parameters
    ----------
    X : pd.DataFrame
        Ma trận đặc trưng gốc.
    importance_file : str
        Đường dẫn tới ``importance_scores.csv`` (output của step 1).
    top_k : int
        Số lượng features muốn giữ lại.

    Returns
    -------
    pd.DataFrame
        Ma trận đặc trưng chỉ chứa top-K features.
    """
    importance_df: pd.DataFrame = pd.read_csv(importance_file)
    top_features: list[str] = importance_df["feature"].head(top_k).tolist()

    # Kiểm tra tính hợp lệ
    missing: set[str] = set(top_features) - set(X.columns)
    if missing:
        raise ValueError(
            f"Các feature sau không tìm thấy trong dữ liệu: {missing}"
        )

    X_selected: pd.DataFrame = X[top_features]
    print(f"[INFO] Đã chọn top {top_k} features từ '{importance_file}'.")
    return X_selected


# ---------------------------------------------------------------------------
# 2. Định nghĩa 8 mô hình
# ---------------------------------------------------------------------------
def build_models(random_state: int = 42) -> list[tuple[str, Any]]:
    """Tạo danh sách 8 mô hình với cấu hình tối ưu (bao gồm GPU).

    Returns
    -------
    list[tuple[str, Any]]
        Danh sách dạng ``(tên, estimator)`` theo thứ tự cố định.
    """
    models: list[tuple[str, Any]] = [
        # --- CPU-based models ---
        (
            "KNN",
            KNeighborsClassifier(
                n_neighbors=5,
                weights="distance",
                n_jobs=-1,
            ),
        ),
        (
            "SVM",
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                random_state=random_state,
            ),
        ),
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=300,
                random_state=random_state,
                n_jobs=-1,
                class_weight="balanced",
            ),
        ),
        (
            "Logistic Regression",
            LogisticRegression(
                max_iter=2000,
                random_state=random_state,
                n_jobs=-1,
                class_weight="balanced",
                solver="lbfgs",
                multi_class="multinomial",
            ),
        ),
        (
            "Extra Trees",
            ExtraTreesClassifier(
                n_estimators=300,
                random_state=random_state,
                n_jobs=-1,
                class_weight="balanced",
            ),
        ),
        # --- GPU-accelerated models ---
        (
            "XGBoost",
            XGBClassifier(
                n_estimators=300,
                learning_rate=0.1,
                max_depth=6,
                # ===== Cấu hình GPU cho XGBoost (>= 3.1) =====
                device="cuda",           # Sử dụng GPU CUDA (thay thế gpu_hist + gpu_id)
                # ==============================================
                eval_metric="mlogloss",
                random_state=random_state,
                verbosity=0,
            ),
        ),
        (
            "CatBoost",
            CatBoostClassifier(
                iterations=300,
                learning_rate=0.1,
                depth=6,
                # ===== Cấu hình GPU cho CatBoost =====
                task_type="GPU",         # Chạy trên GPU
                devices="0",            # ID card GPU
                # ======================================
                random_seed=random_state,
                verbose=0,
                auto_class_weights="Balanced",
            ),
        ),
        (
            "LightGBM",
            LGBMClassifier(
                n_estimators=300,
                learning_rate=0.1,
                max_depth=6,
                # ===== Cấu hình GPU cho LightGBM =====
                # device="gpu",            # Sử dụng GPU
                # gpu_platform_id=0,
                # gpu_device_id=0,
                # ======================================
                random_state=random_state,
                class_weight="balanced",
                verbose=-1,
            ),
        ),
    ]
    return models


# ---------------------------------------------------------------------------
# 3. Đánh giá: Holdout 80/20
# ---------------------------------------------------------------------------
def evaluate_holdout(
    X: pd.DataFrame,
    y: pd.Series,
    models: list[tuple[str, Any]],
    class_names: list[str],
    output_dir: str,
    random_state: int = 42,
) -> pd.DataFrame:
    """Chia 80% train / 20% test, huấn luyện và đánh giá từng mô hình.

    Parameters
    ----------
    X, y : dữ liệu đã chọn top-K.
    models : danh sách (tên, estimator).
    class_names : tên các lớp.
    output_dir : thư mục lưu kết quả.
    random_state : seed cho train_test_split.

    Returns
    -------
    pd.DataFrame
        Bảng kết quả 4 metrics cho 8 mô hình.
    """
    # ===== Chia dữ liệu — Stratified để giữ tỷ lệ lớp =====
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=random_state, stratify=y,
    )
    print(
        f"[INFO] Holdout split: "
        f"Train={X_train.shape[0]}, Test={X_test.shape[0]}"
    )

    cm_dir: str = os.path.join(output_dir, "confusion_matrices")
    results: list[dict[str, Any]] = []

    for name, model in models:
        print(f"\n>>> Đang huấn luyện: {name} ...")
        start: float = time.time()

        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        except Exception as e:
            # ===== Fallback: nếu GPU lỗi, thử lại với CPU =====
            print(f"[WARNING] {name} GPU failed: {e}")
            print(f"[INFO] Đang thử lại {name} trên CPU ...")
            from sklearn.base import clone
            model_cpu = clone(model)
            # Tắt GPU cho từng loại model
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

        # Lưu Confusion Matrix cho từng mô hình
        save_confusion_matrix(y_test, y_pred, class_names, name, cm_dir)

    return pd.DataFrame(results)[
        ["Model", "Precision", "Recall", "F1-Score", "Accuracy", "Time (s)"]
    ]


# ---------------------------------------------------------------------------
# 4. Đánh giá: Cross-Validation 5-fold
# ---------------------------------------------------------------------------
def evaluate_cv(
    X: pd.DataFrame,
    y: pd.Series,
    models: list[tuple[str, Any]],
    class_names: list[str],
    output_dir: str,
    n_folds: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Đánh giá bằng Stratified K-Fold Cross-Validation.

    Parameters
    ----------
    X, y : dữ liệu đã chọn top-K.
    models : danh sách (tên, estimator).
    class_names : tên các lớp.
    output_dir : thư mục lưu kết quả.
    n_folds : số fold.
    random_state : seed cho StratifiedKFold.

    Returns
    -------
    pd.DataFrame
        Bảng kết quả trung bình ± std cho 4 metrics.
    """
    # ===== Stratified K-Fold để đảm bảo phân phối lớp đồng đều =====
    skf = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=random_state,
    )

    scoring: dict[str, str] = {
        "Accuracy": "accuracy",
        "Precision": "precision_weighted",
        "Recall": "recall_weighted",
        "F1-Score": "f1_weighted",
    }

    cm_dir: str = os.path.join(output_dir, "confusion_matrices")
    results: list[dict[str, Any]] = []

    for name, model in models:
        print(f"\n>>> Đang cross-validate: {name} ({n_folds}-fold) ...")
        start: float = time.time()

        try:
            cv_results = cross_validate(
                model, X, y, cv=skf, scoring=scoring, n_jobs=1,
            )
            # Dự đoán trên tất cả folds để tạo Confusion Matrix
            y_pred_cv: np.ndarray = cross_val_predict(
                model, X, y, cv=skf, n_jobs=1,
            )
        except Exception as e:
            # ===== Fallback: nếu GPU lỗi, thử lại với CPU =====
            print(f"[WARNING] {name} GPU failed: {e}")
            print(f"[INFO] Đang thử lại {name} trên CPU ...")
            from sklearn.base import clone
            model_cpu = clone(model)
            if hasattr(model_cpu, "device"):
                model_cpu.set_params(device="cpu")
            if hasattr(model_cpu, "task_type"):
                model_cpu.set_params(task_type="CPU")
            cv_results = cross_validate(
                model_cpu, X, y, cv=skf, scoring=scoring, n_jobs=1,
            )
            y_pred_cv = cross_val_predict(
                model_cpu, X, y, cv=skf, n_jobs=1,
            )

        elapsed: float = time.time() - start

        row: dict[str, Any] = {"Model": name}
        print(f"\n{'=' * 55}")
        print(f"  {name}  ({n_folds}-Fold CV)")
        print(f"{'=' * 55}")
        for metric_name in scoring:
            key: str = f"test_{metric_name}"
            mean_val: float = round(cv_results[key].mean() * 100, 2)
            row[metric_name] = mean_val
            print(f"  {metric_name:<12}: {mean_val:.2f}%")
        print(f"  {'Time (s)':<12}: {elapsed:.2f}")
        print(f"{'=' * 55}")

        row["Time (s)"] = round(elapsed, 2)
        results.append(row)

        # Lưu Confusion Matrix (tổng hợp từ tất cả folds)
        save_confusion_matrix(y, y_pred_cv, class_names, name, cm_dir)

    return pd.DataFrame(results)[
        ["Model", "Precision", "Recall", "F1-Score", "Accuracy", "Time (s)"]
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 2: Học Máy chuẩn Fair (CV trên TrainVal -> Retrain -> Đánh giá trên Test ẩn)."
    )
    parser.add_argument(
        "--data_trainval", type=str,
        default=r"E:\PROJECTWORSHOP\Eggplant_regconition\code_file\A_feature_selection\csv_data\features_trainval.csv",
        help="CSV features của tập TrainVal.",
    )
    parser.add_argument(
        "--data_test", type=str,
        default=r"E:\PROJECTWORSHOP\Eggplant_regconition\code_file\A_feature_selection\csv_data\features_test.csv",
        help="CSV features của tập Test ẩn.",
    )
    parser.add_argument(
        "--importance_file", type=str,
        default=r"E:\PROJECTWORSHOP\Eggplant_regconition\code_file\A_feature_selection\output_csv\importance_scores.csv",
        help="Đường dẫn tới file importance_scores.csv",
    )
    parser.add_argument(
        "--top_k", type=int, required=True,
        help="Số lượng features muốn giữ lại.",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=r"E:\PROJECTWORSHOP\Eggplant_regconition\code_file\A_feature_selection\output_csv",
        help="Thư mục lưu kết quả CSV",
    )
    parser.add_argument(
        "--scale", type=str, choices=["true", "false"], default="true",
        help="Chuẩn hóa features bằng StandardScaler.",
    )
    return parser.parse_args()


def main() -> None:
    args: argparse.Namespace = parse_args()

    # 1. Đọc 2 dữ liệu
    X_trainval, y_trainval = load_data(args.data_trainval)
    X_test, y_test = load_data(args.data_test)

    # 2. Encode nhãn
    le = LabelEncoder()
    y_tv_encoded: np.ndarray = le.fit_transform(y_trainval)
    y_test_encoded: np.ndarray = le.transform(y_test)
    class_names: list[str] = le.classes_.tolist()
    print(f"[INFO] Classes: {class_names}")

    # 3. Chọn top-K features cho CẢ 2 tập
    X_tv_selected: pd.DataFrame = select_top_k_features(X_trainval, args.importance_file, args.top_k)
    X_test_selected: pd.DataFrame = select_top_k_features(X_test, args.importance_file, args.top_k)

    # 4. Tạo mô hình (Kèm Scale nếu có)
    models: list = build_models()
    use_scale: bool = args.scale.lower() == "true"
    if use_scale:
        models = [(name, Pipeline([("scaler", StandardScaler()), ("clf", model)])) for name, model in models]

    # 5. Pipeline Đánh Giá Chuẩn Học Thuật
    print(f"\n{'#' * 60}")
    print(f"  BƯỚC 1: CROSS-VALIDATION (5 Folds) TRÊN TRAINVAL")
    print(f"{'#' * 60}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 5.1 CV Eval
    cv_results_df = evaluate_cv(X_tv_selected, y_tv_encoded, models, class_names, args.output_dir)
    
    print(f"\n{'#' * 60}")
    print(f"  BƯỚC 2: RETRAIN TRÊN 100% TRAINVAL & ĐÁNH GIÁ TRÊN TEST ẨN")
    print(f"{'#' * 60}")
    
    # 5.2 Holdout Eval (custom logic: train on tv, evaluate on test)
    cm_dir = os.path.join(args.output_dir, "confusion_matrices")
    holdout_results = []
    
    for name, model in models:
        start: float = time.time()
        try:
            model.fit(X_tv_selected, y_tv_encoded)
            y_pred = model.predict(X_test_selected)
        except Exception as e:
            print(f"[ERROR] GPU fail: {e}, fallback CPU...")
            from sklearn.base import clone
            model_cpu = clone(model)
            if hasattr(model_cpu, "device"): model_cpu.set_params(device="cpu")
            if hasattr(model_cpu, "task_type"): model_cpu.set_params(task_type="CPU")
            model_cpu.fit(X_tv_selected, y_tv_encoded)
            y_pred = model_cpu.predict(X_test_selected)
            
        elapsed: float = time.time() - start
        metrics = evaluate_model(y_test_encoded, y_pred, name)
        metrics["Model"] = name
        metrics["Time (s)"] = round(elapsed, 2)
        holdout_results.append(metrics)
        save_confusion_matrix(y_test_encoded, y_pred, class_names, name, cm_dir)

    holdout_df = pd.DataFrame(holdout_results)[["Model", "Precision", "Recall", "F1-Score", "Accuracy", "Time (s)"]]
    
    # Lưu CSV
    cv_csv = os.path.join(args.output_dir, f"results_CV_top{args.top_k}.csv")
    holdout_csv = os.path.join(args.output_dir, f"results_HOLDOUT_TEST_top{args.top_k}.csv")
    cv_results_df.to_csv(cv_csv, index=False)
    holdout_df.to_csv(holdout_csv, index=False)
    
    print(f"\n\n{'*' * 65}")
    print("  BẢNG TỔNG KẾT: CROSS-VALIDATION (TRAINVAL)")
    print(f"{'*' * 65}")
    print(cv_results_df.to_string(index=False))
    
    print(f"\n\n{'*' * 65}")
    print("  BẢNG TỔNG KẾT: HOLDOUT (TEST ẨN)")
    print(f"{'*' * 65}")
    print(holdout_df.to_string(index=False))


if __name__ == "__main__":
    main()
