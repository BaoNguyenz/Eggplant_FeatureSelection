"""
Dataset & DataLoader Module for DenseNet121 Training Pipeline
==============================================================
- Custom Dataset tương thích với Albumentations.
- Stratified split: train_val/ → Train (8/9) + Val (1/9).
- Test set giữ nguyên từ thư mục test/.
- WeightedRandomSampler để oversample class thiểu số.
- Augmentation chuyên sâu với Albumentations (CLAHE, HSV, v.v.).

Cấu trúc thư mục kỳ vọng:
    Fold_X/
    ├── train_val/
    │   ├── Healthy Leaf/
    │   ├── Insect_Pest_Disease/
    │   └── ...
    └── test/
        ├── Healthy Leaf/
        └── ...
"""

import cv2
# Vô hiệu hóa đa luồng ngầm của OpenCV — ngăn rò rỉ RAM
# khi chạy trong worker processes của PyTorch DataLoader trên Windows
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ---------------------------------------------------------------------------
# 1. Custom Dataset (Albumentations-compatible)
# ---------------------------------------------------------------------------
class EggplantDataset(Dataset):
    """
    Dataset đọc ảnh từ danh sách đường dẫn, áp dụng Albumentations transform.

    Args:
        image_paths (list[str | Path]): Đường dẫn tới từng ảnh.
        labels (list[int]): Nhãn tương ứng (class index).
        transform (albumentations.Compose | None): Pipeline augmentation.
    """

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = [str(p) for p in image_paths]
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Đọc ảnh bằng OpenCV (BGR) rồi chuyển sang RGB
        image = cv2.imread(self.image_paths[idx])
        if image is None:
            raise FileNotFoundError(
                f"Không thể đọc ảnh: {self.image_paths[idx]}"
            )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = self.labels[idx]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label


# ---------------------------------------------------------------------------
# 2. Augmentation Pipelines
# ---------------------------------------------------------------------------
def get_train_transforms(image_size: int = 224) -> A.Compose:
    """
    Augmentation mạnh dành cho tập Train.
    Bao gồm CLAHE để làm rõ vết bệnh trên lá.

    Returns:
        albumentations.Compose pipeline.
    """
    return A.Compose([
        A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.7, 1.0),
            ratio=(0.9, 1.1),
            p=1.0
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, p=0.4),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.HueSaturationValue(
            hue_shift_limit=15,
            sat_shift_limit=25,
            val_shift_limit=15,
            p=0.4
        ),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(8, 32),
            hole_width_range=(8, 32),
            fill=0,
            p=0.3,
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_val_test_transforms(image_size: int = 224) -> A.Compose:
    """
    Transform chuẩn cho tập Val / Test: Resize → CenterCrop → Normalize.

    Returns:
        albumentations.Compose pipeline.
    """
    return A.Compose([
        A.Resize(256, 256),
        A.CenterCrop(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
# 3. Utility: Quét ảnh từ thư mục
# ---------------------------------------------------------------------------
def _scan_images(directory):
    """
    Quét tất cả ảnh trong thư mục có cấu trúc class subfolder.

    Args:
        directory (Path): Thư mục gốc chứa các subfolder theo tên class.

    Returns:
        tuple: (image_paths: list[Path], labels: list[int], class_names: list[str])
    """
    directory = Path(directory)
    class_dirs = sorted([d for d in directory.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_dirs]

    image_paths = []
    labels = []

    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    for class_idx, class_dir in enumerate(class_dirs):
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in valid_ext:
                image_paths.append(img_path)
                labels.append(class_idx)

    return image_paths, labels, class_names


# ---------------------------------------------------------------------------
# 4. Tính WeightedRandomSampler
# ---------------------------------------------------------------------------
def _make_weighted_sampler(labels, num_classes):
    """
    Tạo WeightedRandomSampler để oversample class thiểu số trong tập Train.

    Công thức weight mỗi class:
        w_c = N_total / (num_classes * count_c)

    Mỗi sample sẽ được gán weight = w_{label_of_sample}.

    Args:
        labels (list[int]): Nhãn của toàn bộ tập Train.
        num_classes (int): Số lượng class.

    Returns:
        WeightedRandomSampler
    """
    class_counts = np.bincount(labels, minlength=num_classes).astype(float)
    # Tránh chia cho 0
    class_counts = np.maximum(class_counts, 1.0)

    total = float(len(labels))
    class_weights = total / (num_classes * class_counts)

    # Gán weight cho từng sample
    sample_weights = np.array([class_weights[lbl] for lbl in labels])
    sample_weights = torch.from_numpy(sample_weights).double()

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler


# ---------------------------------------------------------------------------
# 5. Tính Class Weights cho Loss Function
# ---------------------------------------------------------------------------
def compute_class_weights(labels, num_classes):
    """
    Tính trọng số cho CrossEntropyLoss(weight=...).
    Công thức: w_c = N / (num_classes * count_c), chuẩn hóa sao cho sum = num_classes.

    Args:
        labels (list[int]): Nhãn của tập Train.
        num_classes (int): Số class.

    Returns:
        torch.FloatTensor: Vector weights có shape (num_classes,).
    """
    class_counts = np.bincount(labels, minlength=num_classes).astype(float)
    class_counts = np.maximum(class_counts, 1.0)

    total = float(len(labels))
    weights = total / (num_classes * class_counts)

    # Normalize: sum(weights) = num_classes
    weights = weights / weights.sum() * num_classes

    return torch.FloatTensor(weights)


# ---------------------------------------------------------------------------
# 6. Main: Tạo DataLoaders
# ---------------------------------------------------------------------------
def create_dataloaders(data_dir, batch_size=32, num_workers=2, seed=42):
    """
    Tạo DataLoader cho Train / Val / Test.

    - train_val/ → chia stratified 8/9 Train + 1/9 Val.
    - test/ → giữ nguyên.
    - Train DataLoader dùng WeightedRandomSampler.

    Args:
        data_dir (str | Path): Đường dẫn Fold (chứa train_val/ và test/).
        batch_size (int): Kích thước batch.
        num_workers (int): Số worker load dữ liệu.
        seed (int): Random seed.

    Returns:
        dict: {
            "train_loader", "val_loader", "test_loader",
            "class_names", "num_classes",
            "train_labels", "class_weights"
        }
    """
    data_dir = Path(data_dir)
    train_val_dir = data_dir / "train_val"
    test_dir = data_dir / "test"

    # Validate thư mục
    for d, name in [(train_val_dir, "train_val"), (test_dir, "test")]:
        if not d.exists():
            raise FileNotFoundError(
                f"Không tìm thấy '{name}' trong {data_dir}."
            )

    # --- Quét ảnh ---
    tv_paths, tv_labels, class_names_tv = _scan_images(train_val_dir)
    test_paths, test_labels, class_names_test = _scan_images(test_dir)

    if class_names_tv != class_names_test:
        raise ValueError(
            f"Class không khớp!\n  train_val: {class_names_tv}\n  test: {class_names_test}"
        )

    class_names = class_names_tv
    num_classes = len(class_names)

    # --- Stratified split: train_val → Train + Val ---
    val_ratio = 1.0 / 9.0  # 1/9 → ~11.1% của train_val
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        tv_paths, tv_labels,
        test_size=val_ratio,
        stratify=tv_labels,
        random_state=seed,
    )

    # --- Logging ---
    total = len(train_paths) + len(val_paths) + len(test_paths)
    print(f"\n{'='*60}")
    print(f"[DATA] Fold: {data_dir.name}")
    print(f"[DATA] Classes ({num_classes}): {class_names}")
    print(f"{'='*60}")
    print(f"  Train : {len(train_paths):>5} images ({len(train_paths)/total*100:.1f}%)")
    print(f"  Val   : {len(val_paths):>5} images ({len(val_paths)/total*100:.1f}%)")
    print(f"  Test  : {len(test_paths):>5} images ({len(test_paths)/total*100:.1f}%)")
    print(f"  Total : {total:>5} images")

    # Log phân phối class trong Train
    train_labels_arr = np.array(train_labels)
    print(f"\n[DATA] Train class distribution:")
    for i, name in enumerate(class_names):
        cnt = int((train_labels_arr == i).sum())
        print(f"  {name:30s}: {cnt:>5}")

    # --- Transforms ---
    train_tf = get_train_transforms(224)
    val_test_tf = get_val_test_transforms(224)

    # --- Datasets ---
    train_dataset = EggplantDataset(train_paths, train_labels, transform=train_tf)
    val_dataset = EggplantDataset(val_paths, val_labels, transform=val_test_tf)
    test_dataset = EggplantDataset(test_paths, test_labels, transform=val_test_tf)

    # --- WeightedRandomSampler cho Train ---
    sampler = _make_weighted_sampler(train_labels, num_classes)

    # --- DataLoaders ---
    # persistent_workers=True: giữ worker processes sống xuyên suốt training,
    # tránh việc tạo/hủy worker mỗi epoch gây rò rỉ RAM trên Windows.
    use_persistent = num_workers > 0

    # Khi có sampler, KHÔNG dùng shuffle
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=use_persistent,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=use_persistent,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=use_persistent,
    )

    # --- Class weights cho Loss ---
    class_weights = compute_class_weights(train_labels, num_classes)
    print(f"\n[DATA] Class weights for loss: {class_weights.tolist()}")

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "class_names": class_names,
        "num_classes": num_classes,
        "train_labels": train_labels,
        "class_weights": class_weights,
    }


# ---------------------------------------------------------------------------
# Quick Test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fold_dir = r"D:\Eggplant_leaf_v2\Fold_1"
    data = create_dataloaders(fold_dir, batch_size=16, num_workers=0, seed=42)

    print(f"\n[TEST] train batches: {len(data['train_loader'])}")
    print(f"[TEST] val batches:   {len(data['val_loader'])}")
    print(f"[TEST] test batches:  {len(data['test_loader'])}")

    images, labels = next(iter(data["train_loader"]))
    print(f"[TEST] Batch shape: {images.shape}, Labels: {labels.shape}")
    print("[TEST] ✅ dataset.py OK!")
