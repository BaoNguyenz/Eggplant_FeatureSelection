"""
Data Setup for DenseNet121 Training Pipeline
=============================================
Load dữ liệu từ thư mục Fold vật lý (train_val / test).
Chia train_val thành Train (8 phần) và Val (1 phần) bằng stratified split.
Test set lấy nguyên từ thư mục test vật lý.

Cấu trúc thư mục Fold kỳ vọng:
    Fold_X/
    ├── train_val/
    │   ├── class_1/
    │   ├── class_2/
    │   └── ...
    └── test/
        ├── class_1/
        ├── class_2/
        └── ...
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np


class EggplantDataset(Dataset):
    """
    Custom Dataset cho ảnh lá cà tím (Eggplant Leaf Disease Detection).
    
    Args:
        image_paths (list): Danh sách đường dẫn đến các ảnh
        labels (list): Danh sách nhãn tương ứng
        transform (torchvision.transforms): Các phép biến đổi ảnh
    """
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load ảnh
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        # Áp dụng transform
        if self.transform:
            image = self.transform(image)
        
        return image, label


class AddGaussianNoise:
    """
    Custom transform để thêm Gaussian Noise vào ảnh với cường độ ngẫu nhiên.
    Theo bảng augmentation: scale=(0, 0.05 * 255) → std random trong [0, 0.05] sau khi ToTensor().
    
    Args:
        mean (float): Giá trị trung bình của noise
        std_range (tuple): Range của std để random (min_std, max_std)
    """
    def __init__(self, mean=0.0, std_range=(0.0, 0.05)):
        self.mean = mean
        self.std_range = std_range
    
    def __call__(self, tensor):
        # Random std từ range mỗi lần gọi
        std = torch.empty(1).uniform_(self.std_range[0], self.std_range[1]).item()
        noise = torch.randn_like(tensor) * std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)
    
    def __repr__(self):
        return f'{self.__class__.__name__}(mean={self.mean}, std_range={self.std_range})'


def _scan_images(directory):
    """
    Quét tất cả ảnh trong thư mục có cấu trúc class subfolder.
    
    Args:
        directory (Path): Thư mục gốc chứa các subfolder theo tên class
    
    Returns:
        tuple: (image_paths, labels, class_names)
            - image_paths: list[Path]
            - labels: list[int]  (chỉ số class)
            - class_names: list[str]
    """
    directory = Path(directory)
    class_dirs = sorted([d for d in directory.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_dirs]
    
    image_paths = []
    labels = []
    
    for class_idx, class_dir in enumerate(class_dirs):
        image_files = (
            list(class_dir.glob('*.jpg')) +
            list(class_dir.glob('*.jpeg')) +
            list(class_dir.glob('*.png'))
        )
        for img_path in image_files:
            image_paths.append(img_path)
            labels.append(class_idx)
    
    return image_paths, labels, class_names


def create_dataloaders(data_dir, batch_size=32, num_workers=4, seed=42):
    """
    Tạo DataLoader cho train/val/test từ thư mục Fold vật lý.
    
    Cấu trúc thư mục kỳ vọng:
        data_dir/              (VD: Fold_1)
        ├── train_val/         (90% dữ liệu gốc)
        │   ├── class_1/
        │   └── class_2/
        └── test/              (10% dữ liệu gốc)
            ├── class_1/
            └── class_2/
    
    Logic chia:
        - test  → lấy nguyên thư mục test (10% tổng)
        - train_val → chia stratified 8/9 Train + 1/9 Val
          → Train ≈ 80% tổng, Val ≈ 10% tổng
    
    Args:
        data_dir (str or Path): Đường dẫn Fold (chứa train_val/ và test/)
        batch_size (int): Kích thước batch
        num_workers (int): Số worker để load dữ liệu
        seed (int): Random seed cho reproducibility
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, class_names, all_labels)
    """
    data_dir = Path(data_dir)
    
    # --- Xác định đường dẫn train_val và test ---
    train_val_dir = data_dir / 'train_val'
    test_dir = data_dir / 'test'
    
    if not train_val_dir.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục 'train_val' trong {data_dir}. "
            f"Hãy chắc chắn data_dir trỏ tới 1 thư mục Fold (VD: Fold_1)."
        )
    if not test_dir.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục 'test' trong {data_dir}. "
            f"Hãy chắc chắn data_dir trỏ tới 1 thư mục Fold (VD: Fold_1)."
        )
    
    # --- 1. Quét ảnh từ train_val ---
    tv_paths, tv_labels, class_names_tv = _scan_images(train_val_dir)
    
    # --- 2. Quét ảnh từ test ---
    test_paths, test_labels, class_names_test = _scan_images(test_dir)
    
    # Kiểm tra class nhất quán
    if class_names_tv != class_names_test:
        raise ValueError(
            f"Danh sách class không khớp giữa train_val và test!\n"
            f"  train_val classes: {class_names_tv}\n"
            f"  test classes:      {class_names_test}"
        )
    
    class_names = class_names_tv
    num_classes = len(class_names)
    
    print(f"\n[INFO] Fold directory: {data_dir}")
    print(f"[INFO] Found {num_classes} classes: {class_names}")
    
    for cls_idx, cls_name in enumerate(class_names):
        n_tv = sum(1 for l in tv_labels if l == cls_idx)
        n_test = sum(1 for l in test_labels if l == cls_idx)
        print(f"  Class '{cls_name}': train_val={n_tv}, test={n_test}")
    
    # --- 3. Chia train_val thành Train (8/9) và Val (1/9) ---
    #   Tổng thể: Train=80%, Val=10%, Test=10%
    tv_paths = np.array(tv_paths)
    tv_labels = np.array(tv_labels)
    
    val_ratio = 1.0 / 9.0  # 1/9 của train_val → 10% tổng
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        tv_paths, tv_labels,
        test_size=val_ratio,
        stratify=tv_labels,
        random_state=seed
    )
    
    test_paths = np.array(test_paths)
    test_labels = np.array(test_labels)
    
    total_images = len(train_paths) + len(val_paths) + len(test_paths)
    
    print(f"\n[INFO] Dataset split (Train ≈ 80% / Val ≈ 10% / Test = 10%):")
    print(f"  Train: {len(train_paths)} images ({len(train_paths)/total_images*100:.1f}%)")
    print(f"  Val:   {len(val_paths)} images ({len(val_paths)/total_images*100:.1f}%)")
    print(f"  Test:  {len(test_paths)} images ({len(test_paths)/total_images*100:.1f}%)")
    print(f"  Total: {total_images} images")
    
    # Gom tất cả nhãn lại cho tính class weights
    all_labels = np.concatenate([train_labels, val_labels, test_labels]).tolist()
    
    # --- 4. Định nghĩa transforms ---
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    image_size = 224
    
    # Train transform: Strong augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=25),
        transforms.ColorJitter(brightness=(0.7, 1.3), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=0.15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        AddGaussianNoise(mean=0, std_range=(0.0, 0.08)),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    # Val/Test transform: chỉ resize và normalize
    val_test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    # --- 5. Tạo Datasets ---
    train_dataset = EggplantDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = EggplantDataset(val_paths, val_labels, transform=val_test_transform)
    test_dataset = EggplantDataset(test_paths, test_labels, transform=val_test_transform)
    
    # --- 6. Tạo DataLoaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, class_names, all_labels


if __name__ == "__main__":
    """
    Unit test cho data_setup.py — Kiểm tra đọc dữ liệu từ thư mục Fold.
    """
    # Trỏ vào Fold cụ thể để test
    fold_dir = r"E:\PROJECTWORSHOP\Eggplant_regconition\10_Fold_CV_Dataset\Fold_1"
    
    print("\n" + "=" * 70)
    print("TEST: Load data từ thư mục Fold vật lý")
    print("=" * 70)
    try:
        train_loader, val_loader, test_loader, class_names, all_labels = create_dataloaders(
            data_dir=fold_dir,
            batch_size=16,
            num_workers=0,
            seed=42
        )
        
        print(f"\n✓ DataLoaders created successfully!")
        print(f"  Classes: {class_names}")
        print(f"  train_loader batches: {len(train_loader)}")
        print(f"  val_loader batches:   {len(val_loader)}")
        print(f"  test_loader batches:  {len(test_loader)}")
        
        # Kiểm tra 1 batch
        images, labels = next(iter(train_loader))
        print(f"  Batch shape: {images.shape}, Labels shape: {labels.shape}")
        print("✅ TEST PASSED!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
    
    print("\n" + "=" * 70)
    print("All data_setup.py tests done.")
    print("=" * 70)
