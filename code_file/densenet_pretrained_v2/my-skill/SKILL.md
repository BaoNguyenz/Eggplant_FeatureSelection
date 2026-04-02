---
name: ai-cv-engineer
description: >
  Bộ kỹ năng chuyên sâu cho AI/Computer Vision Engineer. Sử dụng khi cần
  thiết kế pipeline huấn luyện (training), tối ưu hóa mô hình (optimization),
  xử lý dữ liệu ảnh, đánh giá hiệu suất, và triển khai mô hình deep learning.
---

# AI / Computer Vision Engineer Skill

Bộ skill này cung cấp hướng dẫn chi tiết cho agent khi thực hiện các tác vụ liên quan đến **huấn luyện mô hình deep learning** (đặc biệt là image classification), **tối ưu hóa hiệu suất**, và **đánh giá kết quả** trong lĩnh vực Computer Vision.

---

## When to use this skill

- Khi cần **thiết kế pipeline huấn luyện** (training pipeline) cho bài toán phân loại ảnh, object detection, hoặc segmentation.
- Khi cần **fine-tune** một pretrained model (DenseNet, ResNet, EfficientNet, ViT, ...).
- Khi gặp vấn đề **mất cân bằng dữ liệu** (class imbalance) và cần chiến lược xử lý.
- Khi cần **tối ưu hyperparameter**, chọn learning rate, scheduler, augmentation.
- Khi cần **đánh giá mô hình** (evaluation metrics, visualization, statistical tests).
- Khi cần **debug** vấn đề training: overfitting, underfitting, memory leak, NaN loss.
- Khi cần **triển khai** mô hình (export ONNX, TorchScript, inference optimization).

---

## How to use it

### 1. Thiết kế Training Pipeline

Luôn tuân thủ cấu trúc **Modular** khi xây dựng pipeline:

```
project/
├── dataset.py      # Data loading, augmentation, sampling
├── model.py        # Model architecture, freeze/unfreeze
├── train.py        # Training loop, optimizer, scheduler
├── evaluate.py     # Inference, metrics, visualization
└── utils.py        # Helpers: seed, logging, plotting
```

**Nguyên tắc bắt buộc:**
- Luôn đặt `seed_everything()` ở đầu script để đảm bảo **reproducibility**.
- Tách rõ tập dữ liệu thành **Train / Val / Test** — không bao giờ để dữ liệu test rò rỉ vào train.
- Lưu toàn bộ hyperparameters ra file JSON để truy vết thực nghiệm.
- Dùng `argparse` cho mọi tham số có thể thay đổi.

---

### 2. Data Loading & Augmentation

#### Thư viện ưu tiên
- **Albumentations** (ưu tiên hàng đầu) — nhanh hơn torchvision transforms 3-5x, hỗ trợ CLAHE, GridDistortion, CoarseDropout.
- **torchvision.transforms** — dùng khi project đơn giản hoặc không cài được albumentations.

#### Augmentation theo tình huống

| Bài toán | Augmentation khuyến nghị |
|----------|-------------------------|
| Bệnh lá cây | CLAHE, HueSaturationValue, RandomBrightnessContrast, GridDistortion |
| Y tế (X-ray, MRI) | CLAHE, ElasticTransform, ShiftScaleRotate, GaussianNoise |
| Tổng quát | RandomResizedCrop, HorizontalFlip, ColorJitter, GaussianBlur |
| Dữ liệu ít (<500 ảnh) | Thêm MixUp, CutMix, hoặc Mosaic augmentation |

#### Template augmentation cho Train:
```python
A.Compose([
    A.RandomResizedCrop(size=(224, 224), scale=(0.7, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomRotate90(p=0.3),
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.4),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.CLAHE(clip_limit=2.0, p=0.3),
    A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(8, 32), hole_width_range=(8, 32), p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])
```

#### Template augmentation cho Val/Test:
```python
A.Compose([
    A.Resize(256, 256),
    A.CenterCrop(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])
```

---

### 3. Xử lý Class Imbalance

Khi dữ liệu mất cân bằng (ví dụ: class_counts = [1451, 546, 602, 1362, 63, 65]), áp dụng **một hoặc kết hợp** các chiến lược sau:

#### Chiến lược 1: WeightedRandomSampler (Oversampling)
```python
# Công thức: w_c = N_total / (num_classes * count_c)
class_weights = total / (num_classes * class_counts)
sample_weights = [class_weights[label] for label in labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(labels), replacement=True)
```
- **Khi nào dùng:** Khi tỉ lệ imbalance > 5:1.
- **Lưu ý:** Khi dùng sampler, KHÔNG dùng `shuffle=True` trong DataLoader.

#### Chiến lược 2: Weighted Loss Function
```python
# Normalize weights sao cho sum = num_classes
weights = weights / weights.sum() * num_classes
criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weights).to(device))
```
- **Khi nào dùng:** Luôn nên dùng khi có imbalance.

#### Chiến lược 3: Label Smoothing
```python
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
```
- **Khi nào dùng:** Khi model overfit trên class đa số.

#### Lưu ý quan trọng:
> Nếu dùng **cả WeightedRandomSampler lẫn Weighted Loss** cùng lúc (Dual Imbalance Handling),
> có thể xảy ra **over-compensation** — model thiên vị quá mức cho class thiểu số.
> Theo dõi per-class F1 để phát hiện và điều chỉnh.

---

### 4. Kiến trúc Model & Fine-tuning

#### Chọn Pretrained Model

| Model | Params | Accuracy (ImageNet) | Khi nào dùng |
|-------|--------|--------------------|-|
| DenseNet121 | 8M | 74.4% | Dataset nhỏ-vừa, cần feature reuse |
| ResNet50 | 25.6M | 76.1% | Baseline mạnh, phổ biến |
| EfficientNet-B0/B3 | 5.3M/12M | 77.1%/81.6% | Cần balance accuracy vs speed |
| ConvNeXt-Tiny | 28.6M | 82.1% | Modern CNN, hiệu suất cao |
| ViT-B/16 | 86M | 81.1% | Dataset lớn (>10K images) |

#### Custom Classifier Head (template)
```python
model.classifier = nn.Sequential(
    nn.Linear(in_features, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.4),
    nn.Linear(512, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.4),
    nn.Linear(256, num_classes),
)
```

#### Discriminative Learning Rate
Chia model thành các nhóm layers với LR khác nhau — layers gần input giữ LR rất nhỏ (giữ low-level features), layers sâu hơn và classifier head dùng LR lớn hơn:

```
init layers    : lr = 1e-6,  weight_decay = 1e-4
denseblock1-2  : lr = 1e-6 → 5e-6,  weight_decay = 1e-4
denseblock3-4  : lr = 1e-5 → 5e-5,  weight_decay = 1e-2
classifier     : lr = 1e-3 → 1e-4,  weight_decay = 1e-2
```

---

### 5. Chiến lược Huấn luyện 2 Giai đoạn

#### Giai đoạn 1 — Warmup (5-10 epoch đầu)
- **Freeze** toàn bộ backbone, chỉ train classifier head.
- Dùng AdamW với LR cao cho classifier (1e-3).
- Không dùng scheduler.
- Mục đích: Để classifier hội tụ trước khi mở backbone.

#### Giai đoạn 2 — Fine-tune (các epoch còn lại)
- **Unfreeze** toàn bộ model.
- Áp dụng Discriminative LR.
- Dùng scheduler:
  - `CosineAnnealingLR` — ổn định, dễ config.
  - `OneCycleLR` — hội tụ nhanh hơn, step per-batch.

#### Cách chọn số Warmup Epochs
| Tổng Epochs | Warmup Epochs | Lý do |
|-------------|---------------|-------|
| 30-40       | 5             | Đủ để classifier ổn định |
| 50-100      | 10            | Ngân sách epochs lớn, cho phép warmup lâu hơn |
| 100+        | 10-15         | Không nên vượt quá 15% tổng epochs |

---

### 6. Optimizer & Scheduler

#### Optimizer khuyến nghị: AdamW
```python
optimizer = torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=1e-2)
```
- **AdamW** luôn tốt hơn Adam thuần túy nhờ decoupled weight decay.
- Với dataset nhỏ, `weight_decay=1e-2` là giá trị tốt.

#### Scheduler phổ biến

| Scheduler | Đặc điểm | Config |
|-----------|----------|--------|
| CosineAnnealingLR | LR giảm dần theo cosine | `T_max=fine_tune_epochs, eta_min=1e-6` |
| OneCycleLR | LR tăng rồi giảm, step per-batch | `pct_start=0.3, div_factor=10` |
| ReduceLROnPlateau | Giảm LR khi metric không cải thiện | `patience=5, factor=0.5` |

---

### 7. Early Stopping

#### Cách đặt tham số

| Parameter | Hướng dẫn |
|-----------|-----------|
| `patience` | = 10-20% tổng epochs. VD: 100 epochs → patience=10~15 |
| `min_delta` | = 10x nhỏ hơn bước cải thiện kỳ vọng. VD: F1 thường nhích ~0.001 → min_delta=1e-4 |
| `mode` | `"max"` cho F1/Accuracy, `"min"` cho Loss |
| `warmup_epochs` | = Số epoch của Stage 1 — KHÔNG bao giờ early stop trong warmup |

#### Quy tắc ngón tay cái cho `min_delta`:
- LR = 1e-3 → `min_delta = 1e-4` (bước cải thiện lớn)
- LR = 1e-4 → `min_delta = 5e-5` (bước cải thiện nhỏ hơn)
- LR = 1e-5 → `min_delta = 1e-5` (micro-improvement)

> **Nguyên tắc:** LR càng nhỏ, model cải thiện càng chậm → `min_delta` phải nhỏ tương ứng,
> nếu không Early Stopping sẽ trigger quá sớm.

---

### 8. Đánh giá Mô hình (Evaluation)

#### Metrics bắt buộc cho Classification
- **Accuracy** — tổng quan nhưng misleading khi imbalance.
- **Macro F1** — metric chính, cân bằng giữa các class.
- **Weighted F1** — phản ánh hiệu suất thực tế theo tỉ lệ dữ liệu.
- **Per-class Precision/Recall** — để phát hiện class yếu.
- **Confusion Matrix** — trực quan hóa lỗi phân loại.
- **ROC/AUC (One-vs-Rest)** — đánh giá khả năng tách class.

#### Khi nào model tốt (cho bài toán 6 class bệnh lá)?
| Metric | Tốt | Khá | Cần cải thiện |
|--------|-----|-----|---------------|
| Macro F1 | > 0.90 | 0.80-0.90 | < 0.80 |
| Macro AUC | > 0.95 | 0.90-0.95 | < 0.90 |
| Accuracy | > 0.92 | 0.85-0.92 | < 0.85 |

---

### 9. Debug & Troubleshooting

#### Overfitting (Train Acc cao, Val Acc thấp)
1. Tăng Dropout (0.3 → 0.5).
2. Thêm augmentation mạnh hơn (GridDistortion, CoarseDropout).
3. Tăng weight_decay (1e-2 → 5e-2).
4. Dùng Label Smoothing (0.05 → 0.15).
5. Giảm model complexity (ít layer hơn trong classifier head).

#### Underfitting (Cả Train và Val Acc đều thấp)
1. Tăng LR (1e-4 → 1e-3).
2. Giảm regularization (Dropout, weight_decay).
3. Tăng model capacity (head lớn hơn, hoặc dùng model lớn hơn).
4. Train nhiều epoch hơn.

#### RAM Leak trên Windows
1. Đặt `cv2.setNumThreads(0)` và `cv2.ocl.setUseOpenCL(False)` ở đầu dataset.py.
2. Dùng `persistent_workers=True` trong DataLoader.
3. Thêm `del images, labels, outputs, loss` trong training loop.
4. Gọi `gc.collect()` + `torch.cuda.empty_cache()` cuối mỗi epoch.
5. Nếu vẫn leak: dùng `num_workers=0`.

#### NaN Loss
1. Giảm LR (quá cao gây explosion).
2. Thêm gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`.
3. Kiểm tra dữ liệu có ảnh lỗi (corrupt image).

#### CUDA Out of Memory
1. Giảm `batch_size`.
2. Dùng `torch.cuda.amp` (Mixed Precision Training).
3. Dùng gradient accumulation thay vì tăng batch_size.

---

### 10. Best Practices Checklist

Trước khi submit kết quả training, kiểm tra:

- [ ] Seed đã được đặt cố định (42).
- [ ] Train/Val/Test chia đúng, không data leakage.
- [ ] Augmentation chỉ áp dụng cho Train, KHÔNG cho Val/Test.
- [ ] Best model được lưu theo Val F1 (không phải Train F1).
- [ ] Hyperparameters được log ra file JSON.
- [ ] Classification Report, Confusion Matrix, ROC/AUC đã được xuất.
- [ ] Không có memory leak (RAM không tăng liên tục qua epochs).
- [ ] Code tuân thủ PEP-8, có docstrings và comments rõ ràng.

---

### 11. Lệnh Chạy Tham khảo

```bash
# Training cơ bản (40 epochs)
python train.py --epochs 40 --lr 1e-3 --batch_size 32

# Training dài (100 epochs) với early stopping
python train.py --epochs 100 --warmup_epochs 10 --lr 1e-3 --patience 15

# Training với LR thấp (1e-4), cần giảm es_min_delta
python train.py --epochs 100 --lr 1e-4 --patience 15 --es_min_delta 5e-5

# Multi-GPU
python train.py --epochs 100 --batch_size 64 --gpu_ids 0 1

# Evaluation
python evaluate.py --checkpoint "path/to/best_model.pth"

# Quick test (2 epochs, kiểm tra pipeline)
python train.py --epochs 2 --warmup_epochs 1 --batch_size 8 --num_workers 0
```
