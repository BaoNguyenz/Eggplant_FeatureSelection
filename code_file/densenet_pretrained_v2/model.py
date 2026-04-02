"""
Model Module for DenseNet121 Fine-tuning Pipeline
===================================================
- Load DenseNet121 pretrained (ImageNet).
- Thay thế classifier head: Linear(1024,512) → BN → ReLU → Dropout → Linear(512,256) → BN → ReLU → Dropout → Linear(256,6).
- Freeze / Unfreeze backbone utilities.
- Discriminative Learning Rate configuration cho AdamW.
- Multi-GPU DataParallel wrapper.
"""

import torch
import torch.nn as nn
from torchvision import models


# ---------------------------------------------------------------------------
# 1. Khởi tạo Model
# ---------------------------------------------------------------------------
def create_densenet121(num_classes: int = 6, dropout: float = 0.4) -> nn.Module:
    """
    Tạo DenseNet121 pretrained với custom classifier head.

    Classifier head mới:
        Linear(1024, 512) → BN → ReLU → Dropout(p) → Linear(512, 256) → BN → ReLU → Dropout(p) → Linear(256, num_classes)

    Args:
        num_classes: Số lượng class đầu ra.
        dropout: Tỉ lệ Dropout.

    Returns:
        nn.Module: DenseNet121 model.
    """
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

    in_features = model.classifier.in_features  # 1024

    model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(256, num_classes),
    )

    print(f"[MODEL] DenseNet121 loaded (pretrained=ImageNet)")
    print(f"[MODEL] Classifier head: {in_features} → 512 → BN → ReLU → Dropout({dropout}) → 256 → BN → ReLU → Dropout({dropout}) → {num_classes}")

    return model


# ---------------------------------------------------------------------------
# 2. Freeze / Unfreeze Backbone
# ---------------------------------------------------------------------------
def freeze_backbone(model: nn.Module) -> None:
    """
    Freeze toàn bộ backbone (features), chỉ để classifier head trainable.
    Dùng cho Giai đoạn 1 (Warmup).
    """
    # Xử lý DataParallel wrapper
    base_model = model.module if hasattr(model, "module") else model

    for param in base_model.features.parameters():
        param.requires_grad = False

    # Đảm bảo classifier luôn trainable
    for param in base_model.classifier.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in base_model.parameters())
    print(f"[MODEL] Backbone FROZEN — trainable: {trainable:,} / {total:,} params")


def unfreeze_backbone(model: nn.Module) -> None:
    """
    Unfreeze toàn bộ model. Dùng cho Giai đoạn 2 (Fine-tune).
    """
    base_model = model.module if hasattr(model, "module") else model

    for param in base_model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    print(f"[MODEL] Backbone UNFROZEN — all {trainable:,} params trainable")


# ---------------------------------------------------------------------------
# 3. Discriminative Learning Rate (AdamW)
# ---------------------------------------------------------------------------
def get_discriminative_params(model: nn.Module, base_lr: float = 1e-3):
    """
    Phân nhóm parameter với Discriminative Learning Rate cho AdamW.

    Cấu hình:
        - denseblock1:   lr=1e-6,  weight_decay=1e-4  (low-level features)
        - denseblock2:   lr=5e-6,  weight_decay=1e-4
        - denseblock3:   lr=1e-5,  weight_decay=1e-2
        - denseblock4:   lr=5e-5,  weight_decay=1e-2
        - transition*:   lr theo block liền trước
        - init conv/bn:  lr=1e-6,  weight_decay=1e-4
        - classifier:    lr=base_lr, weight_decay=1e-2

    Args:
        model: DenseNet121 model (có thể wrapped trong DataParallel).
        base_lr: Learning rate cho classifier head.

    Returns:
        list[dict]: Parameter groups cho torch.optim.AdamW.
    """
    base_model = model.module if hasattr(model, "module") else model

    # Nhóm tham số
    groups = {
        "init": {"params": [], "lr": 1e-6, "weight_decay": 1e-4},
        "denseblock1": {"params": [], "lr": 1e-6, "weight_decay": 1e-4},
        "transition1": {"params": [], "lr": 1e-6, "weight_decay": 1e-4},
        "denseblock2": {"params": [], "lr": 5e-6, "weight_decay": 1e-4},
        "transition2": {"params": [], "lr": 5e-6, "weight_decay": 1e-4},
        "denseblock3": {"params": [], "lr": 1e-5, "weight_decay": 1e-2},
        "transition3": {"params": [], "lr": 1e-5, "weight_decay": 1e-2},
        "denseblock4": {"params": [], "lr": 5e-5, "weight_decay": 1e-2},
        "norm5": {"params": [], "lr": 5e-5, "weight_decay": 1e-2},
        "classifier": {"params": [], "lr": base_lr, "weight_decay": 1e-2},
    }

    # Phân loại từng parameter vào nhóm tương ứng
    for name, param in base_model.named_parameters():
        if not param.requires_grad:
            continue

        assigned = False
        # Classifier head
        if name.startswith("classifier"):
            groups["classifier"]["params"].append(param)
            assigned = True
        # Backbone features
        elif name.startswith("features."):
            feat_name = name[len("features."):]  # Bỏ prefix "features."
            for key in ["denseblock1", "transition1",
                        "denseblock2", "transition2",
                        "denseblock3", "transition3",
                        "denseblock4", "norm5"]:
                if feat_name.startswith(key):
                    groups[key]["params"].append(param)
                    assigned = True
                    break
            if not assigned:
                # conv0, norm0, relu0, pool0 — init layers
                groups["init"]["params"].append(param)
                assigned = True

        if not assigned:
            # Fallback: gán vào classifier group
            groups["classifier"]["params"].append(param)

    # Lọc nhóm rỗng
    param_groups = []
    for group_name, group_cfg in groups.items():
        if len(group_cfg["params"]) > 0:
            n_params = sum(p.numel() for p in group_cfg["params"])
            print(
                f"  [OPT] {group_name:20s}: "
                f"lr={group_cfg['lr']:.1e}, "
                f"wd={group_cfg['weight_decay']:.1e}, "
                f"params={n_params:>10,}"
            )
            param_groups.append(group_cfg)

    return param_groups


# ---------------------------------------------------------------------------
# 4. Multi-GPU Wrapper
# ---------------------------------------------------------------------------
def wrap_data_parallel(model: nn.Module, gpu_ids=None) -> nn.Module:
    """
    Wrap model bằng DataParallel nếu có nhiều GPU.

    Args:
        model: PyTorch model.
        gpu_ids: List GPU IDs (VD: [0, 1]). None → dùng tất cả GPU.

    Returns:
        nn.Module: Model (có thể wrapped trong DataParallel).
    """
    if not torch.cuda.is_available():
        print("[MODEL] CUDA không khả dụng — chạy trên CPU.")
        return model

    num_gpus = torch.cuda.device_count()
    if num_gpus <= 1 and gpu_ids is None:
        model = model.cuda()
        print(f"[MODEL] Single GPU: {torch.cuda.get_device_name(0)}")
        return model

    if gpu_ids is None:
        gpu_ids = list(range(num_gpus))

    model = nn.DataParallel(model, device_ids=gpu_ids)
    model = model.cuda()
    gpu_names = [torch.cuda.get_device_name(i) for i in gpu_ids]
    print(f"[MODEL] DataParallel on GPUs {gpu_ids}: {gpu_names}")
    return model


# ---------------------------------------------------------------------------
# Quick Test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = create_densenet121(num_classes=6, dropout=0.4)

    # Test freeze
    freeze_backbone(model)

    # Test unfreeze
    unfreeze_backbone(model)

    # Test discriminative LR
    print("\n[TEST] Discriminative LR groups:")
    param_groups = get_discriminative_params(model, base_lr=1e-3)
    print(f"[TEST] Total groups: {len(param_groups)}")

    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    print(f"[TEST] Output shape: {out.shape}")
    print("[TEST] ✅ model.py OK!")
