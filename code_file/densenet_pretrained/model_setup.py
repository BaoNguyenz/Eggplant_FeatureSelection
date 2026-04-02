"""
Model Setup for DenseNet121 Training Pipeline
Load pretrained DenseNet121 and customize classifier for target number of classes.
Hỗ trợ Dropout trong classifier để chống overfitting.
"""

import torch
import torch.nn as nn
from torchvision import models


def create_densenet121(num_classes, pretrained=True, dropout_rate=0.2, device='cuda'):
    """
    Tạo mô hình DenseNet121 với pretrained weights từ ImageNet.
    Thay thế lớp classifier để phù hợp với số lượng lớp của bài toán.
    Có tích hợp Dropout trước Linear để chống overfitting.
    
    Args:
        num_classes (int): Số lượng lớp cần phân loại
        pretrained (bool): Sử dụng pretrained weights từ ImageNet hay không
        dropout_rate (float): Tỷ lệ Dropout (0.0 = không dropout, 0.5 = mạnh)
        device (str): 'cuda' hoặc 'cpu'
    
    Returns:
        torch.nn.Module: DenseNet121 model đã được customize
    """
    print(f"Creating DenseNet121 model (pretrained={pretrained}, dropout={dropout_rate})...")
    
    # Load DenseNet121 pretrained trên ImageNet
    if pretrained:
        try:
            # PyTorch >= 1.13
            weights = models.DenseNet121_Weights.IMAGENET1K_V1
            model = models.densenet121(weights=weights)
        except AttributeError:
            # PyTorch < 1.13
            model = models.densenet121(pretrained=True)
    else:
        model = models.densenet121(pretrained=False)
    
    # DenseNet121 architecture:
    # - Features: Convolutional layers (densenet blocks)
    # - Classifier: Fully connected layer
    #
    # model.classifier: Linear(in_features=1024, out_features=1000) for ImageNet
    
    # Lấy số features đầu vào của classifier
    num_features = model.classifier.in_features
    
    print(f"Original classifier: Linear({num_features} -> 1000)")
    print(f"New classifier: Dropout({dropout_rate}) -> Linear({num_features} -> {num_classes})")
    
    # Thay thế classifier: Dropout + Linear
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(num_features, num_classes)
    )
    
    # Chuyển model lên device
    model = model.to(device)
    
    # Tính tổng số parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Summary:")
    print(f"  Architecture: DenseNet121")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Dropout rate: {dropout_rate}")
    print(f"  Device: {device}")
    
    return model


def freeze_backbone(model, freeze=True):
    """
    Đóng băng (freeze) các layers của backbone (features) để chỉ train classifier.
    Hữu ích khi dataset nhỏ hoặc muốn train nhanh hơn.
    
    Args:
        model: DenseNet121 model
        freeze (bool): True để freeze backbone, False để unfreeze
    
    Returns:
        torch.nn.Module: Model đã được freeze/unfreeze
    """
    for param in model.features.parameters():
        param.requires_grad = not freeze
    
    status = "frozen" if freeze else "unfrozen"
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nBackbone {status}. Trainable parameters: {trainable_params:,}")
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing model_setup.py...")
    
    # Test với CPU (để tránh lỗi nếu không có GPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Tạo model cho 5 classes (ví dụ)
    num_classes = 5
    model = create_densenet121(num_classes=num_classes, pretrained=True, dropout_rate=0.2, device=device)
    
    print("\n✓ Model created successfully!")
    
    # Kiểm tra classifier có Dropout không
    print(f"\nClassifier structure:")
    print(model.classifier)
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224).to(device)  # Batch size 2
    output = model(dummy_input)
    print(f"\nTest forward pass:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    
    # Test freeze backbone
    print("\n" + "="*50)
    model_frozen = freeze_backbone(model, freeze=True)
    
    print("\n" + "="*50)
    model_unfrozen = freeze_backbone(model, freeze=False)
