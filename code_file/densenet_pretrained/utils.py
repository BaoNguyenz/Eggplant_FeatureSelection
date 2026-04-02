"""
Utility Functions for DenseNet169 Training Pipeline
Includes: Class weight calculation, metrics computation, directory creation, Focal Loss, and visualization
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def get_class_weights(labels, num_classes):
    """
    Tính trọng số cho từng lớp dựa trên số lượng mẫu (để xử lý dữ liệu mất cân bằng).
    
    Args:
        labels (list or np.array): Danh sách nhãn của tất cả mẫu
        num_classes (int): Số lượng lớp
    
    Returns:
        torch.Tensor: Tensor chứa trọng số cho từng lớp
    """
    labels = np.array(labels)
    class_counts = np.bincount(labels, minlength=num_classes)
    
    # Tránh chia cho 0
    class_counts = np.maximum(class_counts, 1)
    
    # Tính trọng số nghịch đảo tần suất
    total_samples = len(labels)
    class_weights = total_samples / (num_classes * class_counts)
    
    return torch.FloatTensor(class_weights)


def compute_metrics(y_true, y_pred):
    """
    Tính toán các metrics: Accuracy, Precision, Recall, F1-Score (Weighted).
    
    Args:
        y_true (list or np.array): Nhãn thực tế
        y_pred (list or np.array): Nhãn dự đoán
    
    Returns:
        dict: Dictionary chứa các metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def check_dir(directory):
    """
    Tạo thư mục nếu chưa tồn tại (an toàn cho Windows với pathlib).
    
    Args:
        directory (str or Path): Đường dẫn thư mục cần tạo
    
    Returns:
        Path: Đối tượng Path của thư mục
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


class FocalLoss(nn.Module):
    """
    Focal Loss cho bài toán Multi-class Classification.
    Giúp tập trung vào các mẫu khó (hard examples) và xử lý mất cân bằng dữ liệu.
    
    Paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    
    Args:
        alpha (torch.Tensor or None): Trọng số cho từng lớp (shape: [num_classes])
        gamma (float): Tham số focusing (gamma > 0 giảm loss của các mẫu dễ)
        reduction (str): Cách tính loss cuối cùng ('mean', 'sum', 'none')
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Logits từ model (shape: [batch_size, num_classes])
            targets (torch.Tensor): Ground truth labels (shape: [batch_size])
        
        Returns:
            torch.Tensor: Focal loss value
        """
        # Tính Cross Entropy loss (không reduction)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Tính확률 예측 (probabilities)
        pt = torch.exp(-ce_loss)  # pt = probability of true class
        
        # Focal term: (1 - pt)^gamma
        focal_term = (1 - pt) ** self.gamma
        
        # Focal Loss = focal_term * ce_loss
        focal_loss = focal_term * ce_loss
        
        # Áp dụng alpha (class weights) nếu có
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss




def plot_training_history(history, output_dir):
    """
    Plot training history (loss, accuracy, F1-score) và lưu thành file PNG.
    
    Args:
        history (dict): Dictionary chứa training history với keys:
                       'train_loss', 'val_loss', 'train_acc', 'val_acc',
                       'train_f1', 'val_f1'
        output_dir (str or Path): Đường dẫn thư mục để lưu plot
    """
    output_dir = Path(output_dir)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Training History', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot 1: Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_title('Loss Curves', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
    axes[1].set_title('Accuracy Curves', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: F1-Score
    axes[2].plot(epochs, history['train_f1'], 'b-', label='Train F1', linewidth=2)
    axes[2].plot(epochs, history['val_f1'], 'r-', label='Val F1', linewidth=2)
    axes[2].set_title('F1-Score Curves', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1-Score')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / 'training_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved training curves to: {output_path}")


def plot_confusion_matrix(y_true, y_pred, class_names, output_dir):
    """
    Vẽ confusion matrix và lưu thành file PNG.
    
    Args:
        y_true (list or np.array): Nhãn thực tế
        y_pred (list or np.array): Nhãn dự đoán
        class_names (list): Danh sách tên các lớp
        output_dir (str or Path): Đường dẫn thư mục để lưu plot
    """
    output_dir = Path(output_dir)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Create labels with both count and percentage
    labels = np.empty_like(cm, dtype='<U20')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            labels[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
    
    # Plot heatmap
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=True,
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, linecolor='gray')
    
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / 'confusion_matrix.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved confusion matrix to: {output_path}")


if __name__ == "__main__":
    # Test các hàm utility
    print("Testing utils.py...")
    
    # Test 1: Class weights
    labels = [0, 0, 0, 1, 1, 2]
    weights = get_class_weights(labels, num_classes=3)
    print(f"\nClass weights: {weights}")
    
    # Test 2: Metrics
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 2, 1]
    metrics = compute_metrics(y_true, y_pred)
    print(f"\nMetrics: {metrics}")
    
    # Test 3: Directory creation
    test_dir = check_dir("./test_output")
    print(f"\nCreated directory: {test_dir}")
    
    # Test 4: Focal Loss
    print("\nTesting Focal Loss...")
    focal_loss = FocalLoss(alpha=weights, gamma=2.0)
    inputs = torch.randn(4, 3)  # Batch size 4, 3 classes
    targets = torch.tensor([0, 1, 2, 1])
    loss = focal_loss(inputs, targets)
    print(f"Focal Loss: {loss.item():.4f}")
    
    print("\n✓ All tests passed!")

