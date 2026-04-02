"""
Main Training Script for DenseNet121
Includes: Argparse, training loop, validation, early stopping, model saving,
          CosineAnnealingLR scheduler, and visualization.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
import time
from tqdm import tqdm
import numpy as np

# Import custom modules
from utils import get_class_weights, compute_metrics, check_dir, FocalLoss, plot_training_history, plot_confusion_matrix
from data_setup import create_dataloaders
from model_setup import create_densenet121



def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Train model trong một epoch.
    
    Returns:
        dict: Training metrics (loss, accuracy, etc.)
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    # Compute metrics
    epoch_loss = running_loss / len(train_loader.dataset)
    metrics = compute_metrics(all_labels, all_preds)
    metrics['loss'] = epoch_loss
    
    return metrics


def validate(model, val_loader, criterion, device, epoch, phase="Val", return_preds=False):
    """
    Validate model trên validation/test set.
    
    Args:
        return_preds (bool): If True, return predictions for confusion matrix
    
    Returns:
        dict: Validation metrics (loss, accuracy, etc.)
        tuple: (metrics, all_preds, all_labels) if return_preds=True
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(val_loader, desc=f"Epoch {epoch} [{phase}]")
    
    with torch.no_grad():
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    # Compute metrics
    epoch_loss = running_loss / len(val_loader.dataset)
    metrics = compute_metrics(all_labels, all_preds)
    metrics['loss'] = epoch_loss
    
    if return_preds:
        return metrics, all_preds, all_labels
    return metrics


def train_model(args):
    """
    Main training function.
    """
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Training DenseNet121 on Eggplant Leaf Disease Detection")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Loss type: {args.loss_type}")
    print(f"Dropout rate: {args.dropout_rate}")
    print(f"Epochs: {args.epochs}")
    print(f"LR Scheduler: CosineAnnealingLR (eta_min=1e-6)")
    print(f"{'='*60}\n")
    
    # Create output directory
    output_dir = check_dir(args.output_dir)
    
    # Create dataloaders (từ Fold vật lý)
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader, class_names, all_labels = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed
    )
    
    num_classes = len(class_names)
    
    # Save class names
    class_names_file = output_dir / 'class_names.json'
    with open(class_names_file, 'w', encoding='utf-8') as f:
        json.dump(class_names, f, indent=2, ensure_ascii=False)
    print(f"\nSaved class names to: {class_names_file}")
    
    # Calculate class weights
    class_weights = get_class_weights(all_labels, num_classes)
    print(f"\nClass weights: {class_weights.numpy()}")
    
    # Create model (DenseNet121 với Dropout)
    model = create_densenet121(
        num_classes=num_classes,
        pretrained=True,
        dropout_rate=args.dropout_rate,
        device=device
    )
    
    # Setup criterion (loss function)
    if args.loss_type == 'focal':
        print("\nUsing Focal Loss")
        criterion = FocalLoss(alpha=class_weights.to(device), gamma=2.0, reduction='mean')
    else:
        print("\nUsing Cross Entropy Loss")
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Setup learning rate scheduler — CosineAnnealingLR
    # LR sẽ giảm dần từ args.lr xuống eta_min=1e-6 theo hình cosine trong T_max epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    # Lưu cấu hình huấn luyện
    training_config = {
        'data_dir': str(args.data_dir),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'dropout_rate': args.dropout_rate,
        'loss_type': args.loss_type,
        'early_stopping': args.early_stopping,
        'scheduler': 'CosineAnnealingLR',
        'scheduler_T_max': args.epochs,
        'scheduler_eta_min': 1e-6,
        'seed': args.seed,
        'num_classes': num_classes,
        'class_names': class_names,
    }
    config_file = output_dir / 'training_config.json'
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(training_config, f, indent=2, ensure_ascii=False)
    print(f"Saved training config to: {config_file}")
    
    # Training tracking
    best_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [],
        'lr': []
    }
    
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch}/{args.epochs} (LR: {current_lr:.6f})")
        print("-" * 40)
        
        # Train
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, epoch, phase="Val")
        
        # Update scheduler (CosineAnnealingLR cập nhật theo epoch)
        scheduler.step()
        
        # Log metrics
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['train_f1'].append(train_metrics['f1_score'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1_score'])
        history['lr'].append(current_lr)
        
        # Print metrics
        print(f"\n[Train] Loss: {train_metrics['loss']:.4f} | "
              f"Acc: {train_metrics['accuracy']:.4f} | "
              f"Prec: {train_metrics['precision']:.4f} | "
              f"Rec: {train_metrics['recall']:.4f} | "
              f"F1: {train_metrics['f1_score']:.4f}")
        
        print(f"[Val]   Loss: {val_metrics['loss']:.4f} | "
              f"Acc: {val_metrics['accuracy']:.4f} | "
              f"Prec: {val_metrics['precision']:.4f} | "
              f"Rec: {val_metrics['recall']:.4f} | "
              f"F1: {val_metrics['f1_score']:.4f}")
        
        # Check if best model (based on F1-Score)
        current_f1 = val_metrics['f1_score']
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            best_model_path = output_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_f1': best_f1,
                'class_names': class_names,
                'metrics': val_metrics,
                'dropout_rate': args.dropout_rate,
                'training_config': training_config,
            }, best_model_path)
            
            print(f"\n✓ New best model saved! (F1: {best_f1:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if args.early_stopping > 0 and patience_counter >= args.early_stopping:
            print(f"\n⚠ Early stopping triggered after {epoch} epochs (patience: {args.early_stopping})")
            break
    
    # Training complete
    training_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"{'='*60}")
    print(f"Total training time: {training_time/60:.2f} minutes")
    print(f"Best epoch: {best_epoch} (F1: {best_f1:.4f})")
    
    # Save training history
    history_file = output_dir / 'training_history.json'
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nSaved training history to: {history_file}")
    
    # Load best model
    checkpoint = torch.load(output_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # ===================================================================
    # Test on test set
    # ===================================================================
    print(f"\n{'='*60}")
    print("Evaluating on Test Set...")
    print(f"{'='*60}\n")
    
    test_metrics = validate(model, test_loader, criterion, device, epoch=best_epoch, phase="Test")
    
    print(f"\n[Test] Loss: {test_metrics['loss']:.4f} | "
          f"Acc: {test_metrics['accuracy']:.4f} | "
          f"Prec: {test_metrics['precision']:.4f} | "
          f"Rec: {test_metrics['recall']:.4f} | "
          f"F1: {test_metrics['f1_score']:.4f}")
    
    # Save test results
    test_results = {
        'test_metrics': test_metrics,
        'best_epoch': best_epoch,
        'best_val_f1': best_f1,
        'training_time_minutes': training_time / 60,
        'class_names': class_names
    }
    
    test_results_file = output_dir / 'test_results.json'
    with open(test_results_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    print(f"\nSaved test results to: {test_results_file}")
    
    # ===================================================================
    # Generate visualizations
    # ===================================================================
    print(f"\n{'='*60}")
    print("Generating Visualizations...")
    print(f"{'='*60}\n")
    
    # Plot training curves
    plot_training_history(history, output_dir)
    
    # Plot confusion matrix trên test set
    cm_metrics, cm_preds, cm_labels = validate(
        model, test_loader, criterion, device, epoch=best_epoch, phase="Test (CM)", return_preds=True
    )
    plot_confusion_matrix(cm_labels, cm_preds, class_names, output_dir)
    
    print(f"\n{'='*60}")
    print("All done! ✓")
    print(f"{'='*60}\n")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train DenseNet121 for Eggplant Leaf Disease Detection"
    )
    
    # Paths
    parser.add_argument(
        '--data_dir',
        type=str,
        default=r"E:\PROJECTWORSHOP\Eggplant_regconition\10_Fold_CV_Dataset\Fold_1",
        help='Path to Fold directory (chứa train_val/ và test/)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=r"E:\PROJECTWORSHOP\Eggplant_regconition\code_file\densenet_pretrained\output",
        help='Path to output directory for saving models and results'
    )
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Dropout
    parser.add_argument(
        '--dropout_rate',
        type=float,
        default=0.2,
        help='Dropout rate trước lớp Linear classifier (0.0 = không dropout)'
    )
    
    # Loss function
    parser.add_argument(
        '--loss_type',
        type=str,
        default='focal',
        choices=['cross_entropy', 'focal'],
        help='Loss function type: cross_entropy or focal'
    )
    
    # Early stopping
    parser.add_argument(
        '--early_stopping',
        type=int,
        default=15,
        help='Early stopping patience (0 to disable)'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(args)
