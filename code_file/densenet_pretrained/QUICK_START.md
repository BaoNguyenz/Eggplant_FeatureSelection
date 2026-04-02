# Quick Reference Guide - DenseNet121 Training

## 📋 Quick Start Checklist

### Step 1: Install Dependencies
```bash
cd "E:\PROJECTWORSHOP\Eggplant Leaf Disease Detection Dataset\code_file\densenet_pretrained"
python install_dependencies.py
```
Choose: 1 (CPU) or 2 (GPU CUDA 11.8)

### Step 2: Validate Setup
```bash
python setup_validation.py
```
Expected output: ✓ All tests passed!

### Step 3: Run Training
```bash
python train.py
```

---

## 🎯 Common Commands

### Basic Training
```bash
python train.py
```

### Custom Parameters
```bash
# Long training with larger batch
python train.py --epochs 100 --batch_size 64

# Lower learning rate
python train.py --lr 0.0001

# Cross Entropy instead of Focal Loss
python train.py --loss_type cross_entropy

# Disable early stopping
python train.py --early_stopping 0
```

### Windows-Specific (if DataLoader errors)
```bash
python train.py --num_workers 0
```

---

## 📊 Understanding Output

### During Training
```
Epoch 10/50
----------------------------------------
Epoch 10 [Train]: 100%|████████| 125/125 [02:15<00:00]
Epoch 10 [Val]:   100%|████████| 16/16 [00:10<00:00]

[Train] Loss: 0.5234 | Acc: 0.8250 | Prec: 0.8150 | Rec: 0.8200 | F1: 0.8175
[Val]   Loss: 0.6123 | Acc: 0.7850 | Prec: 0.7750 | Rec: 0.7800 | F1: 0.7775

✓ New best model saved! (F1: 0.7775)
```

**What to watch:**
- **Train Loss decreasing**: Model is learning ✓
- **Val Loss stable/decreasing**: Good generalization ✓
- **Large gap between Train/Val**: Overfitting ⚠
- **Early stopping triggered**: Model stopped improving

### Final Output Files
```
densenet_pretrained/
├── best_model.pth           # Load this for inference
├── class_names.json         # ["Bacterial_Leaf_Blight", ...]
├── training_history.json    # All metrics per epoch
└── test_results.json        # Final test performance
```

---

## 🔍 Checking Results

### Load Test Results
```python
import json

with open('test_results.json', 'r') as f:
    results = json.load(f)
    
print(f"Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
print(f"Test F1-Score: {results['test_metrics']['f1_score']:.4f}")
print(f"Best Epoch: {results['best_epoch']}")
print(f"Training Time: {results['training_time_minutes']:.2f} minutes")
```

### Load Best Model for Inference
```python
import torch
from model_setup import create_densenet121

# Load checkpoint
checkpoint = torch.load('best_model.pth')

# Create model
num_classes = len(checkpoint['class_names'])
model = create_densenet121(num_classes=num_classes, pretrained=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Loaded model from epoch {checkpoint['epoch']}")
print(f"Best F1-Score: {checkpoint['best_f1']:.4f}")
print(f"Classes: {checkpoint['class_names']}")
```

### Make Prediction
```python
from PIL import Image
from torchvision import transforms

# Load and preprocess image
img = Image.open('path/to/image.jpg').convert('RGB')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

# Predict
with torch.no_grad():
    output = model(img_tensor)
    _, predicted = torch.max(output, 1)
    class_idx = predicted.item()
    
print(f"Predicted class: {checkpoint['class_names'][class_idx]}")
```

---

## 🛠️ Troubleshooting

### Problem: "CUDA out of memory"
**Solution:**
```bash
python train.py --batch_size 16  # Reduce from default 32
```

### Problem: "Too many open files"
**Solution:**
```bash
python train.py --num_workers 0  # Windows-specific
```

### Problem: Model overfitting
**Symptoms:**
- Train Acc: 0.95, Val Acc: 0.70 (large gap)

**Solutions:**
1. Already using data augmentation ✓
2. Already using early stopping ✓
3. Try lower learning rate:
   ```bash
   python train.py --lr 0.0001
   ```
4. Check if dataset is too small

### Problem: Training too slow
**Solutions:**
1. Check GPU usage:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should be True
   ```
2. Increase batch size (if GPU has memory):
   ```bash
   python train.py --batch_size 64
   ```
3. Use more workers (if CPU allows):
   ```bash
   python train.py --num_workers 8
   ```

### Problem: Val Loss not improving
**Symptoms:**
- Early stopping triggered at epoch 15

**Solutions:**
1. Increase patience:
   ```bash
   python train.py --early_stopping 20
   ```
2. Check learning rate (might be too high/low)
3. Try different loss:
   ```bash
   python train.py --loss_type cross_entropy
   ```

---

## 📈 Performance Tips

### For Best Accuracy
1. Train longer: `--epochs 100`
2. Use Focal Loss: `--loss_type focal` (default)
3. Lower LR: `--lr 0.0001`
4. Larger batch: `--batch_size 64`

### For Faster Training
1. Smaller batch: `--batch_size 16`
2. Freeze backbone (modify `model_setup.py`):
   ```python
   from model_setup import create_densenet121, freeze_backbone
   model = create_densenet121(...)
   model = freeze_backbone(model, freeze=True)
   ```

### For Better Generalization
1. Data augmentation already enabled ✓
2. Early stopping already enabled ✓
3. Class weighting already enabled ✓

---

## 📦 File Structure Summary

```
densenet_pretrained/
├── utils.py                    # Helper functions
├── data_setup.py               # Data loading
├── model_setup.py              # Model creation
├── train.py                    # Main script ← RUN THIS
├── setup_validation.py         # Test setup
├── install_dependencies.py     # Install packages
├── README.md                   # Full documentation
└── QUICK_START.md             # This file
```

---

## 🚀 Typical Workflow

```bash
# 1. Install (first time only)
python install_dependencies.py

# 2. Validate (recommended)
python setup_validation.py

# 3. Train
python train.py --epochs 50 --batch_size 32 --loss_type focal

# 4. Check results
# Open test_results.json to see final metrics

# 5. Use model
# Load best_model.pth for inference
```

---

## 💡 Parameter Cheat Sheet

| Parameter | Default | Recommended Range | Description |
|-----------|---------|-------------------|-------------|
| `--epochs` | 50 | 50-100 | More = better fit, longer time |
| `--batch_size` | 32 | 16-64 | Higher = faster, needs more GPU RAM |
| `--lr` | 0.001 | 0.0001-0.01 | Lower = stable, slower convergence |
| `--loss_type` | focal | focal/cross_entropy | Focal better for imbalanced data |
| `--early_stopping` | 10 | 5-20 | Patience epochs before stopping |
| `--num_workers` | 4 | 0-8 | Set to 0 if errors on Windows |

---

**Good luck! 🚀**
