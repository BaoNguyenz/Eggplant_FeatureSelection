"""
Configuration Module for DenseNet121 Feature Extraction
========================================================
Manages all paths, hyperparameters, and device settings.
Optimized for Windows with pathlib for path handling.
"""

from pathlib import Path
from typing import Tuple
import torch


# ============================================================================
# PATH CONFIGURATION
# ============================================================================

# Datasets to process (Name, Path, Output_CSV)
DATASETS_TO_PROCESS = [
    ("TrainVal", Path(r"E:\PROJECTWORSHOP\Eggplant_regconition\10_Fold_CV_Dataset\Fold_1\train_val"), "features_trainval_fold1.csv"),
    ("Test", Path(r"E:\PROJECTWORSHOP\Eggplant_regconition\10_Fold_CV_Dataset\Fold_1\test"), "features_test_fold1.csv")
]

# Default output directory (can be overridden via CLI)
DEFAULT_OUTPUT_DIR: Path = Path(r"E:\PROJECTWORSHOP\Eggplant_regconition\code_file\densenet_feature_extraction\output_fold1")


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Model configuration - DenseNet121 with custom weights
MODEL_NAME: str = "densenet121"

# Custom pretrained weights path (trained on eggplant disease dataset)
CUSTOM_WEIGHTS_PATH: Path = Path(r"E:\PROJECTWORSHOP\Eggplant_regconition\code_file\densenet_pretrained\checkpoints\output_fold1\best_model.pth")

# Use custom weights instead of ImageNet
USE_CUSTOM_WEIGHTS: bool = True  # Set to False to use ImageNet weights
PRETRAINED: bool = True  # Use pretrained weights (ImageNet or custom)

# Feature dimension for DenseNet121 (output of final conv block)
FEATURE_DIM: int = 1024


# ============================================================================
# DATA PROCESSING CONFIGURATION
# ============================================================================

# Image preprocessing parameters (ImageNet standard)
IMG_SIZE: Tuple[int, int] = (224, 224)
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# DataLoader parameters
BATCH_SIZE: int = 32
NUM_WORKERS: int = 4  # Adjust based on your CPU cores
PIN_MEMORY: bool = True  # Speed up GPU transfer


# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

# Automatically detect CUDA availability
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

# Output CSV filename (kept for backward compatibility, unused by default now)
OUTPUT_CSV_NAME: str = "features_output_fold1.csv"

# File encoding for Windows compatibility
FILE_ENCODING: str = "utf-8"
