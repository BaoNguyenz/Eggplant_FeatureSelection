"""
Utility Module for DenseNet121 Feature Extraction
==================================================
Helper functions for directory management and CSV saving.
"""

from pathlib import Path
from typing import List
import csv
import numpy as np

import config


def check_and_create_dir(directory: Path) -> None:
    """
    Check if directory exists, create if it doesn't.
    
    Args:
        directory: Path to directory
    """
    directory = Path(directory)
    
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    else:
        print(f"✓ Directory exists: {directory}")


def validate_data_dir(data_dir: Path) -> None:
    """
    Validate that the data directory exists and contains class folders.
    
    Args:
        data_dir: Path to data directory
        
    Raises:
        FileNotFoundError: If directory doesn't exist
        ValueError: If directory is empty or has no class folders
    """
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    if not data_dir.is_dir():
        raise ValueError(f"Path is not a directory: {data_dir}")
    
    # Check for class subdirectories
    class_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    
    if len(class_dirs) == 0:
        raise ValueError(f"No class directories found in {data_dir}")
    
    print(f"✓ Data directory validated: {data_dir}")
    print(f"  Found {len(class_dirs)} class directories")


def save_features_to_csv(
    features: np.ndarray,
    labels: List[str],
    filenames: List[str],
    output_path: Path
) -> None:
    """
    Save extracted features to CSV file.
    
    CSV Format:
        filename, label, feature_0, feature_1, ..., feature_1023
    
    Args:
        features: numpy array of shape (n_samples, 1024)
        labels: list of class labels
        filenames: list of image filenames
        output_path: path to save CSV file
    """
    output_path = Path(output_path)
    
    # Validate input dimensions
    n_samples = features.shape[0]
    feature_dim = features.shape[1]
    
    if len(labels) != n_samples or len(filenames) != n_samples:
        raise ValueError(
            f"Dimension mismatch: features={n_samples}, "
            f"labels={len(labels)}, filenames={len(filenames)}"
        )
    
    print(f"\n{'='*70}")
    print(f"Saving features to CSV...")
    print(f"{'='*70}")
    print(f"  Output path: {output_path}")
    print(f"  Number of samples: {n_samples}")
    print(f"  Feature dimension: {feature_dim}")
    
    try:
        # Open CSV file with UTF-8 encoding for Windows
        with open(output_path, 'w', newline='', encoding=config.FILE_ENCODING) as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            header = ['filename', 'label'] + [f'feature_{i}' for i in range(feature_dim)]
            writer.writerow(header)
            
            # Write data rows
            for i in range(n_samples):
                row = [filenames[i], labels[i]] + features[i].tolist()
                writer.writerow(row)
        
        print(f"✓ Features saved successfully!")
        print(f"  File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
        print(f"{'='*70}\n")
        
    except Exception as e:
        raise RuntimeError(f"Error saving CSV file: {e}")


def print_extraction_summary(
    num_images: int,
    num_classes: int,
    feature_dim: int,
    device: str,
    output_path: Path
) -> None:
    """
    Print a summary of the feature extraction process.
    
    Args:
        num_images: total number of images processed
        num_classes: number of classes
        feature_dim: dimension of feature vectors
        device: device used (cuda/cpu)
        output_path: path to output CSV file
    """
    print(f"\n{'='*70}")
    print(f"FEATURE EXTRACTION SUMMARY")
    print(f"{'='*70}")
    print(f"  Model: {config.MODEL_NAME.upper()}")
    print(f"  Device: {device}")
    print(f"  Total images processed: {num_images}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Feature dimension: {feature_dim}")
    print(f"  Output file: {output_path.name}")
    print(f"  Output directory: {output_path.parent}")
    print(f"{'='*70}\n")
