"""
Dataset Module for DenseNet121 Feature Extraction
==================================================
Custom Dataset class for loading and preprocessing images.
"""

from pathlib import Path
from typing import Tuple, List
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import re

import config


class ImageDataset(Dataset):
    """
    Custom Dataset for loading images from directory structure.
    
    Expected Directory Structure:
    data_dir/
        ├── class_1/
        │   ├── img1.jpg
        │   ├── img2.jpg
        ├── class_2/
        │   ├── img1.jpg
        │   ├── img2.jpg
    """
    
    def __init__(self, data_dir: Path, transform: transforms.Compose = None) -> None:
        """
        Initialize the dataset.
        
        Args:
            data_dir: Root directory containing class subdirectories
            transform: torchvision transforms to apply to images
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.image_paths: List[Path] = []
        self.labels: List[str] = []
        self.class_names: List[str] = []
        
        # Load all image paths and labels
        self._load_image_paths()
        
    def _load_image_paths(self) -> None:
        """
        Recursively load all image paths from class subdirectories.
        """
        # Supported image extensions
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        # Get all class directories
        class_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        
        if not class_dirs:
            raise ValueError(f"No class directories found in {self.data_dir}")
            
        def natural_sort_key(s):
            """Sort string naturally (e.g., 'image_2' before 'image_10')."""
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]
        
        # Sort class names for consistent and natural ordering
        self.class_names = sorted([d.name for d in class_dirs], key=natural_sort_key)
        
        # Load all images
        for class_dir in sorted(class_dirs, key=lambda d: natural_sort_key(d.name)):
            class_name = class_dir.name
            
            # Get all images in class and sort them naturally
            img_paths = [p for p in class_dir.iterdir() if p.suffix.lower() in valid_extensions]
            img_paths = sorted(img_paths, key=lambda p: natural_sort_key(p.name))
            
            for img_path in img_paths:
                self.image_paths.append(img_path)
                self.labels.append(class_name)
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.data_dir}")
        
        print(f"✓ Loaded {len(self.image_paths)} images from {len(self.class_names)} classes")
        print(f"  Classes: {', '.join(self.class_names)}")
    
    def __len__(self) -> int:
        """Return the total number of images."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, str]:
        """
        Load and return a single image.
        
        Args:
            idx: Index of the image
            
        Returns:
            Tuple of (transformed_image, label, image_filename)
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image and convert to RGB
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {e}")
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Return image, label, and filename (for tracking in CSV)
        return image, label, img_path.name


def get_transform() -> transforms.Compose:
    """
    Get the standard ImageNet preprocessing transform for DenseNet121.
    
    Returns:
        Composed transforms for image preprocessing
    """
    return transforms.Compose([
        transforms.Resize(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.IMAGENET_MEAN,
            std=config.IMAGENET_STD
        )
    ])
