"""
Model Module for DenseNet121 Feature Extraction
================================================
Defines the feature extractor using DenseNet121 pretrained on ImageNet.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Tuple

import config


class DenseNetFeatureExtractor(nn.Module):
    """
    DenseNet121 Feature Extractor.
    
    Architecture:
        - Uses pretrained DenseNet121 (ImageNet or custom trained weights)
        - Removes the classifier (fully connected) layer
        - Extracts features from the final convolutional block
        - Applies Global Average Pooling (GAP)
        - Outputs 1024-dimensional feature vectors
    """
    
    def __init__(self, pretrained: bool = True, custom_weights_path: str = None) -> None:
        """
        Initialize the DenseNet121 feature extractor.
        
        Args:
            pretrained: Whether to use pretrained weights
            custom_weights_path: Path to custom trained weights (if None, use ImageNet)
        """
        super(DenseNetFeatureExtractor, self).__init__()
        
        # Load DenseNet121 architecture
        if custom_weights_path and config.USE_CUSTOM_WEIGHTS:
            # Load custom trained weights
            print(f"Loading custom pretrained {config.MODEL_NAME} from:")
            print(f"  {custom_weights_path}")
            
            # Create model architecture (DenseNet121)
            densenet = models.densenet121(weights=None)
            
            # Load checkpoint
            checkpoint = torch.load(
                custom_weights_path,
                map_location='cpu',
                weights_only=False
            )
            
            # Extract model_state_dict from checkpoint
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            else:
                state_dict = checkpoint
                print(f"✓ Loaded state_dict directly")
            
            # Remove classifier keys from state_dict (we only need features)
            # The checkpoint was trained with custom number of classes (6),
            # but we only extract features, so classifier weights don't matter
            state_dict_filtered = {k: v for k, v in state_dict.items() 
                                  if not k.startswith('classifier.')}
            
            print(f"  Filtered out classifier layer (only loading features)")
            print(f"  Keys loaded: {len(state_dict_filtered)} / {len(state_dict)}")
            
            # Load weights into model (strict=False to ignore classifier mismatch)
            missing_keys, unexpected_keys = densenet.load_state_dict(
                state_dict_filtered, 
                strict=False
            )
            
            # Verify only classifier keys are missing
            if missing_keys:
                classifier_missing = all(k.startswith('classifier.') for k in missing_keys)
                if classifier_missing:
                    print(f"✓ Classifier keys ignored (expected for feature extraction)")
                else:
                    print(f"⚠ Warning: Non-classifier keys missing: {missing_keys}")
            
            print(f"✓ Custom weights loaded successfully!")
            
        elif pretrained:
            # Load ImageNet pretrained weights
            print(f"Loading pretrained {config.MODEL_NAME} from ImageNet...")
            try:
                # PyTorch >= 1.13
                weights = models.DenseNet121_Weights.IMAGENET1K_V1
                densenet = models.densenet121(weights=weights)
            except AttributeError:
                # PyTorch < 1.13
                densenet = models.densenet121(pretrained=True)
            print(f"✓ ImageNet weights loaded successfully!")
            
        else:
            print(f"Loading {config.MODEL_NAME} without pretrained weights...")
            densenet = models.densenet121(weights=None)
        
        # Extract only the convolutional features block
        # DenseNet121 structure: features (conv blocks) -> classifier (FC)
        self.features = densenet.features
        
        # Add Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Set to evaluation mode (disable dropout, batchnorm updates)
        self.eval()
        
        print(f"✓ {config.MODEL_NAME} feature extractor initialized")
        print(f"  Feature dimension: {config.FEATURE_DIM}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to extract features.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            Feature tensor of shape (batch_size, 1024)
        """
        # Pass through convolutional blocks
        features = self.features(x)
        
        # Apply ReLU (DenseNet applies ReLU after final batch norm)
        features = torch.relu(features)
        
        # Apply Global Average Pooling
        # Input: (batch_size, 1024, 7, 7)
        # Output: (batch_size, 1024, 1, 1)
        features = self.global_avg_pool(features)
        
        # Flatten to (batch_size, 1024)
        features = torch.flatten(features, 1)
        
        return features
    
    def get_feature_dim(self) -> int:
        """
        Get the output feature dimension.
        
        Returns:
            Feature dimension (1024 for DenseNet121)
        """
        return config.FEATURE_DIM


def create_feature_extractor() -> Tuple[DenseNetFeatureExtractor, torch.device]:
    """
    Factory function to create and configure the feature extractor.
    
    Returns:
        Tuple of (model, device)
    """
    # Determine custom weights path if using custom weights
    custom_path = str(config.CUSTOM_WEIGHTS_PATH) if config.USE_CUSTOM_WEIGHTS else None
    
    # Create model with custom or ImageNet weights
    model = DenseNetFeatureExtractor(
        pretrained=config.PRETRAINED,
        custom_weights_path=custom_path
    )
    
    # Move to device
    model = model.to(config.DEVICE)
    
    print(f"✓ Model moved to device: {config.DEVICE}")
    
    return model, config.DEVICE
