"""
Main Extraction Script for DenseNet121 Feature Extraction
==========================================================
Entry point for extracting features from images using DenseNet121 with custom trained weights.

Usage:
    python extract.py
    python extract.py --data_dir "path/to/data" --output_dir "path/to/output"
"""

import argparse
from pathlib import Path
from typing import Tuple, List
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import ImageDataset, get_transform
from model import create_feature_extractor
from utils import (
    check_and_create_dir,
    validate_data_dir,
    save_features_to_csv,
    print_extraction_summary
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DenseNet121 Feature Extraction (TrainVal & Test)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--output_dir', type=str, default=str(config.DEFAULT_OUTPUT_DIR),
        help='Path to directory where output CSVs will be saved'
    )
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--num_workers', type=int, default=config.NUM_WORKERS)
    return parser.parse_args()


def extract_features(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, List[str], List[str]]:
    model.eval()
    all_features = []
    all_labels = []
    all_filenames = []
    
    print(f"\n{'='*70}")
    print(f"Extracting features...")
    print(f"{'='*70}")
    
    with torch.no_grad():
        for images, labels, filenames in tqdm(dataloader, desc="Processing batches"):
            images = images.to(device)
            features = model(images)
            features = features.cpu().numpy()
            
            all_features.append(features)
            all_labels.extend(labels)
            all_filenames.extend(filenames)
    
    all_features = np.vstack(all_features)
    print(f"✓ Feature extraction complete!")
    print(f"  Shape: {all_features.shape}")
    
    return all_features, all_labels, all_filenames


def main() -> None:
    args = parse_arguments()
    output_dir = Path(args.output_dir)
    
    print(f"\n{'='*70}")
    print(f"DENSENET121 FEATURE EXTRACTION (CUSTOM WEIGHTS)")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {config.DEVICE}")
    print(f"{'='*70}\n")
    
    try:
        check_and_create_dir(output_dir)
    except Exception as e:
        print(f"❌ Error creating output directory: {e}")
        return
        
    try:
        print(f"\n{'='*70}")
        print(f"Initializing model once...")
        print(f"{'='*70}")
        model, device = create_feature_extractor()
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return

    transform = get_transform()

    for name, data_dir_path, csv_name in config.DATASETS_TO_PROCESS:
        print(f"\n{'*'*70}")
        print(f"PROCESSING DATASET: {name}")
        print(f"Path: {data_dir_path}")
        print(f"{'*'*70}")
        
        try:
            validate_data_dir(data_dir_path)
            dataset = ImageDataset(data_dir_path, transform=transform)
            dataloader = DataLoader(
                dataset, batch_size=args.batch_size, shuffle=False, 
                num_workers=args.num_workers, pin_memory=config.PIN_MEMORY
            )
            
            features, labels, filenames = extract_features(model, dataloader, device)
            
            output_csv = output_dir / csv_name
            save_features_to_csv(features, labels, filenames, output_csv)
            
            print_extraction_summary(
                num_images=len(dataset), num_classes=len(dataset.class_names),
                feature_dim=model.get_feature_dim(), device=str(device), output_path=output_csv
            )
        except Exception as e:
            print(f"❌ Error processing dataset {name}: {e}")

    print("\n✅ All feature extraction phases completed successfully!\n")


if __name__ == "__main__":
    main()
