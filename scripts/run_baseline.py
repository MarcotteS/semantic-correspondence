#!/usr/bin/env python3

import os
import sys
import glob
from tqdm import tqdm
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torchvision.transforms as transforms
from collections import defaultdict
import pickle
from datetime import datetime
import pandas as pd

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.dataset import SPairDataset, collate_fn_correspondence
from src.models import DINOv2Extractor, DINOv3Extractor, SAMExtractor
from src.correspondence import CorrespondenceMatcher
from src.evaluation import evaluate_model
from src.analyzer import ResultsAnalyzer


def run_baseline(
    extractor,
    model_name: str,
    image_size: int,
    batch_size: int = 16,
    category: str = 'all'
):
    """
    Generic baseline evaluation for the feature extractors.
    
    Args:
        extractor: Feature extractor instance (DINOv2, DINOv3, or SAM)
        model_name: Name of the model for saving results
        image_size: Size to resize images
        batch_size: Batch size for dataloader
        category: SPair category to evaluate on
    """
    # Setup dataset
    dataset = SPairDataset(
        datapath='./data/',
        split='test',
        img_size=image_size,
        category=category
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn_correspondence
    )

    # Initialize matcher
    matcher = CorrespondenceMatcher(extractor)

    # Evaluate
    print(f"Evaluating {model_name}...")
    metrics = evaluate_model(matcher, dataloader)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dict = {
        'metrics': metrics,
        'model': model_name,
        'timestamp': timestamp,
        'image_size': image_size
    }

    results_file = f'results_{model_name}_{image_size}_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"Results saved to {results_file}")

    # Generate analysis
    analyzer = ResultsAnalyzer(metrics)
    save_dir = f'./results/{model_name}'
    analyzer.generate_report(save_dir=save_dir)

    # Print summary
    summary_df = analyzer.create_summary_table(threshold=0.10)
    print(summary_df)
    
    return metrics


def dinov2_baseline():
    """Baseline for DINOv2"""
    image_size = 518  # Images must be multiples of 14 for DINOv2
    extractor = DINOv2Extractor(model_name="dinov2_vitb14")
    
    return run_baseline(
        extractor=extractor,
        model_name="dinov2_vitb14",
        image_size=image_size,
        batch_size=16
    )


def dinov3_baseline():
    """Baseline for DINOv3"""
    checkpoint_path = "https://dinov3.llamameta.net/dinov3_vitb16/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiNmt0Y2lrdHZodmt4dG92dm41ajZrNmdsIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3Njc3NDM4ODZ9fX1dfQ__&Signature=Oo15SCsLwsN9zW-rhSAzIkZGaRPl3VBGnzOjgpGgs8zJC5RsjAPjaA2emXitvCNWCUDkBkxO-wrYyhLVjR5LB9euV7S4dPRunVY1W1tIqH7hbfgQJtNJis3hdz0SDJe4YRQc6BU3GEKxal5m7d69lIBcz3BPyTE9FWkgcbKgOLESrZuY32zmZFhQrpr7TaQQbUjBY9XWb2ais8B5FjgrcuEOj%7E5ECznFTRSsfc57Ai1WtVjt5UD-jSJPahxsPj2EMbe3TYAzkxzhXIjv7pT75QBcmgk2lNuXC3YsijBhNBn%7EAbaC3H5c9mDaE4xiGQ8II-OM-geu3QE2P1rLXLfvJQ__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1644049136628051"
    image_size = 512  # Images must be multiples of 16 for DINOv3
    
    extractor = DINOv3Extractor(repo_dir='dinov3', weights=checkpoint_path)
    
    return run_baseline(
        extractor=extractor,
        model_name="dinov3_vitb16",
        image_size=image_size,
        batch_size=16
    )


def sam_baseline():
    """Baseline for SAM"""
    image_size = 512
    checkpoint_path = "sam_vit_b_01ec64.pth"
    
    extractor = SAMExtractor(model_type="vit_b", checkpoint_path=checkpoint_path)
    
    return run_baseline(
        extractor=extractor,
        model_name="sam_vit_b",
        image_size=image_size,
        batch_size=1  # SAM uses batch_size=1
    )


if __name__ == "__main__":
    dinov2_baseline()
    dinov3_baseline()
    sam_baseline()