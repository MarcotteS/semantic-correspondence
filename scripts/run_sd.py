#!/usr/bin/env python3

import sys
import torch
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler, StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from typing import Tuple, Optional, List, Union
from pathlib import Path
from torch.utils.data import DataLoader
import os
import json
from datetime import datetime

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.dataset import SPairDataset, collate_fn_correspondence
from src.sd_model import StableDiffusionExtractor, MultiLayerSDExtractor
from src.correspondence import CorrespondenceMatcher
from src.evaluation import evaluate_model
from src.analyzer import ResultsAnalyzer

WEIGHTS_PATH = "/weights/sd-1-5-weights"


def run_sd_baseline(
    timestep: int,
    layers: Union[str, List[str]],
    model_suffix: str = "",
    image_size: int = 512,
    batch_size: int = 4,
    category: str = 'all',
    weights_path: str = WEIGHTS_PATH
):
    """
    Run SD baseline evaluation with specified timestep and layer(s)
    
    Args:
        timestep: Diffusion timestep to extract features from
        layers: Single layer name (str) or list of layer names for ensemble
        model_suffix: Optional suffix for model name
        image_size: Image size for evaluation
        batch_size: Batch size for dataloader
        category: SPair category to evaluate on
        weights_path: Path to SD weights
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
        num_workers=2,
        collate_fn=collate_fn_correspondence
    )

    # Initialize extractor (single or multi-layer)
    if isinstance(layers, str):
        # Single layer
        extractor = StableDiffusionExtractor(
            weights=weights_path,
            model_name="sd-1-5",
            timestep=timestep,
            layer_name=layers
        )
        layer_str = layers
    else:
        # Multiple layers (ensemble)
        extractor = MultiLayerSDExtractor(
            weights=weights_path,
            timestep=timestep,
            layers=layers
        )
        # Simplify layer names for results files
        layer_str = "+".join([layer.replace("up_blocks.", "up").replace("mid_block", "mid") 
                              for layer in layers])

    # Create matcher
    matcher = CorrespondenceMatcher(extractor)

    # Evaluate
    print(f"Evaluating SD (timestep={timestep}, layers={layers})...")
    metrics = evaluate_model(matcher, dataloader)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"sd-1-5-t{timestep}-{layer_str.replace('.', '_')}"
    if model_suffix:
        model_name += f"-{model_suffix}"

    results_dict = {
        'metrics': metrics,
        'model': model_name,
        'config': {
            'weights': 'stable-diffusion-v1-5',
            'timestep': timestep,
            'layers': layers if isinstance(layers, list) else [layers],
            'image_size': image_size,
            'batch_size': batch_size
        },
        'timestamp': timestamp
    }

    results_file = f'results_{model_name}_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"Results saved to {results_file}")

    # Generate analysis
    analyzer = ResultsAnalyzer(metrics)
    summary = analyzer.create_summary_table(threshold=0.10)
    print("\nPCK@0.10 Summary:")
    print(summary)

    save_dir = f'./results/{model_name}'
    analyzer.generate_report(save_dir=save_dir)
    
    return metrics


def main():
    """Run multiple SD baselines with different configurations"""
    # Single layer
    run_sd_baseline(
        timestep=261, # to change, try t=100 for instance
        layers="up_blocks.1", # to change, choose among: mid_block, up_blocks.i, down_blocks.i (i=0 to 3)
        model_suffix="baseline"
    )
    
    # Multi-layer ensemble
    run_sd_baseline(
        timestep=100,
        layers=["mid_block", "up_blocks.1"], # try different combinations
        model_suffix="ensemble"
    )
    


if __name__ == "__main__":
    main()