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
from src.sd_model import StableDiffusionExtractor, MultiLayerSDExtractor, SDDINOFusion
from src.models import DINOv2Extractor
from src.correspondence import CorrespondenceMatcher
from src.evaluation import evaluate_model
from src.analyzer import ResultsAnalyzer

from scripts.run_sd import run_sd_baseline, download_weights_if_needed
from scripts.run_baseline import dinov2_baseline

WEIGHTS_PATH = "/content/sd-1-5-weights"  # to update

def run_fusion(
    timestep: int,
    layers: Union[str, List[str]],
    image_size: int = 518,
    batch_size: int = 4,
    category: str = 'all',
    weights_path: str = WEIGHTS_PATH,
    alpha: float=0.5
):
    """
    Run SD baseline evaluation with specified timestep and layer(s)
    
    Args:
        timestep: Diffusion timestep to extract features from
        layers: Single layer name (str) or list of layer names for ensemble
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

    # Initialize SD extractor (single or multi-layer)
    if isinstance(layers, str):
        sd_extractor = StableDiffusionExtractor(
            weights=weights_path,
            model_name="sd-1-5",
            timestep=timestep,
            layer_name=layers
        )
        layer_str = layers
    else:
        sd_extractor = MultiLayerSDExtractor(
            weights=weights_path,
            timestep=timestep,
            layers=layers
        )
        layer_str = "+".join([layer.replace("up_blocks.", "up").replace("mid_block", "mid") 
                              for layer in layers])

    # Initialize DINOv2 Extractor
    dino_extractor = DINOv2Extractor(model_name="dinov2_vitb14")


    # Fusion matcher
    fusion = SDDINOFusion(sd_extractor, dino_extractor, alpha)
    matcher = CorrespondenceMatcher(fusion)

    # Evaluate
    print(f"Evaluating SD (timestep={timestep}, layers={layers})...")
    metrics = evaluate_model(matcher, dataloader)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"fusion-dinov2-sd-t{timestep}-{layer_str.replace('.', '_')}"

    results_dict = {
        'metrics': metrics,
        'model': model_name,
        'config_sd': {
            'weights': 'stable-diffusion-v1-5',
            'timestep': timestep,
            'layers': layers if isinstance(layers, list) else [layers],
        },
        'image_size': image_size,
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
    # Download weights once
    download_weights_if_needed()
    
    run_fusion(
        timestep=261, # to change, try t=100 for instance
        layers="up_blocks.1", # single layer or ensemble (list)
        alpha=0.5 # can change
    )
    
    

if __name__ == "__main__":
    main()