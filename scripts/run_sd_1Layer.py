import sys
import torch
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler, StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from typing import Tuple, Optional
from pathlib import Path
from torch.utils.data import DataLoader
import os
import json
from datetime import datetime

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.dataset import SPairDataset, collate_fn_correspondence
from src.sd_model import StableDiffusionExtractor
from src.correspondence import CorrespondenceMatcher
from src.evaluation import evaluate_model
from src.analyzer import ResultsAnalyzer

WEIGHTS_PATH = "/content/sd-1-5-weights"  # to update


def download_weights_if_needed(weights_path: str = WEIGHTS_PATH):
    """Download SD weights if not already present."""
    if os.path.exists(weights_path):
        print("Weights already downloaded, skipping...")
    else:
        print(f"Downloading weights to {weights_path}...")
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16
        )
        pipe.save_pretrained(weights_path)
        print(f"Weights saved to {weights_path}")


def run_sd_baseline(
    timestep: int,
    layer_name: str,
    model_suffix: str = "",
    image_size: int = 512,
    batch_size: int = 4,
    category: str = 'all',
    weights_path: str = WEIGHTS_PATH
):
    """
    Run SD baseline evaluation with specified timestep and layer.
    
    Args:
        timestep: Diffusion timestep to extract features from
        layer_name: Layer to extract features from (e.g., "up_blocks.1")
        model_suffix: Optional suffix for model name (e.g., "t261", "layer2")
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

    # Initialize extractor
    extractor = StableDiffusionExtractor(
        weights=weights_path,
        model_name="sd-1-5",
        timestep=timestep,
        layer_name=layer_name
    )

    # Create matcher
    matcher = CorrespondenceMatcher(extractor)

    # Evaluate
    print(f"Evaluating SD (timestep={timestep}, layer={layer_name})...")
    metrics = evaluate_model(matcher, dataloader)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"sd-1-5-t{timestep}-{layer_name.replace('.', '_')}"
    if model_suffix:
        model_name += f"-{model_suffix}"

    results_dict = {
        'metrics': metrics,
        'model': model_name,
        'config': {
            'weights': 'stable-diffusion-v1-5',
            'timestep': timestep,
            'layer': layer_name,
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
    """Run multiple SD baselines with different configurations."""
    # Download weights once
    download_weights_if_needed()
    
    # Default DIFT configuration
    run_sd_baseline(
        timestep=261, # can change (t<1000), try t=100 
        layer_name="up_blocks.1", # can change (up_blocks.0, up_blocks.2, mid_block, ...)
        model_suffix="baseline"
    )


if __name__ == "__main__":
    main()