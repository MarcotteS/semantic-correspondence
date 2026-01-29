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

from src.fine_tuning.CorrespondenceMatcher2 import CorrespondenceMatcher2
from src.fine_tuning.ptFilesLoader import load_matcher_from_drive_ckpt
from src.evaluation import CorrespondenceEvaluator, evaluate_model
from src.analyzer import ResultsAnalyzer
from src.correspondence import CorrespondenceMatcher
from src.models import DINOv2Extractor, DINOv3Extractor, SAMExtractor
from src.dataset import SPairDataset,collate_fn_correspondence
   






MODEL = "dinov2"          # "dinov2" | "dinov3" | "sam"
N_EPOCHS = 1
BATCH_SIZE = 16
N_UNFREEZE_LAYERS = 1
IMAGE_SIZE=518

SAM_BASE_CKPT = repo_root / "weights" / "sam_vit_b_01ec64.pth"
DINOV3_REPO = str(repo_root / "backbones" / "dinov3")
DINOV3_CKPT = str(repo_root / "weights" / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")

def create_extractor(model: str):
    if model == "dinov2":
        ext = DINOv2Extractor(model_name="dinov2_vitb14")
    elif model == "dinov3":
        ext = DINOv3Extractor(model_name="dinov3_vitb16", repo_dir=DINOV3_REPO, weights=DINOV3_CKPT)
    elif model == "sam":
        ext = SAMExtractor(model_type="vit_b", checkpoint_path=str(SAM_BASE_CKPT), image_size=IMAGE_SIZE)
    else:
        raise ValueError("Unknown model")
    return ext


def build_train_loader(image_size: int):
    dataset = SPairDataset(
        datapath='./data/',
        split='trn',
        img_size=image_size,
        category='all'
    )

    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn_correspondence
    )
    return train_loader



def run_training(extractor, train_loader):
    ckpt_drive = "/content/last.pt"
    matcher = CorrespondenceMatcher2(extractor)
    matcher.ckpt_dir = "/content"
    matcher.exp_name = f"{MODEL}with{N_EPOCHS}epochsImages{IMAGE_SIZE}with{N_UNFREEZE_LAYERS}Layers"
    matcher.resume = True
    matcher.save_every_epoch = 1

    from fine_tuning.train import train_stage2
    matcher = train_stage2(
        matcher=matcher,
        train_loader=train_loader,
        n_epochs=N_EPOCHS,
        n_last_blocks=N_UNFREEZE_LAYERS,
        lr=2e-5,
        weight_decay=0.01,
        tau=0.07,
        max_batches_per_epoch=1500,
        use_amp=True, isSam=False
    )
    return matcher


def build_test_loader(image_size: int):
    dataset = SPairDataset(
        datapath='./data/',
        split='test',
        img_size=image_size,
        category='all'
    )

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn_correspondence
    )
    return dataloader


def run_evaluation(extractor, dataloader, image_size: int):
    os.chdir("/content")
    matcher_eval = CorrespondenceMatcher(extractor)
    metrics_after = evaluate_model(
        matcher_eval,
        dataloader,
        run_name="{MODEL}with{N_EPOCHS}epochsImages{IMAGE_SIZE}with{N_UNFREEZE_LAYERS}LayersMetrics"
    )
    return metrics_after


def save_and_report(metrics_after, image_size: int):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = "dinov2_vitb14"
    results_dict = {
        'metrics': metrics_after,
        'model': model_name,
        'timestamp': timestamp
    }
    with open(f'results_{model_name}_{image_size}_{timestamp}.json', 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"Results saved!")
    analyzer = ResultsAnalyzer(metrics_after)
    analyzer.generate_report(save_dir='./results/{MODEL}with{N_EPOCHS}epochsImages{IMAGE_SIZE}with{N_UNFREEZE_LAYERS}Layers')
    summary_df = analyzer.create_summary_table(threshold=0.10)
    print(summary_df)



def main():
    image_size = IMAGE_SIZE 

    train_loader = build_train_loader(image_size)
    extractor = create_extractor(MODEL)
    run_training(extractor, train_loader)
    dataloader = build_test_loader(image_size)
    metrics_after = run_evaluation(extractor, dataloader, image_size)

    save_and_report(metrics_after, image_size)


if __name__ == "__main__":
    main()
