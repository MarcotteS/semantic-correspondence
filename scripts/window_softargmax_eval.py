#!/usr/bin/env python3
"""
Evaluate models with Window Soft-Argmax on SPair-71k.
"""

import sys
from pathlib import Path
from datetime import datetime
from itertools import product

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.dataset import SPairDataset, collate_fn_correspondence
from src.models import DINOv2Extractor, DINOv3Extractor, SAMExtractor
from src.correspondence_softargmax import WindowSoftArgmaxMatcher
from src.evaluation import CorrespondenceEvaluator


# --- Config ---

RUN_DINOV2 = False
RUN_DINOV3 = False
RUN_SAM = False
RUN_ALL_FINETUNED = True
RUN_HYPERPARAM_SEARCH = False

# Checkpoints (None = baseline)
DINOV2_CHECKPOINT = None
DINOV3_CHECKPOINT = None
SAM_CHECKPOINT = None

# Paths
SAM_BASE_CKPT = repo_root / "backbones" / "segment-anything" / "checkpoints" / "sam_vit_b_01ec64.pth"
DINOV3_REPO = str(repo_root / "backbones" / "dinov3")
DINOV3_CKPT = str(repo_root / "backbones" / "dinov3" / "checkpoints" / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")
FINETUNED_DIR = repo_root / "fintunned"

# Hyperparams
WINDOW_SIZE = 5
TEMPERATURE = 0.05
BATCH_SIZE = 16
NUM_WORKERS = 4
CATEGORY = "all"

# For grid search
SEARCH_WINDOW_SIZES = [3, 5, 7, 9, 11]
SEARCH_TEMPS = [0.01, 0.05, 0.1, 0.2, 0.5]
SEARCH_MODEL = "dinov2"

IMG_SIZES = {"dinov2": 518, "dinov3": 512, "sam": 512}

FINETUNED_MODELS = {
    "DINOv2_1ep_2layers": ("dinov2", FINETUNED_DIR / "DinOV2with1epochsImages518with2Layers.pt"),
    "DINOv2_2ep_1layer": ("dinov2", FINETUNED_DIR / "DinOV2with2epochsImages518with1Layers.pt"),
    "DINOv3_3ep_1layer": ("dinov3", FINETUNED_DIR / "DINOv3with3epochsImages518with1Layers.pt"),
    "DINOv3_3ep_2layers": ("dinov3", FINETUNED_DIR / "DINO3with3epochsImages518with2Layers.pt"),
    "SAM_1ep_1layer": ("sam", FINETUNED_DIR / "SAMwith1epochsImages512with1Layers.pt"),
    "DINOv2_1ep_1layer": ("dinov2", FINETUNED_DIR / "DinOV2with1epochsImages518with1Layers.pt"),
    "DINOv3_1ep_1layer": ("dinov3", FINETUNED_DIR / "DINOv3with1epochsImages518with1Layers.pt"),
}


def create_extractor(backbone, checkpoint=None):
    img_size = IMG_SIZES[backbone]
    print(f"  Loading {backbone}...")

    if backbone == "dinov2":
        ext = DINOv2Extractor(model_name="dinov2_vitb14")
    elif backbone == "dinov3":
        ext = DINOv3Extractor(model_name="dinov3_vitb16", repo_dir=DINOV3_REPO, weights=DINOV3_CKPT)
    elif backbone == "sam":
        ext = SAMExtractor(model_type="vit_b", checkpoint_path=str(SAM_BASE_CKPT), image_size=img_size)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    if checkpoint is not None:
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        print(f"  Loading {checkpoint.name}")
        ckpt = torch.load(checkpoint, map_location=ext.device)
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        msg = ext.model.load_state_dict(state_dict, strict=False)
        if msg.missing_keys or msg.unexpected_keys:
            print(f"  (missing: {len(msg.missing_keys)}, unexpected: {len(msg.unexpected_keys)})")

    ext.model.eval()
    return ext, img_size


def evaluate(extractor, img_size, ws=WINDOW_SIZE, temp=TEMPERATURE, desc="Eval"):
    dataset = SPairDataset(datapath=str(repo_root / "data"), split='test', img_size=img_size, category=CATEGORY)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, collate_fn=collate_fn_correspondence, pin_memory=True)

    matcher = WindowSoftArgmaxMatcher(extractor, window_size=ws, temperature=temp)
    evaluator = CorrespondenceEvaluator(thresholds=[0.05, 0.10, 0.15, 0.20])

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            preds = matcher.find_correspondences(batch["src_img"], batch["trg_img"], batch["src_kps"])
            for b in range(batch["src_img"].shape[0]):
                evaluator.update(preds[b], {
                    "trg_kps": batch["trg_kps"][b], "pckthres": batch["pckthres"][b],
                    "n_pts": batch["n_pts"][b], "kps_ids": batch["kps_ids"][b],
                    "pair_idx": batch["pair_idx"][b], "category": batch["category"][b],
                })

    return evaluator.get_metrics()


def print_pck(name, metrics):
    print(f"\n  {name}:")
    for t, pck in metrics['overall'].items():
        print(f"    PCK@{t:.2f}: {pck:.2f}%")


def run_single(backbone, checkpoint=None):
    name = backbone + (f" ({checkpoint.name})" if checkpoint else " (baseline)")
    print(f"\n{'='*60}\nEvaluating: {name}\nws={WINDOW_SIZE}, temp={TEMPERATURE}\n{'='*60}")

    ext, img_size = create_extractor(backbone, checkpoint)
    metrics = evaluate(ext, img_size, desc=backbone)
    print_pck(name, metrics)

    del ext
    torch.cuda.empty_cache()
    return metrics


def run_all_finetuned():
    print(f"{'='*60}")
    print(f"Window Soft-Argmax Evaluation - All Fine-Tuned Models")
    print(f"ws={WINDOW_SIZE}, temp={TEMPERATURE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}")

    results = []
    n = len(FINETUNED_MODELS)

    for i, (name, (backbone, ckpt)) in enumerate(FINETUNED_MODELS.items(), 1):
        print(f"\n[{i}/{n}] {name}")

        try:
            ext, img_size = create_extractor(backbone, ckpt)
            metrics = evaluate(ext, img_size, desc=name)

            results.append({
                "model": name, "backbone": backbone,
                **{f"pck_{t}": metrics["overall"][t] for t in [0.05, 0.10, 0.15, 0.20]}
            })
            print_pck(name, metrics)

            del ext
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  [ERROR] {e}")

    if results:
        df = pd.DataFrame(results)
        print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
        print(df.to_string(index=False))

        out_dir = repo_root / "results" / "window_softargmax"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"finetuned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(out_path, index=False)
        print(f"\nSaved to: {out_path}")

        best = df.loc[df["pck_0.1"].idxmax()]
        print(f"Best: {best['model']} (PCK@0.10: {best['pck_0.1']:.2f}%)")


def run_search():
    print(f"{'='*60}\nHyperparam search: {SEARCH_MODEL}")
    print(f"Window sizes: {SEARCH_WINDOW_SIZES}")
    print(f"Temperatures: {SEARCH_TEMPS}\n{'='*60}")

    ext, img_size = create_extractor(SEARCH_MODEL)
    results = []
    total = len(SEARCH_WINDOW_SIZES) * len(SEARCH_TEMPS)

    for i, (ws, temp) in enumerate(product(SEARCH_WINDOW_SIZES, SEARCH_TEMPS), 1):
        print(f"[{i}/{total}] ws={ws}, temp={temp}")
        metrics = evaluate(ext, img_size, ws=ws, temp=temp, desc=f"ws={ws}")
        results.append({
            "window_size": ws, "temperature": temp,
            **{f"pck_{t}": metrics["overall"][t] for t in [0.05, 0.10, 0.15, 0.20]}
        })
        print(f"  -> PCK@0.10: {results[-1]['pck_0.1']:.2f}%")

    df = pd.DataFrame(results)
    out_dir = repo_root / "results" / "hyperparam_search"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{SEARCH_MODEL}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(out_path, index=False)

    best = df.loc[df["pck_0.1"].idxmax()]
    print(f"\nBest: ws={int(best['window_size'])}, temp={best['temperature']}, PCK@0.10={best['pck_0.1']:.2f}%")
    print(f"Saved to: {out_path}")


def main():
    if torch.cuda.is_available():
        print(f"\nUsing GPU: {torch.cuda.get_device_name(0)}")

    if RUN_ALL_FINETUNED:
        run_all_finetuned()
    if RUN_DINOV2:
        run_single("dinov2", DINOV2_CHECKPOINT)
    if RUN_DINOV3:
        run_single("dinov3", DINOV3_CHECKPOINT)
    if RUN_SAM:
        run_single("sam", SAM_CHECKPOINT)
    if RUN_HYPERPARAM_SEARCH:
        run_search()

    if not any([RUN_DINOV2, RUN_DINOV3, RUN_SAM, RUN_ALL_FINETUNED, RUN_HYPERPARAM_SEARCH]):
        print("\nNothing to run. Set one of the RUN_* flags to True.")


if __name__ == "__main__":
    main()
