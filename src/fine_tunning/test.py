import torch
import os
import sys
import glob
from tqdm import tqdm
import random
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
from datetime import datetime, time
import pandas as pd
class CorrespondenceMatcher2:
    def __init__(self, feature_extractor):
        self.extractor = feature_extractor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def find_correspondences(self, src_img, trg_img, src_kps):
        """
        Version pour fine-tuning: renvoie les scores de similarité (différentiables)
        plutôt que des coordonnées.
        Returns:
            sim_matrix: [B, N, L]
            grid_size: (h_p, w_p)
            valid_mask: [B, N]
        """
        # Batch handling
        is_batched = src_img.dim() == 4
        if not is_batched:
            src_img = src_img.unsqueeze(0)
            trg_img = trg_img.unsqueeze(0)
            src_kps = src_kps.unsqueeze(0)

        # Move to device
        src_img = src_img.to(self.device)
        trg_img = trg_img.to(self.device)
        src_kps = src_kps.to(self.device)

        B, N, _ = src_kps.shape

        # Extract features
        src_feats, (h_p, w_p) = self.extractor.extract(src_img, no_grad=False)
        trg_feats, _ = self.extractor.extract(trg_img, no_grad=False)

        D = src_feats.shape[-1]
        patch_size = self.extractor.patch_size

        # Normalize for cosine similarity
        src_feats = F.normalize(src_feats, dim=-1)
        trg_feats = F.normalize(trg_feats, dim=-1)

        # Valid keypoints
        valid_mask = (src_kps[..., 0] >= 0)  # [B,N]

        # Source keypoints (pixels) -> patch indices
        kps_grid = (src_kps / patch_size).long()
        grid_x = kps_grid[..., 0].clamp(0, w_p - 1)
        grid_y = kps_grid[..., 1].clamp(0, h_p - 1)
        flat_indices = grid_y * w_p + grid_x  # [B,N]

        # Gather source features at keypoint locations
        flat_indices_expanded = flat_indices.unsqueeze(-1).expand(-1, -1, D)  # [B,N,D]
        src_kp_feats = torch.gather(src_feats, 1, flat_indices_expanded)      # [B,N,D]

        # Similarity scores (DIFFÉRENTIABLE)
        sim_matrix = torch.bmm(src_kp_feats, trg_feats.transpose(1, 2))       # [B,N,L]

        # Si input non batché, on peut squeeze les masks si tu veux, mais pas obligatoire
        return sim_matrix, (h_p, w_p), valid_mask


def kps_to_flat_indices(kps, patch_size, h_p, w_p):
    kps_grid = (kps / patch_size).long()
    grid_x = kps_grid[..., 0].clamp(0, w_p - 1)
    grid_y = kps_grid[..., 1].clamp(0, h_p - 1)
    return grid_y * w_p + grid_x  # [B,N]


def correspondence_loss_ce(sim_matrix, trg_kps, patch_size, h_p, w_p, valid_src_mask, tau=0.07):
    """
    sim_matrix: [B,N,L]
    trg_kps: [B,N,2] keypoints annotés dans l'image cible (-2 si invalide)
    """
    B, N, L = sim_matrix.shape
    trg_kps = trg_kps.to(sim_matrix.device)

    valid = valid_src_mask & (trg_kps[..., 0] >= 0)  # [B,N]
    labels = kps_to_flat_indices(trg_kps, patch_size, h_p, w_p)  # [B,N]

    logits = (sim_matrix / tau).reshape(B * N, L)
    labels = labels.reshape(B * N)
    mask = valid.reshape(B * N)

    logits = logits[mask]
    labels = labels[mask]

    if logits.shape[0] == 0:
        return None

    return F.cross_entropy(logits, labels)



# utilise TA loss existante
# from ton_fichier_loss import correspondence_loss_ce

def unfreeze_last_blocks_dino(model, n_last_blocks=1):
    """
    Gèle tout puis dégèle les n derniers blocs du ViT (DINO).
    """
    for p in model.parameters():
        p.requires_grad = False

    if not hasattr(model, "blocks"):
        raise AttributeError("Le modèle n'a pas d'attribut .blocks (ViT). Adapte unfreeze_last_blocks_dino().")

    for blk in model.blocks[-n_last_blocks:]:
        for p in blk.parameters():
            p.requires_grad = True


def make_optimizer(model, lr=2e-5, weight_decay=0.01):
    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) == 0:
        raise ValueError("Aucun paramètre entraînable. Vérifie unfreeze_last_blocks_dino().")
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

def train_stage2(
    matcher,
    train_loader,
    n_epochs=1,              # <-- TOTAL epochs à atteindre
    n_last_blocks=1,
    lr=2e-5,
    weight_decay=0.01,
    tau=0.07,
    max_batches_per_epoch=None,
    use_amp=True,
):
    import os
    import torch
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from IPython.display import clear_output

    loss_history = []
    step_history = []

    plt.ion()   # mode interactif
    fig, ax = plt.subplots()
    line, = ax.plot([], [])
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Training loss (live)")
    last_plot_time = time.time()

    def _ckpt_path():
        ckpt_dir = getattr(matcher, "ckpt_dir", None)
        if not ckpt_dir:
            return None
        exp_name = getattr(matcher, "exp_name", "stage2")
        return os.path.join(ckpt_dir, exp_name, "last.pt")

    def _save_ckpt(path, model, optimizer, scaler, epoch, step):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "epoch": epoch,  # 0-based
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": (scaler.state_dict() if (scaler is not None and scaler.is_enabled()) else None),
            "meta": {
                "lr": lr,
                "weight_decay": weight_decay,
                "tau": tau,
                "n_last_blocks": n_last_blocks,
            },
        }
        torch.save(payload, path)

    def _load_ckpt(path, model, optimizer, scaler):
        ckpt = torch.load(path, map_location=matcher.device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scaler is not None and scaler.is_enabled() and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        return int(ckpt.get("epoch", 0)), int(ckpt.get("step", 0))

    # 0) Model
    model = matcher.extractor.model

    # 1) Unfreeze last layers
    unfreeze_last_blocks_dino(model, n_last_blocks=n_last_blocks)

    # 2) Optimizer
    optimizer = make_optimizer(model, lr=lr, weight_decay=weight_decay)

    # 3) Train mode
    model.train()

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and torch.cuda.is_available()))
    patch_size = matcher.extractor.patch_size

    nb_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("nb trainable params:", nb_trainable)

    ckpt_file = _ckpt_path()
    do_resume = getattr(matcher, "resume", True)
    save_every_epoch = int(getattr(matcher, "save_every_epoch", 1))

    start_epoch_offset = 0
    global_step = 0

    # ---- RESUME / EARLY EXIT ----
    if ckpt_file is not None and do_resume and os.path.exists(ckpt_file):
        last_epoch, global_step = _load_ckpt(ckpt_file, model, optimizer, scaler)
        completed = last_epoch + 1  # epochs déjà faites (car epoch est 0-based)
        print(f"[ckpt] chargé: epoch={completed} step={global_step} ({ckpt_file})")

        # Si on a déjà atteint (ou dépassé) n_epochs total -> on ne fait rien
        if completed >= n_epochs:
            print(f"[ckpt] déjà à {completed} epochs (objectif={n_epochs}) -> skip training")
            return matcher

        start_epoch_offset = completed  # reprendre au prochain epoch
        print(f"[ckpt] reprise à epoch={start_epoch_offset+1}/{n_epochs}")
    elif ckpt_file is not None:
        os.makedirs(os.path.dirname(ckpt_file), exist_ok=True)
        print(f"[ckpt] pas de checkpoint -> from scratch ({ckpt_file})")

    # ---- TRAIN UNIQUEMENT LES EPOCHS MANQUANTES ----
    for epoch in range(start_epoch_offset, n_epochs):
        running = 0.0
        steps = 0

        pbar = tqdm(train_loader, desc=f"Stage2 epoch {epoch+1}/{n_epochs}")
        for i, batch in enumerate(pbar):
            if max_batches_per_epoch is not None and i >= max_batches_per_epoch:
                break

            src_img = batch["src_img"]
            trg_img = batch["trg_img"]
            src_kps = batch["src_kps"]
            trg_kps = batch["trg_kps"]

            optimizer.zero_grad(set_to_none=True)

            if scaler.is_enabled():
                with torch.amp.autocast("cuda"):
                    sim_matrix, (h_p, w_p), valid_mask = matcher.find_correspondences(src_img, trg_img, src_kps)
                    loss = correspondence_loss_ce(sim_matrix, trg_kps, patch_size, h_p, w_p, valid_mask, tau=tau)
                if loss is None:
                    continue
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                sim_matrix, (h_p, w_p), valid_mask = matcher.find_correspondences(src_img, trg_img, src_kps)
                loss = correspondence_loss_ce(sim_matrix, trg_kps, patch_size, h_p, w_p, valid_mask, tau=tau)
                if loss is None:
                    continue
                loss.backward()
                optimizer.step()

            running += float(loss.item())
            steps += 1
            global_step += 1
            loss_history.append(float(loss.item()))
            step_history.append(global_step)

# update toutes les 10 secondes
            if time.time() - last_plot_time > 10:
                line.set_xdata(step_history)
                line.set_ydata(loss_history)
                ax.relim()
                ax.autoscale_view()

                clear_output(wait=True)
                display(fig)

                last_plot_time = time.time()


            pbar.set_postfix(loss=running / max(1, steps))

        avg = running / max(1, steps)
        print(f"Epoch {epoch+1}: avg loss = {avg:.6f}")

        if ckpt_file is not None and save_every_epoch > 0 and ((epoch + 1) % save_every_epoch == 0):
            _save_ckpt(ckpt_file, model, optimizer, scaler, epoch, global_step)
            print(f"[ckpt] sauvegardé: {ckpt_file}")

    return matcher


def save_checkpoint(path, model, optimizer, epoch):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path)

def load_checkpoint(path, model, optimizer):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["epoch"]


def plot_loss_steps(step_history, loss_history, smooth=50):
    steps = np.array(step_history)
    losses = np.array(loss_history)

    plt.figure()
    plt.plot(steps, losses, label="batch loss")

    if smooth and smooth > 1:
        kernel = np.ones(smooth) / smooth
        smoothed = np.convolve(losses, kernel, mode="valid")
        plt.plot(steps[smooth-1:], smoothed, label=f"moving avg ({smooth})")

    plt.xlabel("global step")
    plt.ylabel("loss")
    plt.title("Loss per mini-batch")
    plt.legend()
    plt.show()