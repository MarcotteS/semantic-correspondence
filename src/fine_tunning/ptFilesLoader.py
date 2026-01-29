"""
this methods allows to load a finetuned model from a local .pt checkpoint
the .pt files are available on drive: https://drive.google.com/drive/u/0/folders/130C33edJ_vrh-boOOQ2n9IdUUNqlIvVi
they should be upload on collab, please wait that the upload is finished before calling this method.

Example of usage:
ckpt_path = "/content/DinOV2with1epochsImages518with1Layers.pt"


matcher = load_matcher_from_drive_ckpt(
    ckpt_path,
    feature_extractor=extractor,
)

with extractor the excrator corresponding to the checkpoint, dinov2, dinov3 or Sam
"""
from fine_tunning.CorrespondenceMatcher2 import CorrespondenceMatcher2
import torch
import os
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt



def load_matcher_from_drive_ckpt(
    ckpt_path: str,
    feature_extractor,
    device: str | None = None,
    load_optimizer: bool = False,
    load_scaler: bool = False,
):
   
    # device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # 1) créer matcher
    matcher = CorrespondenceMatcher2(feature_extractor)
    matcher.device = device

    model = matcher.extractor.model
    model.to("cpu")  # chargement safe

    # 2) load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if "model" not in ckpt:
        raise KeyError("Checkpoint invalide: clé 'model' absente")

    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()

    outputs = [matcher]

    # 3) optimizer (optionnel)
    if load_optimizer:
        optimizer = make_optimizer(model)
        optimizer.load_state_dict(ckpt["optimizer"])
        outputs.append(optimizer)

    # 4) scaler AMP (optionnel)
    if load_scaler:
        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
        if ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        outputs.append(scaler)

    return outputs[0] if len(outputs) == 1 else tuple(outputs)