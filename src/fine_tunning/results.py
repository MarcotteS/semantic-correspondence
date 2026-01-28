# the .pt files are available on drive: https://drive.google.com/drive/u/0/folders/130C33edJ_vrh-boOOQ2n9IdUUNqlIvVi
#because genretae the metrics is very long, some metrics are also available on the drive
#it is redcomended to use these methods to get a matcher from a .pt file:
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
    """
    Charge un CorrespondenceMatcher2 depuis un checkpoint .pt stocké sur Drive.

    Args:
        ckpt_path (str): chemin vers le .pt (ex: /content/drive/MyDrive/exp/last.pt)
        feature_extractor: extractor déjà construit (même arch que lors du training)
        device (str | None): "cuda", "cpu" ou None (auto)
        load_optimizer (bool): si True, retourne aussi l'optimizer
        load_scaler (bool): si True, retourne aussi le GradScaler

    Returns:
        matcher
        (+ optimizer, scaler si demandés)
    """

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