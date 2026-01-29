import torch
import os
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from .CorrespondenceMatcher2 import correspondence_loss_ce
#this method is used for train the models, if the matcher name already exists and associeted with a .pt file
#the fonction return the checkpoint or resume the training if all the epochs are not yet finished
#the checkpoints containts information such as the wheights, loss history etc...
def train_stage2(
    matcher,
    train_loader,
    n_epochs=1,              #just this two hyperparameters were changed fort the training
    n_last_blocks=1,         #because the training was to long to experiment with too many hyperparameters and collab limitations were quickly reached
    lr=2e-5,
    weight_decay=0.01,
    tau=0.07,
    max_batches_per_epoch=None,
    use_amp=True,isSam=False
):
    import os
    import torch
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from IPython.display import clear_output, display
    import time as pytime

    # --- live plot setup ---
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [])
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Training loss (live)")
    last_plot_time = pytime.time()

    # --- in-memory history (will be saved into checkpoint) ---
    loss_history = []
    step_history = []

    def _ckpt_path():
        ckpt_dir = getattr(matcher, "ckpt_dir", None)
        if not ckpt_dir:
            return None
        exp_name = getattr(matcher, "exp_name", "stage2")
        return os.path.join(ckpt_dir, exp_name, "last.pt")

    def _save_ckpt(path, model, optimizer, scaler, epoch, step, loss_hist, step_hist):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "epoch": epoch,  # 0-based
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": (scaler.state_dict() if (scaler is not None and scaler.is_enabled()) else None),
            "loss_history": loss_hist,
            "step_history": step_hist,
            "meta": {
                "lr": lr,
                "weight_decay": weight_decay,
                "tau": tau,
                "n_last_blocks": n_last_blocks,
            },
        }
        torch.save(payload, path)

    def _load_ckpt(path, model, optimizer, scaler):
        ckpt = torch.load(path, map_location=matcher.device,weight_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scaler is not None and scaler.is_enabled() and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])

        loss_hist = ckpt.get("loss_history", [])
        step_hist = ckpt.get("step_history", [])
        return int(ckpt.get("epoch", 0)), int(ckpt.get("step", 0)), loss_hist, step_hist

    # 0) Model
    model = matcher.extractor.model

    # 1) Unfreeze last layers
    if isSam:
        unfreeze_last_blocks_dino(model.image_encoder, n_last_blocks=n_last_blocks)
    else:
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
        last_epoch, global_step, loss_history, step_history = _load_ckpt(ckpt_file, model, optimizer, scaler)
        completed = last_epoch + 1
        print(f"[ckpt] chargé: epoch={completed} step={global_step} ({ckpt_file})")

        # repaint the plot once after loading history
        if len(step_history) > 0:
            line.set_xdata(step_history)
            line.set_ydata(loss_history)
            ax.relim()
            ax.autoscale_view()
            clear_output(wait=True)
            display(fig)

        if completed >= n_epochs:
            print(f"[ckpt] déjà à {completed} epochs (objectif={n_epochs}) -> skip training")
            return matcher

        start_epoch_offset = completed
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
            step_history.append(int(global_step))

            # update plot every 10 seconds
            if pytime.time() - last_plot_time > 10:
                line.set_xdata(step_history)
                line.set_ydata(loss_history)
                ax.relim()
                ax.autoscale_view()
                clear_output(wait=True)
                display(fig)
                last_plot_time = pytime.time()

            pbar.set_postfix(loss=running / max(1, steps))

        avg = running / max(1, steps)
        print(f"Epoch {epoch+1}: avg loss = {avg:.6f}")

        if ckpt_file is not None and save_every_epoch > 0 and ((epoch + 1) % save_every_epoch == 0):
            _save_ckpt(
                ckpt_file,
                model,
                optimizer,
                scaler,
                epoch,
                global_step,
                loss_history,
                step_history,
            )
            print(f"[ckpt] sauvegardé: {ckpt_file}")

    return matcher


def plot_loss_steps(step_history, loss_history, smooth=50):
    steps = np.array(step_history)
    losses = np.array(loss_history)

    plt.figure()
    plt.plot(steps, losses, label="batch loss")

    if smooth and smooth > 1 and len(losses) >= smooth:
        kernel = np.ones(smooth) / smooth
        smoothed = np.convolve(losses, kernel, mode="valid")
        plt.plot(steps[smooth-1:], smoothed, label=f"moving avg ({smooth})")

    plt.xlabel("global step")
    plt.ylabel("loss")
    plt.title("Loss per mini-batch")
    plt.legend()
    plt.show()



def unfreeze_last_blocks_dino(model, n_last_blocks=1):
    for p in model.parameters():
        p.requires_grad = False

    if not hasattr(model, "blocks"):
        raise AttributeError("Le modèle n'a pas d'attribut .blocks (ViT).")

    for blk in model.blocks[-n_last_blocks:]:
        for p in blk.parameters():
            p.requires_grad = True
def unfreeze_last_blocks_vit(obj, n_last_blocks=1):
    """
    Accepte soit:
    - un ViT direct (DINO): obj.blocks
    - un SAM: obj.image_encoder.blocks
    - un wrapper (matcher): cherche un sous-attribut qui contient le ViT/SAM
    """

    # 1) Si on nous donne directement DINO ViT
    if hasattr(obj, "blocks"):
        container = obj

    # 2) Si on nous donne directement SAM
    elif hasattr(obj, "image_encoder") and hasattr(obj.image_encoder, "blocks"):
        container = obj.image_encoder

    else:
        # 3) Sinon: on suppose que c'est un wrapper (matcher) et on cherche un sous-module probable
        candidates = [
            "backbone", "model", "net", "encoder", "extractor", "feature_extractor",
            "sam", "dino", "vit", "image_encoder"
        ]

        found = None
        for name in candidates:
            if hasattr(obj, name):
                found = getattr(obj, name)
                # DINO
                if hasattr(found, "blocks"):
                    container = found
                    break
                # SAM
                if hasattr(found, "image_encoder") and hasattr(found.image_encoder, "blocks"):
                    container = found.image_encoder
                    break
        else:
            raise AttributeError(
                "Je ne trouve pas les blocs ViT. "
                "Ni obj.blocks (DINO), ni obj.image_encoder.blocks (SAM), "
                "ni via les attributs usuels (model/encoder/extractor/etc.)."
            )

    # Freeze tout le container (ViT ou image_encoder)
    for p in container.parameters():
        p.requires_grad = False

    # Unfreeze les derniers blocs
    for blk in container.blocks[-n_last_blocks:]:
        for p in blk.parameters():
            p.requires_grad = True

    # (Option utile SAM) garder pos_embed entraînable si présent
    if hasattr(container, "pos_embed") and isinstance(container.pos_embed, torch.nn.Parameter):
        container.pos_embed.requires_grad = True




def make_optimizer(model, lr=2e-5, weight_decay=0.01):
    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) == 0:
        raise ValueError("Aucun paramètre entraînable.")
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
