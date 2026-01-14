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
    n_epochs=1,
    n_last_blocks=1,
    lr=2e-5,
    weight_decay=0.01,
    tau=0.07,
    max_batches_per_epoch=None,
    use_amp=True,
):
    """
    Fine-tuning étape 2 avec ta fonction matcher.find_correspondences() + correspondence_loss_ce().
    """
    model = matcher.extractor.model

    # 1) Unfreeze last layers
    unfreeze_last_blocks_dino(model, n_last_blocks=n_last_blocks)

    # 2) Optimizer
    optimizer = make_optimizer(model, lr=lr, weight_decay=weight_decay)

    # 3) Train mode
    model.train()

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and torch.cuda.is_available()))
    patch_size = matcher.extractor.patch_size
    print("nb trainable params:", sum(p.requires_grad for p in model.parameters()))
    for epoch in range(n_epochs):
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
                with torch.cuda.amp.autocast():
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
                if i == 0:  # juste au 1er batch
                     print("cuda available:", torch.cuda.is_available())
                     print("matcher.device:", matcher.device)

                     # est-ce qu'il y a des params entraînables ?
                     model = matcher.extractor.model
                     print("trainable params:", sum(p.requires_grad for p in model.parameters()))

                     # est-ce que le forward est différentiable ?
                     print("sim_matrix.requires_grad:", sim_matrix.requires_grad)
                     print("loss.requires_grad:", loss.requires_grad)

                if loss is None:
                    continue
                loss.backward()
                optimizer.step()

            running += float(loss.item())
            steps += 1
            pbar.set_postfix(loss=running / max(1, steps))

        print(f"Epoch {epoch+1}: avg loss = {running / max(1, steps):.6f}")

    return matcher  # le matcher contient le modèle mis à jour


