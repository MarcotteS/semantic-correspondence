import torch
import torch.nn.functional as F


class DINOv2Extractor:
    def __init__(self, model_name="dinov2_vitb14"):
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.patch_size = self.model.patch_size

    def extract(self, img: torch.Tensor, no_grad: bool = True):
        """
        Extract features from image(s).

        Args:
            img: [3,H,W] or [B,3,H,W]
            no_grad: True for inference/eval, False for fine-tuning (keeps gradients)

        Returns:
            features: [B, H_p*W_p, D]
            (H_p, W_p)
        """
        # Ensure batch dimension
        if img.dim() == 3:
            img = img.unsqueeze(0)

        assert img.dim() == 4, f"Expected 4D tensor, got {img.dim()}D"
        B, C, H, W = img.shape
        assert C == 3, f"Expected 3 channels, got {C}"

        img = img.to(self.device)

        # mode: in training loop handle model.train(), otherwise eval
        if no_grad:
            self.model.eval()
            with torch.no_grad():
                out = self.model.forward_features(img)
        else:
            # IMPORTANT: no no_grad here
            out = self.model.forward_features(img)

        features = out["x_norm_patchtokens"]

        H_p = H // self.patch_size
        W_p = W // self.patch_size

        return features, (H_p, W_p)


class DINOv3Extractor:
    def __init__(
        self,
        model_name="dinov3_vitb16",
        repo_dir="<PATH/TO/A/LOCAL/DIRECTORY/WHERE/THE/DINOV3/REPO/WAS/CLONED>",
        weights="<CHECKPOINT/URL/OR/PATH>",
    ):
        """
        Initialize DINOv3 feature extractor.
        - repo_dir: local path of the cloned dinov3 repo (source="local")
        - weights: checkpoint / URL / path
        """
        self.model = torch.hub.load(repo_dir, model_name, source="local", weights=weights)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # IMPORTANT: do not force eval() here, since training loop handles model.train()
        # (you can leave eval() by default, it won't stop the finetuning as long as
        #  you put model.train() before)
        self.model.eval()

        self.patch_size = self.model.patch_size

    def extract(self, img: torch.Tensor, no_grad: bool = True):
        """
        Extract features from image(s).

        Args:
            img: [3,H,W] or [B,3,H,W]
            no_grad: True for inference, False for fine-tuning (keeps the gradients)

        Returns:
            features: [B, H_p*W_p, D]
            (H_p, W_p)
        """
        # Ensure batch dimension
        if img.dim() == 3:
            img = img.unsqueeze(0)

        assert img.dim() == 4, f"Expected 4D tensor, got {img.dim()}D"
        B, C, H, W = img.shape
        assert C == 3, f"Expected 3 channels, got {C}"

        img = img.to(self.device)

        # Mode: training loop handles model.train(), otherwise eval
        if no_grad:
            self.model.eval()
            with torch.no_grad():
                out = self.model.forward_features(img)
        else:
            # IMPORTANT: no no_grad here
            out = self.model.forward_features(img)

        # DINO format
        features = out["x_norm_patchtokens"]

        H_p = H // self.patch_size
        W_p = W // self.patch_size

        return features, (H_p, W_p)


import torch
import torch.nn.functional as F

class SAMExtractor:
    def __init__(
        self,
        model_type="vit_b",
        checkpoint_path="sam_vit_b_01ec64.pth",
        image_size=512,
        finetune_pos_embed=True,
        freeze_image_encoder=False,
    ):
        """
        Initialize SAM (Segment Anything Model) feature extractor.

        Args:
            model_type: 'vit_b', 'vit_l', 'vit_h'
            checkpoint_path: path to SAM checkpoint
            image_size: target size (e.g. 512). Interpolates pos_embed from 1024->image_size.
            finetune_pos_embed: if True, pos_embed remains trainable after interpolation
            freeze_image_encoder: if True, freezes image_encoder params (except pos_embed if finetune_pos_embed)
        """
        from segment_anything import sam_model_registry

        self.model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Patch size for SAM image encoder (1024 -> 64 tokens => 16 px/patch)
        self.patch_size = 16
        self.original_size = 1024
        self.image_size = image_size
        self.finetune_pos_embed = finetune_pos_embed

        # Default mode (tu peux override avec model.train() dans ton training loop)
        self.model.eval()

        # Interpolate pos embed if needed
        if self.image_size != self.original_size:
            self._interpolate_pos_embed()

        # Optionnel: freeze/défreeze
        if freeze_image_encoder:
            self.freeze_image_encoder(keep_pos_embed_trainable=finetune_pos_embed)

    def freeze_image_encoder(self, keep_pos_embed_trainable: bool = True):
        """Freeze SAM image encoder parameters (optionally keep pos_embed trainable)."""
        for p in self.model.image_encoder.parameters():
            p.requires_grad = False

        if keep_pos_embed_trainable and hasattr(self.model.image_encoder, "pos_embed"):
            self.model.image_encoder.pos_embed.requires_grad = True

    def unfreeze_image_encoder(self):
        """Unfreeze SAM image encoder parameters."""
        for p in self.model.image_encoder.parameters():
            p.requires_grad = True

    def _interpolate_pos_embed(self):
        """
        Interpolate positional embeddings to match the target image size.
        SAM pos_embed is [1, H, W, D] for 64x64 (1024/16).
        For 512x512 => 32x32.
        """
        pos_embed = self.model.image_encoder.pos_embed  # [1, H, W, D]
        original_grid = self.original_size // self.patch_size
        new_grid = self.image_size // self.patch_size

        if original_grid == new_grid:
            return

        # [1, H, W, D] -> [1, D, H, W]
        pos = pos_embed.permute(0, 3, 1, 2)

        pos_interp = F.interpolate(
            pos,
            size=(new_grid, new_grid),
            mode="bilinear",
            align_corners=False,
        )

        # back to [1, H, W, D]
        pos_interp = pos_interp.permute(0, 2, 3, 1).contiguous()

        # IMPORTANT: garder trainable si demandé
        self.model.image_encoder.pos_embed = torch.nn.Parameter(
            pos_interp,
            requires_grad=bool(self.finetune_pos_embed),
        )

    def extract(self, img: torch.Tensor, no_grad: bool = True):
        """
        Extract patch-token features from SAM image encoder.

        Args:
            img: [3, H, W] or [B, 3, H, W], with H=W=image_size
            no_grad: True for inference, False for fine-tuning (keeps gradients)

        Returns:
            features: [B, H_p*W_p, D]
            (H_p, W_p)
        """
        if img.dim() == 3:
            img = img.unsqueeze(0)

        assert img.dim() == 4, f"Expected 4D tensor, got {img.dim()}D"
        B, C, H, W = img.shape
        assert C == 3, f"Expected 3 channels, got {C}"
        assert H == self.image_size and W == self.image_size, \
            f"SAM requires {self.image_size}x{self.image_size} images, got {H}x{W}"

        img = img.to(self.device)

        # Mode: ton training loop peut faire model.train() ; ici on suit le flag no_grad
        if no_grad:
            self.model.eval()
            with torch.no_grad():
                feats = self.model.image_encoder(img)  # [B, D, H_p, W_p]
        else:
            feats = self.model.image_encoder(img)      # grads OK

        B, D, H_p, W_p = feats.shape
        feats = feats.permute(0, 2, 3, 1).reshape(B, H_p * W_p, D)  # [B, H_p*W_p, D]

        return feats, (H_p, W_p)
