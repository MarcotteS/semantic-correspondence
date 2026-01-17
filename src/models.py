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

        # mode: en training loop tu gères model.train(), sinon eval
        if no_grad:
            self.model.eval()
            with torch.no_grad():
                out = self.model.forward_features(img)
        else:
            # IMPORTANT: pas de no_grad ici
            out = self.model.forward_features(img)

        features = out["x_norm_patchtokens"]

        H_p = H // self.patch_size
        W_p = W // self.patch_size

        return features, (H_p, W_p)

class DINOv3Extractor:
    def __init__(self, model_name="dinov3_vitb16",
                 repo_dir="<PATH/TO/A/LOCAL/DIRECTORY/WHERE/THE/DINOV3/REPO/WAS/CLONED>",
                 weights="<CHECKPOINT/URL/OR/PATH>", ):
        """Initialize DINOv3 feature extractor."""
        self.model = torch.hub.load(repo_dir, model_name, source="local", weights=weights)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        self.patch_size = self.model.patch_size

    @torch.no_grad()
    def extract(self, img: torch.Tensor):
        """
        Extract features from image(s).

        Args:
            img: torch.Tensor of shape:
                - [3, H, W] for single image
                - [B, 3, H, W] for batch of images
                All images must have the same H, W (already resized/normalized)

        Returns:
            features: torch.Tensor [B, H_p*W_p, D] - patch token features
            spatial_dims: tuple (H_p, W_p) - spatial patch grid size
        """
        # Ensure batch dimension
        if img.dim() == 3:
            img = img.unsqueeze(0)  # [1, 3, H, W]

        assert img.dim() == 4, f"Expected 4D tensor, got {img.dim()}D"
        B, C, H, W = img.shape
        assert C == 3, f"Expected 3 channels, got {C}"

        img = img.to(self.device)

        # Forward pass
        out = self.model.forward_features(img)

        # Extract patch tokens (shape: [B, H_p*W_p, D])
        features = out["x_norm_patchtokens"]

        H_p = H // self.patch_size
        W_p = W // self.patch_size

        return features, (H_p, W_p)

class SAMExtractor:
    def __init__(self, model_type="vit_b", checkpoint_path="sam_vit_b_01ec64.pth", image_size=512):
        """
        Initialize SAM (Segment Anything Model) feature extractor.

        Args:
            model_type: SAM model type ('vit_b', 'vit_l', 'vit_h')
            checkpoint_path: Path to SAM checkpoint file
            image_size: Target image size (default: 512). Will interpolate positional embeddings
                       from the pretrained 1024x1024 model to match this size.
        """
        from segment_anything import sam_model_registry

        self.model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

        # SAM's image encoder outputs 64x64 patches for 1024x1024 input
        # So patch_size = 1024 / 64 = 16
        self.patch_size = 16
        self.original_size = 1024
        self.image_size = image_size
        
        # Interpolate positional embeddings if using a different image size
        if self.image_size != self.original_size:
            self._interpolate_pos_embed()

    def _interpolate_pos_embed(self):
        """
        Interpolate positional embeddings to match the target image size.
        SAM's positional embeddings are for 64x64 patches (1024x1024 image).
        For 512x512 images, we need 32x32 patches.
        """
        
        # Get the positional embedding from SAM's image encoder
        # Shape: [1, H, W, D]
        pos_embed = self.model.image_encoder.pos_embed
        
        # Original grid size (64x64 for 1024x1024 images)
        original_grid_size = self.original_size // self.patch_size
        
        # New grid size (e.g., 32x32 for 512x512 images)
        new_grid_size = self.image_size // self.patch_size
        
        if original_grid_size == new_grid_size:
            return  # No interpolation needed
        
        # pos_embed shape: [1, H, W, D]
        # Permute to [1, D, H, W] for interpolation
        embed_dim = pos_embed.shape[-1]
        pos_embed_reshaped = pos_embed.permute(0, 3, 1, 2)  # [1, D, H, W]
        
        # Interpolate using bicubic interpolation
        pos_embed_interpolated = F.interpolate(
            pos_embed_reshaped,
            size=(new_grid_size, new_grid_size),
            mode='bilinear',
            align_corners=False
        )
        
        # Permute back to [1, H, W, D]
        pos_embed_interpolated = pos_embed_interpolated.permute(0, 2, 3, 1)
        
        # Update the positional embedding
        self.model.image_encoder.pos_embed = torch.nn.Parameter(
            pos_embed_interpolated, requires_grad=False
        )

    @torch.no_grad()
    def extract(self, img: torch.Tensor):
        """
        Extract features from image(s).

        Args:
            img: torch.Tensor of shape:
                - [3, H, H] for single image
                - [B, 3, H, H] for batch of images
                where H is the configured image_size (default 512)

        Returns:
            features: torch.Tensor [B, H_p*W_p, D] - patch token features
            spatial_dims: tuple (H_p, W_p) - spatial patch grid size
        """
        # Ensure batch dimension
        if img.dim() == 3:
            img = img.unsqueeze(0)  # [1, 3, H, W]

        assert img.dim() == 4, f"Expected 4D tensor, got {img.dim()}D"
        B, C, H, W = img.shape
        assert C == 3, f"Expected 3 channels, got {C}"
        assert H == self.image_size and W == self.image_size, \
            f"SAM requires {self.image_size}x{self.image_size} images, got {H}x{W}"

        img = img.to(self.device)

        # Forward pass through SAM's image encoder
        # Output shape: [B, D, H_p, W_p] (e.g., [B, 256, 32, 32] for 512x512 input)
        features = self.model.image_encoder(img)

        B, D, H_p, W_p = features.shape

        # Reshape to match DINO format: [B, H_p*W_p, D]
        # Permute [B, D, H_p, W_p] -> [B, H_p, W_p, D]
        # Then reshape to [B, H_p*W_p, D]
        features = features.permute(0, 2, 3, 1).reshape(B, H_p * W_p, D)

        return features, (H_p, W_p)
