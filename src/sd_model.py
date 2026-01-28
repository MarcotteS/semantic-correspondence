import torch
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import os
from typing import Tuple

class StableDiffusionExtractor:
    """
    Stable Diffusion 1.5 Feature Extractor for Semantic Correspondence.
    Extracts features from U-Net intermediate layers during denoising process.
    Based on DIFT: "Emergent Correspondence from Image Diffusion" (NeurIPS 2023)
    """

    def __init__(
        self,
        weights: str,
        model_name: str = "sd-1-5",
        timestep: int = 261,
        layer_name: str = "up_blocks.0",
        patch_size: int = 16,
    ):
        """
        Initialize SD 1.5 feature extractor.

        Args:
            weights: Path to local SD weights directory
            model_name: Label for this model (for results tracking)
            timestep: Denoising timestep to extract features (DIFT uses 261)
            layer_name: Which U-Net layer to extract from
                       Options: "up_blocks.0", "up_blocks.1", "up_blocks.2",
                               "down_blocks.0", "down_blocks.1", "down_blocks.2", "mid_block"
            patch_size: Virtual patch size (compatibility with ViT-style extractors)
        """
        print(f"Initializing SD extractor from: {weights}")

        if not os.path.exists(weights):
            raise ValueError(f"Weights directory not found: {weights}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Load U-Net
        self.unet = UNet2DConditionModel.from_pretrained(
            weights,
            subfolder="unet",
            torch_dtype=self.dtype,
            local_files_only=True
        ).to(self.device)

        # Load VAE (for encoding images to latent space)
        self.vae = AutoencoderKL.from_pretrained(
            weights,
            subfolder="vae",
            torch_dtype=self.dtype,
            local_files_only=True
        ).to(self.device)

        # Load text encoder (even if we use empty prompts)
        self.text_encoder = CLIPTextModel.from_pretrained(
            weights,
            subfolder="text_encoder",
            torch_dtype=self.dtype,
            local_files_only=True
        ).to(self.device)

        # Load tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(
            weights,
            subfolder="tokenizer",
            local_files_only=True
        )

        # Load scheduler to add noise
        self.scheduler = DDPMScheduler.from_pretrained(
            weights,
            subfolder="scheduler",
            local_files_only=True
        )

        # Set models to eval mode
        self.unet.eval()
        self.vae.eval()
        self.text_encoder.eval()

        # Store config
        self.timestep = timestep
        self.layer_name = layer_name
        self.patch_size = patch_size
        self.model_name = model_name
        self.features = None

        print(f" SD Extractor initialized!")
        print(f"  Model: {model_name}")
        print(f"  Timestep: {timestep}")
        print(f"  Layer: {layer_name}")
        print(f"  Device: {self.device}")

    def eval(self):
        """Set all models to eval mode"""
        self.unet.eval()
        self.vae.eval()
        self.text_encoder.eval()
        return self

    @property
    def model(self):
        """Compatibility property, returns self so that extractor.model.eval() works"""
        return self

    def _register_hook(self):
        """Register forward hook to capture intermediate features"""
        def hook_fn(module, input, output):
            self.features = output

        # Navigate to the specified layer
        layer = self.unet
        for name in self.layer_name.split('.'):
            layer = getattr(layer, name)

        handle = layer.register_forward_hook(hook_fn)
        return handle

    @torch.no_grad()
    def extract(self, img: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Extract features from image(s)."""

        # Ensure batch dimension
        if img.dim() == 3:
            img = img.unsqueeze(0)

        assert img.dim() == 4, f"Expected 4D tensor, got {img.dim()}D"
        B, C, H, W = img.shape
        assert C == 3, f"Expected 3 channels, got {C}"

        img = img.to(self.device, dtype=self.dtype)

        # Handle different normalization schemes
        img_min, img_max = img.min().item(), img.max().item()

        if img_min < -0.5:
            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device, dtype=self.dtype).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device, dtype=self.dtype).view(1, 3, 1, 1)

            img = img * std + mean
            img = torch.clamp(img, 0, 1)  # Ensure [0, 1] range

        elif img_max > 1.5:
            img = img / 255.0

        # Convert to [-1, 1] for SD
        img = img * 2.0 - 1.0

        # Encode image with VAE
        latents = self.vae.encode(img).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        # Timestep
        t = torch.tensor([self.timestep], device=self.device).long().expand(B)

        # Add noise
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, t)

        # Text embeddings (empty prompt)
        text_embeddings = self._get_empty_text_embeddings(B)

        # Forward pass through U-Net
        handle = self._register_hook()
        _ = self.unet(noisy_latents, t, encoder_hidden_states=text_embeddings).sample
        handle.remove()

        # Process captured features
        features = self.features
        B, C_feat, H_feat, W_feat = features.shape
        features = features.permute(0, 2, 3, 1).reshape(B, H_feat * W_feat, C_feat)
        features = F.normalize(features, dim=-1)


        return features, (H_feat, W_feat)

    def _get_empty_text_embeddings(self, batch_size: int) -> torch.Tensor:
        """Get text embeddings for empty prompt (unconditional generation)."""
        # Tokenize empty prompt
        text_input = self.tokenizer(
            [""] * batch_size,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        # Get embeddings from text encoder
        text_embeddings = self.text_encoder(
            text_input.input_ids.to(self.device)
        )[0]

        return text_embeddings



class MultiLayerSDExtractor:
    """Ensemble multiple SD layers"""

    def __init__(self, weights, timestep=100, layers=None):
        if layers is None:
            # Try some combinations
            layers = ["up_blocks.0", "up_blocks.1", "up_blocks.2"]
            # or: layers = ["mid_block", "up_blocks.0", "up_blocks.1"] ...

        self.extractors = []
        for layer in layers:
            ext = StableDiffusionExtractor(
                weights=weights,
                timestep=timestep,
                layer_name=layer
            )
            self.extractors.append(ext)

        self.device = self.extractors[0].device
        self.patch_size = self.extractors[0].patch_size

    def eval(self):
        for ext in self.extractors:
            ext.eval()
        return self

    @property
    def model(self):
        return self

    @torch.no_grad()
    def extract(self, img):
        all_features = []
        target_size = 32  # Upsample all to 32x32

        for extractor in self.extractors:
            features, (H, W) = extractor.extract(img)
            B = features.shape[0]

            # Upsample to common resolution if needed
            if H != target_size or W != target_size:
                feat_2d = features.reshape(B, H, W, -1).permute(0, 3, 1, 2)
                feat_2d = F.interpolate(
                    feat_2d,
                    size=(target_size, target_size),
                    mode='bilinear',
                    align_corners=False
                )
                features = feat_2d.permute(0, 2, 3, 1).reshape(B, target_size*target_size, -1)

            all_features.append(features)

        # Concatenate and normalize
        fused = torch.cat(all_features, dim=-1)
        fused = F.normalize(fused, dim=-1)

        return fused, (target_size, target_size)


class SDDINOFusion:
    """Fuse SD and DINOv2 features"""

    def __init__(self, sd_extractor, dino_extractor, alpha=0.5):
        self.sd_extractor = sd_extractor
        self.dino_extractor = dino_extractor
        self.device = sd_extractor.device
        self.patch_size = dino_extractor.patch_size
        self.alpha = alpha  # Weight to apply to SD features

    def eval(self):
        if hasattr(self.sd_extractor, 'eval'):
            self.sd_extractor.eval()
        if hasattr(self.dino_extractor, 'eval'):
            self.dino_extractor.eval()

        if hasattr(self.sd_extractor, 'model'):
            self.sd_extractor.model.eval()
        if hasattr(self.dino_extractor, 'model'):
            self.dino_extractor.model.eval()

        return self

    @property
    def model(self):
        return self

    @torch.no_grad()
    def extract(self, img):
        B, C, H_img, W_img = img.shape

        # Extract SD features (512x512)
        if H_img != 512:
            img_sd = F.interpolate(img, size=(512, 512), mode='bilinear', align_corners=False)
        else:
            img_sd = img
        sd_feat, (H_sd, W_sd) = self.sd_extractor.extract(img_sd)

        # Extract DINO features (518x518)
        if H_img != 518:
            img_dino = F.interpolate(img, size=(518, 518), mode='bilinear', align_corners=False)
        else:
            img_dino = img
        dino_feat, (H_dino, W_dino) = self.dino_extractor.extract(img_dino)

        # Upsample SD to match DINO resolution (37x37)
        if (H_sd, W_sd) != (H_dino, W_dino):
            sd_feat_2d = sd_feat.reshape(B, H_sd, W_sd, -1).permute(0, 3, 1, 2)
            sd_feat_2d = F.interpolate(sd_feat_2d, size=(H_dino, W_dino),
                                       mode='bilinear', align_corners=False)
            sd_feat = sd_feat_2d.permute(0, 2, 3, 1).reshape(B, H_dino * W_dino, -1)

        # L2 normalize each feature independently
        sd_feat_norm = F.normalize(sd_feat, dim=-1)
        dino_feat_norm = F.normalize(dino_feat, dim=-1)

        # Apply weighted fusion
        sd_weighted = self.alpha * sd_feat_norm
        dino_weighted = (1 - self.alpha) * dino_feat_norm

        # Concatenate
        fused_feat = torch.cat([sd_weighted, dino_weighted], dim=-1)

        return fused_feat, (H_dino, W_dino)