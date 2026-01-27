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
        print("  Loading UNet...")
        self.unet = UNet2DConditionModel.from_pretrained(
            weights,
            subfolder="unet",
            torch_dtype=self.dtype,
            local_files_only=True
        ).to(self.device)

        # Load VAE (for encoding images to latent space)
        print("  Loading VAE...")
        self.vae = AutoencoderKL.from_pretrained(
            weights,
            subfolder="vae",
            torch_dtype=self.dtype,
            local_files_only=True
        ).to(self.device)

        # Load text encoder (even though we use empty prompts)
        print("  Loading text encoder...")
        self.text_encoder = CLIPTextModel.from_pretrained(
            weights,
            subfolder="text_encoder",
            torch_dtype=self.dtype,
            local_files_only=True
        ).to(self.device)

        # Load tokenizer
        print("  Loading tokenizer...")
        self.tokenizer = CLIPTokenizer.from_pretrained(
            weights,
            subfolder="tokenizer",
            local_files_only=True
        )

        # Load scheduler (for adding noise)
        print("  Loading scheduler...")
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

        # Move to device
        img = img.to(self.device, dtype=self.dtype)

        # CRITICAL: Handle different normalization schemes
        img_min, img_max = img.min().item(), img.max().item()

        if img_min < -0.5:
            # ImageNet stats: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device, dtype=self.dtype).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device, dtype=self.dtype).view(1, 3, 1, 1)

            # Denormalize: img_denorm = img * std + mean
            img = img * std + mean
            img = torch.clamp(img, 0, 1)  # Ensure [0, 1] range

        elif img_max > 1.5:
            # [0, 255] range
            img = img / 255.0

        # Now img should be in [0, 1] range
        # Convert to [-1, 1] for Stable Diffusion
        img = img * 2.0 - 1.0

        # Encode image to latent space using VAE
        latents = self.vae.encode(img).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        # Prepare timestep
        t = torch.tensor([self.timestep], device=self.device).long().expand(B)

        # Add noise (simulate diffusion forward process)
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, t)

        # Get text embeddings (empty prompt)
        text_embeddings = self._get_empty_text_embeddings(B)

        # Forward pass through U-Net with feature hook
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
