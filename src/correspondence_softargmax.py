import torch
import torch.nn.functional as F


class WindowSoftArgmaxMatcher:
    """
    Correspondence matcher using window soft-argmax for sub-pixel refinement.
    
    Instead of using simple argmax (which gives discrete pixel locations), this:
    1. Finds the peak location with argmax
    2. Applies soft-argmax only within a small window around the peak
    
    This allows sub-pixel refinement and makes predictions more robust to noise.
    """

    def __init__(self, feature_extractor, window_size=5, temperature=0.1):
        """
        Args:
            feature_extractor: DINOv2Extractor or similar
            window_size: Size of the window around peak for soft-argmax (default: 5)
            temperature: Temperature for softmax (lower = sharper, default: 0.1)
        """
        self.extractor = feature_extractor
        self.window_size = window_size
        self.temperature = temperature
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def find_correspondences(self, src_img, trg_img, src_kps):
        """
        Find correspondences for source keypoints in target image using window soft-argmax

        Args:
            src_img: torch.Tensor [B, 3, H, W] - source images (normalized)
            trg_img: torch.Tensor [B, 3, H, W] - target images (normalized)
            src_kps: torch.Tensor [B, N, 2] - source keypoints (x, y) in pixels
                     Valid keypoints have positive coordinates, invalid are -2

        Returns:
            pred_kps: torch.Tensor [B, N, 2] - predicted target keypoints (sub-pixel precision)
        """
        # 1. Handle Batch Dimensions
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

        # 2. Extract Features [B, L, D] where L = H_p * W_p
        src_feats, (h_p, w_p) = self.extractor.extract(src_img)
        trg_feats, _ = self.extractor.extract(trg_img)

        D = src_feats.shape[-1]
        patch_size = self.extractor.patch_size

        # 3. Normalize Features (L2)
        src_feats = F.normalize(src_feats, dim=-1)
        trg_feats = F.normalize(trg_feats, dim=-1)

        # 4. Convert Source Keypoints to Grid Indices
        valid_mask = (src_kps[..., 0] >= 0)  # [B, N]

        kps_grid = (src_kps / patch_size).long()
        grid_x = kps_grid[..., 0].clamp(0, w_p - 1)
        grid_y = kps_grid[..., 1].clamp(0, h_p - 1)

        flat_indices = grid_y * w_p + grid_x

        # 5. Gather Source Features at Keypoint Locations
        flat_indices_expanded = flat_indices.unsqueeze(-1).expand(-1, -1, D)
        src_kp_feats = torch.gather(src_feats, 1, flat_indices_expanded)

        # 6. Compute Similarity Matrix
        # [B, N, D] @ [B, D, L] -> [B, N, L]
        sim_matrix = torch.bmm(src_kp_feats, trg_feats.transpose(1, 2))

        # Reshape similarity to 2D grid for window extraction
        # [B, N, L] -> [B, N, H_p, W_p]
        sim_grid = sim_matrix.view(B, N, h_p, w_p)

        # 7. WINDOW SOFT-ARGMAX (Key difference from baseline)
        pred_kps = self._window_soft_argmax(sim_grid, patch_size)

        # 8. Restore Mask for Padding
        pred_kps[~valid_mask] = -2

        # Remove batch dimension if input was unbatched
        if not is_batched:
            pred_kps = pred_kps.squeeze(0)

        return pred_kps

    def _window_soft_argmax(self, sim_grid, patch_size):
        """
        Apply window soft-argmax for sub-pixel refinement
        
        Args:
            sim_grid: [B, N, H_p, W_p] - similarity scores in 2D grid
            patch_size: Size of each patch in pixels
            
        Returns:
            pred_kps: [B, N, 2] - predicted keypoints with sub-pixel precision
        """
        B, N, h_p, w_p = sim_grid.shape
        half_window = self.window_size // 2

        # Step 1: Find peak location with argmax (discrete)
        sim_flat = sim_grid.view(B, N, -1)
        peak_indices = torch.argmax(sim_flat, dim=-1)  # [B, N]
        
        peak_y = peak_indices // w_p
        peak_x = peak_indices % w_p

        # Initialize output
        pred_kps = torch.zeros(B, N, 2, device=sim_grid.device, dtype=torch.float32)

        # Step 2: For each keypoint, extract window around peak and apply soft-argmax
        for b in range(B):
            for n in range(N):
                py = peak_y[b, n].item()
                px = peak_x[b, n].item()

                # Define window bounds (with boundary checks)
                y_min = max(0, py - half_window)
                y_max = min(h_p, py + half_window + 1)
                x_min = max(0, px - half_window)
                x_max = min(w_p, px + half_window + 1)

                # Extract window
                window = sim_grid[b, n, y_min:y_max, x_min:x_max]  # [h_w, w_w]

                # Apply temperature scaling and softmax
                window_flat = window.flatten() / self.temperature
                weights = F.softmax(window_flat, dim=0)  # [h_w * w_w]

                # Create coordinate grids for the window (in patch coordinates)
                h_w, w_w = window.shape
                y_coords = torch.arange(y_min, y_max, device=sim_grid.device, dtype=torch.float32)
                x_coords = torch.arange(x_min, x_max, device=sim_grid.device, dtype=torch.float32)
                
                yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
                yy_flat = yy.flatten()
                xx_flat = xx.flatten()

                # Weighted average (soft-argmax) in patch coordinates
                pred_y_patch = (weights * yy_flat).sum()
                pred_x_patch = (weights * xx_flat).sum()

                # Convert to pixel coordinates (center of patch + sub-pixel offset)
                pred_kps[b, n, 0] = (pred_x_patch + 0.5) * patch_size
                pred_kps[b, n, 1] = (pred_y_patch + 0.5) * patch_size

        return pred_kps
