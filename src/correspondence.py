import torch
import torch.nn.functional as F


class CorrespondenceMatcher:
    """
    Handles correspondence matching between source and target features
    """

    def __init__(self, feature_extractor):
        """
        Args:
            feature_extractor: DINOv2Extractor or similar
        """
        self.extractor = feature_extractor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def find_correspondences(self, src_img, trg_img, src_kps):
        """
        Find correspondences for source keypoints in target image

        Args:
            src_img: torch.Tensor [B, 3, H, W] - source images (normalized)
            trg_img: torch.Tensor [B, 3, H, W] - target images (normalized)
            src_kps: torch.Tensor [B, N, 2] - source keypoints (x, y) in pixels
                     Valid keypoints have positive coordinates, invalid are -2

        Returns:
            pred_kps: torch.Tensor [B, N, 2] - predicted target keypoints
        """
        # 1. Handle Batch Dimensions
        # If inputs are unbatched [3, H, W], unsqueeze to [1, 3, H, W]
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
        # DINOv2Extractor handles the batch dimension naturally
        src_feats, (h_p, w_p) = self.extractor.extract(src_img)
        trg_feats, _ = self.extractor.extract(trg_img)

        D = src_feats.shape[-1]
        patch_size = self.extractor.patch_size

        # 3. Normalize Features (L2)
        # Important for cosine similarity: dot product of normalized vectors
        src_feats = F.normalize(src_feats, dim=-1)
        trg_feats = F.normalize(trg_feats, dim=-1)

        # 4. Convert Source Keypoints to Grid Indices
        # Create a mask for valid keypoints (not -2)
        valid_mask = (src_kps[..., 0] >= 0)  # [B, N]

        # Convert pixel coords to patch coords
        # Clamp to ensure indices are within [0, grid_size-1]
        # (This handles the -2 padding safely by clamping them to 0)
        kps_grid = (src_kps / patch_size).long()
        grid_x = kps_grid[..., 0].clamp(0, w_p - 1)
        grid_y = kps_grid[..., 1].clamp(0, h_p - 1)

        # Calculate flat indices [B, N] for gathering
        flat_indices = grid_y * w_p + grid_x

        # 5. Gather Source Features at Keypoint Locations
        # Expand indices to [B, N, D] to match feature dimension
        flat_indices_expanded = flat_indices.unsqueeze(-1).expand(-1, -1, D)

        # Gather: Select specific patch features for each keypoint
        # src_kp_feats shape: [B, N, D]
        src_kp_feats = torch.gather(src_feats, 1, flat_indices_expanded)

        # 6. Compute Similarity Matrix (Batch Matrix Multiplication)
        # [B, N, D] @ [B, D, L] -> [B, N, L]
        # This computes sim between every source keypoint and every target patch
        sim_matrix = torch.bmm(src_kp_feats, trg_feats.transpose(1, 2))

        # 7. Find Best Matches
        # Argmax over the target patch dimension (L)
        best_match_indices = torch.argmax(sim_matrix, dim=-1)  # [B, N]

        # 8. Convert Matches back to Pixels
        pred_y_idx = best_match_indices // w_p
        pred_x_idx = best_match_indices % w_p

        # Scale back to pixel coordinates (center of the patch)
        pred_x_px = (pred_x_idx.float() + 0.5) * patch_size
        pred_y_px = (pred_y_idx.float() + 0.5) * patch_size

        # Stack coordinates [B, N, 2]
        pred_kps = torch.stack([pred_x_px, pred_y_px], dim=-1)

        # 9. Restore Mask for Padding
        # Set predicted keypoints for invalid source points back to -2
        pred_kps[~valid_mask] = -2

        # Remove batch dimension if input was unbatched
        if not is_batched:
            pred_kps = pred_kps.squeeze(0)

        return pred_kps
