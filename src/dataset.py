import os
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import json
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm


class SPairDataset(Dataset):
    """
    SPair-71k dataset for semantic correspondence

    Args:
        datapath: Root path to SPair-71k dataset
        split: 'trn', 'val', or 'test'
        img_size: Size to resize images to (default: 224)
        category: Specific category or 'all' (default: 'all')
        mean: Normalization mean (default: ImageNet)
        std: Normalization std (default: ImageNet)
    """

    CATEGORIES = [
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
        "car", "cat", "chair", "cow", "dog", "horse", "motorbike",
        "person", "pottedplant", "sheep", "train", "tvmonitor"
    ]

    CLASS_DICT = {
        'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4,
        'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9,
        'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14,
        'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19
    }

    def __init__(
        self,
        datapath,
        split='trn',
        img_size=224,
        category='all',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ):
        super().__init__()

        assert split in ['trn', 'val', 'test'], f"Split must be trn/val/test, got {split}"
        assert category == 'all' or category in self.CATEGORIES, \
            f"Category must be 'all' or one of {self.CATEGORIES}"

        self.split = split
        self.img_size = (img_size, img_size)
        self.max_pts = 20

        # Setup paths
        base_path = os.path.join(os.path.abspath(datapath), 'SPair-71k')
        self.img_path = os.path.join(base_path, 'JPEGImages')
        self.seg_path = os.path.join(base_path, 'Segmentation')
        self.ann_path = os.path.join(base_path, 'PairAnnotation', split)
        split_file = os.path.join(base_path, 'Layout/large', f'{split}.txt')

        # Load pair list
        self.train_data = open(split_file).read().strip().split('\n')

        # Filter by category if needed
        if category != 'all':
            self.train_data = [pair for pair in self.train_data if category in pair]

        # Extract image names
        self.src_imnames = [x.split('-')[1] + '.jpg' for x in self.train_data]
        self.trg_imnames = [x.split('-')[2].split(':')[0] + '.jpg' for x in self.train_data]

        # Get class list
        self.cls = sorted(os.listdir(self.img_path))

        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        # Load annotations
        print(f"Loading SPair-71k {split} annotations...")
        anntn_files = [glob.glob(f'{self.ann_path}/{name}.json')[0]
                       for name in self.train_data]

        self.src_kps, self.trg_kps = [], []
        self.src_bbox, self.trg_bbox = [], []
        self.cls_ids = []
        self.kps_ids = []  # Store keypoint IDs
        self.vpvar, self.scvar, self.trncn, self.occln = [], [], [], []

        for anntn_file in tqdm(anntn_files):
            with open(anntn_file) as f:
                anntn = json.load(f)

            self.src_kps.append(torch.tensor(anntn['src_kps']).t().float())
            self.trg_kps.append(torch.tensor(anntn['trg_kps']).t().float())
            self.src_bbox.append(torch.tensor(anntn['src_bndbox']).float())
            self.trg_bbox.append(torch.tensor(anntn['trg_bndbox']).float())
            self.cls_ids.append(self.cls.index(anntn['category']))

            # Store keypoint IDs (as strings)
            self.kps_ids.append(anntn.get('kps_ids', []))

            self.vpvar.append(torch.tensor(anntn['viewpoint_variation']))
            self.scvar.append(torch.tensor(anntn['scale_variation']))
            self.trncn.append(torch.tensor(anntn['truncation']))
            self.occln.append(torch.tensor(anntn['occlusion']))

        # Create unique identifiers
        self.src_identifiers = [f"{self.cls[ids]}-{name[:-4]}"
                                for ids, name in zip(self.cls_ids, self.src_imnames)]
        self.trg_identifiers = [f"{self.cls[ids]}-{name[:-4]}"
                                for ids, name in zip(self.cls_ids, self.trg_imnames)]

    def __len__(self):
        return len(self.src_imnames)

    def __getitem__(self, idx):
        """Return a sample pair"""
        batch = {}

        # Image names and category
        batch['src_imname'] = self.src_imnames[idx]
        batch['trg_imname'] = self.trg_imnames[idx]
        batch['category_id'] = self.cls_ids[idx]
        batch['category'] = self.cls[batch['category_id']]
        batch['pair_idx'] = idx  # Add pair index for per-image tracking

        # Load images
        src_pil = self._get_image(self.src_imnames[idx], batch['category_id'])
        trg_pil = self._get_image(self.trg_imnames[idx], batch['category_id'])
        batch['src_imsize'] = src_pil.size  # (W, H)
        batch['trg_imsize'] = trg_pil.size

        # Transform images
        batch['src_img'] = self.transform(src_pil)
        batch['trg_img'] = self.transform(trg_pil)

        h1, w1 = batch['src_img'].shape[1:]
        h2, w2 = batch['trg_img'].shape[1:]

        # Load masks
        batch['src_mask'] = self._get_mask(batch['category'], batch['src_imname'], (h1, w1))
        batch['trg_mask'] = self._get_mask(batch['category'], batch['trg_imname'], (h2, w2))

        # Scale bounding boxes
        batch['src_bbox'] = self._scale_bbox(self.src_bbox[idx], batch['src_imsize'], (h1, w1))
        batch['trg_bbox'] = self._scale_bbox(self.trg_bbox[idx], batch['trg_imsize'], (h2, w2))

        # Scale keypoints
        batch['src_kps'], num_pts = self._get_points(
            self.src_kps[idx], batch['src_imsize'], (h1, w1))
        batch['trg_kps'], _ = self._get_points(
            self.trg_kps[idx], batch['trg_imsize'], (h2, w2))
        batch['n_pts'] = torch.tensor(num_pts)

        # Convert keypoints from 2xN to Nx2
        batch['src_kps'] = batch['src_kps'].permute(1, 0)
        batch['trg_kps'] = batch['trg_kps'].permute(1, 0)

        # Regularize coordinates
        batch['src_kps'][:num_pts] = self._regularise_coordinates(
            batch['src_kps'][:num_pts], h1, w1, eps=1e-4)
        batch['trg_kps'][:num_pts] = self._regularise_coordinates(
            batch['trg_kps'][:num_pts], h2, w2, eps=1e-4)

        # Pad keypoint IDs to max_pts (use "-1" as sentinel for invalid)
        kps_ids = self.kps_ids[idx][:num_pts]  # Only take valid keypoints
        # Pad to max_pts
        padded_kps_ids = kps_ids + ["-1"] * (self.max_pts - len(kps_ids))
        batch['kps_ids'] = padded_kps_ids

        # PCK thresholds (using bbox)
        batch['src_pckthres'] = self._get_pckthres(batch['src_bbox'])
        batch['trg_pckthres'] = self._get_pckthres(batch['trg_bbox'])
        batch['pckthres'] = batch['trg_pckthres'].clone()

        # Additional metadata
        batch['vpvar'] = self.vpvar[idx]
        batch['scvar'] = self.scvar[idx]
        batch['trncn'] = self.trncn[idx]
        batch['occln'] = self.occln[idx]
        batch['src_identifier'] = self.src_identifiers[idx]
        batch['trg_identifier'] = self.trg_identifiers[idx]
        batch['datalen'] = len(self.train_data)

        return batch

    def _get_image(self, imname, cls_id):
        """Load image from disk"""
        path = os.path.join(self.img_path, self.cls[cls_id], imname)
        return Image.open(path).convert('RGB')

    def _get_mask(self, category, imname, scaled_imsize):
        """Load and process segmentation mask"""
        mask_path = os.path.join(self.seg_path, category, imname.split('.')[0] + '.png')
        tensor_mask = torch.tensor(np.array(Image.open(mask_path)))

        # Filter to only the target class
        class_id = self.CLASS_DICT[category] + 1
        tensor_mask = (tensor_mask == class_id).float() * 255

        # Resize mask
        tensor_mask = F.interpolate(
            tensor_mask.unsqueeze(0).unsqueeze(0),
            size=scaled_imsize,
            mode='bilinear',
            align_corners=True
        ).int().squeeze()

        return tensor_mask

    def _scale_bbox(self, bbox, ori_imsize, scaled_imsize):
        """Scale bounding box from original to scaled image size"""
        bbox = bbox.clone()
        bbox[0::2] *= scaled_imsize[1] / ori_imsize[0]  # x coordinates
        bbox[1::2] *= scaled_imsize[0] / ori_imsize[1]  # y coordinates
        return bbox

    def _get_points(self, pts, ori_imsize, scaled_imsize):
        """Scale keypoints and pad to max_pts"""
        xy, n_pts = pts.size()
        pad_pts = torch.zeros((xy, self.max_pts - n_pts)) - 2

        x_crds = pts[0] * (scaled_imsize[1] / ori_imsize[0])
        y_crds = pts[1] * (scaled_imsize[0] / ori_imsize[1])
        kps = torch.cat([torch.stack([x_crds, y_crds]), pad_pts], dim=1)

        return kps, n_pts

    def _get_pckthres(self, bbox):
        """Compute PCK threshold from bounding box"""
        if len(bbox.shape) == 2:
            bbox = bbox.squeeze(0)
        bbox_w = bbox[2] - bbox[0]
        bbox_h = bbox[3] - bbox[1]
        return torch.max(bbox_w, bbox_h).float()

    def _regularise_coordinates(self, coord, H, W, eps=0):
        """Clamp coordinates to image boundaries"""
        coord = coord.clone()
        coord[..., 0] = torch.clamp(coord[..., 0], min=0+eps, max=W-1-eps)
        coord[..., 1] = torch.clamp(coord[..., 1], min=0+eps, max=H-1-eps)
        return coord


def collate_fn_correspondence(batch):
    """
    Custom collate function for batching correspondence samples.
    Handles variable-length keypoint IDs properly.

    Args:
        batch: list of dicts from SPairDataset.__getitem__()

    Returns:
        dict with batched tensors and lists for variable-length data
    """
    # Stack image tensors
    src_imgs = torch.stack([b['src_img'] for b in batch])
    trg_imgs = torch.stack([b['trg_img'] for b in batch])

    # Stack keypoint tensors
    src_kps = torch.stack([b['src_kps'] for b in batch])
    trg_kps = torch.stack([b['trg_kps'] for b in batch])

    # Stack masks and bboxes
    src_masks = torch.stack([b['src_mask'] for b in batch])
    trg_masks = torch.stack([b['trg_mask'] for b in batch])
    src_bboxes = torch.stack([b['src_bbox'] for b in batch])
    trg_bboxes = torch.stack([b['trg_bbox'] for b in batch])

    # Stack scalar tensors
    n_pts = torch.stack([b['n_pts'] for b in batch])
    pckthres = torch.stack([b['pckthres'] for b in batch])

    # Stack metadata tensors
    vpvar = torch.stack([b['vpvar'] for b in batch])
    scvar = torch.stack([b['scvar'] for b in batch])
    trncn = torch.stack([b['trncn'] for b in batch])
    occln = torch.stack([b['occln'] for b in batch])

    # Keep non-tensor data as lists
    src_imnames = [b['src_imname'] for b in batch]
    trg_imnames = [b['trg_imname'] for b in batch]
    categories = [b['category'] for b in batch]
    pair_indices = [b['pair_idx'] for b in batch]
    src_identifiers = [b['src_identifier'] for b in batch]
    trg_identifiers = [b['trg_identifier'] for b in batch]

    # Handle kps_ids: keep as list of lists (one list per sample in batch)
    kps_ids_list = [b['kps_ids'] for b in batch]

    # Store image sizes
    src_imsizes = [b['src_imsize'] for b in batch]
    trg_imsizes = [b['trg_imsize'] for b in batch]

    return {
        'src_img': src_imgs,
        'trg_img': trg_imgs,
        'src_kps': src_kps,
        'trg_kps': trg_kps,
        'src_mask': src_masks,
        'trg_mask': trg_masks,
        'src_bbox': src_bboxes,
        'trg_bbox': trg_bboxes,
        'n_pts': n_pts,
        'pckthres': pckthres,
        'vpvar': vpvar,
        'scvar': scvar,
        'trncn': trncn,
        'occln': occln,
        'src_imname': src_imnames,
        'trg_imname': trg_imnames,
        'category': categories,
        'pair_idx': pair_indices,
        'src_identifier': src_identifiers,
        'trg_identifier': trg_identifiers,
        'kps_ids': kps_ids_list,
        'src_imsize': src_imsizes,
        'trg_imsize': trg_imsizes,
    }
# functions for visualising samples with keypoints
def denorm(img_chw, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)):
    mean = torch.tensor(mean)[:, None, None]
    std  = torch.tensor(std)[:, None, None]
    x = img_chw.cpu() * std + mean
    return x.clamp(0, 1)

def draw_image_with_keypoints(
    ax,
    image,
    keypoints,
    title=None,
    colors=None,
    kp_size=35,
    show_indices=True,
    fontsize=9
):
    ax.imshow(image)
    if title is not None:
        ax.set_title(title)
    ax.axis("off")

    n = keypoints.shape[0]

    for i in range(n):
        c = colors[i] if colors is not None else None

        ax.scatter(
            keypoints[i, 0],
            keypoints[i, 1],
            s=kp_size,
            color=c
        )

        if show_indices:
            ax.text(
                keypoints[i, 0] + 2,
                keypoints[i, 1] + 2,
                str(i),
                fontsize=fontsize,
                color="white",
                bbox=dict(
                    facecolor="black",
                    alpha=0.5,
                    pad=1,
                    edgecolor="none"
                ),
            )


def visualize_sample(
    ds,
    idx=None,
    kp_size=35,
    show_indices=True,
    use_colors=True,
    fontsize=9
):
    if idx is None:
        idx = random.randrange(len(ds))

    b = ds[idx]
    n = int(b["n_pts"].item())

    src = denorm(b["src_img"]).permute(1, 2, 0).numpy()
    trg = denorm(b["trg_img"]).permute(1, 2, 0).numpy()

    src_kps = b["src_kps"][:n].cpu()
    trg_kps = b["trg_kps"][:n].cpu()

    # Same color per keypoint index across both images
    if use_colors:
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i % 20) for i in range(n)]
    else:
        colors = None

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    draw_image_with_keypoints(
        ax[0],
        src,
        src_kps,
        title=f"SRC ({b['category']}) idx={idx}",
        colors=colors,
        kp_size=kp_size,
        show_indices=show_indices,
        fontsize=fontsize,
    )

    draw_image_with_keypoints(
        ax[1],
        trg,
        trg_kps,
        title="TRG",
        colors=colors,
        kp_size=kp_size,
        show_indices=show_indices,
        fontsize=fontsize,
    )

    plt.tight_layout()
    plt.show()