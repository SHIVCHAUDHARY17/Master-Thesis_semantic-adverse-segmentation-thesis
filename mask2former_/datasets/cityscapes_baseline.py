# datasets/cityscapes_baseline.py
import os
from collections import namedtuple
from typing import List, Optional, Tuple

import numpy as np
from torch.utils import data
from torch.utils.data import Subset, ConcatDataset
from PIL import Image
import torch


class Cityscapes(data.Dataset):
    """
    Cityscapes loader (19-class trainId space) with optional index filtering.

    - Standard splits: 'train', 'val', 'test'
    - Deterministic order within each split (sorted by city / filename)
    - Optional 'indices' lets you make sub-splits (e.g., 85/15) outside.
    """
    CityscapesClass = namedtuple(
        'CityscapesClass',
        ['name', 'id', 'train_id', 'category', 'category_id', 'has_instances', 'ignore_in_eval', 'color']
    )
    classes = [
        CityscapesClass('unlabeled',            0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle',          1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi',           3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static',               4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic',              5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground',               6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road',                 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk',             8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking',              9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track',           10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building',             11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall',                 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence',                13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail',           14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge',               15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel',               16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole',                 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup',            18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light',        19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign',         20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation',           21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain',              22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky',                  23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person',               24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider',                25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car',                  26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck',                27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus',                  28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan',              29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer',              30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train',                31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle',           32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle',              33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate',        -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color, dtype=np.uint8)

    def __init__(
        self,
        root: str,
        split: str = 'train',
        mode: str = 'fine',
        target_type: str = 'semantic',
        transform=None,
        indices: Optional[List[int]] = None,
        debug_print: bool = True,
    ):
        if split not in ['train', 'val', 'test']:
            raise ValueError('split must be one of: "train", "val", "test"')

        self.root = os.path.expanduser(root)
        self.mode = 'gtFine'  # Cityscapes fine annotations
        self.target_type = target_type
        self.transform = transform
        self.split = split

        self.images_dir = os.path.join(self.root, 'leftImg8bit', split)
        self.targets_dir = os.path.join(self.root, self.mode, split)

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            raise RuntimeError(
                f'Dataset not found or incomplete: {self.images_dir} / {self.targets_dir}'
            )

        # Load (image_path, label_path) pairs with deterministic ordering
        pairs = []
        for city in sorted(os.listdir(self.images_dir)):
            city_img_dir = os.path.join(self.images_dir, city)
            if not os.path.isdir(city_img_dir):
                continue
            for fname in sorted(os.listdir(city_img_dir)):
                if not fname.endswith('_leftImg8bit.png'):
                    continue
                img_path = os.path.join(city_img_dir, fname)
                tgt_name = fname.replace(
                    '_leftImg8bit.png',
                    f'_{self._get_target_suffix(self.mode, self.target_type)}'
                )
                tgt_path = os.path.join(self.targets_dir, city, tgt_name)
                if os.path.exists(tgt_path):
                    pairs.append((img_path, tgt_path))

        # Apply sub-selection if indices are provided (for 85/15 etc.)
        if indices is not None:
            pairs = [pairs[i] for i in indices]

        self.images = [p[0] for p in pairs]
        self.targets = [p[1] for p in pairs]

        # class conversion: Cityscapes labelIds -> 19 trainIds
        self.class_conversion_dict = {
            7: 0,   8: 1,  11: 2, 12: 3, 13: 4,
            17: 5,  18: 255,  # polegroup -> ignore
            19: 6,  20: 7,  21: 8,  22: 9,  23: 10,
            24: 11, 25: 12, 26: 13, 27: 14, 28: 15,
            31: 16, 32: 17, 33: 18,
            # Ignore everything else
            0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
            9: 255, 10: 255, 14: 255, 15: 255, 16: 255, 29: 255, 30: 255, -1: 255
        }

        if debug_print:
            print(f"Cityscapes {split} set: {len(self.images)} images")

    def _get_target_suffix(self, mode, target_type):
        if target_type == 'instance':
            return f'{mode}_instanceIds.png'
        elif target_type == 'semantic':
            return f'{mode}_labelIds.png'
        elif target_type == 'color':
            return f'{mode}_color.png'
        elif target_type == 'polygon':
            return f'{mode}_polygons.json'
        elif target_type == 'depth':
            return f'{mode}_disparity.png'
        else:
            raise ValueError(f"Unknown target_type: {target_type}")

    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target(cls, target):
        """
        Map 0..18 to colors; map ignore(255) to index 19 (last color = black).
        """
        target = target.copy()
        target[target == 255] = 19
        return cls.train_id_to_color[target]

    def __getitem__(self, index):
        # load
        img_path = self.images[index]
        lbl_path = self.targets[index]

        image = Image.open(img_path).convert('RGB')
        label_img = Image.open(lbl_path)  # labelIds

        if self.transform:
            image, label_out = self.transform(image, label_img)
        else:
            label_out = label_img

        # --- normalize label to numpy uint8, regardless of transform output type
        if isinstance(label_out, Image.Image):
            label_np = np.array(label_out, dtype=np.uint8)
            H_img, W_img = (image.size[1], image.size[0]) if isinstance(image, Image.Image) else image.shape[-2:]
        elif torch.is_tensor(label_out):
            # expect [H, W] or [1, H, W]
            label_np = label_out.squeeze().detach().cpu().numpy().astype(np.uint8)
            if torch.is_tensor(image):
                H_img, W_img = image.shape[-2:]
            elif isinstance(image, Image.Image):
                W_img, H_img = image.size
            else:
                H_img, W_img = image.shape[-2:]
        else:
            # already numpy-like
            label_np = np.array(label_out, dtype=np.uint8)
            if isinstance(image, Image.Image):
                W_img, H_img = image.size
            elif torch.is_tensor(image):
                H_img, W_img = image.shape[-2:]
            else:
                H_img, W_img = image.shape[-2:]

        # --- enforce SAME spatial size as image (nearest keeps IDs crisp)
        if label_np.shape[0] != H_img or label_np.shape[1] != W_img:
            label_np = np.array(
                Image.fromarray(label_np).resize((W_img, H_img), Image.NEAREST),
                dtype=np.uint8
            )

        # convert Cityscapes labelIds -> 19-class trainIds
        label_train = 255 * np.ones(label_np.shape, dtype=np.int64)
        for k, v in self.class_conversion_dict.items():
            label_train[label_np == k] = v

        name = os.path.basename(lbl_path)
        weather, time = 0, 0
        data_domain = 1  # 1: real

        return (
            image,
            torch.from_numpy(label_train).long(),
            name,
            torch.tensor(weather),
            torch.tensor(time),
            data_domain
        )


# ---------- Helpers for your exact 85/15 requirement ----------

def split_indices_85_15(n_total: int, seed: int = 1) -> Tuple[List[int], List[int]]:
    """
    Deterministically split [0..n_total-1] into 85% train, 15% val.
    """
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=gen).tolist()
    n_train = int(round(0.85 * n_total))
    return perm[:n_train], perm[n_train:]


def build_cityscapes_train85_val15_datasets(
    root: str,
    train_transform,
    val_transform,
    seed: int = 1,
    include_official_val_in_train: bool = True
) -> Tuple[data.Dataset, data.Dataset]:
    """
    Returns (train_ds, val_ds) where:
      - train_ds = 85% of official train split (2975) [+ optionally ALL official val(500)]
      - val_ds   = the remaining 15% of official train split
    Exactly matches your “train = 85% train + official val; val = 15% train”.
    """
    ds_train_aug = Cityscapes(root=root, split='train', transform=train_transform, debug_print=False)
    ds_train_val = Cityscapes(root=root, split='train', transform=val_transform, debug_print=False)

    n_total = len(ds_train_aug)
    idx85, idx15 = split_indices_85_15(n_total, seed=seed)

    train85 = Subset(ds_train_aug, idx85)
    val15   = Subset(ds_train_val, idx15)

    if include_official_val_in_train:
        ds_val_official_for_training = Cityscapes(root=root, split='val', transform=train_transform, debug_print=False)
        train_final = ConcatDataset([train85, ds_val_official_for_training])
    else:
        train_final = train85

    print(f"[split] CS/train total={n_total} -> train85={len(train85)}, val15={len(val15)}")
    if include_official_val_in_train:
        print(f"[split] Using CS/val (500) as training too -> train_final={len(train_final)}")

    return train_final, val15

