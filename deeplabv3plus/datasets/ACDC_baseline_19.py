# datasets/ACDC.py — 19-class Cityscapes-trainId compatible loader
# Adds deterministic 85/15 split on ACDC/train with cached JSON,
# and keeps condition-wise filtering for evaluation.

import os
import re
import json
import glob
import random
import collections
import numpy as np
import torch
from torch.utils import data
from PIL import Image


class ACDC(data.Dataset):
    CityscapesClass = collections.namedtuple(
        "CityscapesClass",
        ["name", "id", "train_id", "category", "category_id",
         "has_instances", "ignore_in_eval", "color"]
    )

    # Cityscapes taxonomy (trainId space, 19 evaluable classes)
    classes = [
        CityscapesClass("unlabeled", 0, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("ego vehicle", 1, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("rectification border", 2, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("out of roi", 3, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("static", 4, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("dynamic", 5, 255, "void", 0, False, True, (111, 74, 0)),
        CityscapesClass("ground", 6, 255, "void", 0, False, True, (81, 0, 81)),
        CityscapesClass("road", 7, 0, "flat", 1, False, False, (128, 64, 128)),
        CityscapesClass("sidewalk", 8, 1, "flat", 1, False, False, (244, 35, 232)),
        CityscapesClass("parking", 9, 255, "flat", 1, False, True, (250, 170, 160)),
        CityscapesClass("rail track", 10, 255, "flat", 1, False, True, (230, 150, 140)),
        CityscapesClass("building", 11, 2, "construction", 2, False, False, (70, 70, 70)),
        CityscapesClass("wall", 12, 3, "construction", 2, False, False, (102, 102, 156)),
        CityscapesClass("fence", 13, 4, "construction", 2, False, False, (190, 153, 153)),
        CityscapesClass("guard rail", 14, 255, "construction", 2, False, True, (180, 165, 180)),
        CityscapesClass("bridge", 15, 255, "construction", 2, False, True, (150, 100, 100)),
        CityscapesClass("tunnel", 16, 255, "construction", 2, False, True, (150, 120, 90)),
        CityscapesClass("pole", 17, 5, "object", 3, False, False, (153, 153, 153)),
        CityscapesClass("polegroup", 18, 255, "object", 3, False, True, (153, 153, 153)),
        CityscapesClass("traffic light", 19, 6, "object", 3, False, False, (250, 170, 30)),
        CityscapesClass("traffic sign", 20, 7, "object", 3, False, False, (220, 220, 0)),
        CityscapesClass("vegetation", 21, 8, "nature", 4, False, False, (107, 142, 35)),
        CityscapesClass("terrain", 22, 9, "nature", 4, False, False, (152, 251, 152)),
        CityscapesClass("sky", 23, 10, "sky", 5, False, False, (70, 130, 180)),
        CityscapesClass("person", 24, 11, "human", 6, True, False, (220, 20, 60)),
        CityscapesClass("rider", 25, 12, "human", 6, True, False, (255, 0, 0)),
        CityscapesClass("car", 26, 13, "vehicle", 7, True, False, (0, 0, 142)),
        CityscapesClass("truck", 27, 14, "vehicle", 7, True, False, (0, 0, 70)),
        CityscapesClass("bus", 28, 15, "vehicle", 7, True, False, (0, 60, 100)),
        CityscapesClass("caravan", 29, 255, "vehicle", 7, True, True, (0, 0, 90)),
        CityscapesClass("trailer", 30, 255, "vehicle", 7, True, True, (0, 0, 110)),
        CityscapesClass("train", 31, 16, "vehicle", 7, True, False, (0, 80, 100)),
        CityscapesClass("motorcycle", 32, 17, "vehicle", 7, True, False, (0, 0, 230)),
        CityscapesClass("bicycle", 33, 18, "vehicle", 7, True, False, (119, 11, 32)),
        CityscapesClass("license plate", -1, 255, "vehicle", 7, False, True, (0, 0, 142)),
    ]

    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])  # for ignore=255 mapped to 19
    train_id_to_color = np.array(train_id_to_color, dtype=np.uint8)

    def __init__(self,
                 root,
                 split="train85",                   # 'train85' | 'val15' | 'val' | 'test'
                 transform=None,
                 test_class=None,                   # 'rain'|'fog'|'snow'|'night' or None
                 holdout_seed=1,
                 cache_splits_dir="artifacts/splits/acdc"):
        """
        Directory layout expected:
            root/
              rgb/{rain,fog,snow,night}/{train,val,test}/...png
              gt/{rain,fog,snow,night}/{train,val,test}/..._gt_labelTrainIds.png

        Splits:
          - 'train85' : 85% of ACDC/train (deterministic; cached to JSON)
          - 'val15'   : 15% of ACDC/train (training-time validation only)
          - 'val'     : official ACDC/val (≈406) — use only for final evaluation
          - 'test'    : optional
        """
        self.root = root
        self.split = split
        self.transform = transform
        self.condition = test_class  # None => all
        self.holdout_seed = int(holdout_seed)
        self.cache_splits_dir = cache_splits_dir

        # Collect (img, label, cond) pairs from a given subfolder
        def collect_from(sub):
            img_root = os.path.join(self.root, "rgb")
            lbl_root = os.path.join(self.root, "gt")
            conds = ["rain", "fog", "snow", "night"]
            items = []

            for c in conds:
                if self.condition is not None and c != self.condition:
                    continue

                img_dir = os.path.join(img_root, c, sub)
                lbl_dir = os.path.join(lbl_root, c, sub)
                lbls = sorted(glob.glob(os.path.join(lbl_dir, "**", "*gt_labelTrainIds.png"),
                                        recursive=True))
                for lp in lbls:
                    stem = re.sub(r"_gt_labelTrainIds\.png$", "", os.path.basename(lp))
                    # ACDC filenames for rgb typically share the same stem
                    cands = glob.glob(os.path.join(img_dir, "**", f"{stem}*.png"), recursive=True)
                    if not cands:
                        continue
                    items.append((cands[0], lp, c))  # (img, label, condition)
            return items

        # Build the final list according to requested split
        if self.split in ("train85", "val15"):
            train_items = collect_from("train")
            # deterministic 85/15 with cached JSON
            os.makedirs(self.cache_splits_dir, exist_ok=True)
            list_file = os.path.join(
                self.cache_splits_dir, f"acdc_train_85_15_seed{self.holdout_seed}.json"
            )
            if os.path.isfile(list_file):
                with open(list_file, "r") as f:
                    data = json.load(f)  # {"train": [...], "val15": [...]}
            else:
                rng = random.Random(self.holdout_seed)
                idxs = list(range(len(train_items)))
                rng.shuffle(idxs)
                k = int(round(len(train_items) * 0.15))
                val_idx = set(idxs[:k])
                train_list, val_list = [], []
                for i, (img, lab, cond) in enumerate(train_items):
                    pair = {"img": img, "label": lab, "cond": cond}
                    (val_list if i in val_idx else train_list).append(pair)
                data = {"train": train_list, "val15": val_list}
                with open(list_file, "w") as f:
                    json.dump(data, f, indent=2)

            items = data["train"] if self.split == "train85" else data["val15"]
            self.images = [it["img"] for it in items]
            self.targets = [it["label"] for it in items]
            self.conditions = [it["cond"] for it in items]

            print(f"[ACDC] split={self.split} (seed={self.holdout_seed}) "
                  f"count={len(self.images)} | "
                  f"rain={sum(c=='rain' for c in self.conditions)}, "
                  f"fog={sum(c=='fog' for c in self.conditions)}, "
                  f"snow={sum(c=='snow' for c in self.conditions)}, "
                  f"night={sum(c=='night' for c in self.conditions)}")

        elif self.split in ("val", "test"):
            sub = "val" if self.split == "val" else "test"
            items = collect_from(sub)
            self.images = [i[0] for i in items]
            self.targets = [i[1] for i in items]
            self.conditions = [i[2] for i in items]

            print(f"[ACDC] split={self.split} sub='{sub}' total={len(self.images)} "
                  f"(rain={sum(cc=='rain' for cc in self.conditions)}, "
                  f"fog={sum(cc=='fog' for cc in self.conditions)}, "
                  f"snow={sum(cc=='snow' for cc in self.conditions)}, "
                  f"night={sum(cc=='night' for cc in self.conditions)})")
        else:
            raise ValueError(f"Unknown split: {self.split} (use 'train85', 'val15', 'val', or 'test')")

    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target(cls, target):
        # map ignore(255) -> 19th color for visualization
        target = target.copy()
        target[target == 255] = 19
        return cls.train_id_to_color[target]

    def __getitem__(self, index):
        # image
        img_path = self.images[index]
        image = Image.open(img_path).convert("RGB")
        name = os.path.basename(img_path)

        # label (already in Cityscapes trainIds)
        lbl_path = self.targets[index]
        label_np = np.array(Image.open(lbl_path), dtype=np.uint8)

        # transforms
        if self.transform:
            image, label_t = self.transform(image, Image.fromarray(label_np))

            # --- convert label to numpy uint8, regardless of type returned by the transform ---
            if isinstance(label_t, Image.Image):
                label_np = np.array(label_t, dtype=np.uint8)
            elif torch.is_tensor(label_t):
                # expected shape [H, W] (or [1, H, W]); move to CPU numpy
                label_np = label_t.squeeze().detach().cpu().numpy().astype(np.uint8)
            else:
                # e.g., already a numpy array
                label_np = np.array(label_t, dtype=np.uint8)

            # --- enforce SAME spatial size as the (possibly Tensor) image ---
            if torch.is_tensor(image):
                H, W = image.shape[-2:]            # image is CHW
            elif isinstance(image, Image.Image):
                W, H = image.size                   # PIL returns (W, H)
            else:
                H, W = image.shape[:2]              # numpy HWC

            if label_np.shape[0] != H or label_np.shape[1] != W:
                # keep labels crisp
                label_np = np.array(
                    Image.fromarray(label_np).resize((W, H), Image.NEAREST),
                    dtype=np.uint8
                )

        # weather/time ids
        weather_id = 0  # normal
        time_id = 0     # day
        if "/rain/" in img_path:
            weather_id = 1
        elif "/fog/" in img_path:
            weather_id = 2
        elif "/snow/" in img_path:
            weather_id = 3
        if "/night/" in img_path:
            time_id = 1

        data_domain = 0  # keep as in your pipeline
        return (
            image,
            torch.from_numpy(label_np).long(),
            name,
            torch.tensor(weather_id),
            torch.tensor(time_id),
            data_domain,
        )

