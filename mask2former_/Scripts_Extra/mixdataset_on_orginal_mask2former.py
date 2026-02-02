# mixdataset_on_orginal_mask2former.py
# ------------------------------------------------------------------
# MASK2FORMER SEGMENTATION-ONLY ABLATION: MIXED CS+ACDC TRAINING
#  - No WAS/TAS heads or losses
#  - No encoder freezing
#  - MODE 0: MIXED training dataset = CS(train85)+ACDC(train85), VAL = CS(val15)+ACDC(val15)
#  - MODE 1: Cityscapes-only baseline
#  - MODE 11: Cityscapes test
#  - MODE 21: ACDC test
#
# NOTE (checkpoint naming):
#  - Mixed seg-only (mode 0) checkpoints:
#       checkpoints/mixdataset_on_orginal_mask2former_segonly_latest.pth
#       checkpoints/mixdataset_on_orginal_mask2former_segonly_best.pth
#  - Defaults for test modes (11/21) now point to *_segonly_best.pth
# ------------------------------------------------------------------
# ==============================================================
# HOW TO RUN (COMMANDS)
# --------------------------------------------------------------
# NOTE:
# - MODE 0  = TRAIN (mixed CS85+ACDC85, validate on CS15+ACDC15)
# - MODE 11 = TEST Cityscapes (official val by default)
# - MODE 21 = TEST ACDC (val), optionally filtered by condition
# - Default test checkpoint (if --ckpt not provided):
#     checkpoints/mixdataset_on_orginal_mask2former_segonly_best.pth
#
# -----------------------
# MODE 0: TRAIN (MIXED)
# -----------------------
# CUDA_VISIBLE_DEVICES=0 python mixdataset_on_orginal_mask2former.py \
#   --mode 0 \
#   --crop_size 640 --batch_size 3 --val_batch_size 3 \
#   --total_itrs 90000 --lr 1e-5 --loss_type cross_entropy \
#   --val_interval 5000 --per_condition_val \
#   --enable_vis --vis_port 13570 --vis_env mix_segonly
#
# -----------------------
# MODE 11: TEST CITYSCAPES
# -----------------------
# (official val)
# CUDA_VISIBLE_DEVICES=0 python mixdataset_on_orginal_mask2former.py \
#   --mode 11 --cs_eval_split val \
#   --test_batch_size 3 --crop_size 640
#
# (optional: test split if your loader supports it)
# CUDA_VISIBLE_DEVICES=0 python mixdataset_on_orginal_mask2former.py \
#   --mode 11 --cs_eval_split test \
#   --test_batch_size 3 --crop_size 640
#
# -----------------------
# MODE 21: TEST ACDC (ALL)
# -----------------------
# CUDA_VISIBLE_DEVICES=0 python mixdataset_on_orginal_mask2former.py \
#   --mode 21 \
#   --test_batch_size 3 --crop_size 640
#
# -----------------------
# MODE 21: TEST ACDC PER-WEATHER (FILTERED)
# -----------------------
# rain
# CUDA_VISIBLE_DEVICES=0 python mixdataset_on_orginal_mask2former.py \
#   --mode 21 --ACDC_test_class rain \
#   --test_batch_size 3 --crop_size 640
#
# fog
# CUDA_VISIBLE_DEVICES=0 python mixdataset_on_orginal_mask2former.py \
#   --mode 21 --ACDC_test_class fog \
#   --test_batch_size 3 --crop_size 640
#
# snow
# CUDA_VISIBLE_DEVICES=0 python mixdataset_on_orginal_mask2former.py \
#   --mode 21 --ACDC_test_class snow \
#   --test_batch_size 3 --crop_size 640
#
# night
# CUDA_VISIBLE_DEVICES=0 python mixdataset_on_orginal_mask2former.py \
#   --mode 21 --ACDC_test_class night \
#   --test_batch_size 3 --crop_size 640
#
# -----------------------
# OPTIONAL: FORCE A CHECKPOINT (recommended for reproducibility)
# -----------------------
# Add this to any TEST command:
#   --ckpt checkpoints/mixdataset_on_orginal_mask2former_segonly_best.pth
# ==============================================================

import torch
import torch.nn as nn
import numpy as np
import random
import os
from tqdm import tqdm
import network
import utils
import argparse
from torch.utils import data
from datasets.cityscapes_baseline import Cityscapes, build_cityscapes_train85_val15_datasets
from datasets.ACDC_baseline_19 import ACDC
from utils import ext_transforms as et
from metrics import StreamSegMetrics
from utils.visualizer import Visualizer
from PIL import Image
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
import math
import json

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# ---- Constant label IDs (kept; Cityscapes typically returns these) ----
CLEAR_ID = 0
DAY_ID = 0


def load_labelmap(path_or_none):
    """Load label mapping from JSON/YAML file for 10-class remap."""
    remap = np.full(256, 255, dtype=np.uint8)  # Default: ignore

    if path_or_none is None:
        cfg = {
            "groups": {
                "0": 0, "1": 1, "2": 2, "5": 3, "6": 4,
                "7": 5, "8": 6, "10": 7, "11": 8, "13": 9
            },
            "ignore": [3, 4, 9, 12, 14, 15, 16, 17, 18, 255]
        }
    else:
        with open(path_or_none, "r") as f:
            if path_or_none.endswith((".yml", ".yaml")) and HAS_YAML:
                cfg = yaml.safe_load(f)
            else:
                cfg = json.load(f)

    for k, v in cfg["groups"].items():
        remap[int(k)] = int(v)
    for ig in cfg.get("ignore", []):
        remap[int(ig)] = 255

    return remap


def remap_to_10c(preds_np, targets_np, remap_vec):
    """Remap predictions and targets using the mapping vector."""
    preds_10 = remap_vec[preds_np]
    targets_10 = remap_vec[targets_np]
    return preds_10, targets_10


PALETTE_10C = np.array([
    [128, 64, 128],    # road (0)
    [244, 35, 232],    # sidewalk (1)
    [70, 70, 70],      # building (2)
    [153, 153, 153],   # pole (3)
    [250, 170, 30],    # traffic light (4)
    [220, 220, 0],     # traffic sign (5)
    [107, 142, 35],    # vegetation (6)
    [70, 130, 180],    # sky (7)
    [220, 20, 60],     # person (8)
    [0, 0, 142]        # car (9)
], dtype=np.uint8)


def colorize_10c(mask_hw):
    """Colorize 10-class masks."""
    h, w = mask_hw.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    valid = (mask_hw < 10)
    out[valid] = PALETTE_10C[mask_hw[valid]]
    return out


def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=int, default=1, choices=[0, 1, 11, 21],
                        help="0: train MIXED CS+ACDC, "
                             "1: train CS only, 11: test CS, 21: test ACDC")
    parser.add_argument("--cs_eval_split", type=str, default="val", choices=["val", "test"],
                        help="Cityscapes split for test-only mode")
    parser.add_argument("--eval_mode", type=str, default="19", choices=["19", "10"],
                        help="Evaluation head-space: 19 (native) or 10-class remap")
    parser.add_argument("--labelmap", type=str, default=None,
                        help="Path to JSON/YAML mapping trainId->10-class or 255(ignore)")
    parser.add_argument("--save_color_10c", action="store_true", default=False,
                        help="Save colored 10-class predictions")
    parser.add_argument("--separate_10c_dirs", action="store_true", default=True,
                        help="Write 10-class results into *_10c dirs")
    parser.add_argument("--out_tag", type=str, default="",
                        help="Extra tag for result dirs/filenames")

    parser.add_argument("--per_condition_val", action="store_true", default=True,
                        help="Report mIoU by ACDC condition (rain/fog/snow/night)")

    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes', 'ACDC', 'cityscapes_ACDC'])
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="Fallback dataset root")

    parser.add_argument(
        "--cs_split_strategy", type=str, default="standard",
        choices=[
            "standard",
            "train85_val15_plus_officialval_in_train",
            "train85_val15_only_train_split"
        ],
        help=("Cityscapes split policy.")
    )
    parser.add_argument("--cs_holdout_seed", type=int, default=1,
                        help="Seed for 85/15 split of Cityscapes train AND ACDC split.")
    parser.add_argument("--data_root_cs", type=str, default=None, help="Cityscapes root.")
    parser.add_argument("--data_root_acdc", type=str, default=None, help="ACDC root.")
    parser.add_argument("--data_root_awss", type=str, default=None, help="AWSS root.")

    available_models = sorted(name for name in network.modeling.__dict__
                              if name.islower() and not name.startswith("_")
                              and callable(network.modeling.__dict__[name]))
    parser.add_argument("--model", type=str, default='mask2former', choices=available_models)
    parser.add_argument("--separable_conv", action='store_true', default=False)
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False)
    parser.add_argument("--total_itrs", type=int, default=30000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'])
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--val_batch_size", type=int, default=4)
    parser.add_argument("--test_batch_size", type=int, default=4)
    parser.add_argument("--crop_size", type=int, default=513)

    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--continue_training", action='store_true', default=False)
    parser.add_argument("--finetune", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'])
    parser.add_argument("--gpu_id", type=str, default='0')
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--print_interval", type=int, default=10)
    parser.add_argument("--val_interval", type=int, default=100)
    parser.add_argument("--download", action='store_true', default=False)

    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'])

    parser.add_argument("--ACDC_test_class", type=str, default=None,
                        help="ACDC condition (rain/fog/snow/night)")

    parser.add_argument("--enable_vis", action='store_true', default=False)
    parser.add_argument("--vis_port", type=str, default='13570')
    parser.add_argument("--vis_env", type=str, default='main')
    parser.add_argument("--vis_num_samples", type=int, default=8)

    parser.add_argument("--num_classes", type=int, default=None)

    return parser


# -------------------------------------------------------------------------
# ACDC split builder: KEEP IT SIMPLE & IDENTICAL TO YOUR EXISTING 85/15
# Uses your ACDC_baseline_19.py cached JSON split (seeded).
# -------------------------------------------------------------------------
def build_acdc_train85_val15_same_as_before(root, train_transform, val_transform, seed=1):
    ds_train85 = ACDC(root=root, split='train85', transform=train_transform, holdout_seed=seed)
    ds_val15 = ACDC(root=root, split='val15', transform=val_transform, holdout_seed=seed)

    # print condition distribution (proof)
    for tag, ds in [("train85", ds_train85), ("val15", ds_val15)]:
        conds = getattr(ds, "conditions", None)
        if conds is not None:
            uniq, cnt = np.unique(np.array(conds), return_counts=True)
            dist = {str(u): int(c) for u, c in zip(uniq, cnt)}
            print(f"[ACDC] {tag} condition distribution:", dist)

    return ds_train85, ds_val15


def get_dataset(opts, tr_ds_name=None):
    """
    Returns (train_dst, val_dst, tst_dst) for the requested dataset name.
    """
    name = tr_ds_name or opts.dataset

    if name == 'cityscapes':
        train_transform = et.ExtCompose([
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        val_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        root_cs = getattr(opts, "data_root_cs", None) or getattr(opts, "data_root", None)

        if getattr(opts, "cs_split_strategy", "standard") == "standard":
            train_dst = Cityscapes(root=root_cs, split='train', transform=train_transform)
            val_dst = Cityscapes(root=root_cs, split='val', transform=val_transform)
        else:
            if opts.cs_split_strategy == "train85_val15_plus_officialval_in_train":
                include_val = True
            elif opts.cs_split_strategy == "train85_val15_only_train_split":
                include_val = False
            else:
                raise ValueError(f"Unknown cs_split_strategy: {opts.cs_split_strategy}")

            train_dst, val_dst = build_cityscapes_train85_val15_datasets(
                root=root_cs,
                train_transform=train_transform,
                val_transform=val_transform,
                seed=getattr(opts, "cs_holdout_seed", 1),
                include_official_val_in_train=include_val
            )

        tst_split = getattr(opts, "cs_eval_split", "test")
        tst_dst = Cityscapes(root=root_cs, split=tst_split, transform=val_transform)
        return train_dst, val_dst, tst_dst

    elif name == 'ACDC':
        val_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        train_dst = []  # Not used in test-only
        val_dst = ACDC(root=opts.data_root_acdc, split='val', transform=val_transform)

        tst_dst_all = ACDC(root=opts.data_root_acdc, split='val', transform=val_transform)

        if getattr(opts, "ACDC_test_class", None):
            from torch.utils.data import Subset
            condition_indices = []
            requested_condition = opts.ACDC_test_class.lower()
            print(f"[MANUAL FILTER] Filtering ACDC val for condition: {requested_condition}")
            for i in range(len(tst_dst_all)):
                if tst_dst_all.conditions[i] == requested_condition:
                    condition_indices.append(i)
            tst_dst = Subset(tst_dst_all, condition_indices)
            print(f"[MANUAL FILTER] Loaded {len(tst_dst)} {requested_condition} images from ACDC val")
        else:
            tst_dst = tst_dst_all

        return train_dst, val_dst, tst_dst

    elif name == 'cityscapes_ACDC':
        # --- MIXED TRAIN/VAL DATASETS (what your professor wants) ---
        from torch.utils.data import ConcatDataset

        # Cityscapes train85/val15 (controlled by opts.cs_split_strategy)
        train_cs, val_cs, _ = get_dataset(opts, 'cityscapes')

        # ACDC train85/val15 EXACTLY as before (cached JSON split)
        train_transform = et.ExtCompose([
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        val_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_acdc, val_acdc = build_acdc_train85_val15_same_as_before(
            root=opts.data_root_acdc,
            train_transform=train_transform,
            val_transform=val_transform,
            seed=getattr(opts, "cs_holdout_seed", 1)
        )

        train_mix = ConcatDataset([train_cs, train_acdc])
        val_mix = ConcatDataset([val_cs, val_acdc])

        return train_mix, val_mix, []

    else:
        raise NotImplementedError(f"Unknown dataset: {name}")


def validate(opts, model, loader, device, metrics,
             ret_samples_ids=None, vis=None, denorm=None,
             save_dir="results", max_vis=8, tag=""):
    """
    Validation: segmentation-only.
    Optional 10-class remap + per-condition mIoU using dataset labels.
    """
    if getattr(opts, "eval_mode", "19") == "10":
        if not hasattr(opts, "_metrics_10"):
            opts._metrics_10 = StreamSegMetrics(10)
            opts._remap_vec = load_labelmap(opts.labelmap)
        metrics_local = opts._metrics_10
    else:
        metrics_local = metrics

    metrics_local.reset()

    is_10c = (getattr(opts, "eval_mode", "19") == "10")
    if is_10c and getattr(opts, "separate_10c_dirs", False):
        save_dir = f"{save_dir}_10c"
    if getattr(opts, "out_tag", ""):
        save_dir = f"{save_dir}_{opts.out_tag}"
    os.makedirs(save_dir, exist_ok=True)

    from torch.utils.data import Subset, ConcatDataset

    def _unwrap_base_dataset(ds):
        while isinstance(ds, (Subset, ConcatDataset)):
            ds = ds.dataset if isinstance(ds, Subset) else ds.datasets[0]
        return ds

    _base_ds = _unwrap_base_dataset(loader.dataset)
    _decode_fn = getattr(_base_ds, "decode_target", None)

    per_cond = getattr(opts, "per_condition_val", False)
    if per_cond:
        ncls = getattr(metrics_local, "n_classes", getattr(opts, "num_classes", 19))
        cond_metrics = {k: StreamSegMetrics(ncls) for k in ["rain", "fog", "snow", "night"]}
        time_metrics = {k: StreamSegMetrics(ncls) for k in ["day", "night"]}

    condition_counts = {'rain': 0, 'fog': 0, 'snow': 0, 'night': 0, 'clear': 0}

    model.eval()
    shown = 0
    with torch.no_grad():
        for i, (images, labels, names, weather_ids, time_ids, domain) in tqdm(enumerate(loader)):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            # Count dataset condition distribution (proof)
            if weather_ids is not None and time_ids is not None:
                for j in range(len(weather_ids)):
                    w = int(weather_ids[j].item())
                    t = int(time_ids[j].item())
                    if t == 1:
                        condition_counts['night'] += 1
                    elif w == 1:
                        condition_counts['rain'] += 1
                    elif w == 2:
                        condition_counts['fog'] += 1
                    elif w == 3:
                        condition_counts['snow'] += 1
                    else:
                        condition_counts['clear'] += 1

            seg_logits, _, _ = model(images)
            seg_logits = nn.functional.interpolate(
                seg_logits, size=labels.shape[-2:], mode='bilinear', align_corners=False
            )

            preds = seg_logits.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            # save/vis (19-class)
            B = images.size(0)
            images_np = images.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()

            for k in range(B):
                if denorm is not None:
                    img_uint8 = (denorm(images_np[k]) * 255).transpose(1, 2, 0).astype(np.uint8)
                else:
                    x = images_np[k].transpose(1, 2, 0)
                    x = np.clip(x * 255.0, 0, 255)
                    img_uint8 = x.astype(np.uint8)

                if _decode_fn is not None:
                    gt_color = np.asarray(_decode_fn(labels_np[k])).astype(np.uint8)
                    pred_color = np.asarray(_decode_fn(preds[k])).astype(np.uint8)
                else:
                    gg = (labels_np[k].astype(np.uint8) % 20) * 12
                    pp = (preds[k].astype(np.uint8) % 20) * 12
                    gt_color = np.stack([gg] * 3, axis=2)
                    pred_color = np.stack([pp] * 3, axis=2)

                base = None
                try:
                    if isinstance(names, (list, tuple)) and k < len(names):
                        base = names[k]
                except Exception:
                    pass
                if not base:
                    base = f"{tag}_{i:06d}_{k}"

                if getattr(opts, "save_val_results", False):
                    suffix = "_10c" if is_10c and getattr(opts, "separate_10c_dirs", False) else ""
                    tag_sfx = (f"_{opts.out_tag}" if getattr(opts, "out_tag", "") else "")
                    Image.fromarray(img_uint8).save(os.path.join(save_dir, f"{base}{suffix}{tag_sfx}_image.png"))
                    Image.fromarray(gt_color).save(os.path.join(save_dir, f"{base}{suffix}{tag_sfx}_gt.png"))
                    Image.fromarray(pred_color).save(os.path.join(save_dir, f"{base}{suffix}{tag_sfx}_pred19.png"))

                if vis is not None and shown < max_vis:
                    triptych = np.concatenate([img_uint8, gt_color, pred_color], axis=1)
                    vis.vis_image(f"[{tag}] {base}", triptych.transpose(2, 0, 1))
                    shown += 1

            if getattr(opts, "eval_mode", "19") == "10":
                preds, targets = remap_to_10c(preds, targets, opts._remap_vec)
                if getattr(opts, "save_color_10c", False):
                    save_dir_10c = os.path.join(save_dir, "colored")
                    os.makedirs(save_dir_10c, exist_ok=True)
                    for k in range(B):
                        pred10_color = colorize_10c(preds[k].astype(np.uint8))
                        base_k = None
                        try:
                            if isinstance(names, (list, tuple)) and k < len(names):
                                base_k = str(names[k]).rsplit(".", 1)[0]
                        except Exception:
                            pass
                        if not base_k:
                            base_k = f"{tag}_{i:06d}_{k}"
                        tag_sfx = (f"_{opts.out_tag}" if getattr(opts, "out_tag", "") else "")
                        Image.fromarray(pred10_color).save(
                            os.path.join(save_dir_10c, f"{base_k}_pred10{tag_sfx}.png")
                        )

            metrics_local.update(targets, preds)

            if per_cond and weather_ids is not None:
                w_np = weather_ids.cpu().numpy()
                for cname, cid in zip(["rain", "fog", "snow"], [1, 2, 3]):
                    mask = (w_np == cid)
                    if mask.any():
                        cond_metrics[cname].update(targets[mask], preds[mask])

            if per_cond and time_ids is not None:
                t_np = time_ids.cpu().numpy()
                for tname, tid in zip(["day", "night"], [0, 1]):
                    mask = (t_np == tid)
                    if mask.any():
                        time_metrics[tname].update(targets[mask], preds[mask])

        score = metrics_local.get_results()

    extra = {"condition_counts": condition_counts}
    if per_cond:
        extra["cond_mIoU"] = {k: cond_metrics[k].get_results()["Mean IoU"] for k in cond_metrics}
        extra["time_mIoU"] = {k: time_metrics[k].get_results()["Mean IoU"] for k in time_metrics}

    print(f"[VALIDATION PROOF] Condition distribution: {condition_counts}")
    return score, [], extra


def apply_mode_presets(opts, _parser_defaults):
    if opts.mode == 0:
        opts.test_only = False
        opts.save_val_results = False
        opts.dataset = 'cityscapes_ACDC'

        # ✅ CRITICAL FAIRNESS FIX:
        # Mixed ablation must NOT leak official CS val (500) into training.
        # Use ONLY CS train split for 85/15 holdout.
        opts.cs_split_strategy = "train85_val15_only_train_split"

    elif opts.mode == 1:
        opts.test_only = False
        opts.save_val_results = False
        opts.dataset = 'cityscapes'

    elif opts.mode == 11:
        opts.test_only = True
        opts.save_val_results = True
        opts.dataset = 'cityscapes'
        if opts.ckpt is None:
            opts.ckpt = "checkpoints/mixdataset_on_orginal_mask2former_segonly_best.pth"
        if getattr(opts, "eval_mode", "19") == "10":
            if not getattr(opts, "labelmap", None):
                opts.labelmap = "configs/cityscapes_10class.json"
            opts.save_color_10c = True

    elif opts.mode == 21:
        opts.test_only = True
        opts.save_val_results = True
        opts.dataset = 'ACDC'
        if opts.ckpt is None:
            opts.ckpt = "checkpoints/mixdataset_on_orginal_mask2former_segonly_best.pth"
        if getattr(opts, "eval_mode", "19") == "10":
            if not getattr(opts, "labelmap", None):
                opts.labelmap = "configs/cityscapes_10class.json"
            opts.save_color_10c = True

    if opts.test_only:
        if opts.batch_size == _parser_defaults.batch_size:
            opts.batch_size = 1
        if opts.test_batch_size == _parser_defaults.test_batch_size:
            opts.test_batch_size = 1
        if opts.val_batch_size == _parser_defaults.val_batch_size:
            opts.val_batch_size = 1

    return opts


def main(ACDC_test_class=None, n_itrs=90000):
    opts = get_argparser().parse_args()
    _parser_defaults = get_argparser().parse_args([])

    if ACDC_test_class is not None:
        opts.ACDC_test_class = ACDC_test_class
    print(f"[DEBUG] Using ACDC_test_class={opts.ACDC_test_class}")

    opts.finetune = False
    opts.pretrained_model = None

    opts.data_root_cs = os.environ.get("CS_ROOT", "/home/ubuntu22user2/shiv/datasets/cityscapes")
    opts.data_root_acdc = os.environ.get("ACDC_ROOT", "/home/ubuntu22user2/shiv/datasets/ACDC")
    opts.total_itrs = n_itrs
    opts.test_class = None
    opts.val_interval = 5000

    opts = apply_mode_presets(opts, _parser_defaults)

    print("===================================================")
    print("  MASK2FORMER SEGMENTATION-ONLY: MIX DATASET TRAIN  ")
    print("---------------------------------------------------")
    print(f" Mode            : {opts.mode}")
    print(f" Dataset setting : {opts.dataset}")
    print(f" CS root         : {opts.data_root_cs}")
    print(f" ACDC root       : {opts.data_root_acdc}")
    print(f" Total iters     : {opts.total_itrs}")
    if opts.mode == 0:
        print(" TRAINING REGIME : SINGLE MIXED DATASET (CS85+ACDC85), VAL (CS15+ACDC15)")
        print(f" CS split strat  : {opts.cs_split_strategy}  (NO OFFICIAL VAL IN TRAIN)")
    elif opts.mode == 1:
        print(" TRAINING REGIME : Cityscapes-only")
    if opts.test_only:
        print(" RUN TYPE        : TEST-ONLY")
    else:
        print(" RUN TYPE        : TRAINING")
    print(" WAS/TAS         : DISABLED")
    print(" Encoder freezing: DISABLED")
    print("===================================================")

    utils.mkdir('runs')
    with open('runs/last_config.txt', 'w') as f:
        for k, v in vars(opts).items():
            f.write(f"{k}: {v}\n")
    print("[cfg] saved runs/last_config.txt")

    opts.model = "mask2former_segonly"
    opts.output_stride = 16
    opts.crop_val = True

    opts.num_classes = 19
    denorm = utils.Denormalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    vis = Visualizer(port=opts.vis_port, env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:
        vis.vis_table("Options", vars(opts))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DEVICE] Using device: {device}")

    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # ----------------------
    # DATASET / DATALOADER
    # ----------------------
    if opts.test_only:
        _, _, tst_dst = get_dataset(opts)
        test_loader = data.DataLoader(
            tst_dst,
            batch_size=opts.test_batch_size,
            shuffle=False,
            num_workers=0
        )
        print(f"[DATA] Dataset: {opts.dataset}, Test set size: {len(tst_dst)}")
    else:
        if opts.mode == 0:
            # MIXED DATASET
            train_mix, val_mix, _ = get_dataset(opts, 'cityscapes_ACDC')

            train_loader = data.DataLoader(
                train_mix, batch_size=opts.batch_size, shuffle=True,
                num_workers=4, pin_memory=True, drop_last=True
            )
            val_loader = data.DataLoader(
                val_mix, batch_size=opts.val_batch_size, shuffle=False,
                num_workers=2, pin_memory=True
            )

            print("[DATA] MIX Train size:", len(train_loader.dataset))
            print("[DATA] MIX Val size  :", len(val_loader.dataset))

        elif opts.mode == 1:
            train_dst_cs, val_dst_cs, _ = get_dataset(opts, 'cityscapes')
            train_loader = data.DataLoader(
                train_dst_cs, batch_size=opts.batch_size, shuffle=True,
                num_workers=0, drop_last=True
            )
            val_loader = data.DataLoader(
                val_dst_cs, batch_size=opts.val_batch_size, shuffle=False,
                num_workers=0
            )
            print("[DATA] CS Train size:", len(train_loader.dataset))
            print("[DATA] CS Val size  :", len(val_loader.dataset))
        else:
            raise RuntimeError("Unexpected mode in training branch")

    # ----------------------
    # MODEL
    # ----------------------
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes)
    print("[DEBUG] opts.model =", opts.model)
    print("[DEBUG] model.enable_wastas =", getattr(model, "enable_wastas", "MISSING"))
    print("[MODEL] Using model:", type(model))
    model = model.to(device)

    # BN momentum
    if isinstance(model, nn.DataParallel):
        if hasattr(model.module, 'backbone'):
            utils.set_bn_momentum(model.module.backbone, momentum=0.01)
    else:
        if hasattr(model, 'backbone'):
            utils.set_bn_momentum(model.backbone, momentum=0.01)

    metrics = StreamSegMetrics(opts.num_classes)

    # ----------------------
    # OPTIMIZER (same as yours)
    # ----------------------
    def collect_backbone_param_ids(model: nn.Module) -> set:
        ids = set()
        for attr in ["backbone", "encoder", "trunk", "body", "feature_extractor"]:
            mod = getattr(model, attr, None)
            if isinstance(mod, nn.Module):
                for p in mod.parameters():
                    ids.add(id(p))
        BACKBONE_CLASS_HINTS = ("backbone", "resnet", "swin", "convnext", "hrnet")
        for name, module in model.named_modules():
            cls = module.__class__.__name__.lower()
            if any(h in cls for h in BACKBONE_CLASS_HINTS):
                for p in module.parameters(recurse=False):
                    ids.add(id(p))
                for ch in module.children():
                    for p in ch.parameters(recurse=True):
                        ids.add(id(p))
        return ids

    backbone_param_ids = collect_backbone_param_ids(model)

    base_lr = opts.lr
    head_lr = base_lr * 5

    groups = [
        {"params": [], "weight_decay": 0.05, "lr": base_lr},  # backbone decay
        {"params": [], "weight_decay": 0.00, "lr": base_lr},  # backbone no-decay
        {"params": [], "weight_decay": 0.05, "lr": head_lr},  # heads decay
        {"params": [], "weight_decay": 0.00, "lr": head_lr},  # heads no-decay
    ]

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_backbone = (id(p) in backbone_param_ids)
        is_no_decay = (
            n.endswith(".bias")
            or any(k in n.lower() for k in ["norm", "bn", "bias", "layernorm", "ln"])
            or "pos_embed" in n.lower() or "absolute_pos_embed" in n.lower()
        )
        idx = (0 if is_backbone and not is_no_decay else
               1 if is_backbone and is_no_decay else
               2 if (not is_backbone) and not is_no_decay else
               3)
        groups[idx]["params"].append(p)

    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)

    print("AdamW param groups:",
          f"backbone(decay)={len(groups[0]['params'])}",
          f"backbone(no-decay)={len(groups[1]['params'])}",
          f"heads(decay)={len(groups[2]['params'])}",
          f"heads(no-decay)={len(groups[3]['params'])}",
          f"lr_backbone={base_lr}, lr_heads={head_lr}")

    # ----------------------
    # SCHEDULER
    # ----------------------
    # Keep your behavior: MODE 0 no accumulation; MODE 1 accumulation.
    accumulation_steps = 1 if opts.mode == 0 else 8
    warmup_steps = 1000
    total_steps = math.ceil(opts.total_itrs / accumulation_steps)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    # ----------------------
    # LOSS
    # ----------------------
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    else:
        raise NotImplementedError(f"Unsupported loss type: {opts.loss_type}")

    # ----------------------
    # CKPT HELPERS
    # ----------------------
    def save_ckpt(path):
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
            "notes": {"cur_itrs": cur_itrs, "best_score": best_score},
        }, path)
        print(f"[CKPT] Model saved at {path}")

    utils.mkdir('checkpoints')

    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0

    # restore
    if not opts.finetune:
        if opts.ckpt is not None and os.path.isfile(opts.ckpt):
            print(f"[CKPT] Restoring model from checkpoint: {opts.ckpt}")
            checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint["model_state"], strict=False)
            model = model.to(device)

            if opts.continue_training:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                scheduler.load_state_dict(checkpoint["scheduler_state"])
                cur_itrs = checkpoint.get("cur_itrs", 0)
                best_score = checkpoint.get("best_score", 0.0)
                print(f"[CKPT] Resumed training from iteration {cur_itrs}")

            del checkpoint
        else:
            print("[CKPT] Starting fresh training.")
            model = model.to(device)

    # ----------------------
    # TEST-ONLY
    # ----------------------
    if opts.test_only:
        print("[MODE] TEST-ONLY SEGMENTATION.")
        model.eval()
        test_score, _, extra = validate(
            opts=opts, model=model, loader=test_loader, device=device, metrics=metrics,
            ret_samples_ids=None, vis=vis, denorm=denorm,
            save_dir=f"results/test_{opts.dataset.lower()}",
            max_vis=opts.vis_num_samples, tag=opts.dataset.upper()
        )

        is_10c = (getattr(opts, "eval_mode", "19") == "10") and hasattr(opts, "_metrics_10")
        if is_10c:
            print(opts._metrics_10.to_str(test_score))
        else:
            print(metrics.to_str(test_score))
        return

    # ----------------------
    # TRAINING
    # ----------------------
    scaler = GradScaler()
    interval_loss = 0.0

    print("[TRAIN] Starting training loop...")
    print("[TRAIN] CONFIRMATION: seg-only, no WAS/TAS, no freezing.")
    if opts.mode == 0:
        print("[TRAIN] MODE 0: single MIXED dataset loader (CS85+ACDC85).")

    while cur_itrs < opts.total_itrs:
        model.train()
        cur_epochs += 1
        print(f"[EPOCH] Starting epoch {cur_epochs} (cur_itrs={cur_itrs})")

        if opts.mode == 0:
            # SINGLE MIXED TRAIN LOADER
            for step, (images, labels, names, weather_ids, time_ids, domain) in enumerate(train_loader):
                cur_itrs += 1

                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)

                optimizer.zero_grad(set_to_none=True)

                with autocast():
                    seg_logits, _, _ = model(images)
                    seg_logits = nn.functional.interpolate(
                        seg_logits, size=labels.shape[-2:], mode='bilinear', align_corners=False
                    )
                    loss_seg = criterion(seg_logits, labels)
                    total_loss = loss_seg

                    if torch.isnan(total_loss) or torch.isinf(total_loss):
                        print(f"[FATAL] NaN/Inf in loss at itr={cur_itrs}")
                        save_ckpt('checkpoints/abort_nan_mixdataset.pth')
                        raise RuntimeError("NaN/Inf detected")

                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                np_loss = float(loss_seg.detach().cpu().numpy())
                interval_loss += np_loss

                if vis is not None:
                    vis.vis_scalar('Loss/seg', cur_itrs, np_loss)

                if cur_itrs % 10 == 0:
                    avg_loss = interval_loss / 10.0
                    print(f"[TRAIN] Epoch {cur_epochs}, Itrs {cur_itrs}/{opts.total_itrs}, Loss(seg)={avg_loss:.4f}")
                    interval_loss = 0.0

                if cur_itrs % opts.val_interval == 0:
                    ck_latest = 'checkpoints/mixdataset_on_orginal_mask2former_segonly_latest.pth'
                    ck_best = 'checkpoints/mixdataset_on_orginal_mask2former_segonly_best.pth'
                    save_ckpt(ck_latest)

                    print("[VAL] Running validation on MIX val (CS15+ACDC15)...")
                    model.eval()
                    val_score, _, extra = validate(
                        opts=opts, model=model, loader=val_loader, device=device, metrics=metrics,
                        ret_samples_ids=None, vis=vis, denorm=denorm,
                        save_dir="results/val_mix", max_vis=opts.vis_num_samples, tag="MIX"
                    )
                    print(metrics.to_str(val_score))

                    if val_score['Mean IoU'] > best_score:
                        best_score = val_score['Mean IoU']
                        save_ckpt(ck_best)
                        print(f"[CKPT] NEW BEST | MIX mIoU={best_score:.3f}")

                    if vis is not None:
                        vis.vis_scalar("[Val] Mean IoU MIX", cur_itrs, val_score['Mean IoU'])
                        vis.vis_table("[Val] Class IoU MIX", val_score['Class IoU'])

                    model.train()

                if cur_itrs >= opts.total_itrs:
                    print("[TRAIN] Reached total iterations.")
                    return

        elif opts.mode == 1:
            # keep your MODE 1 exactly (accumulation)
            accumulation_steps = 8
            print(f"[MODE 1] Cityscapes-only training, accumulation_steps={accumulation_steps}")

            for step, (images, labels, _, _, _, _) in enumerate(train_loader):
                cur_itrs += 1
                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)

                if step % accumulation_steps == 0:
                    optimizer.zero_grad(set_to_none=True)

                with autocast():
                    seg_logits, _, _ = model(images)
                    seg_logits = nn.functional.interpolate(
                        seg_logits, size=labels.shape[-2:], mode='bilinear', align_corners=False
                    )
                    loss_segmentation = criterion(seg_logits, labels) / accumulation_steps

                scaler.scale(loss_segmentation).backward()

                if (step + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()

                np_loss = float(loss_segmentation.detach().cpu().numpy() * accumulation_steps)
                interval_loss += np_loss

                if vis is not None:
                    vis.vis_scalar('Loss', cur_itrs, np_loss)

                if cur_itrs % 10 == 0:
                    avg_loss = interval_loss / 10.0
                    print(f"[TRAIN] Epoch {cur_epochs}, Itrs {cur_itrs}/{opts.total_itrs}, Loss={avg_loss:.4f}")
                    interval_loss = 0.0

                if cur_itrs % opts.val_interval == 0:
                    ck_latest = 'checkpoints/main_mask2former_cityscapes_segonly_latest.pth'
                    ck_best = 'checkpoints/main_mask2former_cityscapes_segonly_best.pth'
                    save_ckpt(ck_latest)

                    print("[VAL] Running validation on CS val...")
                    model.eval()
                    val_score, _, extra = validate(
                        opts=opts, model=model, loader=val_loader, device=device, metrics=metrics,
                        ret_samples_ids=None, vis=vis, denorm=denorm,
                        save_dir="results/val_cs", max_vis=opts.vis_num_samples, tag="CS"
                    )
                    print(metrics.to_str(val_score))

                    if val_score['Mean IoU'] > best_score:
                        best_score = val_score['Mean IoU']
                        save_ckpt(ck_best)

                    model.train()

                if cur_itrs >= opts.total_itrs:
                    print("[TRAIN] Training completed.")
                    return


if __name__ == '__main__':
    main()

