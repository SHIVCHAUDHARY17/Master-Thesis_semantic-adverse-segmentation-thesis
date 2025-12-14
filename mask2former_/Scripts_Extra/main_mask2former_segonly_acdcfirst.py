# main_mask2former_segonly_acdcfirst.py
# ------------------------------------------------------------------
# MASK2FORMER SEGMENTATION-ONLY BASELINE FOR CS + ACDC
#  - No WAS/TAS heads or losses
#  - No encoder freezing (backbone always trainable)
#  - MODE 0: epoch-wise alternation ACDC(train85) -> Cityscapes(train85)
#  - MODE 1: Cityscapes-only baseline
#  - MODE 11: Cityscapes test
#  - MODE 21: ACDC test
# ------------------------------------------------------------------

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
import matplotlib
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
import math
import time
import json

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# ---- Constant label IDs for Cityscapes WS/TAS (kept for possible future use) ----
CLEAR_ID = 0
DAY_ID = 0


def load_labelmap(path_or_none):
    """Load label mapping from JSON/YAML file for 10-class remap."""
    remap = np.full(256, 255, dtype=np.uint8)  # Default: ignore

    if path_or_none is None:
        # Default 10-class mapping (paper subset)
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


# Palette for 10-class visualization
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

    # Core mode / eval switches
    parser.add_argument("--mode", type=int, default=1, choices=[0, 1, 11, 21],
                        help="0: train CS+ACDC (ACDC->CS alternation), "
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

    # Per-condition evaluation
    parser.add_argument("--per_condition_val", action="store_true", default=True,
                        help="Report mIoU by ACDC condition (rain/fog/snow/night)")

    # Dataset selection + root
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes', 'ACDC', 'cityscapes_ACDC'])
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="Fallback dataset root")

    # Cityscapes split options + explicit roots
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
                        help="Seed for 85/15 split of Cityscapes train.")
    parser.add_argument("--data_root_cs", type=str, default=None, help="Cityscapes root.")
    parser.add_argument("--data_root_acdc", type=str, default=None, help="ACDC root.")
    parser.add_argument("--data_root_awss", type=str, default=None, help="AWSS root.")

    # Model / training
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

    # VOC
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'])

    # ACDC
    parser.add_argument("--ACDC_test_class", type=str, default=None,
                        help="ACDC condition (rain/fog/snow/night)")

    # Visdom
    parser.add_argument("--enable_vis", action='store_true', default=False)
    parser.add_argument("--vis_port", type=str, default='13570')
    parser.add_argument("--vis_env", type=str, default='main')
    parser.add_argument("--vis_num_samples", type=int, default=8)

    # Num classes (optional override)
    parser.add_argument("--num_classes", type=int, default=None)

    return parser


def get_dataset(opts, tr_ds_name=None):
    """
    Returns (train_dst, val_dst, tst_dst) for the requested dataset name.
    Supports Cityscapes standard or 85/15 split via build_cityscapes_train85_val15_datasets.
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
        train_dst = []  # No training for ACDC in this branch
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

    elif name == 'AWSS':
        from datasets import AWSS  # only needed if you actually use AWSS
        train_transform = et.ExtCompose([
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.1673, 0.1685, 0.1948],
                            std=[0.0801, 0.0775, 0.0805]),
        ])
        val_transform = et.ExtCompose([
            et.ExtCenterCrop(opts.crop_size),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.1673, 0.1685, 0.1948],
                            std=[0.0801, 0.0775, 0.0805]),
        ])
        train_dst = AWSS(root=opts.data_root_awss, split='train', transform=train_transform)
        val_dst = AWSS(root=opts.data_root_awss, split='val', transform=val_transform)
        tst_dst = []
        return train_dst, val_dst, tst_dst

    else:
        raise NotImplementedError(f"Unknown dataset: {name}")


def validate(opts, model, loader, device, metrics,
             ret_samples_ids=None, vis=None, denorm=None,
             save_dir="results", max_vis=8, tag=""):
    """
    Validation: segmentation-only.
    Optional 10-class remap + per-condition mIoU using dataset weather/time labels.
    """
    # 19-class vs 10-class
    if getattr(opts, "eval_mode", "19") == "10":
        if not hasattr(opts, "_metrics_10"):
            opts._metrics_10 = StreamSegMetrics(10)
            opts._remap_vec = load_labelmap(opts.labelmap)
        metrics_local = opts._metrics_10
    else:
        metrics_local = metrics

    metrics_local.reset()
    ret_samples = []

    is_10c = (getattr(opts, "eval_mode", "19") == "10")
    if is_10c and getattr(opts, "separate_10c_dirs", False):
        save_dir = f"{save_dir}_10c"
    if getattr(opts, "out_tag", ""):
        save_dir = f"{save_dir}_{opts.out_tag}"
    os.makedirs(save_dir, exist_ok=True)

    shown = 0

    from torch.utils.data import Subset, ConcatDataset

    def _unwrap_base_dataset(ds):
        while isinstance(ds, (Subset, ConcatDataset)):
            ds = ds.dataset if isinstance(ds, Subset) else ds.datasets[0]
        return ds

    _base_ds = _unwrap_base_dataset(loader.dataset)
    _decode_fn = getattr(_base_ds, "decode_target", None)

    per_cond = getattr(opts, "per_condition_val", False)
    if per_cond:
        ncls = getattr(metrics_local, "n_classes",
                       getattr(metrics, "n_classes",
                               getattr(opts, "num_classes", 19)))
        cond_metrics = {k: StreamSegMetrics(ncls) for k in ["rain", "fog", "snow", "night"]}
        time_metrics = {k: StreamSegMetrics(ncls) for k in ["day", "night"]}

    condition_counts = {'rain': 0, 'fog': 0, 'snow': 0, 'night': 0, 'clear': 0}

    model.eval()
    with torch.no_grad():
        for i, (images, labels, names, weather_ids, time_ids, domain) in tqdm(enumerate(loader)):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            # Count conditions (dataset labels only)
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

            # Forward: segmentation only
            seg_logits, _, _ = model(images)
            seg_logits = nn.functional.interpolate(
                seg_logits, size=labels.shape[-2:], mode='bilinear', align_corners=False
            )

            preds = seg_logits.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            # Visualisation / saving (19-class)
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

            # 10-class remap
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
                # ACDC mapping: 0:clear, 1:rain, 2:fog, 3:snow; night via time_ids
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
                # For "night" condition metrics, time_ids == 1 already tracked above
                # We won't double-count "clear" vs "night" here, it's separated.

        score = metrics_local.get_results()

    extra = {}
    if per_cond:
        extra["cond_mIoU"] = {k: cond_metrics[k].get_results()["Mean IoU"] for k in cond_metrics}
        extra["time_mIoU"] = {k: time_metrics[k].get_results()["Mean IoU"] for k in time_metrics}

    print(f"[VALIDATION PROOF] Condition distribution: {condition_counts}")
    extra["condition_counts"] = condition_counts

    return score, ret_samples, extra


def preview_fullres(model, loader, device, denorm, vis, out_dir, tag, max_vis=4):
    """Quick full-res preview from a dataloader (no random crop)."""
    import os
    from PIL import Image
    import numpy as np
    from torch.utils.data import Subset, ConcatDataset

    ds = loader.dataset
    while isinstance(ds, (Subset, ConcatDataset)):
        ds = ds.dataset if isinstance(ds, Subset) else ds.datasets[0]
    decode = getattr(ds, "decode_target", None)
    os.makedirs(out_dir, exist_ok=True)

    model.eval()
    shown = 0
    with torch.no_grad():
        for images, labels, names, *_ in loader:
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            logits, _, _ = model(images)
            logits = torch.nn.functional.interpolate(
                logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            preds = logits.argmax(1).cpu().numpy()

            imgs_np = images.detach().cpu().numpy()
            gts_np = labels.detach().cpu().numpy()
            B = images.size(0)

            for k in range(B):
                img_u8 = (denorm(imgs_np[k]) * 255).transpose(1, 2, 0).astype(np.uint8)
                if decode is not None:
                    gt_color = np.asarray(decode(gts_np[k])).astype(np.uint8)
                    pred_color = np.asarray(decode(preds[k])).astype(np.uint8)
                else:
                    gg = (gts_np[k].astype(np.uint8) % 20) * 12
                    pp = (preds[k].astype(np.uint8) % 20) * 12
                    gt_color = np.stack([gg] * 3, axis=2)
                    pred_color = np.stack([pp] * 3, axis=2)

                base = (str(names[k]) if isinstance(names, (list, tuple)) else f"{tag}_{shown:03d}").split(".")[0]
                Image.fromarray(img_u8).save(os.path.join(out_dir, f"{base}_image.png"))
                Image.fromarray(gt_color).save(os.path.join(out_dir, f"{base}_gt.png"))
                Image.fromarray(pred_color).save(os.path.join(out_dir, f"{base}_pred.png"))

                if vis is not None:
                    trip = np.concatenate([img_u8, gt_color, pred_color], axis=1)
                    vis.vis_image(f"[PREVIEW]{tag} {base}", trip.transpose(2, 0, 1))

                shown += 1
                if shown >= max_vis:
                    break
            if shown >= max_vis:
                break
    model.train()


def apply_mode_presets(opts, _parser_defaults):
    """Apply MODE presets."""
    if opts.mode == 0:
        # train on Cityscapes + ACDC (alternating, ACDC->CS)
        opts.test_only = False
        opts.save_val_results = False
        opts.dataset = 'cityscapes_ACDC'

    elif opts.mode == 1:
        # train on Cityscapes only
        opts.test_only = False
        opts.save_val_results = False
        opts.dataset = 'cityscapes'

    elif opts.mode == 11:
        # test on Cityscapes
        opts.test_only = True
        opts.save_val_results = True
        opts.dataset = 'cityscapes'
        if opts.ckpt is None:
            opts.ckpt = "checkpoints/best_mask2former_cityscapes_ACDC_os16.pth"
        if getattr(opts, "eval_mode", "19") == "10":
            if not getattr(opts, "labelmap", None):
                opts.labelmap = "configs/cityscapes_10class.json"
            opts.save_color_10c = True

    elif opts.mode == 21:
        # test on ACDC
        opts.test_only = True
        opts.save_val_results = True
        opts.dataset = 'ACDC'
        if opts.ckpt is None:
            opts.ckpt = "checkpoints/best_mask2former_cityscapes_ACDC_os16.pth"
        if getattr(opts, "eval_mode", "19") == "10":
            if not getattr(opts, "labelmap", None):
                opts.labelmap = "configs/cityscapes_10class.json"
            opts.save_color_10c = True

    # test mode niceties
    if opts.test_only:
        if opts.batch_size == _parser_defaults.batch_size:
            opts.batch_size = 1
        if opts.test_batch_size == _parser_defaults.test_batch_size:
            opts.test_batch_size = 1
        if opts.val_batch_size == _parser_defaults.val_batch_size:
            opts.val_batch_size = 1

    return opts


def make_infinite(loader):
    """Safe infinite iterator."""
    while True:
        for batch in loader:
            yield batch


def main(ACDC_test_class=None, n_itrs=90000):
    opts = get_argparser().parse_args()
    _parser_defaults = get_argparser().parse_args([])

    if ACDC_test_class is not None:
        opts.ACDC_test_class = ACDC_test_class
    print(f"[DEBUG] Using ACDC_test_class={opts.ACDC_test_class}")

    opts.finetune = False
    opts.pretrained_model = None

    # Dataset roots (adapt to your paths)
    opts.data_root_cs = os.environ.get("CS_ROOT", "/home/ubuntu22user2/shiv/datasets/cityscapes")
    opts.data_root_acdc = os.environ.get("ACDC_ROOT", "/home/ubuntu22user2/shiv/datasets/ACDC")
    opts.total_itrs = n_itrs
    opts.test_class = None
    opts.val_interval = 5000  # long training -> less frequent validation

    opts = apply_mode_presets(opts, _parser_defaults)

    # >>> EXPLICIT RUN CONFIG SUMMARY <<<
    print("===================================================")
    print("     MASK2FORMER SEGMENTATION-ONLY BASELINE RUN     ")
    print("---------------------------------------------------")
    print(f" Mode            : {opts.mode}")
    print(f" Dataset setting : {opts.dataset}")
    print(f" CS root         : {opts.data_root_cs}")
    print(f" ACDC root       : {opts.data_root_acdc}")
    print(f" Total iters     : {opts.total_itrs}")
    if opts.mode == 0:
        print(" TRAINING REGIME : [EPOCH-WISE] ACDC(train85) -> Cityscapes(train85)")
    elif opts.mode == 1:
        print(" TRAINING REGIME : Cityscapes-only (train85/val15 split if configured)")
    if opts.test_only:
        print(" RUN TYPE        : TEST-ONLY")
    else:
        print(" RUN TYPE        : TRAINING")
    print(" WAS/TAS         : COMPLETELY DISABLED (NOT USED IN THIS SCRIPT)")
    print(" Encoder freezing: COMPLETELY DISABLED (BACKBONE ALWAYS TRAINABLE)")
    print("===================================================")

    utils.mkdir('runs')
    with open('runs/last_config.txt', 'w') as f:
        for k, v in vars(opts).items():
            f.write(f"{k}: {v}\n")
    print("[cfg] saved runs/last_config.txt")

    # Force model settings
    opts.model = "mask2former"
    opts.output_stride = 16
    opts.crop_val = True

    # num_classes + denorm
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
        denorm = utils.Denormalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    elif opts.dataset.lower() in ['cityscapes', 'acdc', 'cityscapes_acdc', 'cityscapes_acdc'.upper()]:
        opts.num_classes = 19
        denorm = utils.Denormalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    else:
        opts.num_classes = 19
        denorm = utils.Denormalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    # Visualisation
    vis = Visualizer(port=opts.vis_port, env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:
        vis.vis_table("Options", vars(opts))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DEVICE] Using device: {device}")

    # Seeds
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Dataset / DataLoader
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
            # CS + ACDC training

            train_dst_cs, val_dst_cs, _ = get_dataset(opts, 'cityscapes')
            train_loader_cs = data.DataLoader(
                train_dst_cs, batch_size=opts.batch_size, shuffle=True,
                num_workers=4, pin_memory=True, drop_last=True
            )
            val_loader_cs = data.DataLoader(
                val_dst_cs, batch_size=opts.val_batch_size, shuffle=False,
                num_workers=2, pin_memory=True
            )

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

            train_dst_acdc = ACDC(
                root=opts.data_root_acdc, split='train85',
                transform=train_transform, holdout_seed=opts.cs_holdout_seed
            )
            val_dst_acdc = ACDC(
                root=opts.data_root_acdc, split='val15',
                transform=val_transform, holdout_seed=opts.cs_holdout_seed
            )

            train_loader_acdc = data.DataLoader(
                train_dst_acdc, batch_size=opts.batch_size, shuffle=True,
                num_workers=4, pin_memory=True, drop_last=True
            )
            val_loader_acdc = data.DataLoader(
                val_dst_acdc, batch_size=opts.val_batch_size, shuffle=False,
                num_workers=2, pin_memory=True
            )

        elif opts.mode == 1:
            train_dst_cs, val_dst_cs, _ = get_dataset(opts, 'cityscapes')
            train_loader_cs = data.DataLoader(
                train_dst_cs, batch_size=opts.batch_size,
                shuffle=True, num_workers=0, drop_last=True
            )
            val_loader_cs = data.DataLoader(
                val_dst_cs, batch_size=opts.val_batch_size,
                shuffle=False, num_workers=0
            )

        print("[DATA] CS Train size:", len(train_loader_cs.dataset))
        if opts.mode == 0:
            print("[DATA] ACDC Train85 size:", len(train_loader_acdc.dataset))
        print("[DATA] CS Val size:", len(val_loader_cs.dataset))
        if opts.mode == 0:
            print("[DATA] ACDC Val15 size:", len(val_loader_acdc.dataset))

    # Model
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes)
    print("[MODEL] Using model:", type(model))
    model = model.to(device)

    # BN momentum
    if isinstance(model, nn.DataParallel):
        if hasattr(model.module, 'backbone'):
            utils.set_bn_momentum(model.module.backbone, momentum=0.01)
    else:
        if hasattr(model, 'backbone'):
            utils.set_bn_momentum(model.backbone, momentum=0.01)

    # Optional previews
    if opts.enable_vis and not opts.test_only:
        if opts.mode == 0:
            preview_fullres(model, val_loader_cs, device, denorm, vis,
                            out_dir="results/preview_cs", tag="CS", max_vis=4)
            preview_fullres(model, val_loader_acdc, device, denorm, vis,
                            out_dir="results/preview_acdc", tag="ACDC", max_vis=4)
        elif opts.mode == 1:
            preview_fullres(model, val_loader_cs, device, denorm, vis,
                            out_dir="results/preview_cs", tag="CS", max_vis=4)

    metrics = StreamSegMetrics(opts.num_classes)

    # Optimizer
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

    # LR scheduler
    accumulation_steps = 1 if opts.mode == 0 else 8
    warmup_steps = 1000
    total_steps = math.ceil(opts.total_itrs / accumulation_steps)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    # Loss
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    else:
        raise NotImplementedError(f"Unsupported loss type: {opts.loss_type}")

    # Checkpoint helpers
    def save_ckpt(path):
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score_cs": best_score_cs,
            "best_score_acdc": best_score_acdc,
            "notes": {
                "cur_itrs": cur_itrs,
                "best_score_cs": best_score_cs,
                "best_score_acdc": best_score_acdc
            },
        }, path)
        print(f"[CKPT] Model saved at {path}")

    utils.mkdir('checkpoints')

    best_score_cs = 0.0
    best_score_acdc = 0.0
    cur_itrs = 0
    cur_epochs = 0

    # Restore / finetune logic
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
                best_score_cs = checkpoint.get("best_score_cs", 0.0)
                best_score_acdc = checkpoint.get("best_score_acdc", 0.0)
                print(f"[CKPT] Resumed training from iteration {cur_itrs}")

            del checkpoint
        else:
            print("[CKPT] Starting fresh training.")
            model = model.to(device)
    else:
        print(f"[CKPT] Fine-tuning from pretrained model: {opts.pretrained_model}")
        checkpoint = torch.load(opts.pretrained_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"], strict=False)
        model = model.to(device)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        cur_itrs = checkpoint.get("cur_itrs", 0)
        best_score_cs = checkpoint.get("best_score_cs", 0.0)
        best_score_acdc = checkpoint.get("best_score_acdc", 0.0)

    # TEST-ONLY BRANCH
    if opts.test_only:
        print("[MODE] TEST-ONLY SEGMENTATION (no WAS/TAS, no freezing).")
        model.eval()
        test_score, ret_samples, extra = validate(
            opts=opts, model=model, loader=test_loader, device=device, metrics=metrics,
            ret_samples_ids=None, vis=vis, denorm=denorm,
            save_dir=f"results/test_{opts.dataset.lower()}",
            max_vis=opts.vis_num_samples, tag=opts.dataset.upper()
        )
        is_10c = (getattr(opts, "eval_mode", "19") == "10") and hasattr(opts, "_metrics_10")

        if is_10c:
            print(opts._metrics_10.to_str(test_score))
            class_iou = test_score.get('Class IoU', {})
            IoU_scores = np.array([class_iou.get(i, 0.0) for i in range(10)])
            print("IoU (10-class):", IoU_scores)
        else:
            print(metrics.to_str(test_score))
            class_iou = test_score.get('Class IoU', {})
            if isinstance(class_iou, dict):
                IoU_scores = np.array([class_iou.get(i, 0.0) for i in range(getattr(metrics, "n_classes", 19))])
            else:
                IoU_scores = np.array(class_iou)
            print("IoU (19-class):", IoU_scores)
        return

    # TRAINING BRANCH
    if not opts.test_only:
        if opts.mode == 0:
            vis_sample_id_cs = (
                np.random.randint(0, len(val_loader_cs), opts.vis_num_samples, np.int32)
                if opts.enable_vis else None
            )
            vis_sample_id_acdc = (
                np.random.randint(0, len(val_loader_acdc), opts.vis_num_samples, np.int32)
                if opts.enable_vis else None
            )
        elif opts.mode == 1:
            vis_sample_id_cs = (
                np.random.randint(0, len(val_loader_cs), opts.vis_num_samples, np.int32)
                if opts.enable_vis else None
            )
            vis_sample_id_acdc = None

    scaler = GradScaler()
    interval_loss = 0
    cs_seen = 0
    acdc_seen = 0

    print("[TRAIN] Starting training loop...")
    print("[TRAIN] CONFIRMATION: segmentation-only, no WAS/TAS, no encoder freezing, "
          "ACDC->CS alternation in MODE 0.")

    # MAIN TRAINING LOOP
    while cur_itrs < opts.total_itrs:
        model.train()
        cur_epochs += 1
        print(f"[EPOCH] Starting epoch {cur_epochs} (cur_itrs={cur_itrs})")

        if opts.mode == 0:
            # ===== ACDC PASS FIRST =====
            print(f"[EPOCH {cur_epochs}] ACDC pass (train85)")
            for step_acdc, (images, labels, names, weather_ids, time_ids, _) in enumerate(train_loader_acdc):
                domain_name = "ACDC"
                cur_itrs += 1

                dev = next(model.parameters()).device
                images = images.to(dev, dtype=torch.float32)
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
                        print(f"[FATAL] NaN/Inf in total_loss at itr={cur_itrs}")
                        save_ckpt('checkpoints/abort_nan.pth')
                        raise RuntimeError("NaN/Inf detected")

                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                np_loss = float(loss_seg.detach().cpu().numpy())
                interval_loss += np_loss
                acdc_seen += 1

                if vis is not None:
                    vis.vis_scalar('Loss/seg', cur_itrs, np_loss)

                if cur_itrs % 10 == 0:
                    avg_loss = interval_loss / 10.0
                    print(f"[TRAIN] Epoch {cur_epochs}, Itrs {cur_itrs}/{opts.total_itrs}, "
                          f"Loss(seg)={avg_loss:.4f}, Domain={domain_name}")
                    interval_loss = 0.0

                if cur_itrs % opts.val_interval == 0:
                    ck_latest = 'checkpoints/main_mask2former_segonly_acdc_cs_alt_latest.pth'
                    ck_best   = 'checkpoints/main_mask2former_segonly_acdc_cs_alt_best.pth'
                    save_ckpt(ck_latest)
                    print("[VAL] Running validation on CS + ACDC...")
                    model.eval()
                    val_score_cs, _, extra_cs = validate(
                        opts=opts, model=model, loader=val_loader_cs, device=device, metrics=metrics,
                        ret_samples_ids=vis_sample_id_cs,
                        vis=vis, denorm=denorm,
                        save_dir="results/cs", max_vis=opts.vis_num_samples, tag="CS"
                    )
                    val_score_acdc, _, extra_acdc = validate(
                        opts=opts, model=model, loader=val_loader_acdc, device=device, metrics=metrics,
                        ret_samples_ids=vis_sample_id_acdc,
                        vis=vis, denorm=denorm,
                        save_dir="results/acdc", max_vis=opts.vis_num_samples, tag="ACDC"
                    )
                    print(metrics.to_str(val_score_cs), metrics.to_str(val_score_acdc))

                    if val_score_cs['Mean IoU'] > best_score_cs:
                        best_score_cs = val_score_cs['Mean IoU']
                        best_score_acdc = val_score_acdc['Mean IoU']
                        save_ckpt(ck_best)
                        print(f"[CKPT] NEW BEST | CS mIoU={best_score_cs:.3f} | "
                              f"ACDC mIoU={best_score_acdc:.3f}")

                    if vis is not None:
                        vis.vis_scalar("[Val] Mean IoU CS", cur_itrs, val_score_cs['Mean IoU'])
                        vis.vis_table("[Val] Class IoU CS", val_score_cs['Class IoU'])
                        vis.vis_scalar("[Val] Mean IoU ACDC", cur_itrs, val_score_acdc['Mean IoU'])
                        vis.vis_table("[Val] Class IoU ACDC", val_score_acdc['Class IoU'])

                    model.train()

                if cur_itrs >= opts.total_itrs:
                    print("[TRAIN] Reached total iterations during ACDC phase.")
                    return

            # ===== CITYSCAPES PASS =====
            print(f"[EPOCH {cur_epochs}] Cityscapes pass (train85)")
            for step_cs, (images, labels, names, weather_ids, time_ids, _) in enumerate(train_loader_cs):
                domain_name = "Cityscapes"
                cur_itrs += 1

                dev = next(model.parameters()).device
                images = images.to(dev, dtype=torch.float32)
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
                        print(f"[FATAL] NaN/Inf in total_loss at itr={cur_itrs}")
                        save_ckpt('checkpoints/abort_nan.pth')
                        raise RuntimeError("NaN/Inf detected")

                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                np_loss = float(loss_seg.detach().cpu().numpy())
                interval_loss += np_loss
                cs_seen += 1

                if vis is not None:
                    vis.vis_scalar('Loss/seg', cur_itrs, np_loss)

                if cur_itrs % 10 == 0:
                    avg_loss = interval_loss / 10.0
                    print(f"[TRAIN] Epoch {cur_epochs}, Itrs {cur_itrs}/{opts.total_itrs}, "
                          f"Loss(seg)={avg_loss:.4f}, Domain={domain_name}")
                    interval_loss = 0.0

                if cur_itrs % opts.val_interval == 0:
                    ck_latest = 'checkpoints/main_mask2former_segonly_acdc_cs_alt_latest.pth'
                    ck_best   = 'checkpoints/main_mask2former_segonly_acdc_cs_alt_best.pth'
                    save_ckpt(ck_latest)
                    print("[VAL] Running validation on CS + ACDC...")
                    model.eval()
                    val_score_cs, _, extra_cs = validate(
                        opts=opts, model=model, loader=val_loader_cs, device=device, metrics=metrics,
                        ret_samples_ids=vis_sample_id_cs,
                        vis=vis, denorm=denorm,
                        save_dir="results/cs", max_vis=opts.vis_num_samples, tag="CS"
                    )
                    val_score_acdc, _, extra_acdc = validate(
                        opts=opts, model=model, loader=val_loader_acdc, device=device, metrics=metrics,
                        ret_samples_ids=vis_sample_id_acdc,
                        vis=vis, denorm=denorm,
                        save_dir="results/acdc", max_vis=opts.vis_num_samples, tag="ACDC"
                    )
                    print(metrics.to_str(val_score_cs), metrics.to_str(val_score_acdc))

                    if val_score_cs['Mean IoU'] > best_score_cs:
                        best_score_cs = val_score_cs['Mean IoU']
                        best_score_acdc = val_score_acdc['Mean IoU']
                        save_ckpt(ck_best)
                        print(f"[CKPT] NEW BEST | CS mIoU={best_score_cs:.3f} | "
                              f"ACDC mIoU={best_score_acdc:.3f}")

                    if vis is not None:
                        vis.vis_scalar("[Val] Mean IoU CS", cur_itrs, val_score_cs['Mean IoU'])
                        vis.vis_table("[Val] Class IoU CS", val_score_cs['Class IoU'])
                        vis.vis_scalar("[Val] Mean IoU ACDC", cur_itrs, val_score_acdc['Mean IoU'])
                        vis.vis_table("[Val] Class IoU ACDC", val_score_acdc['Class IoU'])

                    model.train()

                if cur_itrs >= opts.total_itrs:
                    print("[TRAIN] Reached total iterations during Cityscapes phase.")
                    return

        elif opts.mode == 1:
            # Cityscapes-only with gradient accumulation
            accumulation_steps = 8
            print(f"[MODE 1] Cityscapes-only training, accumulation_steps={accumulation_steps}")

            for step, (images, labels, _, _, _, _) in enumerate(train_loader_cs):
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
                    total_loss = loss_segmentation

                scaler.scale(total_loss).backward()

                if (step + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()

                np_loss = loss_segmentation.detach().cpu().numpy() * accumulation_steps
                interval_loss += np_loss
                if vis is not None:
                    vis.vis_scalar('Loss', cur_itrs, np_loss)

                if cur_itrs % 10 == 0:
                    interval_loss = interval_loss / 10
                    print(f"[TRAIN] Epoch {cur_epochs}, Itrs {cur_itrs}/{opts.total_itrs}, "
                          f"Loss={interval_loss:.4f}")
                    interval_loss = 0.0

                if cur_itrs % opts.val_interval == 0:
                    ck_latest = 'checkpoints/main_mask2former_cityscapes_segonly_latest.pth'
                    ck_best   = 'checkpoints/main_mask2former_cityscapes_segonly_best.pth'
                    save_ckpt(ck_latest)
                    print("[VAL] Running validation on CS...")
                    model.eval()
                    val_score_cs, _, extra_cs = validate(
                        opts=opts, model=model, loader=val_loader_cs, device=device,
                        metrics=metrics, ret_samples_ids=None
                    )
                    print(metrics.to_str(val_score_cs))

                    if val_score_cs['Mean IoU'] > best_score_cs:
                        best_score_cs = val_score_cs['Mean IoU']
                        save_ckpt(ck_best)

                    if vis is not None:
                        vis.vis_scalar("[Val] Mean IoU CS", cur_itrs, val_score_cs['Mean IoU'])
                        vis.vis_table("[Val] Class IoU CS", val_score_cs['Class IoU'])

                    model.train()

                if cur_itrs >= opts.total_itrs:
                    print("[TRAIN] Training completed! Running final validation on CS...")
                    model.eval()
                    final_score, _, _ = validate(
                        opts=opts, model=model, loader=val_loader_cs,
                        device=device, metrics=metrics
                    )
                    print(f"[FINAL] CS mIoU: {final_score['Mean IoU']:.4f}")
                    return


if __name__ == '__main__':
    main()

