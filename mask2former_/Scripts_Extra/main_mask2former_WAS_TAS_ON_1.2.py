# main_mask2former.py
# The model is weather and time aware!
# This models is trained on: ACDC + CS 1:1
# low level features i.e module.backbone.low_level_features (Atrous Conv.) are frozen when training on CS
# Multi-task learning two losses Segmentation loss and weather_time loss, propagated separtely.
# weather awareness just of the Atrous Convolution
# ------------------------
# Please note that our code is based on Mask2Former pytorch implementation.
# --------------------------

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
#from datasets.cityscapes_baseline import Cityscapes  # ← FIXED version
from datasets.cityscapes_baseline import Cityscapes, build_cityscapes_train85_val15_datasets
from datasets.ACDC_baseline_19 import ACDC


#from datasets import ACDC, AWSS, Cityscapes
#from datasets import ACDC, AWSS
from utils import ext_transforms as et
from metrics import StreamSegMetrics
from utils.visualizer import Visualizer
from PIL import Image
#from freeze_tools import freeze_encoder, unfreeze_encoder
import matplotlib
import matplotlib.pyplot as plt
# Add this after your existing imports
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
    

# ---- Constant label IDs for Cityscapes WS/TAS (adjust if your dataset differs) ----
CLEAR_ID = 0  # WS: clear
DAY_ID   = 0  # TAS: day

def load_labelmap(path_or_none):
    """Load label mapping from JSON/YAML file"""
    remap = np.full(256, 255, dtype=np.uint8)  # Default: map everything to ignore
    
    if path_or_none is None:
        # =============================================================
        # DEFAULT: 10-class SUBSET evaluation (others → 255/ignore)
        # Mapping: 0:road, 1:sidewalk, 2:building, 3:pole, 4:traffic light,
        #          5:traffic sign, 6:vegetation, 7:sky, 8:person, 9:car
        # All other classes are ignored (mapped to 255)
        # =============================================================
        # Default mapping (matches your paper's 10 classes)
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
    
    # Apply the mapping
    for k, v in cfg["groups"].items():
        remap[int(k)] = int(v)
    
    for ig in cfg.get("ignore", []):
        remap[int(ig)] = 255
    
    return remap

def remap_to_10c(preds_np, targets_np, remap_vec):
    """Remap predictions and targets using the mapping vector"""
    preds_10 = remap_vec[preds_np]
    targets_10 = remap_vec[targets_np]
    return preds_10, targets_10

# Palette for 10-class visualization (matches paper's colors)
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
    """Colorize 10-class masks"""
    h, w = mask_hw.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    valid = (mask_hw < 10)  # Only color valid classes (0-9)
    out[valid] = PALETTE_10C[mask_hw[valid]]
    return out


def get_argparser():
    parser = argparse.ArgumentParser()
    
    # Add these RIGHT AFTER the existing arguments
    parser.add_argument("--disable_ws_tas", action="store_true", default=False,
                        help="Turn off WAS/TAS heads & grads (true ablation)")

    parser.add_argument("--ws_weight", type=float, default=1e-5,
                        help="Weight for weather supervision loss")
    parser.add_argument("--tes_weight", type=float, default=1e-5, 
                        help="Weight for time-of-day supervision loss")
    parser.add_argument("--log_heads_every", type=int, default=50,
                        help="Print WS/TES stats every N iterations")
    parser.add_argument("--per_condition_val", action="store_true", default=True,
                        help="Report mIoU by ACDC condition (rain/fog/snow/night)")
    parser.add_argument("--check_backbone_grads", action="store_true", default=False,
                        help="Log backbone grad-norms to verify freeze/unfreeze")
    parser.add_argument("--use_aux_adapter", action="store_true", default=False,
                        help="Build WS/TAS linear adapter (aux_in_proj). Must match the checkpoint.")

    # ---- Core mode / eval switches ----
    parser.add_argument("--mode", type=int, default=1, choices=[0, 1, 11, 21],
                        help="0: train CS+ACDC (alternating), 1: train CS, 11: test CS, 21: test ACDC")

    parser.add_argument("--cs_eval_split", type=str, default="val", choices=["val", "test"],
                        help="Cityscapes split to evaluate when in test-only mode")
    parser.add_argument("--eval_mode", type=str, default="19", choices=["19", "10"],
                        help="Evaluation head-space: 19 (native) or 10-class remap")
    parser.add_argument("--labelmap", type=str, default=None,
                        help="Path to JSON/YAML mapping trainId(0..18)->10-class or 255(ignore)")
    parser.add_argument("--save_color_10c", action="store_true", default=False,
                        help="Save colored masks for 10-class eval")
    # --- keep 10-class results separate ---

    parser.add_argument("--separate_10c_dirs", action="store_true", default=True,
                        help="When eval_mode=10, write into *_10c directories and suffix filenames")
    parser.add_argument("--out_tag", type=str, default="",
                        help="Optional extra tag appended to result dirs/filenames")
                        
    # ---- Dataset selection + generic root ----
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc','cityscapes','ACDC','cityscapes_ACDC'])

    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="Fallback dataset root")

    # ---- Cityscapes split options + explicit roots ----
    parser.add_argument(
        "--cs_split_strategy", type=str, default="standard",
        choices=[
            "standard",
            "train85_val15_plus_officialval_in_train",
            "train85_val15_only_train_split"
        ],
        help=("Cityscapes split policy: 'standard' = 2975 train / 500 val; "
              "'train85_val15_plus_officialval_in_train' = 85% of train + all official val used in training; "
              "'train85_val15_only_train_split' = 85%/15% split of train only; official val kept for eval.")
    )
    parser.add_argument("--cs_holdout_seed", type=int, default=1,
                        help="Seed for 85/15 split of Cityscapes train.")
    parser.add_argument("--data_root_cs", type=str, default=None, help="Cityscapes root.")
    parser.add_argument("--data_root_acdc", type=str, default=None, help="ACDC root.")
    parser.add_argument("--data_root_awss", type=str, default=None, help="AWSS root.")

    # ---- Model / training ----
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

    # ---- VOC ----
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug','2012','2011','2009','2008','2007'])

    # ---- ACDC ----
    parser.add_argument("--ACDC_test_class", type=str, default=None,
                        help="ACDC condition (rain/fog/snow/night)")

    # ---- Visdom ----
    parser.add_argument("--enable_vis", action='store_true', default=False)
    parser.add_argument("--vis_port", type=str, default='13570')
    parser.add_argument("--vis_env", type=str, default='main')
    parser.add_argument("--vis_num_samples", type=int, default=8)

    # ---- Num classes (optional override) ----
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

        # Resolve root (prefer explicit Cityscapes root; otherwise fall back to generic data_root)
        root_cs = getattr(opts, "data_root_cs", None) or getattr(opts, "data_root", None)

        if getattr(opts, "cs_split_strategy", "standard") == "standard":
            train_dst = Cityscapes(root=root_cs, split='train', transform=train_transform)
            val_dst   = Cityscapes(root=root_cs, split='val',   transform=val_transform)
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
        tst_dst   = Cityscapes(root=root_cs, split=tst_split, transform=val_transform)
        return train_dst, val_dst, tst_dst

    elif name == 'ACDC':
        val_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        train_dst = []  # No training for ACDC here
        val_dst   = ACDC(root=opts.data_root_acdc, split='val', transform=val_transform)

        # Load all validation images
        tst_dst_all = ACDC(root=opts.data_root_acdc, split='val', transform=val_transform)
        
        # Manual filtering for specific condition
        if getattr(opts, "ACDC_test_class", None):
            from torch.utils.data import Subset
            
            condition_indices = []
            requested_condition = opts.ACDC_test_class.lower()
            
            print(f"[MANUAL FILTER] Filtering ACDC val for condition: {requested_condition}")
            
            for i in range(len(tst_dst_all)):
                # Check the condition from the dataset's conditions list
                if tst_dst_all.conditions[i] == requested_condition:
                    condition_indices.append(i)
            
            tst_dst = Subset(tst_dst_all, condition_indices)
            print(f"[MANUAL FILTER] Loaded {len(tst_dst)} {requested_condition} images from ACDC val")
        else:
            tst_dst = tst_dst_all
        
        return train_dst, val_dst, tst_dst

    elif name == 'AWSS':
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
        val_dst   = AWSS(root=opts.data_root_awss, split='val',   transform=val_transform)
        tst_dst   = []
        return train_dst, val_dst, tst_dst

    else:
        raise NotImplementedError(f"Unknown dataset: {name}")
    
def validate(opts, model, loader, device, metrics,
             ret_samples_ids=None, vis=None, denorm=None,
             save_dir="results", max_vis=8, tag=""):
    """Run validation with optional 10-class remapping"""
    # Reset appropriate metrics
    if getattr(opts, "eval_mode", "19") == "10":
        if not hasattr(opts, "_metrics_10"):
            opts._metrics_10 = StreamSegMetrics(10)
            opts._remap_vec = load_labelmap(opts.labelmap)
        metrics_local = opts._metrics_10
    else:
        metrics_local = metrics
    
    metrics_local.reset()
    ret_samples = []
    #os.makedirs(save_dir, exist_ok=True)
    # --- keep 10-class results separate (dir naming) ---
    is_10c = (getattr(opts, "eval_mode", "19") == "10")
    if is_10c and getattr(opts, "separate_10c_dirs", False):
        save_dir = f"{save_dir}_10c"
    if getattr(opts, "out_tag", ""):
        save_dir = f"{save_dir}_{opts.out_tag}"
    os.makedirs(save_dir, exist_ok=True)
    shown = 0  # number of triptychs pushed to Visdom for this validation
    # === unwrap base dataset so we can access decode_target even if
    # loader.dataset is a Subset or ConcatDataset (Cityscapes val is a Subset) ===
    from torch.utils.data import Subset, ConcatDataset
    def _unwrap_base_dataset(ds):
        while isinstance(ds, (Subset, ConcatDataset)):
            if isinstance(ds, Subset):
                ds = ds.dataset
            else:  # ConcatDataset
                ds = ds.datasets[0]
        return ds
    _base_ds = _unwrap_base_dataset(loader.dataset)
    _decode_fn = getattr(_base_ds, "decode_target", None)
    
    # Add these for head accuracy tracking
    ws_correct = 0
    tes_correct = 0
    ws_total = 0
    tes_total = 0
    
    # Per-condition metrics
    per_cond = getattr(opts, "per_condition_val", False)
    if per_cond:
        ncls = getattr(metrics_local, "n_classes",
               getattr(metrics, "n_classes",
                       getattr(opts, "num_classes", 19)))
        cond_metrics = {k: StreamSegMetrics(ncls) for k in ["rain","fog","snow","night"]}
        time_metrics = {k: StreamSegMetrics(ncls) for k in ["day","night"]}
    
    # === ADD CONDITION TRACKING FOR PROOF ===
    condition_counts = {'rain': 0, 'fog': 0, 'snow': 0, 'night': 0, 'clear': 0}
    weather_id_to_condition = {
        1: 'rain',    # Based on your ACDC dataset mapping
        2: 'fog',     # /rain/ -> weather_id=1
        3: 'snow',    # /fog/ -> weather_id=2  
        0: 'clear'    # /snow/ -> weather_id=3
    }
    # ========================================
    
    model.eval()
    with torch.no_grad():
        for i, (images, labels, names, weather_ids, time_ids, domain) in tqdm(enumerate(loader)):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            # === COUNT CONDITIONS FOR PROOF ===
            for j in range(len(weather_ids)):
                weather_id_val = weather_ids[j].item()
                time_id_val = time_ids[j].item()
                
                if time_id_val == 1:  # night
                    condition_counts['night'] += 1
                elif weather_id_val == 1:  # rain
                    condition_counts['rain'] += 1
                elif weather_id_val == 2:  # fog
                    condition_counts['fog'] += 1
                elif weather_id_val == 3:  # snow
                    condition_counts['snow'] += 1
                else:  # clear
                    condition_counts['clear'] += 1
            # ==================================

            # Forward pass
            logits, weather_preds, time_preds = model(images)
            logits = nn.functional.interpolate(logits, size=labels.shape[-2:], 
                                             mode='bilinear', align_corners=False)
            
            # Get predictions
            preds = logits.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            # ---------- Save & visualize (19-class) ----------
            B = images.size(0)
            images_np = images.detach().cpu().numpy()   # NCHW
            labels_np = labels.detach().cpu().numpy()   # NHW
            preds_np  = preds                            # NHW (already numpy)

            for k in range(B):
                # 1) de-norm input -> uint8 HxWx3
                if denorm is not None:
                    img_uint8 = (denorm(images_np[k]) * 255).transpose(1, 2, 0).astype(np.uint8)
                else:
                    x = images_np[k].transpose(1, 2, 0)
                    x = np.clip(x * 255.0, 0, 255)
                    img_uint8 = x.astype(np.uint8)

                # 2) colorize GT & Pred using the *real* dataset's decoder (handles Subset/ConcatDataset)
                if _decode_fn is not None:
                    gt_color   = np.asarray(_decode_fn(labels_np[k])).astype(np.uint8)
                    pred_color = np.asarray(_decode_fn(preds_np[k])).astype(np.uint8)
                else:
                    # fallback grayscale if no decoder is available
                    gg = (labels_np[k].astype(np.uint8) % 20) * 12
                    pp = (preds_np[k].astype(np.uint8)  % 20) * 12
                    gt_color   = np.stack([gg]*3, axis=2)
                    pred_color = np.stack([pp]*3, axis=2)

                # 3) filename base
                base = None
                try:
                    if isinstance(names, (list, tuple)) and k < len(names):
                        base = names[k]
                except Exception:
                    pass
                if not base:
                    base = f"{tag}_{i:06d}_{k}"

                # 4) save PNGs (only if requested)
                if getattr(opts, "save_val_results", False):
                    suffix = "_10c" if is_10c and getattr(opts, "separate_10c_dirs", False) else ""
                    tag_sfx = (f"_{opts.out_tag}" if getattr(opts, "out_tag", "") else "")
                    Image.fromarray(img_uint8).save(os.path.join(save_dir, f"{base}{suffix}{tag_sfx}_image.png"))
                    Image.fromarray(gt_color).save(os.path.join(save_dir, f"{base}{suffix}{tag_sfx}_gt.png"))
                    Image.fromarray(pred_color).save(os.path.join(save_dir, f"{base}{suffix}{tag_sfx}_pred19.png"))
                
                

                # 5) live Visdom triptych (limit to max_vis per validate)
                if vis is not None and shown < max_vis:
                    triptych = np.concatenate([img_uint8, gt_color, pred_color], axis=1)  # H x (3W) x 3
                    vis.vis_image(f"[{tag}] {base}", triptych.transpose(2, 0, 1))  # CHW
                    shown += 1            
            
            
            # Apply 10-class remapping if needed
            if getattr(opts, "eval_mode", "19") == "10":
                preds, targets = remap_to_10c(preds, targets, opts._remap_vec)
                # Optional: also save 10-class-colored predictions
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
            
            # Head accuracy calculation
            if weather_ids is not None:
                ws_preds = weather_preds.argmax(1).cpu()
                ws_correct += (ws_preds == weather_ids.cpu()).sum().item()
                ws_total += weather_ids.numel()
                
            if time_ids is not None:
                tes_preds = time_preds.argmax(1).cpu()
                tes_correct += (tes_preds == time_ids.cpu()).sum().item()
                tes_total += time_ids.numel()
            
            # Per-condition metrics
            if per_cond and weather_ids is not None:
                w_np = weather_ids.cpu().numpy()
                for cname, cid in zip(["rain", "fog", "snow", "night"], [0, 1, 2, 3]):
                    mask = (w_np == cid)
                    if mask.any():
                        cond_metrics[cname].update(targets[mask], preds[mask])
            
            if per_cond and time_ids is not None:
                t_np = time_ids.cpu().numpy()
                for tname, tid in zip(["day", "night"], [0, 1]):
                    mask = (t_np == tid)
                    if mask.any():
                        time_metrics[tname].update(targets[mask], preds[mask])
            
            # ... existing visualization code

        score = metrics_local.get_results()
        
    # Prepare extra metrics
    extra = {}
    if not (getattr(opts, "disable_ws_tas", False)):
        if ws_total > 0:
            extra["WS Acc"] = 100.0 * ws_correct / ws_total
        if tes_total > 0:
            extra["TES Acc"] = 100.0 * tes_correct / tes_total
        
    if per_cond:
        extra["cond_mIoU"] = {k: cond_metrics[k].get_results()["Mean IoU"] for k in cond_metrics}
        extra["time_mIoU"] = {k: time_metrics[k].get_results()["Mean IoU"] for k in time_metrics}
    
    # === ADD CONDITION PROOF TO OUTPUT ===
    print(f"[VALIDATION PROOF] Condition distribution: {condition_counts}")
    extra["condition_counts"] = condition_counts
    # ====================================
    
    return score, ret_samples, extra

# --- quick full-res preview from a dataloader (no random crop) ---
def preview_fullres(model, loader, device, denorm, vis, out_dir, tag, max_vis=4):
    import os
    from PIL import Image
    import numpy as np
    from torch.utils.data import Subset, ConcatDataset

    # unwrap to reach the base dataset's decode_target
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

            imgs_np = images.detach().cpu().numpy()   # NCHW
            gts_np  = labels.detach().cpu().numpy()   # NHW
            B = images.size(0)

            for k in range(B):
                img_u8 = (denorm(imgs_np[k]) * 255).transpose(1,2,0).astype(np.uint8)
                if decode is not None:
                    gt_color   = np.asarray(decode(gts_np[k])).astype(np.uint8)
                    pred_color = np.asarray(decode(preds[k])).astype(np.uint8)
                else:  # fallback coloring
                    gg = (gts_np[k].astype(np.uint8) % 20) * 12
                    pp = (preds[k].astype(np.uint8)  % 20) * 12
                    gt_color   = np.stack([gg]*3, axis=2)
                    pred_color = np.stack([pp]*3, axis=2)

                base = (str(names[k]) if isinstance(names, (list, tuple)) else f"{tag}_{shown:03d}").split(".")[0]
                Image.fromarray(img_u8).save(os.path.join(out_dir, f"{base}_image.png"))
                Image.fromarray(gt_color).save(os.path.join(out_dir, f"{base}_gt.png"))
                Image.fromarray(pred_color).save(os.path.join(out_dir, f"{base}_pred.png"))

                if vis is not None:
                    trip = np.concatenate([img_u8, gt_color, pred_color], axis=1)  # H x 3W x 3
                    vis.vis_image(f"[PREVIEW]{tag} {base}", trip.transpose(2,0,1))  # CHW

                shown += 1
                if shown >= max_vis:
                    break
            if shown >= max_vis:
                break
    model.train()
    
def apply_mode_presets(opts, _parser_defaults):
    """
    Apply MODE presets. This sets typical flags for each mode.
    You can still override with CLI flags if needed.
    """
    if opts.mode == 0:
        # train on Cityscapes + ACDC (alternating)

        opts.test_only = False
        opts.save_val_results = False
        opts.dataset = 'cityscapes_ACDC'


    elif opts.mode == 1:
        # train on Cityscapes only (ablation)
        opts.test_only = False
        opts.save_val_results = False
        opts.dataset = 'cityscapes'

    elif opts.mode == 11:
        # test on Cityscapes (10-class eval)
        opts.test_only = True
        opts.save_val_results = True
        opts.dataset = 'cityscapes'
        if opts.ckpt is None:  # Only set if not provided via CLI
            opts.ckpt = "checkpoints/best_mask2former_cityscapes_ACDC_os16.pth"
            #opts.ckpt = "checkpoints/best_mask2former_cityscapes_os16.pth"
        #opts.eval_mode = "10"
        # Only set 10-class helpers if the user actually asked for eval_mode=10
        if getattr(opts, "eval_mode", "19") == "10":
            if not getattr(opts, "labelmap", None):
                opts.labelmap = "configs/cityscapes_10class.json"
            opts.save_color_10c = True

    elif opts.mode == 21:
        # test on ACDC (10-class eval)
        opts.test_only = True
        opts.save_val_results = True
        opts.dataset = 'ACDC'
        if opts.ckpt is None:  # Only set if not provided via CLI
            opts.ckpt = "checkpoints/best_mask2former_cityscapes_ACDC_os16.pth"
            #opts.ckpt = "checkpoints/best_mask2former_cityscapes_os16.pth"
            # Honor CLI: only set 10-class defaults if the user chose it
        if getattr(opts, "eval_mode", "19") == "10":
            if not getattr(opts, "labelmap", None):
                opts.labelmap = "configs/cityscapes_10class.json"
            opts.save_color_10c = True
        #opts.eval_mode = "10"
        #opts.labelmap = "configs/cityscapes_10class.json"
        #opts.save_color_10c = True

    # Common niceties for test modes
    if opts.test_only:
        # evaluation-friendly defaults (only if user didn't override on CLI)
        if opts.batch_size == _parser_defaults.batch_size:
            opts.batch_size = 1
        if opts.test_batch_size == _parser_defaults.test_batch_size:
            opts.test_batch_size = 1
        if opts.val_batch_size == _parser_defaults.val_batch_size:
            opts.val_batch_size = 1
    
    return opts

def make_infinite(loader):
    """Safe infinite iterator that doesn't cache batches in memory"""
    while True:
        for batch in loader:
            yield batch
    
def main(ACDC_test_class=None, n_itrs=90000):
    opts = get_argparser().parse_args()
    _parser_defaults = get_argparser().parse_args([])
    # keep the CLI value; only override if a function arg was explicitly provided
    if ACDC_test_class is not None:
        opts.ACDC_test_class = ACDC_test_class
    print(f"[DEBUG] Using ACDC_test_class={opts.ACDC_test_class}")
    opts.finetune = False
    opts.pretrained_model = None

    opts.data_root_cs = os.environ.get("CS_ROOT",  "/home/ubuntu22user2/shiv/datasets/cityscapes")
    opts.data_root_acdc = os.environ.get("ACDC_ROOT","/home/ubuntu22user2/shiv/datasets/ACDC")
    opts.total_itrs = n_itrs
    opts.test_class = None
    #opts.total_itrs = 90000 # Short run
    #opts.val_batch_size = 1
    opts.val_interval = 5000 # ← Change from 100 to 500
    # If requested, hard-disable WAS/TAS (true ablation)
    if getattr(opts, "disable_ws_tas", False):
        opts.ws_weight = 0.0
        opts.tes_weight = 0.0
    opts = apply_mode_presets(opts, _parser_defaults)
    print(f"[ABLATION] ws_weight={opts.ws_weight} | tes_weight={opts.tes_weight} "
          f"| seg_scale_ACDC=1.2 | seg_scale_CS=1.0")
    utils.mkdir('runs')
    with open('runs/last_config.txt', 'w') as f:
        for k, v in vars(opts).items():
            f.write(f"{k}: {v}\n")
    print("[cfg] saved runs/last_config.txt")
    # Switch model name from DeepLab to Mask2Former
    opts.model = "mask2former"
    #opts.enable_vis = True
    #opts.vis_port = 8097
    #opts.gpu_id = '0'
    #opts.lr = 0.1
    #opts.lr = 2e-05 
    #opts.crop_size = 768
    #opts.lr = 2e-05 
    #opts.crop_size = 640 
    #opts.batch_size = 1 
    opts.output_stride = 16
    opts.crop_val = True

    # ---------------- Dataset-based num_classes and Denormalization ----------------
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    elif opts.dataset.lower() == 'acdc':
        opts.num_classes = 19
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    elif opts.dataset.lower() == 'awss':
        opts.num_classes = 19
        denorm = utils.Denormalize(mean=[0.1987, 0.1846, 0.1884],
                                std=[0.1084, 0.0950, 0.0902])
    else:
        opts.num_classes = 19
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])                                             

    # ---------------- Visualization Setup ----------------
    vis = Visualizer(port=opts.vis_port, env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:
        vis.vis_table("Options", vars(opts))

    # ---- GPU setup (single GPU quick fix) ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # --------------------- Random Seed Setup ---------------------
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Adjust val batch size for VOC when crop_val is off
    if opts.dataset == 'voc' and not opts.crop_val and opts.val_batch_size == _parser_defaults.val_batch_size:
        opts.val_batch_size = 1

    # --------------------- Dataset & DataLoader Setup ---------------------
    if opts.test_only:
        # Only test set required
        _, _, tst_dst = get_dataset(opts)

        test_loader = data.DataLoader(
            tst_dst,
            batch_size=opts.test_batch_size,
            shuffle=False,
            num_workers=0
        )
        print(f"Dataset: {opts.dataset}, Test set size: {len(tst_dst)}")

    else:
        if opts.mode == 0:
            # CS + ACDC training, CS & ACDC validation

            # --- Cityscapes via your helper (85/15 on official train) ---
            train_dst_cs, val_dst_cs, _ = get_dataset(opts, 'cityscapes')
            train_loader_cs = data.DataLoader(
                train_dst_cs, batch_size=opts.batch_size, shuffle=True,
                num_workers=4, pin_memory=True, drop_last=True
            )
            val_loader_cs = data.DataLoader(
                val_dst_cs, batch_size=opts.val_batch_size, shuffle=False,
                num_workers=2, pin_memory=True
            )

            # --- ACDC: use the new class directly (85/15 from ACDC/train) ---
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
            # Tiny ACDC quick-val slice (optional)
            quick_val_acdc = None
            if getattr(opts, "per_condition_val", False):
                from torch.utils.data import Subset
                idxs = list(range(min(24, len(val_loader_acdc.dataset))))
                quick_val_acdc = data.DataLoader(
                    Subset(val_loader_acdc.dataset, idxs),
                    batch_size=1, shuffle=False, num_workers=0
                )

        elif opts.mode == 1:
            # Cityscapes-only training & validation
            train_dst_cs, val_dst_cs, _ = get_dataset(opts, 'cityscapes')
            train_loader_cs = data.DataLoader(
                train_dst_cs, batch_size=opts.batch_size,
                shuffle=True, num_workers=0, drop_last=True
            )
            val_loader_cs = data.DataLoader(
                val_dst_cs, batch_size=opts.val_batch_size,
                shuffle=False, num_workers=0
            )

        print("CS Train set (used for training):", len(train_loader_cs.dataset))
        if opts.mode == 0:
            print("ACDC Train85 (used for training):", len(train_loader_acdc.dataset))
        print("CS Val set (used during training):", len(val_loader_cs.dataset))
        if opts.mode == 0:
            print("ACDC Val15 (used during training):", len(val_loader_acdc.dataset))

    # --------------------- Model Setup ---------------------
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes)
    print("DEBUG: Using model:", type(model))

    # move first (fixes lazy-init quirks; ensures params live on primary GPU)
    model = model.to(device)

    # Set batchnorm momentum (works for single + DP)
    if isinstance(model, nn.DataParallel):
        if hasattr(model.module, 'backbone'):
            utils.set_bn_momentum(model.module.backbone, momentum=0.01)
    else:
        if hasattr(model, 'backbone'):
            utils.set_bn_momentum(model.backbone, momentum=0.01)


    # === Full-resolution previews (use val loaders so there is no random crop) ===
    if opts.enable_vis:
        if opts.mode == 0:
            # You already created val_loader_cs and val_loader_acdc earlier
            preview_fullres(model, val_loader_cs,   device, denorm, vis,
                            out_dir="results/preview_cs",   tag="CS",   max_vis=4)
            preview_fullres(model, val_loader_acdc, device, denorm, vis,
                            out_dir="results/preview_acdc", tag="ACDC", max_vis=4)
        elif opts.mode == 1:
            preview_fullres(model, val_loader_cs,   device, denorm, vis,
                            out_dir="results/preview_cs",   tag="CS",   max_vis=4)

    # --------------------- Metric Setup ---------------------
    metrics = StreamSegMetrics(opts.num_classes)

    # --------------------- Optimizer Setup ---------------------
    import itertools
    # --------------------- Optimizer Setup (backbone vs heads) ---------------------
    base_lr = opts.lr          # e.g., 2e-5
    head_lr = base_lr * 5      # decoder+heads get 5x; tune 3x–10x later if needed

    def collect_backbone_param_ids(model: nn.Module) -> set:
        ids = set()

        # 1) obvious attributes if present
        for attr in ["backbone", "encoder", "trunk", "body", "feature_extractor"]:
            mod = getattr(model, attr, None)
            if isinstance(mod, nn.Module):
                for p in mod.parameters():
                    ids.add(id(p))

        # 2) class-name sweep (covers Swin/ResNet/etc. even if nested)
        BACKBONE_CLASS_HINTS = ("backbone", "resnet", "swin", "convnext", "hrnet")
        for name, module in model.named_modules():
            cls = module.__class__.__name__.lower()
            if any(h in cls for h in BACKBONE_CLASS_HINTS):
                for p in module.parameters(recurse=False):
                    ids.add(id(p))
                # also add children’s params
                for ch in module.children():
                    for p in ch.parameters(recurse=True):
                        ids.add(id(p))

        return ids

    backbone_param_ids = collect_backbone_param_ids(model)

    groups = [
        {"params": [], "weight_decay": 0.05, "lr": base_lr},  # 0: backbone (decay)
        {"params": [], "weight_decay": 0.00, "lr": base_lr},  # 1: backbone (no-decay)
        {"params": [], "weight_decay": 0.05, "lr": head_lr},  # 2: decoder+heads (decay)
        {"params": [], "weight_decay": 0.00, "lr": head_lr},  # 3: decoder+heads (no-decay)
    ]

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_backbone = (id(p) in backbone_param_ids)
        # treat norms/bias/positional emb as no-decay
        is_no_decay = (
            n.endswith(".bias")
            or any(k in n.lower() for k in ["norm", "bn", "bias", "layernorm", "ln"])
            or "pos_embed" in n.lower() or "absolute_pos_embed" in n.lower()
        )
        idx = (0 if  is_backbone and not is_no_decay else
               1 if  is_backbone and     is_no_decay else
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
          
    # Get all parameters that require gradients, separate norm/bias from others
    #decay, no_decay = [], []
    #for n, p in model.named_parameters():
    #    if p.requires_grad:
    #        if n.endswith(".bias") or any(nm in n.lower() for nm in ["norm", "bn", "bias"]):
    #            no_decay.append(p)
    #        else:
    #            decay.append(p)

    # Use AdamW with better settings for transformers
    #optimizer = torch.optim.AdamW(
    #    [{"params": decay, "weight_decay": 0.05},
    #     {"params": no_decay, "weight_decay": 0.0}],
    #    lr=opts.lr,
    #    betas=(0.9, 0.999),
    #    eps=1e-8
    #)

    #print(f"Using AdamW optimizer with {len(decay)} decay params, {len(no_decay)} no-decay params")
    #print(f"Learning rate: {opts.lr}, Weight decay: 0.05 (for decay params only)")


    # --------------------- Learning Rate Scheduler ---------------------
    # Dynamic accumulation steps: 1 for mode 0 (batch alternation), 8 for mode 1
    accumulation_steps = 1 if opts.mode == 0 else 8
    warmup_steps = 1000

    # Total optimizer steps (not iterations)
    total_steps = math.ceil(opts.total_itrs / accumulation_steps)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
    
    # --------------------- Loss Function Setup ---------------------
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    else:
        raise NotImplementedError(f"Unsupported loss type: {opts.loss_type}")

    # --------------------- Checkpoint Save Function ---------------------
    def save_ckpt(path):
        """Save current model checkpoint."""
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
        print(f"Model saved at {path}")

    # >>> ADD THIS HELPER RIGHT HERE (same indent as save_ckpt) <<<
    def set_encoder_trainable(model, trainable: bool):
        if hasattr(model, "backbone"):
            for p in model.backbone.parameters():
                p.requires_grad_(trainable)
        else:
            # Fallback: freeze anything with 'backbone' in the name
            for name, p in model.named_parameters():
                if 'backbone' in name:
                    p.requires_grad_(trainable)

    utils.mkdir('checkpoints')  # create directory if not exists
    
    # -------- freeze only low-level encoder blocks (works across ResNet/Swin) --------
    LOW_LVL_KEYS = [
        "backbone.low_level", "backbone.low_level_features",   # ResNet-style
        "backbone.stem", "backbone.layer1",                    # ResNet-style
        "backbone.patch_embed", "backbone.layers.0", "backbone.stages.0"  # Swin/Transformer early stage
    ]

    def set_low_level_trainable(model, trainable: bool, verbose: bool = False):
        hits = 0
        for n, p in model.named_parameters():
            if any(k in n for k in LOW_LVL_KEYS):
                p.requires_grad_(trainable)
                hits += 1
        if verbose:
            print(f"[freeze] low-level {'TRAINABLE' if trainable else 'FROZEN'} | matched={hits}")

    # Lock BN running stats (safe no-op if your backbone uses LayerNorm)
    def _bn_eval(m: nn.Module):
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()
            
    def grad_norm_named(model, name_substrs):
        total = 0.0
        for n, p in model.named_parameters():
            if any(s in n for s in name_substrs) and p.grad is not None:
                total += p.grad.detach().pow(2).sum().item()
        return total ** 0.5
    # --------------------- Checkpoint Restore / Resume ---------------------
    best_score_cs = 0.0
    best_score_acdc = 0.0
    cur_itrs = 0
    cur_epochs = 0

    if not opts.finetune:
        if opts.ckpt is not None and os.path.isfile(opts.ckpt):
            print(f"Restoring model from checkpoint: {opts.ckpt}")
            checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
            #model.load_state_dict(checkpoint["model_state"])
            model.load_state_dict(checkpoint["model_state"], strict=False)
            model = model.to(device)

            #model = nn.DataParallel(model)
            #model.to(device)

            if opts.continue_training:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                scheduler.load_state_dict(checkpoint["scheduler_state"])
                cur_itrs = checkpoint.get("cur_itrs", 0)
                best_score_cs = checkpoint.get("best_score_cs", 0.0)
                best_score_acdc = checkpoint.get("best_score_acdc", 0.0)
                print(f"Resumed training from iteration {cur_itrs}")

            del checkpoint  # free memory
        else:
            print("[!] Starting fresh training.")
            model = model.to(device)
            #model = nn.DataParallel(model)
            #model.to(device)

    else:  # opts.finetune
        print(f"Fine-tuning from pretrained model: {opts.pretrained_model}")
        checkpoint = torch.load(opts.pretrained_model, map_location=torch.device('cpu'))
        #model.load_state_dict(checkpoint["model_state"])
        model.load_state_dict(checkpoint["model_state"], strict=False)
        #model = nn.DataParallel(model)
        #model.to(device)
        model = model.to(device)


        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        cur_itrs = checkpoint.get("cur_itrs", 0)
        best_score_cs = checkpoint.get("best_score_cs", 0.0)
        best_score_acdc = checkpoint.get("best_score_acdc", 0.0)

        # Optional: freeze encoder layers if needed
        for name, param in model.named_parameters():
            if not name.startswith("module.classifier"):
                param.requires_grad = False

    # ==========   Train Loop   ========== #

    if opts.test_only:  # Testing only
        vis_sample_id = None
        # Uncomment below if you want per-mode visualization
        # if MODE == 11:
        #     vis_sample_id = vis_sample_id_cs
        # elif MODE == 21:
        #     vis_sample_id = vis_sample_id_acdc

        print("[!] testing")
        model.eval()
        test_score, ret_samples, extra = validate(
            opts=opts, model=model, loader=test_loader, device=device, metrics=metrics,
            ret_samples_ids=vis_sample_id,
            vis=vis, denorm=denorm,
            save_dir=f"results/test_{opts.dataset.lower()}",
            max_vis=opts.vis_num_samples, tag=opts.dataset.upper()
        )
        # Use correct metrics printer based on eval mode
        # Use correct metrics printer based on eval mode
        is_10c = (getattr(opts, "eval_mode", "19") == "10") and hasattr(opts, "_metrics_10")

        if is_10c:
            # 10-class: pretty print and list IoUs for all 10 classes (0..9)
            print(opts._metrics_10.to_str(test_score))
            class_iou = test_score.get('Class IoU', {})
            IoU_scores = np.array([class_iou.get(i, 0.0) for i in range(10)])
            print("IoU (10-class):", IoU_scores)
        else:
            # 19-class: pretty print and list IoUs for all 19 classes (0..18)
            print(metrics.to_str(test_score))
            class_iou = test_score.get('Class IoU', {})
            # Handle dict or list
            if isinstance(class_iou, dict):
                IoU_scores = np.array([class_iou.get(i, 0.0) for i in range(getattr(metrics, "n_classes", 19))])
            else:
                IoU_scores = np.array(class_iou)  # already a list/array of length 19
            print("IoU (19-class):", IoU_scores)

    #else:  # Training setup
        #vis_sample_id_cs = np.random.randint(
            #0, len(val_loader_cs), opts.vis_num_samples, np.int32) if opts.enable_vis else None

        #vis_sample_id_acdc = np.random.randint(
            #0, len(val_loader_acdc), opts.vis_num_samples, np.int32) if opts.enable_vis else None
    # --------------------- Train setup ---------------------
    if not opts.test_only:
        if opts.mode == 0:
            # CS + AWSS: we have CS and ACDC validation
            vis_sample_id_cs = (
                np.random.randint(0, len(val_loader_cs), opts.vis_num_samples, np.int32)
                if opts.enable_vis else None
            )
            vis_sample_id_acdc = (
                np.random.randint(0, len(val_loader_acdc), opts.vis_num_samples, np.int32)
                if opts.enable_vis else None
            )
        elif opts.mode == 1:

            # Cityscapes-only: only CS validation
            vis_sample_id_cs = (
                np.random.randint(0, len(val_loader_cs), opts.vis_num_samples, np.int32)
                if opts.enable_vis else None
            )
            vis_sample_id_acdc = None
    # Initialize GradScaler for AMP
    scaler = GradScaler()
    interval_loss = 0
    # alternation debug
    cs_seen = 0
    acdc_seen = 0

    # --- helpers for color decode even if Subset/ConcatDataset ---
    from torch.utils.data import Subset, ConcatDataset
    def _unwrap_base_dataset(ds):
        while isinstance(ds, (Subset, ConcatDataset)):
            ds = ds.dataset if isinstance(ds, Subset) else ds.datasets[0]
        return ds

    _base_cs   = _unwrap_base_dataset(train_loader_cs.dataset)   if opts.mode in (0,1) else None
    _base_acdc = _unwrap_base_dataset(train_loader_acdc.dataset) if opts.mode == 0 else None
    _cs_decode   = getattr(_base_cs,   "decode_target", None) if _base_cs   is not None else None
    _acdc_decode = getattr(_base_acdc, "decode_target", None) if _base_acdc is not None else None

    os.makedirs("results/train_cs", exist_ok=True)
    os.makedirs("results/train_acdc", exist_ok=True)

    # show only a few triptychs at the very start so we can sanity-check quickly
    warm_vis_budget = 8      # total images to push/save at start
    warm_vis_done   = 0
        
    #while True:  # Training loop: optionally set to `while cur_itrs < opts.total_itrs`
    while cur_itrs < opts.total_itrs:
        model.train()
        cur_epochs += 1
        if opts.mode == 0:
            # -------- BATCH-WISE ALTERNATION (ACDC <-> Cityscapes) --------
            # Safe infinite iterators that don't cache batches in memory
            iter_cs   = make_infinite(train_loader_cs)
            iter_acdc = make_infinite(train_loader_acdc)

            # One "epoch" = as many steps as the longer loader
            max_len = max(len(train_loader_cs), len(train_loader_acdc))

            for step in range(max_len):
                use_acdc = (step % 2 == 0)
                t0 = time.time()
                
                if (cur_itrs < 200) or (cur_itrs % 500 == 0):  # longer window + periodic check
                    print(f"[proof] itrs={cur_itrs} step={step} use_acdc={use_acdc} domain={'ACDC' if use_acdc else 'Cityscapes'}")

                if use_acdc:
                    images, labels, names, weather_ids, time_ids, _ = next(iter_acdc)
                    # ACDC: everything learns
                    set_encoder_trainable(model, True)
                    set_low_level_trainable(model, True)
                    domain_name = "ACDC"
                else:
                    images, labels, names, weather_ids, time_ids, _ = next(iter_cs)


                    # Cityscapes: keep higher layers learning, freeze only low-level
                    set_encoder_trainable(model, True)         # leave backbone ON
                    set_low_level_trainable(model, False)      # but lock early features
                    if hasattr(model, "backbone"):
                        model.backbone.apply(_bn_eval)         # stop BN stat drift on CS
                    domain_name = "Cityscapes"
                    
                # ---- Sanity: domain + label invariants ----
                if (cur_itrs < 200) or (cur_itrs % 1000 == 0):
                    print(f"[batch] itr={cur_itrs} | domain={domain_name} | B={images.size(0)} "
                          f"| HxW={images.size(-2)}x{images.size(-1)}")

                # Cityscapes should always be CLEAR/DAY
                if domain_name == "Cityscapes":
                    u_ws = weather_ids.unique()
                    u_ts = time_ids.unique()
                    if not (u_ws.numel() == 1 and int(u_ws.item()) == CLEAR_ID):
                        print(f"[WARN] CS weather_ids not CLEAR({CLEAR_ID}): uniques={u_ws.tolist()}")
                    if not (u_ts.numel() == 1 and int(u_ts.item()) == DAY_ID):
                        print(f"[WARN] CS time_ids not DAY({DAY_ID}): uniques={u_ts.tolist()}")
                # --- update debug counters (optional) ---
                if use_acdc:
                    acdc_seen += 1
                else:
                    cs_seen += 1    

                cur_itrs += 1
                #images = images.to(device, dtype=torch.float32)
                dev = next(model.parameters()).device
                images = images.to(dev, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)
                #weather_ids = weather_ids.to(device)
                #time_ids = time_ids.to(device)
                weather_ids = weather_ids.to(device, dtype=torch.long)
                time_ids    = time_ids.to(device, dtype=torch.long)

                optimizer.zero_grad(set_to_none=True)

                # ----- forward -----
                with autocast():
                    seg_logits, weather_preds, time_preds = model(images)

                    # Make TES completely inert when its weight is zero
                    if opts.tes_weight == 0:
                        time_preds = time_preds.detach()

                    # Original full-disable guard remains
                    if getattr(opts, "disable_ws_tas", False) or (opts.ws_weight == 0 and opts.tes_weight == 0):
                        weather_preds = weather_preds.detach()
                        time_preds    = time_preds.detach()
                    
                    seg_logits = nn.functional.interpolate(
                        seg_logits, size=labels.shape[-2:], mode='bilinear', align_corners=False
                    )
                    loss_seg = criterion(seg_logits, labels)

                    # Heads: compute loss only if ACDC AND any weight > 0
                    is_acdc = (domain_name == "ACDC")

                    # Heads: compute on **every** batch when weights > 0 (DeepLab-style)
                    use_heads = (opts.ws_weight > 0 or opts.tes_weight > 0) and not getattr(opts, "disable_ws_tas", False)
                    if use_heads:
                        loss_ws_ce  = (criterion(weather_preds, weather_ids)
                                       if opts.ws_weight > 0 else torch.tensor(0.0, device=images.device))
                        loss_tes_ce = (criterion(time_preds,   time_ids)
                                       if opts.tes_weight > 0 else torch.tensor(0.0, device=images.device))
                        loss_weather = opts.ws_weight  * loss_ws_ce
                        loss_time    = opts.tes_weight * loss_tes_ce
                    else:
                        loss_ws_ce = torch.tensor(0.0, device=images.device)
                        loss_tes_ce = torch.tensor(0.0, device=images.device)
                        loss_weather = 0.0
                        loss_time    = 0.0

                    # small nudge to ACDC seg to close the domain gap faster
                    seg_scale = 1.2 if is_acdc else 1.0
                    total_loss = seg_scale * loss_seg + loss_weather + loss_time
                    # ---- NaN/Inf guards ----
                    if torch.isnan(total_loss) or torch.isinf(total_loss):
                        print(f"[FATAL] NaN/Inf in total_loss at itr={cur_itrs} "
                              f"| seg={float(loss_seg):.4f} ws={float(loss_ws_ce):.4f} tes={float(loss_tes_ce):.4f}")
                        save_ckpt('checkpoints/abort_nan.pth')
                        raise RuntimeError("NaN/Inf detected")
                # --- quick warm-up visualization at training start ---
                if warm_vis_done < warm_vis_budget:
                    with torch.no_grad():
                        preds_vis = seg_logits.detach().argmax(1).cpu().numpy()
                        imgs_np   = images.detach().cpu().numpy()
                        gts_np    = labels.detach().cpu().numpy()
                        decoder   = _acdc_decode if is_acdc else _cs_decode
                        out_dir   = "results/train_acdc" if is_acdc else "results/train_cs"

                        B = images.size(0)
                        for k in range(min(B, warm_vis_budget - warm_vis_done)):
                            # de-normalize input
                            if denorm is not None:
                                img_u8 = (denorm(imgs_np[k]) * 255).transpose(1, 2, 0).astype(np.uint8)
                            else:
                                x = imgs_np[k].transpose(1, 2, 0)
                                img_u8 = np.clip(x * 255.0, 0, 255).astype(np.uint8)

                            # colorize GT & pred in 19-class trainId space
                            if decoder is not None:
                                gt_color   = np.asarray(decoder(gts_np[k])).astype(np.uint8)
                                pred_color = np.asarray(decoder(preds_vis[k])).astype(np.uint8)
                            else:
                                gg = (gts_np[k].astype(np.uint8) % 20) * 12
                                pp = (preds_vis[k].astype(np.uint8) % 20) * 12
                                gt_color   = np.stack([gg]*3, axis=2)
                                pred_color = np.stack([pp]*3, axis=2)

                            # base name (prefer dataset filename if available)
                            base = None
                            if isinstance(names, (list, tuple)) and k < len(names):
                                base = str(names[k]).rsplit(".", 1)[0]
                            if not base:
                                base = f"{'ACDC' if is_acdc else 'CS'}_{cur_itrs:06d}_{k}"

                            # save individual PNGs
                            Image.fromarray(img_u8).save(os.path.join(out_dir, f"{base}_image.png"))
                            Image.fromarray(gt_color).save(os.path.join(out_dir, f"{base}_gt.png"))
                            Image.fromarray(pred_color).save(os.path.join(out_dir, f"{base}_pred.png"))

                            # live triptych to Visdom (if enabled)
                            if vis is not None:
                                trip = np.concatenate([img_u8, gt_color, pred_color], axis=1)  # H x 3W x 3
                                vis.vis_image(f"[TRAIN]{'ACDC' if is_acdc else 'CS'} {base}",
                                              trip.transpose(2, 0, 1))  # CHW
                            warm_vis_done += 1
                            if warm_vis_done >= warm_vis_budget:
                                break

                # simple top-1 accuracies for the heads
                # simple top-1 accuracies for the heads (only compute on ACDC)
                if is_acdc:
                    ws_acc  = (weather_preds.argmax(1) == weather_ids).float().mean().item()
                    tes_acc = (time_preds.argmax(1)    == time_ids).float().mean().item()
                else:
                    ws_acc, tes_acc = None, None
                # --- ACDC heartbeat print so you see WS/TES on ACDC too ---
                if is_acdc and (cur_itrs % opts.log_heads_every == (opts.log_heads_every // 2)):
                    print(f"[ACDC] itr={cur_itrs} | seg={float(loss_seg):.4f} "
                          f"| WS_CE={float(loss_ws_ce):.4f} ({ws_acc*100:.1f}%) "
                          f"| TES_CE={float(loss_tes_ce):.4f} ({tes_acc*100:.1f}%) "
                          f"| B={images.size(0)} HxW={images.size(-2)}x{images.size(-1)}")                    

                # ----- backward/step -----
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # (optional) check grad norms to prove freeze works
                # (optional) check grad norms to prove freeze works (targeted)
                if opts.check_backbone_grads and (cur_itrs % opts.log_heads_every == 0):
                    gn_backbone = grad_norm_named(model, ["backbone", "encoder", "pixel_level_module.encoder"])
                    gn_early = grad_norm_named(model, ["patch_embed", "layers.0", "stages.0"])
                    gn_late  = grad_norm_named(model, ["layers.3", "stages.3"])
                    print(f"[grads] itr={cur_itrs} dom={domain_name} | enc={gn_backbone:.4e} "
                          f"| early={gn_early:.4e} late={gn_late:.4e}")

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                # LR snapshot (sparse)
                if cur_itrs % 1000 == 0:
                    lrs = [pg['lr'] for pg in optimizer.param_groups]
                    print(f"[lr] itr={cur_itrs} | backbone(decay/no)={lrs[0]:.2e}/{lrs[1]:.2e} "
                          f"| heads(decay/no)={lrs[2]:.2e}/{lrs[3]:.2e}")

                # Speed / throughput
                if cur_itrs % 200 == 0 and cur_itrs > 0:
                    dt = time.time() - t0
                    print(f"[speed] itr={cur_itrs} | {dt*1000:.1f} ms/iter | {images.size(0)/dt:.1f} img/s")

                # Seg-logit magnitude (sparse)
                if cur_itrs % 1000 == 0:
                    with torch.no_grad():
                        smax = float(seg_logits.detach().abs().max())
                        print(f"[mag] itr={cur_itrs} | seg_logits|max|={smax:.2f}")
                # ----- logging -----
                np_loss = loss_seg.detach().cpu().numpy()
                interval_loss += np_loss
                if vis is not None:
                    vis.vis_scalar('Loss/seg', cur_itrs, np_loss)
                    vis.vis_scalar('Loss/ws_ce_raw',  cur_itrs, float(loss_ws_ce))
                    vis.vis_scalar('Loss/tes_ce_raw', cur_itrs, float(loss_tes_ce))
                    if is_acdc:
                        vis.vis_scalar('Acc/WS',  cur_itrs, ws_acc * 100.0)
                        vis.vis_scalar('Acc/TES', cur_itrs, tes_acc * 100.0)

                if (cur_itrs) % opts.log_heads_every == 0:
                    if getattr(opts, "disable_ws_tas", False) or (opts.ws_weight == 0 and opts.tes_weight == 0):
                        print(f"Epoch {cur_epochs}, Itrs {cur_itrs}/{opts.total_itrs}, "
                              f"Loss(seg)={np_loss:.4f}, Domain={domain_name} [WAS/TAS OFF]")
                    else:
                        # Always print the full info, but show actual values only on ACDC batches
                        if is_acdc:
                            # ACDC batch: show actual losses and accuracies
                            acc_ws_txt  = f"{ws_acc*100:.1f}%" if ws_acc is not None else "N/A"
                            acc_tes_txt = f"{tes_acc*100:.1f}%" if tes_acc is not None else "N/A"
                            print(f"Epoch {cur_epochs}, Itrs {cur_itrs}/{opts.total_itrs}, "
                                  f"Loss(seg)={np_loss:.4f}, "
                                  f"WS_CE={float(loss_ws_ce):.4f} ({acc_ws_txt}), "
                                  f"TES_CE={float(loss_tes_ce):.4f} ({acc_tes_txt}), "
                                  f"Domain={domain_name}")
                        else:
                            # Cityscapes batch: now also prints real aux losses/accs
                            acc_ws_txt  = f"{(weather_preds.argmax(1) == weather_ids).float().mean().item()*100:.1f}%"
                            acc_tes_txt = f"{(time_preds.argmax(1)    == time_ids).float().mean().item()*100:.1f}%"
                            print(f"Epoch {cur_epochs}, Itrs {cur_itrs}/{opts.total_itrs}, "
                                  f"Loss(seg)={np_loss:.4f}, "
                                  f"WS_CE={float(loss_ws_ce):.4f} ({acc_ws_txt}), "
                                  f"TES_CE={float(loss_tes_ce):.4f} ({acc_tes_txt}), "
                                  f"Domain={domain_name}")

                        # Heads should be ~100% on CS (labels constant)
                        if domain_name == "Cityscapes":
                            cs_ws_acc  = (weather_preds.argmax(1) == weather_ids).float().mean().item()*100.0
                            cs_tes_acc = (time_preds.argmax(1)    == time_ids).float().mean().item()*100.0
                            if cs_ws_acc < 99.0 or cs_tes_acc < 99.0:
                                print(f"[WARN] CS head acc dropped: WS={cs_ws_acc:.1f}% TES={cs_tes_acc:.1f}% "
                                      f"(labels constant; heads should be ~100%)")
                                  
                #scaler.scale(total_loss).backward()
                #scaler.unscale_(optimizer)
                #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                #scaler.step(optimizer)
                #scaler.update()
                #scheduler.step()

                if (cur_itrs) % 10 == 0:
                    interval_loss = interval_loss / 10
                    print(f"Epoch {cur_epochs}, Itrs {cur_itrs}/{opts.total_itrs}, "
                          f"Loss={interval_loss:.4f}, Domain={domain_name}")
                    interval_loss = 0.0
                # ---- tiny ACDC micro-validation (quick regression check) ----
                if 'quick_val_acdc' in locals() and (quick_val_acdc is not None) and (cur_itrs % 2000 == 0) and (cur_itrs > 0):
                    model.eval()
                    q_metrics = StreamSegMetrics(opts.num_classes)
                    q_metrics.reset()
                    with torch.no_grad():
                        for q_imgs, q_lbls, *_ in quick_val_acdc:
                            q_imgs = q_imgs.to(device, dtype=torch.float32)
                            q_lbls = q_lbls.to(device, dtype=torch.long)
                            q_logits, _, _ = model(q_imgs)
                            q_logits = nn.functional.interpolate(q_logits, size=q_lbls.shape[-2:], mode='bilinear', align_corners=False)
                            q_preds = q_logits.argmax(1).cpu().numpy()
                            q_metrics.update(q_lbls.cpu().numpy(), q_preds)
                    q_score = q_metrics.get_results()
                    print(f"[quick-val] itr={cur_itrs} | ACDC tiny mIoU={q_score['Mean IoU']:.3f}")
                    model.train()

                if (cur_itrs) % opts.val_interval == 0:
                    save_ckpt('checkpoints/latest_mask2former_WAS_TAS_ON_1.2.pth')
                    print("validation...")
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
                    
                    print(metrics.to_str(val_score_cs))
                    if "WS Acc" in extra_cs:
                        print(f"[CS] WS Acc={extra_cs['WS Acc']:.2f}%, TES Acc={extra_cs.get('TES Acc',0):.2f}%")

                        print(metrics.to_str(val_score_acdc))
                    if "WS Acc" in extra_acdc:
                        print(f"[ACDC] WS Acc={extra_acdc['WS Acc']:.2f}%, TES Acc={extra_acdc.get('TES Acc',0):.2f}%")
                    if "cond_mIoU" in extra_acdc:
                        print("[ACDC] Per-condition mIoU:", extra_acdc["cond_mIoU"])
                    if "time_mIoU" in extra_acdc:
                        print("[ACDC] Per-time mIoU:", extra_acdc["time_mIoU"])
                                            

                    if val_score_cs['Mean IoU'] > best_score_cs:
                        best_score_cs = val_score_cs['Mean IoU']
                        best_score_acdc = val_score_acdc['Mean IoU']
                        save_ckpt('checkpoints/best_mask2former_WAS_TAS_ON_1.2.pth')
                        print(f"[ckpt] NEW BEST | CS mIoU={best_score_cs:.3f} | ACDC mIoU={best_score_acdc:.3f}")


                    if vis is not None:
                        vis.vis_scalar("[Val] Mean IoU CS", cur_itrs, val_score_cs['Mean IoU'])
                        vis.vis_table("[Val] Class IoU CS", val_score_cs['Class IoU'])
                        vis.vis_scalar("[Val] Mean IoU ACDC", cur_itrs, val_score_acdc['Mean IoU'])
                        vis.vis_table("[Val] Class IoU ACDC", val_score_acdc['Class IoU'])

                    model.train()

                if cur_itrs >= opts.total_itrs:
                    return
        elif opts.mode == 1:
            # ------- Cityscapes-only simple loop (segmentation loss only) -------
            accumulation_steps = 8  # Effective batch size = 8
            
            for step, (images, labels, _, _, _, _) in enumerate(train_loader_cs):
                cur_itrs += 1
                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)

                # Only zero gradients at the start of accumulation window
                if step % accumulation_steps == 0:
                    optimizer.zero_grad(set_to_none=True)

                # === AMP: Mixed precision forward pass ===
                with autocast():
                    seg_logits, _, _ = model(images)
                    seg_logits = nn.functional.interpolate(
                        seg_logits, size=labels.shape[-2:], mode='bilinear', align_corners=False
                    )
                    # Scale loss by accumulation steps
                    loss_segmentation = criterion(seg_logits, labels) / accumulation_steps
                    total_loss = loss_segmentation

                # === AMP: Scaled backward pass ===
                scaler.scale(total_loss).backward()

                # Only update weights after accumulation steps
                if (step + 1) % accumulation_steps == 0:
                    # Gradient clipping for stability
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()  # Step scheduler on update, not every iteration

                np_loss = loss_segmentation.detach().cpu().numpy() * accumulation_steps
                interval_loss += np_loss
                if vis is not None:
                    vis.vis_scalar('Loss', cur_itrs, np_loss)

                if (cur_itrs) % 10 == 0:
                    interval_loss = interval_loss / 10
                    print("Epoch %d, Itrs %d/%d, Loss=%f" %
                          (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                    interval_loss = 0.0

                if (cur_itrs) % opts.val_interval == 0:
                    save_ckpt('checkpoints/latest_mask2former_WAS_TAS_ON_1.2.pth')
                    print("validation...")

                    model.eval()
                    val_score_cs, _, extra_cs = validate(
                        opts=opts, model=model, loader=val_loader_cs, device=device,
                        metrics=metrics, ret_samples_ids=None
                    )
                    print(metrics.to_str(val_score_cs))

                    # keep best score using the Cityscapes metric slot you already have
                    if val_score_cs['Mean IoU'] > best_score_cs:
                        best_score_cs = val_score_cs['Mean IoU']
                        save_ckpt('checkpoints/best_mask2former_WAS_TAS_ON_1.2.pth')

                    if vis is not None:
                        vis.vis_scalar("[Val] Mean IoU CS", cur_itrs, val_score_cs['Mean IoU'])
                        vis.vis_table("[Val] Class IoU CS", val_score_cs['Class IoU'])

                    model.train()

                if cur_itrs >= opts.total_itrs:
                    print("Training completed! Running final validation...")
                    model.eval()
                    final_score, _, _ = validate(
                        opts=opts, model=model, loader=val_loader_cs, 
                        device=device, metrics=metrics
                    )
                    print(f"FINAL RESULTS - mIoU: {final_score['Mean IoU']:.4f}")
                    return
               

if __name__ == '__main__':
    main()
