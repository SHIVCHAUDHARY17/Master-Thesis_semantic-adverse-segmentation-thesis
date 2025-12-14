# main_with_was_tas.py
# ------------------------------------------------------------
# Cityscapes + ACDC (19-class), NO AWSS
# Alternating batches:
#   - ACDC (even i)  -> UNFREEZE module.backbone.low_level_features.*
#   - CS   (odd  i)  -> FREEZE   module.backbone.low_level_features.*
# Multi-task: seg + tiny (weather/time) aux losses if model returns them
# Train:    CS(train85) + ACDC(train85)
# Val:      CS(val15-of-train) + ACDC(val15-of-train)
# Eval:     CS official val (mode=11), ACDC official val (mode=21; optional per-condition)
# Optim:    SGD(lr=0.01), PolyLR (power=0.9), weight_decay=1e-4
# Loss:     CrossEntropy(ignore=255)
# ------------------------------------------------------------

import os
import random
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils import data

import network
import utils
from utils import ext_transforms as et
from metrics import StreamSegMetrics
from utils.visualizer import Visualizer

# 19-class datasets you provided
from datasets.cityscapes_baseline import Cityscapes, build_cityscapes_train85_val15_datasets
from datasets.ACDC_baseline_19 import ACDC


# -----------------------
# Argparser
# -----------------------
def get_argparser():
    p = argparse.ArgumentParser()

    # Paths
    p.add_argument("--data_root_cs", type=str, default="/path/to/cityscapes")
    p.add_argument("--data_root_acdc", type=str, default="/path/to/ACDC")

    # Modes
    p.add_argument("--mode", type=int, default=0, choices=[0, 11, 21],
                   help="0=train (CS+ACDC alt-batches); 11=eval CS official val; 21=eval ACDC official val")
    p.add_argument("--ACDC_test_class", type=str, default=None,
                   choices=[None, "rain", "fog", "snow", "night"],
                   help="When --mode 21, optionally evaluate a single ACDC condition")

    # Model
    avail = sorted(n for n in network.modeling.__dict__
                   if n.islower() and not (n.startswith("__") or n.startswith("_"))
                   and callable(network.modeling.__dict__[n]))
    p.add_argument("--model", type=str, default="deeplabv3plus_mobilenet", choices=avail)
    p.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    p.add_argument("--separable_conv", action="store_true", default=False)

    # Train/Eval
    p.add_argument("--num_classes", type=int, default=19)
    p.add_argument("--total_itrs", type=int, default=int(30e3))
    p.add_argument("--val_interval", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--val_batch_size", type=int, default=8)
    p.add_argument("--test_batch_size", type=int, default=4)
    p.add_argument("--crop_size", type=int, default=768)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--lr_policy", type=str, default="poly", choices=["poly", "step"])
    p.add_argument("--step_size", type=int, default=10000)
    p.add_argument("--loss_type", type=str, default="cross_entropy",
                   choices=["cross_entropy", "focal_loss"])

    # Aux heads weight (print-friendly, default matches your code)
    p.add_argument("--aux_weight", type=float, default=1e-5,
                   help="Weight for tiny aux (weather/time) losses; set 0 to disable their contribution.")

    # Infra
    p.add_argument("--gpu_id", type=str, default="0")
    p.add_argument("--random_seed", type=int, default=1)
    p.add_argument("--print_interval", type=int, default=10)

    # Optional grad-norm logging (0 disables)
    p.add_argument("--log_grad_norm_every", type=int, default=0,
                   help="If >0, prints global grad-norm every N iterations (after backward, before step).")

    # Checkpoints (optional)
    p.add_argument("--ckpt", type=str, default=None,
                   help="Optional: load weights (evaluation or warm start)")
    p.add_argument("--continue_training", action="store_true", default=False)

    # Visdom
    p.add_argument("--enable_vis", action="store_true", default=False)
    p.add_argument("--vis_port", type=str, default="28300")
    p.add_argument("--vis_env", type=str, default="main")
    p.add_argument("--vis_num_samples", type=int, default=8)

    # Save validation visualizations
    p.add_argument("--save_val_results", action="store_true", default=False)

    return p


# -----------------------
# Transforms
# -----------------------
def _build_transforms(opts):
    train_tf = et.ExtCompose([
        et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
        et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tf = et.ExtCompose([
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


# -----------------------
# Validation
# -----------------------
def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    print(f"[val] BEGIN (loader={type(loader.dataset).__name__}, batches={len(loader)})")
    metrics.reset()
    ret_samples = []

    if opts.save_val_results:
        os.makedirs("results", exist_ok=True)
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img_id = 0

    base_ds = loader.dataset.dataset if hasattr(loader.dataset, "dataset") else loader.dataset

    with torch.no_grad():
        for i, (images, labels, names, weather_ids, time_ids, data_domain) in tqdm(enumerate(loader)):

            
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            out = model(images)
           
            # 🖨️ PRINT: Validation inputs
            print(f"\n=== VALIDATION BATCH {i} ===")
            print(f"Input - Images shape: {images.shape}")
            print(f"Input - Labels shape: {labels.shape}")
            print(f"Input - Weather IDs: {weather_ids}")
            print(f"Input - Time IDs: {time_ids}")
            print(f"Input - Names: {names[:2]}...")  # Show first 2 names
            
 
            print("------------------ testing out -------------------")
            print("type: ", type(out))
            print("len: ", len(out))
            print("type1: ", type(out[0]))


            # 🆕 ADD THIS PRINT TO SEE VALIDATION OUTPUTS:
            print(f"=== VALIDATION OUTPUTS ===")
            print(f"Model output type: {type(out)}")
            if isinstance(out, (list, tuple)):
                print(f"Number of outputs: {len(out)}")
                for j, output in enumerate(out):
                    print(f"  Output {j} shape: {output.shape if hasattr(output, 'shape') else 'No shape'}")
                print(f"  USING ONLY Output 0 (Segmentation) - Outputs 1 & 2 (WAS/TAS) are IGNORED during validation!")
            else:
                print(f"Single output shape: {out.shape}")
            print("===========================")            
            # 🖨️ PRINT: Model outputs during validation
            print(f"Model output type: {type(out)}")
            if isinstance(out, (list, tuple)):
                print(f"Number of outputs: {len(out)}")
                for j, output in enumerate(out):
                    print(f"  Output {j} shape: {output.shape if hasattr(output, 'shape') else 'No shape'}")
            else:
                print(f"Single output shape: {out.shape}")
            
            outputs = out[0] if isinstance(out, (list, tuple)) and len(out) >= 1 else out

            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:
                ret_samples.append((images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                for j in range(len(images)):
                    img_np = images[j].detach().cpu().numpy()
                    pred = preds[j]
                    img_vis = (denorm(img_np) * 255).transpose(1, 2, 0).astype(np.uint8)
                    pred_vis = base_ds.decode_target(pred).astype(np.uint8)
                    Image.fromarray(pred_vis).save(f"results/{names[j]}")
                # avoid heavy per-image overlay viz to keep val fast

        score = metrics.get_results()

    print(f"[val] DONE: mIoU={score.get('Mean IoU', float('nan')):.6f} | "
          f"FWAcc={score.get('FreqW Acc', float('nan')):.6f} | "
          f"OvAcc={score.get('Overall Acc', float('nan')):.6f}")
    return score, ret_samples


# -----------------------
# Helpers
# -----------------------
def _current_lrs(optimizer):
    return [pg.get("lr", None) for pg in optimizer.param_groups]

def _toggle_low_level(model, trainable: bool):
    for name, p in model.named_parameters():
        if "module.backbone.low_level_features." in name:
            p.requires_grad = trainable

def _count_low_level_params(model):
    trainable = 0
    frozen = 0
    for name, p in model.named_parameters():
        if "module.backbone.low_level_features." in name:
            if p.requires_grad:
                trainable += p.numel()
            else:
                frozen += p.numel()
    return trainable, frozen

def _global_grad_norm(model) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total += param_norm.item() ** 2
    return float(total ** 0.5)


# -----------------------
# Main
# -----------------------
def main():
    opts = get_argparser().parse_args()

    # Device & seeds
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n========== RUN CONFIG ==========")
    print(f"Start time           : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device               : {device}")
    print(f"GPU_VISIBLE_DEVICES  : {os.environ.get('CUDA_VISIBLE_DEVICES','')}")
    print(f"Seeds (torch/numpy/random): {opts.random_seed}/{opts.random_seed}/{opts.random_seed}")
    print(f"Mode                 : {opts.mode} (0=train, 11=eval CS, 21=eval ACDC)")
    print(f"Data roots           : CS={opts.data_root_cs} | ACDC={opts.data_root_acdc}")
    print(f"Model                : {opts.model} | output_stride={opts.output_stride} | num_classes={opts.num_classes}")
    print(f"Train schedule       : total_itrs={opts.total_itrs} | val_interval={opts.val_interval} | batch={opts.batch_size}")
    print(f"Optim                : SGD(lr={opts.lr}, wd={opts.weight_decay}, momentum=0.9) | lr_policy={opts.lr_policy}")
    print(f"Loss(main)           : {opts.loss_type} | Aux weight={opts.aux_weight:g}")
    print(f"Visdom               : enabled={opts.enable_vis} port={opts.vis_port} env={opts.vis_env}")
    if opts.log_grad_norm_every > 0:
        print(f"Grad-norm logging    : every {opts.log_grad_norm_every} iters")
    print(f"===============================\n")

    torch.manual_seed(opts.random_seed); np.random.seed(opts.random_seed); random.seed(opts.random_seed)

    # Visdom
    vis = Visualizer(port=opts.vis_port, env=opts.vis_env) if opts.enable_vis else None
    if vis is not None: vis.vis_table("Options", vars(opts))

    # Transforms
    train_tf, val_tf = _build_transforms(opts)

    # MODE handling
    MODE = opts.mode
    print(f"[mode] Running mode={MODE} ({'train' if MODE==0 else 'eval'})")

    if MODE == 0:
        # --------- Splits: 85/15 from TRAIN ONLY (official val is for eval only) ----------
        cs_train, cs_val = build_cityscapes_train85_val15_datasets(
            root=opts.data_root_cs,
            train_transform=train_tf,
            val_transform=val_tf,
            seed=opts.random_seed,
            include_official_val_in_train=False  # IMPORTANT
        )
        acdc_train = ACDC(root=opts.data_root_acdc, split="train85",
                          transform=train_tf, holdout_seed=opts.random_seed)
        acdc_val   = ACDC(root=opts.data_root_acdc, split="val15",
                          transform=val_tf, holdout_seed=opts.random_seed)

        train_loader_cs   = data.DataLoader(cs_train,   batch_size=opts.batch_size,     shuffle=True,  num_workers=8, drop_last=True, pin_memory=True)
        train_loader_acdc = data.DataLoader(acdc_train, batch_size=opts.batch_size,     shuffle=True,  num_workers=8, drop_last=True, pin_memory=True)
        val_loader_cs     = data.DataLoader(cs_val,     batch_size=opts.val_batch_size, shuffle=False, num_workers=4, pin_memory=True)
        val_loader_acdc   = data.DataLoader(acdc_val,   batch_size=opts.val_batch_size, shuffle=False, num_workers=4, pin_memory=True)

        print(f"[split] Cityscapes TRAIN: train85={len(cs_train)} | val15={len(cs_val)} | loader_len={len(train_loader_cs)}")
        print(f"[split] ACDC       TRAIN: train85={len(acdc_train)} | val15={len(acdc_val)} | loader_len={len(train_loader_acdc)}")
        print(f"[paths] CS={opts.data_root_cs}")
        print(f"[paths] ACDC={opts.data_root_acdc}")

    elif MODE in (11, 21):
        # Evaluation on OFFICIAL val only
        if MODE == 11:
            tst = Cityscapes(root=opts.data_root_cs, split="val", transform=val_tf, debug_print=True)
            dataset_flag = "cityscapes"
        else:
            tst = ACDC(root=opts.data_root_acdc, split="val", transform=val_tf,
                       test_class=opts.ACDC_test_class)
            dataset_flag = "acdc"
        test_loader = data.DataLoader(tst, batch_size=opts.test_batch_size,
                                      shuffle=False, num_workers=4, pin_memory=True)
        print(f"[eval] Dataset={dataset_flag} | size={len(tst)} | test_loader_len={len(test_loader)} | per-condition={opts.ACDC_test_class}")

    else:
        raise ValueError("--mode must be one of {0, 11, 21}")

    # ----- Model -----
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and "plus" in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    model = nn.DataParallel(model).to(device)

    # ----- Optim / Sched / Loss -----
    optimizer = torch.optim.SGD(params=[
        {"params": model.module.backbone.parameters(),   "lr": 0.1 * opts.lr},
        {"params": model.module.classifier.parameters(), "lr": opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)

    if opts.lr_policy == "poly":
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
        sched_name = f"PolyLR(power=0.9, max_iters={opts.total_itrs})"
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)
        sched_name = f"StepLR(step_size={opts.step_size}, gamma=0.1)"

    if opts.loss_type == "focal_loss":
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
        loss_name = "FocalLoss(ignore=255)"
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction="mean")
        loss_name = "CrossEntropy(ignore=255)"

    print(f"[model] {opts.model} | os={opts.output_stride} | num_classes={opts.num_classes}")
    print(f"[optim] SGD(lr={opts.lr}, wd={opts.weight_decay}, momentum=0.9)")
    print(f"[optim] Param-group LRs (initial): { _current_lrs(optimizer) }  # [backbone_lr, classifier_lr]")
    print(f"[sched] {sched_name}")
    print(f"[loss ] {loss_name} | aux_weight={opts.aux_weight:g}")

    # ----- Optional weight load -----
    if opts.ckpt and os.path.isfile(opts.ckpt):
        ckpt = torch.load(opts.ckpt, map_location="cpu")
        model.module.load_state_dict(ckpt["model_state"], strict=True)
        if opts.continue_training and "optimizer_state" in ckpt and "scheduler_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            print(f"[ckpt] Loaded and CONTINUED from {opts.ckpt}")
        else:
            print(f"[ckpt] Loaded weights from {opts.ckpt} (no optimizer/scheduler state resume)")
        del ckpt

    # ----- TEST-ONLY -----
    if MODE in (11, 21):
        print("[eval] Starting evaluation...")
        model.eval()
        metrics = StreamSegMetrics(opts.num_classes)
        score, _ = validate(opts, model, test_loader, device, metrics, ret_samples_ids=None)
        print(metrics.to_str(score))

        # Full per-class IoU (with class names)
        ciou = score.get("Class IoU", {})
        names = Cityscapes.classes if MODE == 11 else ACDC.classes
        print("[eval] Per-class IoU:")
        for cid in range(opts.num_classes):
            raw = names[cid] if cid < len(names) else f"class_{cid}"
            name_str = raw.name if hasattr(raw, "name") else str(raw)
            print(f"  {cid:2d} {name_str:>15}: {ciou.get(cid, float('nan')):.6f}")
        return

    # ----- TRAIN LOOP (MODE=0) -----
    metrics = StreamSegMetrics(opts.num_classes)
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    os.makedirs("checkpoints", exist_ok=True)

    best_score_cs = 0.0
    best_score_acdc = 0.0
    cur_itrs = 0
    cur_epochs = 0

    vis_id_cs   = np.random.randint(0, len(val_loader_cs),   opts.vis_num_samples, np.int32) if opts.enable_vis else None
    vis_id_acdc = np.random.randint(0, len(val_loader_acdc), opts.vis_num_samples, np.int32) if opts.enable_vis else None

    def save_ckpt(tag):
        path = f"checkpoints/{tag}_{opts.model}_mix_os{opts.output_stride}.pth"
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score_cs": best_score_cs,
            "best_score_acdc": best_score_acdc,
        }, path)
        print(f"[ckpt] Saved → {path}")

    print("[train] BEGIN training")
    print(f"[train] total_itrs={opts.total_itrs} | val_interval={opts.val_interval} | print_interval={opts.print_interval}")
    interval_loss = 0.0
    aux_seen_once = False

    while cur_itrs < opts.total_itrs:
        model.train()
        cur_epochs += 1

        it_cs   = iter(train_loader_cs)
        it_acdc = iter(train_loader_acdc)
        max_iter = min(len(train_loader_cs), len(train_loader_acdc))
        print(f"[epoch] {cur_epochs} | max_iter(per-epoch)={max_iter} | cur_itrs={cur_itrs}")

        for i in range(max_iter):
            # Alternate batches: ACDC even → unfreeze; CS odd → freeze
            if i % 2 == 0:
                (images, labels, names, weather_ids, time_ids, data_domain) = next(it_acdc)
                _toggle_low_level(model, True)
                batch_src = "ACDC"
            else:
                (images, labels, names, weather_ids, time_ids, data_domain) = next(it_cs)
                _toggle_low_level(model, False)
                batch_src = "Cityscapes"

            # 🖨️ PRINT: Training inputs
            print(f"\n=== TRAINING BATCH {i} (Source: {batch_src}) ===")
            print(f"Input - Images shape: {images.shape}")
            print(f"Input - Labels shape: {labels.shape}")
            print(f"Input - Weather IDs: {weather_ids}")
            print(f"Input - Time IDs: {time_ids}")
            print(f"Input - Data Domain: {data_domain}")

            # Show toggle status a few times + sparsely later
            if cur_itrs < 5 or (cur_itrs % 500 == 0):
                tl, fr = _count_low_level_params(model)
                print(f"[train] itr={cur_itrs+1} | src={batch_src:10s} | "
                      f"low_level_features: trainable={tl:,} frozen={fr:,}")

            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            weather_ids = weather_ids.to(device)
            time_ids = time_ids.to(device)

            optimizer.zero_grad()
            out = model(images)

            print("******************** during training ******************************************************")

            print(f"Model output type: {type(out)}")
            if isinstance(out, (list, tuple)):
                print(f"Number of outputs: {len(out)}")
                for j, output in enumerate(out):
                    print(f"  Output {j} shape: {output.shape if hasattr(output, 'shape') else 'No shape'}")
                print(f"  USING ONLY Output 0 (Segmentation) - Outputs 1 & 2 (WAS/TAS) are IGNORED during validation!")
            else:
                print(f"Single output shape: {out.shape}")

            # 🖨️ PRINT: Model outputs during training
            print(f"Model output type: {type(out)}")
            if isinstance(out, (list, tuple)):
                print(f"Number of outputs: {len(out)}")
                for j, output in enumerate(out):
                    print(f"  Output {j} shape: {output.shape if hasattr(output, 'shape') else 'No shape'}")
            else:
                print(f"Single output shape: {out.shape}")

            if isinstance(out, (list, tuple)):
                outputs = out[0]
                weather_preds = out[1] if len(out) > 1 else None
                time_preds    = out[2] if len(out) > 2 else None
            else:
                outputs, weather_preds, time_preds = out, None, None

            # Log once whether aux heads exist
            if not aux_seen_once:
                print(f"[aux] heads_present: weather={weather_preds is not None} | time={time_preds is not None} "
                      f"| aux_weight={opts.aux_weight:g}")
                aux_seen_once = True

            # Main loss
            loss_seg = criterion(outputs, labels)

            # Aux losses (with raw values printed)
            raw_ws = None
            raw_tes = None
            loss_ws = None
            loss_tes = None
            if (weather_preds is not None) and (time_preds is not None) and (opts.aux_weight > 0):
                raw_ws  = criterion(weather_preds, weather_ids)
                raw_tes = criterion(time_preds,    time_ids)
                loss_ws  = raw_ws  * opts.aux_weight
                loss_tes = raw_tes * opts.aux_weight
                loss = loss_seg + loss_ws + loss_tes
                
                # 🖨️ PRINT: Loss breakdown
                print(f"Loss Breakdown:")
                print(f"  - Segmentation: {loss_seg.item():.6f}")
                print(f"  - Weather (raw): {raw_ws.item():.6f} → weighted: {loss_ws.item():.6f}")
                print(f"  - Time (raw): {raw_tes.item():.6f} → weighted: {loss_tes.item():.6f}")
                print(f"  - TOTAL: {loss.item():.6f}")
            else:
                loss = loss_seg
                print(f"Loss: Segmentation only = {loss_seg.item():.6f}")

            loss.backward()

            # Optional grad-norm
            if opts.log_grad_norm_every > 0 and (cur_itrs % opts.log_grad_norm_every == 0):
                gn = _global_grad_norm(model)
                print(f"[grad] itr={cur_itrs} | global_grad_norm={gn:.4f}")

            optimizer.step()

            # Log to running avg + vis
            interval_loss += float(loss_seg.detach().cpu().numpy())
            if vis is not None:
                vis.vis_scalar("Loss(seg)", cur_itrs, float(loss_seg.detach().cpu().numpy()))
                if loss_ws is not None:  vis.vis_scalar("Loss(weather)_weighted", cur_itrs, float(loss_ws.detach().cpu().numpy()))
                if loss_tes is not None: vis.vis_scalar("Loss(time)_weighted",    cur_itrs, float(loss_tes.detach().cpu().numpy()))

            # Console prints at interval
            if cur_itrs % opts.print_interval == 0:
                avg_loss = interval_loss / opts.print_interval
                lrs = _current_lrs(optimizer)
                msg = (f"[itrs] {cur_itrs}/{opts.total_itrs} | src={batch_src:10s} | "
                       f"loss(seg)={avg_loss:.6f} | lrs(backbone,cls)={lrs} | epoch={cur_epochs}")
                print(msg)

                # Detailed aux breakdown (raw vs weighted) if present
                if (raw_ws is not None) and (raw_tes is not None):
                    print(f"       aux(raw): weather={float(raw_ws.detach().cpu()):.6f} "
                          f"time={float(raw_tes.detach().cpu()):.6f} | "
                          f"weighted (x{opts.aux_weight:g}): "
                          f"weather={float(loss_ws.detach().cpu()):.6f} "
                          f"time={float(loss_tes.detach().cpu()):.6f}")
                interval_loss = 0.0

            # Validation
            if cur_itrs % opts.val_interval == 0:
                save_ckpt("latest")
                torch.cuda.empty_cache()
                model.eval()
                print("[val] running validation on CS(val15) and ACDC(val15)...")

                val_score_cs, _ = validate(
                    opts=opts, model=model, loader=val_loader_cs, device=device,
                    metrics=metrics, ret_samples_ids=vis_id_cs
                )
                val_score_acdc, _ = validate(
                    opts=opts, model=model, loader=val_loader_acdc, device=device,
                    metrics=metrics, ret_samples_ids=vis_id_acdc
                )

                print("[val][CS]   " + metrics.to_str(val_score_cs))
                print("[val][ACDC] " + metrics.to_str(val_score_acdc))

                if (val_score_cs["Mean IoU"] > best_score_cs) and (val_score_acdc["Mean IoU"] > best_score_acdc):
                    best_score_cs = val_score_cs["Mean IoU"]
                    best_score_acdc = val_score_acdc["Mean IoU"]
                    print(f"[val] NEW BEST → CS mIoU={best_score_cs:.6f} | ACDC mIoU={best_score_acdc:.6f}")
                    save_ckpt("best")

                # Visdom qualitative
                if vis is not None:
                    vis.vis_scalar("[Val] Mean IoU CS",   cur_itrs, val_score_cs["Mean IoU"])
                    vis.vis_table ("[Val] Class IoU CS",  val_score_cs["Class IoU"])
                    vis.vis_scalar("[Val] Mean IoU ACDC", cur_itrs, val_score_acdc["Mean IoU"])
                    vis.vis_table ("[Val] Class IoU ACDC",val_score_acdc["Class IoU"])

                model.train()

            # Scheduler step *after* optimizer step
            scheduler.step()

            if cur_itrs >= opts.total_itrs:
                print("[train] Reached total_itrs. Finishing.")
                break

        # end for i in epoch
    # end while
    print("[train] DONE.")


if __name__ == "__main__":
    main()
