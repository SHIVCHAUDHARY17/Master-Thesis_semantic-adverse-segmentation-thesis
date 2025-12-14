# main_mask2former.py
# The model is weather and time aware!
# This models is trained on: AWSS + CS 1:1
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
from datasets import Cityscapes, ACDC, AWSS
from utils import ext_transforms as et
from metrics import StreamSegMetrics
from utils.visualizer import Visualizer
from PIL import Image
from freeze_tools import freeze_encoder, unfreeze_encoder
import matplotlib
import matplotlib.pyplot as plt


def get_argparser():
    parser = argparse.ArgumentParser()

    # Dataset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="Path to dataset root directory.")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=[
    'voc',
    'cityscapes',
    'ACDC',
    'AWSS',
    'cityscapes_AWSS'],
                        help='Dataset name')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="Number of segmentation classes")

    # Model Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and
                            not (name.startswith("__") or name.startswith('_')) and callable(
                            network.modeling.__dict__[name]))
    parser.add_argument("--model", type=str, default='mask2former',
                        choices=available_models, help='Model name (mask2former for new setup)')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="Apply separable conv to decoder and ASPP (if using DeepLab)")
    parser.add_argument(
    "--output_stride",
    type=int,
    default=16,
    choices=[
        8,
        16])

    # Training Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=True,
                        help="Save segmentation results to ./results")
    parser.add_argument("--total_itrs", type=int, default=30000,
                        help="Total training iterations (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="Learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help="Crop validation images")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help="Batch size for validation")
    parser.add_argument("--test_batch_size", type=int, default=4,
                        help="Batch size for testing")
    parser.add_argument("--crop_size", type=int, default=513)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="Path to model checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False,
                        help="Resume training from last checkpoint")
    parser.add_argument("--finetune", action='store_true', default=False,
                        help="Whether to fine-tune the model")

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="Loss function")
    parser.add_argument("--gpu_id", type=str, default='0', help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--print_interval", type=int, default=10)
    parser.add_argument("--val_interval", type=int, default=100)
    parser.add_argument("--download", action='store_true', default=False)

    # PASCAL VOC specific
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='VOC year')

    # ACDC test class
    parser.add_argument("--ACDC_test_class", type=str, default=None,
                        help="ACDC test class (rain/fog/snow/night)")

    # Visdom Options
    parser.add_argument("--enable_vis", action='store_true', default=False)
    parser.add_argument("--vis_port", type=str, default='13570')
    parser.add_argument("--vis_env", type=str, default='main')
    parser.add_argument("--vis_num_samples", type=int, default=8)

    return parser


def get_dataset(opts, tr_ds_name=None):
    """Returns datasets and corresponding transforms based on the name."""

    if opts.dataset == 'cityscapes' or tr_ds_name == "cityscapes":
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

        train_dst = Cityscapes(
    root=opts.data_root_cs,
    split='train',
    transform=train_transform)
        val_dst = Cityscapes(
    root=opts.data_root_cs,
    split='val',
    transform=val_transform)
        tst_dst = Cityscapes(
    root=opts.data_root_cs,
    split='test',
    transform=val_transform)

    elif opts.dataset == 'ACDC' or tr_ds_name == "ACDC":
        val_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        train_dst = []  # No training on ACDC
        val_dst = ACDC(
    root=opts.data_root_acdc,
    split='val',
    transform=val_transform)
        tst_dst = ACDC(root=opts.data_root_acdc, split='test', transform=val_transform,
                    test_class=opts.ACDC_test_class) if opts.ACDC_test_class else []

    elif opts.dataset == 'AWSS' or tr_ds_name == "AWSS":
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
        train_dst = AWSS(
    root=opts.data_root_awss,
    split='train',
    transform=train_transform)
        val_dst = AWSS(
    root=opts.data_root_awss,
    split='val',
    transform=val_transform)
        tst_dst = []

    else:
        raise NotImplementedError

    return train_dst, val_dst, tst_dst

def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Run validation, compute metrics, optionally save results."""
    metrics.reset()
    ret_samples = []
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        img_id = 0

    model.eval()
    with torch.no_grad():
        for i, (images, labels, names, weather_ids, time_ids, domain) in tqdm(enumerate(loader)):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            print("Unique label values:", torch.unique(labels))

            #preds, _, _ = model(images)
            # Forward pass
            logits, _, _ = model(images)
            # Resize logits to match label shape
            logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode='bilinear', align_corners=False)
            # ✅ Interpolate logits to match ground truth shape
            #if isinstance(preds, torch.Tensor) and preds.ndim == 4:
                #preds = nn.functional.interpolate(preds, size=labels.shape[-2:], mode='bilinear', align_corners=False)
                
            # Compute predictions
            preds = logits.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()
            
            print("preds shape:", preds.shape)
            print("labels shape:", labels.shape)
        
            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:
                ret_samples.append((images[0].cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].cpu().numpy()
                    pred = preds[i]
                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)
                    Image.fromarray(pred).save(f'results/{names[i]}')

        score = metrics.get_results()
    return score, ret_samples

def main(ACDC_test_class=None, n_itrs=30000, MODE=12345):
    opts = get_argparser().parse_args()
    opts.ACDC_test_class = ACDC_test_class
    opts.finetune = False
    opts.pretrained_model = None

    MODE = 0   # train on Cityscapes + AWSS
    # MODE = 11  # test on Cityscapes
    # MODE = 21  # test on ACDC
    opts.data_root_cs = "/media/ubuntu22/secondSSD/Shiv/datasets/cityscapes"
    opts.data_root_acdc = "/media/ubuntu22/secondSSD/Shiv/datasets/ACDC"
    opts.data_root_awss = "/media/ubuntu22/secondSSD/Shiv/datasets/AWSS"
    opts.total_itrs = n_itrs
    opts.test_class = None
    opts.val_batch_size = 1

    # Dataset logic based on MODE
    if MODE == 0:
        opts.test_only = False
        opts.save_val_results = False
        opts.dataset = 'cityscapes_AWSS'
    elif MODE == 11:
        opts.test_only = True
        opts.save_val_results = True
        opts.dataset = "cityscapes"
        opts.ckpt = "checkpoints/D01_mask2former_cityscapes_AWSS.pth"
    elif MODE == 21:
        opts.test_only = True
        opts.save_val_results = True
        opts.dataset = "ACDC"
        opts.ckpt = "checkpoints/D01_mask2former_cityscapes_AWSS.pth"

    # Switch model name from DeepLab to Mask2Former
    opts.model = "mask2former"
    opts.enable_vis = True
    opts.vis_port = 8097
    opts.gpu_id = '0'
    #opts.lr = 0.1
    opts.lr = 0.001
    opts.crop_size = 768
    opts.batch_size = 1 
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

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # --------------------- Random Seed Setup ---------------------
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Adjust val batch size for VOC when crop_val is off
    if opts.dataset == 'voc' and not opts.crop_val:
        opts.val_batch_size = 1

    # --------------------- Dataset & DataLoader Setup ---------------------
    if opts.test_only:
        # Only test set required
        _, _, tst_dst = get_dataset(opts)

        test_loader = data.DataLoader(
            tst_dst,
            batch_size=opts.test_batch_size,
            shuffle=True,
            num_workers=0
        )
        print(f"Dataset: {opts.dataset}, Test set size: {len(tst_dst)}")

    else:
        # Training mode — load all 3 datasets
        train_dst_cs, val_dst_cs, _ = get_dataset(opts, 'cityscapes')
        train_dst_awss, _, _ = get_dataset(opts, 'AWSS')
        _, val_dst_acdc, _ = get_dataset(opts, 'ACDC')

        # Training dataloaders
        train_loader_cs = data.DataLoader(
            train_dst_cs,
            batch_size=opts.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )
        train_loader_awss = data.DataLoader(
            train_dst_awss,
            batch_size=opts.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )

        # Validation dataloaders
        val_loader_cs = data.DataLoader(
            val_dst_cs,
            batch_size=opts.val_batch_size,
            shuffle=True,
            num_workers=0
        )
        val_loader_acdc = data.DataLoader(
            val_dst_acdc,
            batch_size=opts.val_batch_size,
            shuffle=True,
            num_workers=0
        )

        #print(f"Dataset: Cityscapes + AWSS")
        #print(f"Train sets — Cityscapes: {len(train_dst_cs)}, AWSS: {len(train_dst_awss)}")
        #print(f"Validation sets — Cityscapes: {len(val_dst_cs)}, ACDC: {len(val_dst_acdc)}")

    # --------------------- Model Setup ---------------------
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes)
    print("DEBUG: Using model:", type(model))


    # Set batchnorm momentum (for training stability)
    if hasattr(model, 'backbone'):
        utils.set_bn_momentum(model.backbone, momentum=0.01)

    # --------------------- Metric Setup ---------------------
    metrics = StreamSegMetrics(opts.num_classes)

    # --------------------- Optimizer Setup ---------------------
    if hasattr(model, 'backbone'):
        # Typical for transformer models: separate learning rate for backbone and head
        optimizer = torch.optim.SGD(params=[
            {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
            {'params': model.classifier.parameters(), 'lr': opts.lr},
        ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    else:
        # If backbone is not exposed separately
        optimizer = torch.optim.SGD(model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)

    # --------------------- Learning Rate Scheduler ---------------------
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

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
        }, path)
        print(f"Model saved at {path}")

    utils.mkdir('checkpoints')  # create directory if not exists

    # --------------------- Checkpoint Restore / Resume ---------------------
    best_score_cs = 0.0
    best_score_acdc = 0.0
    cur_itrs = 0
    cur_epochs = 0

    if not opts.finetune:
        if opts.ckpt is not None and os.path.isfile(opts.ckpt):
            print(f"Restoring model from checkpoint: {opts.ckpt}")
            checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint["model_state"])
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
        model.load_state_dict(checkpoint["model_state"])
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
        test_score, ret_samples = validate(
            opts=opts,
            model=model,
            loader=test_loader,
            device=device,
            metrics=metrics,
            ret_samples_ids=vis_sample_id
        )
        print(metrics.to_str(test_score))
        IoU_scores = np.array(list(test_score['Class IoU'].values()))[
            np.array([0, 1, 2, 5, 6, 7, 8, 10, 11, 13])]
        print(IoU_scores)
        return

    else:  # Training setup
        vis_sample_id_cs = np.random.randint(
            0, len(val_loader_cs), opts.vis_num_samples, np.int32) if opts.enable_vis else None

        vis_sample_id_acdc = np.random.randint(
            0, len(val_loader_acdc), opts.vis_num_samples, np.int32) if opts.enable_vis else None

    interval_loss = 0
    while True:  # Training loop: optionally set to `while cur_itrs < opts.total_itrs`
        model.train()
        cur_epochs += 1

        dataloader_iterator_cs = iter(train_loader_cs)
        dataloader_iterator_awss = iter(train_loader_awss)

        for i, (_, _, _, _, _, _) in enumerate(train_loader_awss):

            if i % 2 == 0:
                # AWSS Batch
                images, labels, _, weather_ids, time_ids, data_domain = next(dataloader_iterator_awss)

                for name, param in model.named_parameters():
                    if 'module.backbone.low_level_features.' in name:
                        param.requires_grad = True

            else:
                # Cityscapes Batch
                images, labels, _, weather_ids, time_ids, data_domain = next(dataloader_iterator_cs)

                for name, param in model.named_parameters():
                    if 'module.backbone.low_level_features.' in name:
                        param.requires_grad = False

            cur_itrs += 1
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            print("Unique label values:", torch.unique(labels))
            #print("DEBUG: type(weather_ids):", type(weather_ids))
            #print("DEBUG: weather_ids:", weather_ids)
            weather_ids = weather_ids.to(device)
            time_ids = time_ids.to(device)

            optimizer.zero_grad()

            seg_logits, weather_preds, time_preds = model(images)
            # 🔍 DEBUG: Check shape compatibility
            print("weather_preds shape:", weather_preds.shape)
            print("weather_ids shape:", weather_ids.shape)
            print("time_preds shape:", time_preds.shape)
            print("time_ids shape:", time_ids.shape)
             


            #print("DEBUG TYPE of seg_logits:", type(seg_logits))
            seg_logits = nn.functional.interpolate(seg_logits, size=labels.shape[-2:], mode='bilinear', align_corners=False)
            print("Logits has NaNs:", torch.isnan(seg_logits).any().item())  # ✅ Insert here

            # === DEBUG BLOCK ===
            if cur_itrs % 50 == 0:
                with torch.no_grad():
                    pred_classes = seg_logits.argmax(dim=1)
                    print(f"[DEBUG @ Iter {cur_itrs}] Unique predicted class values: {torch.unique(pred_classes)}")
                    print(f"[DEBUG @ Iter {cur_itrs}] Unique label values: {torch.unique(labels)}")
                    print(f"[DEBUG @ Iter {cur_itrs}] Data Domain: {'AWSS' if i % 2 == 0 else 'Cityscapes'}")
                    print(f"[DEBUG @ Iter {cur_itrs}] Weather IDs: {weather_ids}")
                    print(f"[DEBUG @ Iter {cur_itrs}] Time IDs: {time_ids}")

            loss_segmentation = criterion(seg_logits, labels)
            #loss_segmentation.backward(retain_graph=True)

            loss_weather = criterion(weather_preds, weather_ids) * 0.00001
            loss_time = criterion(time_preds, time_ids) * 0.00001

            #loss_weather.backward(retain_graph=True)
            #loss_time.backward()
            total_loss = loss_segmentation + 1e-5 * loss_weather + 1e-5 * loss_time
            print(f"[Loss @ Iter {cur_itrs}] Seg: {loss_segmentation.item():.4f}, "
            f"Weather: {loss_weather.item():.6f}, Time: {loss_time.item():.6f}")

            #total_loss = loss_segmentation
            total_loss.backward()
            optimizer.step()
            np_loss = loss_segmentation.detach().cpu().numpy()

            interval_loss += np_loss
            if vis is not None:
                vis.vis_scalar('Loss', cur_itrs, np_loss)

            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss / 10
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                    (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                interval_loss = 0.0

            if (cur_itrs) % opts.val_interval == 0:
                save_ckpt('checkpoints/latest_%s_%s_os%d.pth' %
                        (opts.model, opts.dataset, opts.output_stride))
                print("validation...")
                model.eval()
                val_score_cs, ret_samples_cs = validate(
                    opts=opts, model=model, loader=val_loader_cs, device=device, metrics=metrics,
                    ret_samples_ids=vis_sample_id_cs)

                val_score_acdc, ret_samples_acdc = validate(
                    opts=opts, model=model, loader=val_loader_acdc, device=device, metrics=metrics,
                    ret_samples_ids=vis_sample_id_acdc)
                print(metrics.to_str(val_score_cs), metrics.to_str(val_score_acdc))

                if val_score_cs['Mean IoU'] > best_score_cs and val_score_acdc['Mean IoU'] > best_score_acdc:
                    best_score_cs = val_score_cs['Mean IoU']
                    best_score_acdc = val_score_acdc['Mean IoU']
                    save_ckpt('checkpoints/best_%s_%s_os%d.pth' %
                            (opts.model, opts.dataset, opts.output_stride))

                if vis is not None:
                    vis.vis_scalar("[Val] Mean IoU CS", cur_itrs, val_score_cs['Mean IoU'])
                    vis.vis_table("[Val] Class IoU CS", val_score_cs['Class IoU'])

                    vis.vis_scalar("[Val] Mean IoU ACDC", cur_itrs, val_score_acdc['Mean IoU'])
                    vis.vis_table("[Val] Class IoU ACDC", val_score_acdc['Class IoU'])

                    for k, (img, target, lbl) in enumerate(ret_samples_cs):
                        img = (denorm(img) * 255).astype(np.uint8)
                        target = train_dst_cs.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                        lbl = train_dst_cs.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                        concat_img = np.concatenate((img, target, lbl), axis=2)
                        vis.vis_image('Sample %d' % k, concat_img)

                model.train()

            scheduler.step()

            if cur_itrs >= opts.total_itrs:
                return


if __name__ == '__main__':
    ACDC_classes = ['rain', 'fog', 'snow', 'night']
    opts = []

    MODE = 0  # train on cityscapes and AWSS
    # MODE = 11  # test on cityscapes
    # MODE = 21  # test on acdc

    if MODE == 21:
        for ACDC_test_class in ACDC_classes:
            main(ACDC_test_class=ACDC_test_class, MODE=MODE)
        exit()
    main(MODE=MODE)


