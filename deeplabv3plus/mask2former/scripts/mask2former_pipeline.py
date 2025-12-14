# ================================================
# Mask2Former Unified Pipeline (DeepLab-like)
# Part 1: Imports, Args, Dataset Prep
# ================================================

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import Mask2FormerForUniversalSegmentation, AutoImageProcessor

from models.auxiliary_heads import AuxiliaryClassifier
from scripts.utils import freeze_encoder, unfreeze_encoder
from datasets.cityscapes import Cityscapes
from datasets.AWSS import AWSS
from datasets.ACDC import ACDC  # assuming you have an ACDC dataset class like Cityscapes/AWSS

import numpy as np
from visdom import Visdom
from tqdm import tqdm
from PIL import Image
from metrics.streammetrics import StreamSegMetrics
import matplotlib.pyplot as plt

# -----------------------------------------------
# Argparse: replicate DeepLab interface
# -----------------------------------------------
def get_argparser():
    parser = argparse.ArgumentParser(description="Mask2Former Pipeline (DeepLab-like Modes)")

    parser.add_argument("--mode", type=int, required=True,
                        help="0 = Train, 11 = Evaluate Cityscapes, 21 = Evaluate ACDC")

    parser.add_argument("--data_root_cityscapes", type=str, required=True,
                        help="Path to Cityscapes dataset")

    parser.add_argument("--data_root_awss", type=str, required=True,
                        help="Path to AWSS dataset")

    parser.add_argument("--data_root_acdc", type=str, default=None,
                        help="Path to ACDC dataset (for mode 21)")

    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path for evaluation")

    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs for training")

    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size")

    parser.add_argument("--save_results", action="store_true",
                        help="Save predictions during evaluation")

    parser.add_argument("--vis_port", type=str, default='8097',
                        help="Visdom port")

    return parser

# -----------------------------------------------
# Helper: decode mask to RGB
# -----------------------------------------------
def decode_segmap(mask, n_classes=20):
    label_colors = np.array([
        (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
        (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128),
        (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
        (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
        (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0)
    ])
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for l in range(n_classes):
        idx = mask == l
        r[idx], g[idx], b[idx] = label_colors[l]
    rgb = np.stack([r, g, b], axis=0)
    return rgb

# -----------------------------------------------
# Helper: denormalize image
# -----------------------------------------------
def denorm(image):
    mean = np.array([0.485, 0.456, 0.406]).reshape(3,1,1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3,1,1)
    return (image * std + mean).clip(0,1)

# -----------------------------------------------
# Custom collate (to keep PIL images for processor)
# -----------------------------------------------
def custom_collate(batch):
    images, masks, weathers, times = zip(*batch)
    return {
        "image": list(images),
        "mask": torch.stack([torch.as_tensor(m) for m in masks]),
        "weather": torch.tensor(weathers),
        "time": torch.tensor(times)
    }
# ================================================
# Mask2Former Unified Pipeline (DeepLab-like)
# Part 2: Model Setup and Training Loop
# ================================================

def main():
    opts = get_argparser().parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", DEVICE)

    # Visdom
    viz = Visdom(port=opts.vis_port)

    # Model
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-small-ade-semantic"
    ).to(DEVICE)
    processor = AutoImageProcessor.from_pretrained(
        "facebook/mask2former-swin-small-ade-semantic"
    )

    encoder_channels = 768
    was_head = AuxiliaryClassifier(encoder_channels, 4).to(DEVICE)
    tas_head = AuxiliaryClassifier(encoder_channels, 2).to(DEVICE)

    ce_loss = nn.CrossEntropyLoss(ignore_index=255)

    optimizer = optim.Adam(
        list(model.parameters()) + list(was_head.parameters()) + list(tas_head.parameters()),
        lr=1e-4
    )

    if opts.mode == 0:
        # ======= MODE=0 TRAINING =======

        print("Setting up datasets for training...")
        city_dataset = Cityscapes(
            root=opts.data_root_cityscapes,
            split="train"
        )
        awss_dataset = AWSS(
            root=opts.data_root_awss,
            split="train"
        )

        city_loader = DataLoader(
            city_dataset, batch_size=opts.batch_size, shuffle=True, drop_last=True, collate_fn=custom_collate
        )
        awss_loader = DataLoader(
            awss_dataset, batch_size=opts.batch_size, shuffle=True, drop_last=True, collate_fn=custom_collate
        )

        city_iter = iter(city_loader)
        awss_iter = iter(awss_loader)

        num_batches = min(len(city_loader), len(awss_loader)) * 2

        loss_win = viz.line(
            X=[0], Y=[[0,0,0]],
            opts=dict(title="Training Losses", legend=["Seg","WAS","TAS"], xlabel="Batch", ylabel="Loss")
        )

        for epoch in range(opts.epochs):
            model.train()
            was_head.train()
            tas_head.train()
            print(f"\nEpoch {epoch+1}/{opts.epochs}")

            for i in range(num_batches):
                if i % 2 == 0:
                    print("CITYSCAPES **********")
                    try: batch = next(city_iter)
                    except StopIteration:
                        city_iter = iter(city_loader)
                        batch = next(city_iter)
                    freeze_encoder(model)
                else:
                    print("AWSS **********")
                    try: batch = next(awss_iter)
                    except StopIteration:
                        awss_iter = iter(awss_loader)
                        batch = next(awss_iter)
                    unfreeze_encoder(model)

                images = batch["image"]
                masks = batch["mask"].to(DEVICE)
                weather_labels = batch["weather"].to(DEVICE)
                time_labels = batch["time"].to(DEVICE)

                encoded = processor(images=images, return_tensors="pt")
                pixel_values = encoded["pixel_values"].to(DEVICE)

                optimizer.zero_grad()
                outputs = model(pixel_values=pixel_values, return_dict=True)

                class_logits = outputs.class_queries_logits
                mask_logits = outputs.masks_queries_logits
                seg_logits = torch.einsum("bqc,bqhw->bchw", class_logits.softmax(-1), mask_logits)

                target_masks = torch.nn.functional.interpolate(
                    masks.unsqueeze(1).float(),
                    size=seg_logits.shape[-2:], mode="nearest"
                ).squeeze(1).long()

                seg_loss = ce_loss(seg_logits, target_masks)
                encoder_feats = outputs.encoder_last_hidden_state
                was_loss = ce_loss(was_head(encoder_feats), weather_labels)
                tas_loss = ce_loss(tas_head(encoder_feats), time_labels)

                total_loss = seg_loss + 0.5 * was_loss + 0.5 * tas_loss
                total_loss.backward()
                optimizer.step()

                viz.line(
                    X=[i + 1 + epoch*num_batches],
                    Y=[[seg_loss.item(), was_loss.item(), tas_loss.item()]],
                    win=loss_win,
                    update="append"
                )

                print(f"[Batch {i+1}/{num_batches}] Seg: {seg_loss:.3f} | WAS: {was_loss:.3f} | TAS: {tas_loss:.3f}")

                # Visualization every 100 batches
                if (i+1)%100==0:
                    preds = torch.argmax(seg_logits, dim=1)
                    mask_np = target_masks[0].cpu().numpy()
                    pred_np = preds[0].cpu().numpy()

                    img_np = np.array(images[0]).transpose(2,0,1).astype(np.float32)/255.0
                    img_dn = denorm(img_np)
                    mask_h, mask_w = mask_np.shape
                    img_resized = torch.nn.functional.interpolate(
                        torch.from_numpy(img_dn).unsqueeze(0),
                        size=(mask_h, mask_w),
                        mode="bilinear",
                        align_corners=False
                    ).squeeze(0).clamp(0,1).numpy()*255
                    img_resized = img_resized.astype(np.uint8)

                    gt_rgb = decode_segmap(mask_np, 20)
                    pred_rgb = decode_segmap(pred_np, 20)

                    concat = np.concatenate([img_resized, gt_rgb, pred_rgb], axis=2)
                    viz.image(concat, opts=dict(title=f"Input | GT | Pred Batch {i+1}"))

            # Save checkpoint after each epoch
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'was_state_dict': was_head.state_dict(),
                'tas_state_dict': tas_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, f"checkpoints/epoch_{epoch+1}.pt")

        print("Training complete.")
# ================================================
# Mask2Former Unified Pipeline (DeepLab-like)
# Part 3: Evaluation (Cityscapes and ACDC)
# ================================================

    if opts.mode in [11, 21]:
        assert opts.checkpoint is not None, "Please provide --checkpoint for evaluation"

        print("Loading checkpoint:", opts.checkpoint)
        ckpt = torch.load(opts.checkpoint, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        was_head.load_state_dict(ckpt["was_state_dict"])
        tas_head.load_state_dict(ckpt["tas_state_dict"])
        model.eval()
        was_head.eval()
        tas_head.eval()

        # Dataset
        if opts.mode == 11:
            dataset = Cityscapes(
                root=opts.data_root_cityscapes,
                split="val"
            )
            dataset_name = "Cityscapes Validation"
        elif opts.mode == 21:
            assert opts.data_root_acdc is not None, "Provide --data_root_acdc for ACDC evaluation"
            dataset = ACDC(
                root=opts.data_root_acdc,
                split="test"
            )
            dataset_name = "ACDC Test"

        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=custom_collate
        )

        print(f"Running evaluation on {dataset_name} ({len(loader)} samples)...")

        metrics = StreamSegMetrics(n_classes=10)
        metrics.reset()

        os.makedirs("results", exist_ok=True)

        with torch.no_grad():
            for idx, batch in enumerate(tqdm(loader)):
                images = batch["image"]
                masks = batch["mask"].to(DEVICE)

                encoded = processor(images=images, return_tensors="pt")
                pixel_values = encoded["pixel_values"].to(DEVICE)

                outputs = model(pixel_values=pixel_values, return_dict=True)
                class_logits = outputs.class_queries_logits
                mask_logits = outputs.masks_queries_logits

                seg_logits = torch.einsum("bqc,bqhw->bchw", class_logits.softmax(-1), mask_logits)
                preds = torch.argmax(seg_logits, dim=1)

                gt = torch.nn.functional.interpolate(
                    masks.unsqueeze(1).float(),
                    size=seg_logits.shape[-2:], mode="nearest"
                ).squeeze(1).long()

                pred_np = preds[0].cpu().numpy()
                gt_np = gt[0].cpu().numpy()

                # Update metrics
                metrics.update([gt_np], [pred_np])

                if opts.save_results:
                    img_np = np.array(images[0]).transpose(2,0,1).astype(np.float32)/255.0
                    img_dn = denorm(img_np)
                    mask_h, mask_w = gt_np.shape
                    img_resized = torch.nn.functional.interpolate(
                        torch.from_numpy(img_dn).unsqueeze(0),
                        size=(mask_h, mask_w),
                        mode="bilinear",
                        align_corners=False
                    ).squeeze(0).clamp(0,1).numpy()*255
                    img_resized = img_resized.astype(np.uint8)

                    gt_rgb = decode_segmap(gt_np, 20)
                    pred_rgb = decode_segmap(pred_np, 20)

                    concat = np.concatenate([img_resized, gt_rgb, pred_rgb], axis=2)
                    Image.fromarray(concat.transpose(1,2,0)).save(f"results/sample_{idx}.png")

        # Compute and print results
        results = metrics.get_results()
        print("\n=== Evaluation Results ===")
        print(f"Pixel Accuracy: {results['Overall Acc']*100:.2f}%")
        print(f"Mean IoU: {results['Mean IoU']:.4f}")
        for class_id, iou in results['Class IoU'].items():
            print(f"Class {class_id} IoU: {iou:.4f}")

        print("Evaluation complete.")

if __name__ == "__main__":
    main()
