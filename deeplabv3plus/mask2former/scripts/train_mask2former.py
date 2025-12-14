# scripts/train_cityscapes_only.py
import os, sys, time, random, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from datasets.cityscapes import Cityscapes
from utils import ext_transforms as et
from metrics.stream_metrics import StreamSegMetrics

from transformers import (
    AutoImageProcessor,
    Mask2FormerConfig,
    Mask2FormerForUniversalSegmentation,
    get_cosine_schedule_with_warmup,
)

NUM_CLASSES = 19
IGNORE = 255
CITYSCAPES_NAMES = [
    "road","sidewalk","building","wall","fence","pole","traffic light","traffic sign",
    "vegetation","terrain","sky","person","rider","car","truck","bus","train","motorcycle","bicycle"
]

def set_seed(seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def build_transforms(crop_size):
    # keep PIL transforms; HF processor will tensorize + normalize
    return et.ExtCompose([
        et.ExtRandomCrop(size=(crop_size, crop_size)),
        et.ExtRandomHorizontalFlip(),
        et.ExtColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    ])

def collate_with_processor(processor):
    """
    Dataset item: (PIL image, trainId label tensor [HxW], name, weather, time, domain)
    We convert labels to int64 numpy arrays; HF processor builds training targets.
    """
    def _fn(batch):
        imgs, lbls, names, *_ = zip(*batch)
        segmaps = [l.cpu().numpy().astype("int64") for l in lbls]  # {0..18,255}
        proc = processor(images=list(imgs), segmentation_maps=segmaps, return_tensors="pt")
        return proc, segmaps, names
    return _fn

@torch.no_grad()
def evaluate(model, processor, val_loader, device):
    model.eval()
    metrics = StreamSegMetrics(n_classes=NUM_CLASSES)
    for proc, segmaps, _ in val_loader:
        pixel_values = proc["pixel_values"].to(device, non_blocking=True)
        pixel_mask   = proc.get("pixel_mask")
        pixel_mask   = pixel_mask.to(device, non_blocking=True) if pixel_mask is not None else None

        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        # upsample to input size
        target_sizes = [(pixel_values.shape[-2], pixel_values.shape[-1])] * pixel_values.size(0)
        preds = processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
        label_trues = segmaps
        label_preds = [p.cpu().numpy().astype("int64") for p in preds]
        metrics.update(label_trues, label_preds)
    return metrics.get_results()

def build_model_and_processor(args, device):
    # image processor (normalization + label packing)
    processor = AutoImageProcessor.from_pretrained(args.init_card)
    processor.num_labels = NUM_CLASSES
    processor.ignore_index = IGNORE
    processor.do_reduce_labels = False  # Cityscapes

    id2label = {i: n for i, n in enumerate(CITYSCAPES_NAMES)}
    label2id = {v: k for k, v in id2label.items()}

    if args.init_mode == "pretrained":
        # fine-tune from the Facebook Cityscapes-semantic checkpoint you want
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            args.init_card,
            num_labels=NUM_CLASSES,
            ignore_mismatched_sizes=False,  # head already matches 19 classes
            id2label=id2label,
            label2id=label2id,
        ).to(device)
    elif args.init_mode == "scratch":
        # random weights but with the SAME architecture and Swin-B (IN21k) timm backbone
        cfg = Mask2FormerConfig.from_pretrained(args.init_card)
        cfg.num_labels = NUM_CLASSES
        cfg.ignore_value = IGNORE
        cfg.id2label = id2label
        cfg.label2id = label2id

        # critical bits to avoid the earlier error:
        cfg.use_timm_backbone = True
        # timm model name for IN21k Swin-Base @ 384 (used by Mask2Former Swin-B)
        cfg.backbone = "swin_base_patch4_window12_384_in22k"
        # start from RANDOM init (not pretrained)
        cfg.use_pretrained_backbone = False

        model = Mask2FormerForUniversalSegmentation(cfg).to(device)
    else:
        raise ValueError("--init_mode must be 'pretrained' or 'scratch'")

    return model, processor

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="Cityscapes root with leftImg8bit/ and gtFine/")
    ap.add_argument("--out_dir", type=str, default="./outputs/m2f_city_run")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--crop", type=int, default=768)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--save_every", type=int, default=2000)

    # init options
    ap.add_argument("--init_card", type=str,
                    default="facebook/mask2former-swin-base-IN21k-cityscapes-semantic",
                    help="HF repo to init from")
    ap.add_argument("--init_mode", type=str, choices=["pretrained","scratch"], default="pretrained",
                    help="'pretrained' = load from init_card; 'scratch' = build same arch w/ timm Swin-B from random init")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 1) Model + processor
    model, processor = build_model_and_processor(args, device)

    # 2) Data
    train_tfms = build_transforms(args.crop)
    val_tfms   = et.ExtCompose([])  # no-op; keep PIL for processor

    train_set  = Cityscapes(root=args.data_root, split="train", transform=train_tfms)
    val_set    = Cityscapes(root=args.data_root, split="val",   transform=val_tfms)

    collate    = collate_with_processor(processor)
    train_ld   = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True, drop_last=True,
                            collate_fn=collate)
    val_ld     = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, collate_fn=collate)

    # 3) Optim & sched
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * max(len(train_ld), 1)
    warmup = int(args.warmup_ratio * total_steps) if total_steps > 0 else 0
    sched  = get_cosine_schedule_with_warmup(optim, warmup, total_steps)
    scaler = GradScaler(enabled=True)

    best_miou, global_step, run_loss = 0.0, 0, 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for proc, segmaps, names in train_ld:
            pixel_values = proc["pixel_values"].to(device, non_blocking=True)
            pixel_mask   = proc.get("pixel_mask")
            pixel_mask   = pixel_mask.to(device, non_blocking=True) if pixel_mask is not None else None

            # labels: list of dicts (move to device)
            labels = [{"class_labels": d["class_labels"].to(device, non_blocking=True),
                       "mask_labels":  d["mask_labels"].to(device, non_blocking=True)} for d in proc["labels"]]

            optim.zero_grad(set_to_none=True)
            with autocast(True):
                out  = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            sched.step()

            run_loss += float(loss)
            global_step += 1
            if global_step % args.log_every == 0:
                print(f"[E{epoch}/{args.epochs}  S{global_step}]  loss={run_loss/args.log_every:.4f}")
                run_loss = 0.0
            if global_step % args.save_every == 0:
                torch.save(model.state_dict(), os.path.join(args.out_dir, f"step{global_step}.pt"))

        # ---- Eval per epoch
        results = evaluate(model, processor, val_ld, device)
        miou = results.get("Mean IoU", 0.0)
        print("\n==> Validation:")
        print(StreamSegMetrics.to_str(results))
        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best.pt"))

    print(f"Training done. Best mIoU={best_miou:.4f}")

if __name__ == "__main__":
    main()

