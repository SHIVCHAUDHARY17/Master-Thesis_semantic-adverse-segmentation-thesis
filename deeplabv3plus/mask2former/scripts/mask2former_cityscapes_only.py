# scripts/mask2former_cityscapes_only.py
import os, sys, argparse, random, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# Your dataset and utils
from datasets.cityscapes_baseline import Cityscapes
from utils import ext_transforms as et
from metrics.stream_metrics import StreamSegMetrics

from transformers import (
    AutoImageProcessor,
    Mask2FormerConfig,
    Mask2FormerForUniversalSegmentation,
    get_cosine_schedule_with_warmup,
)

# -------------------- constants --------------------
NUM_CLASSES = 19
IGNORE = 255
CITYSCAPES_NAMES = [
    "road","sidewalk","building","wall","fence","pole","traffic light","traffic sign",
    "vegetation","terrain","sky","person","rider","car","truck","bus","train","motorcycle","bicycle"
]

# -------------------- utils --------------------
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def build_transforms(crop_h: int, crop_w: int):
    """Only use ops known to exist in your ext_transforms."""
    return et.ExtCompose([
        et.ExtRandomHorizontalFlip(),
        et.ExtColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        et.ExtRandomCrop(size=(crop_h, crop_w)),
    ])

def val_transforms():
    return et.ExtCompose([])

def build_labels_from_segmap(seg_np: np.ndarray):
    """
    Convert a HxW trainId map into Mask2Former training targets:
      - class_labels: LongTensor [K]
      - mask_labels:  FloatTensor [K, H, W] with 0/1 values
    (Use float here because grid_sample requires floating-point input.)
    """
    seg_np = seg_np.astype("int64", copy=False)
    seg_t = torch.from_numpy(seg_np)                    # [H,W], int64
    classes = torch.unique(seg_t)
    # keep valid classes only (0..18), drop IGNORE (255)
    classes = classes[(classes != IGNORE) & (classes >= 0) & (classes < NUM_CLASSES)]

    if classes.numel() == 0:
        # Ensure at least one entry; zero mask so the loss stays defined
        classes = torch.tensor([0], dtype=torch.long)
        masks = torch.zeros((1, seg_t.shape[0], seg_t.shape[1]), dtype=torch.float32)  # [1,H,W]
    else:
        masks_bool = torch.stack([(seg_t == c) for c in classes], dim=0)               # [K,H,W] bool
        masks = masks_bool.to(dtype=torch.float32)                                     # float32 0/1

    return {"class_labels": classes.to(dtype=torch.long), "mask_labels": masks}

def collate_with_processor(processor):
    """
    Dataset item expected: (PIL image, trainId label tensor [H,W], name, *_)
    We ONLY use the processor for pixel_values; we ALWAYS build labels ourselves
    so empty/ignore-only crops don't crash.
    """
    def _fn(batch):
        imgs, lbls, names, *_ = zip(*batch)
        # Ensure numpy int64 segmaps (0..18 valid, 255 ignore)
        segmaps = [np.asarray(l.cpu(), dtype="int64") for l in lbls]

        # Get pixel_values only; DO NOT pass segmentation_maps here
        proc = processor(
            images=list(imgs),
            return_tensors="pt",
            do_resize=False,
        )

        # Always build labels ourselves (robust to empty/ignore-only crops)
        proc["labels"] = [build_labels_from_segmap(s) for s in segmaps]

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

        # Resize predictions to the label sizes (H, W) for correct metrics
        target_sizes = [(s.shape[0], s.shape[1]) for s in segmaps]
        preds = processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)

        # metrics expect lists of HxW int
        label_trues = segmaps
        label_preds = [p.cpu().numpy().astype("int64") for p in preds]
        metrics.update(label_trues, label_preds)
    return metrics.get_results()

def build_model_and_processor(args, device):
    processor = AutoImageProcessor.from_pretrained(args.init_card)
    # Keep these aligned with Cityscapes trainIds
    processor.num_labels = NUM_CLASSES
    processor.ignore_index = IGNORE
    processor.do_reduce_labels = False

    id2label = {i: n for i, n in enumerate(CITYSCAPES_NAMES)}
    label2id = {v: k for k, v in id2label.items()}

    if args.init_mode == "pretrained":
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            args.init_card,
            num_labels=NUM_CLASSES,
            id2label=id2label,
            label2id=label2id,
        ).to(device)
    elif args.init_mode == "scratch":
        cfg = Mask2FormerConfig.from_pretrained(args.init_card)
        cfg.num_labels = NUM_CLASSES
        cfg.ignore_value = IGNORE
        cfg.id2label = id2label
        cfg.label2id = label2id
        cfg.use_timm_backbone = True
        cfg.backbone = "swin_base_patch4_window12_384_in22k"
        cfg.use_pretrained_backbone = False
        model = Mask2FormerForUniversalSegmentation(cfg).to(device)
    else:
        raise ValueError(f"Unknown init mode: {args.init_mode}")

    return model, processor

def save_checkpoints(args, model, optim, sched, scaler, best=False, epoch=None):
    # Always keep inference-friendly weights:
    if best:
        torch.save(model.state_dict(), os.path.join(args.out_dir, "best.pt"))
    else:
        torch.save(model.state_dict(), os.path.join(args.out_dir, "last.pt"))
    # Save full training state for robust resume (overwrites each epoch)
    state = {
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "sched": sched.state_dict(),
        "scaler": scaler.state_dict() if isinstance(scaler, GradScaler) else None,
        "epoch": epoch,
    }
    torch.save(state, os.path.join(args.out_dir, "last_state.pt"))

def try_load_resume(args, model, optim, sched, scaler, device):
    """
    Returns (start_epoch, resumed) and optionally loads model/optim/sched/scaler.
    Supports two cases:
      - args.resume_from points to last_state.pt  -> full-state resume
      - args.resume_from points to a .pt weights -> weights-only resume
    """
    start_epoch = 1
    resumed = False
    if not args.resume_from:
        return start_epoch, resumed

    ckpt = args.resume_from
    if not os.path.exists(ckpt):
        print(f"[resume] File not found: {ckpt} — skipping resume.")
        return start_epoch, resumed

    try:
        state = torch.load(ckpt, map_location="cpu")
        if isinstance(state, dict) and "model" in state and "optim" in state:
            print(f"[resume] Loading FULL training state from: {ckpt}")
            model.load_state_dict(state["model"], strict=False)
            try:
                optim.load_state_dict(state["optim"])
                sched.load_state_dict(state["sched"])
                if state.get("scaler") and isinstance(scaler, GradScaler):
                    scaler.load_state_dict(state["scaler"])
            except Exception as e:
                print(f"[resume] Optim/sched/scaler restore failed ({e}), continuing with fresh states.")
            start_epoch = int(state.get("epoch", 0)) + 1
            resumed = True
            return start_epoch, resumed
    except Exception:
        pass

    # Weights-only path (e.g., last.pt or best.pt)
    print(f"[resume] Loading MODEL WEIGHTS from: {ckpt}")
    state = torch.load(ckpt, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[resume] missing keys: {len(missing)}  unexpected: {len(unexpected)}")
    # For weights-only resume, suggest lower LR / zero warmup in CLI
    resumed = True
    start_epoch = 1  # starting a new phase
    return start_epoch, resumed

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="./outputs/m2f_city_run")
    ap.add_argument("--epochs", type=int, default=40,
                    help="TOTAL epochs to run in this invocation")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--warmup_ratio", type=float, default=0.1,
                    help="set to 0.0 when resuming with weights-only")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--crop_h", type=int, default=768)
    ap.add_argument("--crop_w", type=int, default=768)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--accum_steps", type=int, default=1)
    ap.add_argument("--init_mode", type=str, choices=["pretrained","scratch"], default="pretrained")
    ap.add_argument("--init_card", type=str, default="facebook/mask2former-swin-base-IN21k-cityscapes-semantic")
    # NEW: resume support
    ap.add_argument("--resume_from", type=str, default="",
                    help="path to last_state.pt (full resume) or last/best .pt (weights-only)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 1) model + processor
    model, processor = build_model_and_processor(args, device)

    # 2) data
    train_tfms = build_transforms(args.crop_h, args.crop_w)
    val_tfms   = val_transforms()

    train_set  = Cityscapes(root=args.data_root, split="train", transform=train_tfms)
    val_set    = Cityscapes(root=args.data_root, split="val",   transform=val_tfms)

    collate    = collate_with_processor(processor)
    train_ld   = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        collate_fn=collate
    )
    val_ld     = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate
    )

    # 3) optim & schedule
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * max(len(train_ld), 1)
    warmup = int(args.warmup_ratio * total_steps) if total_steps > 0 else 0
    sched  = get_cosine_schedule_with_warmup(optim, warmup, total_steps)
    scaler = GradScaler(enabled=True)

    # ---- Resume (full-state or weights-only) ----
    start_epoch, resumed = try_load_resume(args, model, optim, sched, scaler, device)
    if resumed:
        if args.resume_from.endswith(".pt") and "last_state" not in args.resume_from:
            print("[resume] Weights-only resume. Consider a smaller LR and warmup_ratio=0.0 for this phase.")

    best_miou, global_step, run_loss = 0.0, 0, 0.0

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        for proc, segmaps, names in train_ld:
            pixel_values = proc["pixel_values"].to(device, non_blocking=True)
            pixel_mask   = proc.get("pixel_mask")
            pixel_mask   = pixel_mask.to(device, non_blocking=True) if pixel_mask is not None else None

            # Split into the two lists the model expects
            class_labels = [d["class_labels"].to(device, non_blocking=True) for d in proc["labels"]]
            mask_labels  = [d["mask_labels"].to(device, non_blocking=True)  for d in proc["labels"]]

            # 🔍 Debug—only on very first step of this run
            if epoch == start_epoch and global_step == 0:
                print("classes[0]:", class_labels[0].shape, class_labels[0].dtype)
                print("masks[0]:",  mask_labels[0].shape,  mask_labels[0].dtype)

            loss_scale = 1.0 / max(args.accum_steps, 1)

            with autocast(True):
                out  = model(pixel_values=pixel_values,
                             pixel_mask=pixel_mask,
                             class_labels=class_labels,
                             mask_labels=mask_labels)
                loss = out.loss * loss_scale

            scaler.scale(loss).backward()
            if (global_step + 1) % args.accum_steps == 0:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                sched.step()

            run_loss += float(loss); global_step += 1
            if global_step % args.log_every == 0:
                print(f"[E{epoch}/{args.epochs}  S{global_step}]  loss={run_loss/args.log_every:.4f}")
                run_loss = 0.0

        # ---- Eval per epoch (resize preds to label size)
        results = evaluate(model, processor, val_ld, device)
        miou = results.get("Mean IoU", 0.0)
        print("\n==> Validation:")
        print(StreamSegMetrics.to_str(results))

        # Save BEST (on improvement)
        if miou > best_miou:
            best_miou = miou
            save_checkpoints(args, model, optim, sched, scaler, best=True, epoch=epoch)
            print(f"[✓] New best mIoU: {best_miou:.4f} — saved best.pt and last_state.pt")
        else:
            # Still save latest state each epoch
            save_checkpoints(args, model, optim, sched, scaler, best=False, epoch=epoch)
            print("[i] Saved last.pt and last_state.pt (latest model & state)")

    print(f"Training done. Best mIoU={best_miou:.4f}")

if __name__ == "__main__":
    main()

