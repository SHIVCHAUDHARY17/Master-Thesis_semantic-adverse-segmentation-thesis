# scripts/infer_cityscapes.py
import os, sys, argparse, glob
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm

from transformers import AutoImageProcessor, Mask2FormerConfig, Mask2FormerForUniversalSegmentation

CITYSCAPES_PALETTE = [
    (128, 64,128),  # road
    (244, 35,232),  # sidewalk
    (70,  70, 70),  # building
    (102,102,156),  # wall
    (190,153,153),  # fence
    (153,153,153),  # pole
    (250,170, 30),  # traffic light
    (220,220,  0),  # traffic sign
    (107,142, 35),  # vegetation
    (152,251,152),  # terrain
    (70, 130,180),  # sky
    (220, 20, 60),  # person
    (255,  0,  0),  # rider
    (0,   0, 142),  # car
    (0,   0,  70),  # truck
    (0,  60, 100),  # bus
    (0,  80, 100),  # train
    (0,   0, 230),  # motorcycle
    (119, 11, 32),  # bicycle
]

def colorize_label(lbl: np.ndarray) -> Image.Image:
    """
    lbl: HxW int32 in [0..18]
    returns: PIL 'P' image with Cityscapes palette applied
    """
    assert lbl.ndim == 2
    pal_img = Image.fromarray(lbl.astype(np.uint8), mode="P")
    # build flat palette of length 768 (256*3)
    palette = []
    for i in range(256):
        if i < len(CITYSCAPES_PALETTE):
            palette += list(CITYSCAPES_PALETTE[i])
        else:
            palette += [0, 0, 0]
    pal_img.putpalette(palette)
    return pal_img

def overlay_on_image(img: Image.Image, lbl: np.ndarray, alpha=0.5) -> Image.Image:
    """
    img: RGB PIL
    lbl: HxW int32 0..18
    """
    color = colorize_label(lbl).convert("RGBA")
    base = img.convert("RGBA")
    # make color semi-transparent
    o = Image.new("RGBA", color.size, (0,0,0,0))
    o = Image.alpha_composite(o, color)
    # adjust alpha
    r, g, b, a = o.split()
    a = a.point(lambda p: int(alpha*255))
    o = Image.merge("RGBA", (r, g, b, a))
    return Image.alpha_composite(base, o).convert("RGB")

def collect_images(input_path: str):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")
    if os.path.isdir(input_path):
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(input_path, e)))
        files.sort()
        return files
    elif os.path.isfile(input_path):
        return [input_path]
    else:
        raise FileNotFoundError(f"No such file or directory: {input_path}")

def load_model_and_processor(init_card: str, num_labels: int, ckpt_path: str, device: torch.device):
    processor = AutoImageProcessor.from_pretrained(init_card)
    # Keep consistent with training
    processor.num_labels = num_labels
    processor.ignore_index = 255
    processor.do_reduce_labels = False

    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        init_card, num_labels=num_labels
    )
    if ckpt_path.endswith(".pt"):
        state = torch.load(ckpt_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print("[load_state_dict] missing keys:", len(missing), "unexpected:", len(unexpected))
    else:
        # If you saved via save_pretrained(folder), point ckpt_path to that folder
        model = Mask2FormerForUniversalSegmentation.from_pretrained(ckpt_path)

    model.to(device).eval()
    return model, processor

@torch.no_grad()
def predict_image(model, processor, img_pil: Image.Image, device: torch.device):
    # Let processor handle resizing as it sees fit; we’ll map back to original size after.
    inputs = processor(images=img_pil, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    pixel_mask = inputs.get("pixel_mask")
    pixel_mask = pixel_mask.to(device) if pixel_mask is not None else None

    outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
    target_sizes = [img_pil.size[::-1]]  # (H,W) = (orig_h, orig_w)
    seg = processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)[0]
    # seg is a torch.LongTensor HxW with ids 0..num_labels-1
    return seg.cpu().numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init_card", type=str, default="facebook/mask2former-swin-base-IN21k-cityscapes-semantic",
                    help="same card you used during training")
    ap.add_argument("--ckpt_path", type=str, required=True,
                    help="path to best.pt (state_dict) OR a folder saved with save_pretrained")
    ap.add_argument("--input", type=str, required=True,
                    help="image file or directory of images")
    ap.add_argument("--out_dir", type=str, default="./outputs/infer",
                    help="where to save predictions")
    ap.add_argument("--save_overlay", action="store_true", help="also save RGB overlay on source image")
    ap.add_argument("--alpha", type=float, default=0.5, help="overlay transparency")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    model, processor = load_model_and_processor(args.init_card, num_labels=19, ckpt_path=args.ckpt_path, device=device)

    files = collect_images(args.input)
    print(f"Found {len(files)} image(s). Saving to: {args.out_dir}")

    for fp in tqdm(files):
        img = Image.open(fp).convert("RGB")
        pred = predict_image(model, processor, img, device)  # HxW int

        # Save raw label map (0..18)
        base = os.path.splitext(os.path.basename(fp))[0]
        raw_path = os.path.join(args.out_dir, f"{base}_predIds.png")
        Image.fromarray(pred.astype(np.uint8)).save(raw_path)

        # Optional: save a colorized overlay for quick viewing
        if args.save_overlay:
            ov = overlay_on_image(img, pred, alpha=args.alpha)
            ov_path = os.path.join(args.out_dir, f"{base}_overlay.jpg")
            ov.save(ov_path, quality=95)

    print("Done.")

if __name__ == "__main__":
    main()

