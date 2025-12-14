# Mask2Former under Adverse Weather Conditions

This repository contains the code and experimental setup for my Master's thesis on  
**transformer-based semantic segmentation under adverse weather conditions**.

The work is conceptually based on the DeepLabV3+-pipeline from  
“Semantic Segmentation under Adverse Conditions: A Weather and Nighttime-aware Synthetic
Data-based Approach” (Kerim et al.), but replaces DeepLabV3+ with **Mask2Former** (Swin-L)
and focuses on **real-world** data (Cityscapes + ACDC) instead of synthetic AWSS.

Core ideas:

- Train a **single segmentation model** that remains robust on both clear-weather
  (Cityscapes) and adverse-weather (ACDC) scenes.
- Use **alternate-batch training** on Cityscapes and ACDC instead of separate models.
- Add **Weather-Aware Supervisor (WAS)** and **Time-Aware Supervisor (TAS)** heads for
  auxiliary weather/time classification (low-weight multi-task loss).
- Keep the pipeline **reproducible**, with exported `environment.yml` and
  `requirements.txt`.

---

## Repository Structure

```text
Master-Thesis_semantic-adverse-segmentation-thesis/
├── LICENSE
├── README.md
├── environment.yml          # Conda environment export (used in thesis experiments)
├── requirements.txt         # pip-style package list from the working env
└── mask2former_/            # Main code for Mask2Former thesis experiments
    ├── Scripts_Extra/       # Additional training/eval scripts (ablations, variants)
    ├── artifacts/
    │   └── splits/
    │       └── acdc/
    │           └── acdc_train_85_15_seed1.json   # ACDC 85/15 train/val split
    ├── assets/              # Final thesis figures (bar charts, qualitative examples, etc.)
    ├── checkpoints/         # EXPECTED location for .pth files (ignored by git)
    ├── datasets/            # Cityscapes + ACDC dataset wrappers
    ├── metrics/             # StreamSegMetrics (mIoU, per-class IoU, etc.)
    ├── models/              # Auxiliary WAS/TAS heads
    ├── network/             # Mask2Former wrapper + modeling utilities
    ├── utils/               # Training utils, transforms, schedulers, visualizer
    └── main_mask2former_WAS_TAS_ON_1.0.py  # Main training/eval script for final model


Note: mask2former_/checkpoints/ is ignored in git on purpose (large files).

1. Environment Setup

The experiments were run in a Conda environment named mask2former-adverse with:

Python 3.8

PyTorch + CUDA (GPU required for practical training)

Hugging Face transformers (Mask2Former implementation)

Typical vision stack: numpy, tqdm, opencv-python, Pillow, matplotlib, etc.

1.1. Recreate the thesis environment (recommended)

From the repo root:

# Create the environment from the exported YAML
conda env create -f environment.yml

# Activate it
conda activate mask2former-adverse


This should give you an environment matching the one used for the final experiments
(minor differences are possible depending on your Conda setup).

1.2. Minimal setup via requirements.txt (alternative)

If you prefer, you can install from requirements.txt instead:

conda create -n mask2former-adverse python=3.8
conda activate mask2former-adverse

pip install -r requirements.txt


If CUDA / PyTorch versions complain, install the matching PyTorch build for your GPU/driver
from the official PyTorch site, then re-run pip install -r requirements.txt to fill
the remaining packages.

2. Datasets

This project uses only real-world datasets:

Cityscapes – urban scenes, clear weather, daytime.

ACDC – Adverse Conditions Dataset with Correspondences (rain, fog, snow, night).

2.1. Download & folder structure

You must obtain both datasets from their official project pages.

A typical layout (adapt to your own paths):

/path/to/datasets/
├── cityscapes/
│   ├── leftImg8bit/
│   │   ├── train/
│   │   └── val/
│   └── gtFine/
│       ├── train/
│       └── val/
└── ACDC/
    ├── rgb_anon/
    │   ├── train/
    │   └── val/
    └── gt/
        ├── train/
        └── val/

2.2. Point the code to your dataset paths

The main script reads dataset roots either from:

Environment variables
CS_ROOT and ACDC_ROOT, or

CLI arguments --data_root_cs and --data_root_acdc.

Example (bash):

export CS_ROOT=/path/to/datasets/cityscapes
export ACDC_ROOT=/path/to/datasets/ACDC


In the code, these map to:

opts.data_root_cs   = os.environ.get("CS_ROOT",  "/home/.../datasets/cityscapes")
opts.data_root_acdc = os.environ.get("ACDC_ROOT","/home/.../datasets/ACDC")


You can also hard-code your paths in main_mask2former_WAS_TAS_ON_1.0.py if you prefer,
but environment variables keep things cleaner.

3. Checkpoints (Pretrained Models)

The large .pth files are not stored in this repo. You must either:

Train from scratch (slow but fully reproducible), or

Use externally hosted checkpoints (e.g. Google Drive / Hugging Face / Zenodo).

Expected filenames and locations:

mask2former_/checkpoints/
├── best_mask2former_WAS_TAS_ON_1.0.pth                 # final CS+ACDC model (WAS/TAS)
├── best_mask2former_cityscapes_only_os16.pth           # Cityscapes-only baseline
├── main_mask2former_segonly_acdc_cs_alt_best.pth       # CS+ACDC segmentation-only
└── main_mask2former_WAS_TAS_ON_Disable_WAS_TAS_best.pth  # ablation (heads disabled)


These filenames are referenced directly in the scripts under mask2former_/ and
mask2former_/Scripts_Extra/.

(TODO: add public download links once the checkpoints are uploaded.)

4. Training

All main thesis experiments use:

Mask2Former with a Swin-L backbone (pretrained on Cityscapes),

Alternate-batch training: Cityscapes batch, then ACDC batch, etc.,

Early encoder layers partially frozen on Cityscapes batches,

WAS/TAS heads (weather/time) with very low loss weights (1e-5),

85/15 train/val split on the training part of each dataset,
official validation splits used as test sets.

The central script is:

mask2former_/main_mask2former_WAS_TAS_ON_1.0.py

4.1. Final CS+ACDC model (main thesis run)

From inside mask2former_/:

cd mask2former_

CUDA_VISIBLE_DEVICES=0 \
python main_mask2former_WAS_TAS_ON_1.0.py \
  --mode 0 \
  --dataset cityscapes_ACDC \
  --data_root_cs   "$CS_ROOT" \
  --data_root_acdc "$ACDC_ROOT" \
  --cs_split_strategy train85_val15_only_train_split \
  --cs_holdout_seed 1 \
  --num_classes 19 \
  --output_stride 16 \
  --crop_size 640 \
  --batch_size 1 \
  --val_batch_size 1 \
  --total_itrs 90000 \
  --lr 1e-5 \
  --loss_type cross_entropy \
  --ws_weight 1e-5 \
  --tes_weight 1e-5 \
  --per_condition_val \
  --enable_vis --vis_port 13570 --vis_env main


Key points:

--mode 0 → joint training on Cityscapes + ACDC with alternating batches.

--cs_split_strategy train85_val15_only_train_split
→ 85/15 split of the official training split; official val is kept as a test set.

--ws_weight and --tes_weight control the WAS/TAS auxiliary loss weights.

--per_condition_val enables per-condition metrics on ACDC (rain/fog/snow/night).

Visdom is optional; disable by dropping --enable_vis.

5. Evaluation / Inference

The repository also contains scripts in mask2former_/Scripts_Extra/ for specific
evaluation settings and ablations.

5.1. Segmentation-only CS+ACDC model on ACDC (e.g. night)

Example (seg-only variant):

cd mask2former_

CUDA_VISIBLE_DEVICES=0 \
python Scripts_Extra/main_mask2former_segonly_acdcfirst.py \
  --mode 21 \
  --dataset ACDC \
  --ACDC_test_class night \
  --test_batch_size 3 \
  --num_classes 19 \
  --output_stride 16 \
  --crop_size 640 \
  --ckpt checkpoints/main_mask2former_segonly_acdc_cs_alt_best.pth

5.2. Final WAS+TAS model on full ACDC val (overall)
cd mask2former_

CUDA_VISIBLE_DEVICES=0 \
python main_mask2former_WAS_TAS_ON_1.0.py \
  --mode 21 \
  --dataset ACDC \
  --test_batch_size 3 \
  --num_classes 19 \
  --output_stride 16 \
  --crop_size 640 \
  --ckpt checkpoints/best_mask2former_WAS_TAS_ON_1.0.pth


--mode 21 → ACDC test-only mode.

Results are reported as mIoU over the standard 19-class Cityscapes taxonomy.

Optional 10-class remapping for some experiments is supported via --eval_mode 10
and --labelmap, but that is not required for basic reproduction.

6. Assets (Figures used in the Thesis)

The mask2former_/assets/ folder contains the final figures used in the thesis:

barchart_comparison_deeplabv3plus_vs_mask2former_on_cityscapes.png

barchart_comparison_deeplabv3plus_vs_mask2former_on_acdc.png

figure_qualitative_comparison_between_baselines_and_final_proposed_method.png

table_consolidated_ablation_results_stage_1_to_7_miou_cityscapes_acdc.png

These are not needed to run the code, but they document the final performance and
ablation results as reported in the written thesis.

7. Reproducing the Thesis Experiments (Overview)

High-level list of the main experiments:

Cityscapes-only Mask2Former baseline

Train on Cityscapes train 85%, validate on Cityscapes train 15%.

Evaluate on the official Cityscapes val split (treated as a test set).

Segmentation-only CS+ACDC alternate-batch variant

Training on both datasets with only segmentation loss.

Used for ablation comparisons.

Final CS+ACDC alternate-batch model with WAS+TAS (main thesis model)

1:1 batch alternation: CS, ACDC, CS, ACDC, …

Early encoder layers partially frozen on CS batches to preserve clear-weather features.

WAS/TAS heads trained as auxiliary tasks with low weights (1e-5).

Evaluation on:

Cityscapes val (clear-weather performance),

ACDC val (rain, fog, snow, night, and overall).

Exact hyperparameters are set in the scripts and can be overridden via CLI flags.

8. Acknowledgements

Training strategy and the idea of weather/time awareness originate from:
“Semantic Segmentation under Adverse Conditions” (Kerim et al.) and their
official DeepLabV3+-based implementation.

Mask2Former and the Swin-L backbone are based on the implementations from the
original authors and the Hugging Face transformers library.
