# Transformer-Based Semantic Segmentation for Autonomous Driving in Adverse Weather

This repository contains the code and experimental setup for my Master’s thesis:

> **“Transformer-Based Semantic Segmentation for Autonomous Driving in Adverse Weather Conditions Using Weather- and Time-Aware Supervision”**  
> Technische Hochschule Ingolstadt – International Automotive Engineering  
> Author: **Shiv Chaudhary**

The work is conceptually based on the DeepLabV3+-based pipeline of  
**“Semantic Segmentation under Adverse Conditions: A Weather- and Nighttime-aware Synthetic Data-based Approach”** (Kerim et al.)  
but replaces DeepLabV3+ with **Mask2Former (Swin-L)** and focuses on **real-world** adverse conditions (Cityscapes + ACDC), not synthetic AWSS.

Core ideas:

- Train **one Mask2Former model** that stays robust across:
  - clear-weather scenes (Cityscapes)
  - adverse conditions (ACDC: rain, fog, snow, night)
- Use **alternate-batch training**: CS, ACDC, CS, ACDC, …  
- Use **Weather-Aware Supervisor (WAS)** and **Time-Aware Supervisor (TAS)** heads for multi-task learning (weather & time-of-day classification).
- Apply **selective encoder freezing** to preserve clear-weather features while adapting to adverse conditions.

---

## Repository structure

```text
Master-Thesis_semantic-adverse-segmentation-thesis/
├── LICENSE
├── README.md
├── environment.yml          # Conda environment (exported from working setup)
├── requirements.txt         # pip-style dependencies (same env as above)
└── mask2former_/            # Main thesis code
    ├── Scripts_Extra/       # Extra experiment scripts / ablations
    ├── artifacts/
    │   └── splits/
    ├── assets/              # Figures used in the thesis (bar charts, samples, etc.)
    ├── checkpoints/         # EXPECTED location for .pth files (ignored by git)
    ├── datasets/            # Dataset wrappers (Cityscapes + ACDC)
    ├── metrics/             # StreamSegMetrics (mIoU, etc.)
    ├── models/              # WAS/TAS auxiliary heads
    ├── network/             # Mask2Former model wrapper
    ├── utils/               # Transforms, losses, schedulers, visualizer
    └── main_mask2former_WAS_TAS_ON_1.0.py   # Main training / eval script
Note: mask2former_/checkpoints/ is intentionally ignored in .gitignore because the files are large.
You must download checkpoints separately (see below) or train from scratch.



1. Environment setup

The experiments were run in a Conda environment with:

Python 3.8

PyTorch with CUDA (GPU strongly recommended)

Hugging Face transformers (Mask2Former implementation)

Typical vision libs (numpy, tqdm, opencv-python, Pillow, matplotlib, etc.)

You can re-create the exact environment from the exported files:

# Option A: Conda from environment.yml (recommended)
conda env create -f environment.yml
conda activate mask2former-adverse  # or the name inside the yml

# Option B: Create manually + pip from requirements.txt
conda create -n mask2former-adverse python=3.8
conda activate mask2former-adverse

pip install -r requirements.txt


If you want a minimal manual install instead of using the full exports:

# Example (CUDA command should match your system)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install transformers timm tqdm opencv-python pillow matplotlib





2. Datasets

This project uses only real-world datasets:

Cityscapes – clear-weather urban scenes (daytime)

ACDC – Adverse Conditions Dataset with Correspondences (rain, fog, snow, night)

2.1 Download

You must request both datasets from their official websites and agree to their licenses.

After extraction, a typical layout might be:

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

2.2 Tell the code where the datasets are

The main script uses explicit paths for Cityscapes and ACDC:

opts.data_root_cs → Cityscapes root

opts.data_root_acdc → ACDC root

You can either:

Set environment variables:

export CS_ROOT=/path/to/datasets/cityscapes
export ACDC_ROOT=/path/to/datasets/ACDC


and then run the training command (see next section), or

Pass them as CLI arguments:

--data_root_cs /path/to/datasets/cityscapes \
--data_root_acdc /path/to/datasets/ACDC


The dataset code expects 19 Cityscapes classes and applies compatible mappings for ACDC.





3. Checkpoints (pretrained models)

Large .pth files are hosted on Google Drive:

Google Drive folder (all checkpoints):
https://drive.google.com/drive/folders/1NJkVJxFCGNzdk71pG4HeZQxVVC3VWS02?usp=drive_link

Make sure the folder is shared as “Anyone with the link – Viewer”.

3.1 Expected filenames

Download the following files from the Drive folder and place them in:

mask2former_/checkpoints/


Expected files:

best_mask2former_WAS_TAS_ON_1.0.pth
– Final CS+ACDC model with WAS/TAS (main thesis model)

best_mask2former_cityscapes_only_os16.pth
– Cityscapes-only Mask2Former baseline

main_mask2former_segonly_acdc_cs_alt_best.pth
– Segmentation-only CS+ACDC model (no WAS/TAS)

main_mask2former_WAS_TAS_ON_Disable_WAS_TAS_best.pth
– Ablation model with WAS/TAS turned off during training

best_mask2former_WAS_TAS_ON_1.2.pth
– Variant with modified WAS/TAS weighting (used in ablations)

You can also train everything from scratch, but it is time-consuming.






4. Training

The main script for the final CS+ACDC model is:

mask2former_/main_mask2former_WAS_TAS_ON_1.0.py


It supports different --mode values; for the full alternate-batch CS+ACDC training with WAS/TAS, a typical run is:

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


Key behaviour:

Mode 0 → 1:1 alternate-batch training on Cityscapes + ACDC.

Encoder freezing strategy:

ACDC batches: full encoder trainable.

Cityscapes batches: low-level encoder blocks are frozen; higher layers stay trainable.

WAS/TAS:

Weather and time-of-day heads are trained as auxiliary tasks.

Loss weights ws_weight and tes_weight are small (e.g. 1e-5) so segmentation dominates.





5. Evaluation / inference

Additional scripts under mask2former_/Scripts_Extra/ are used for ablations and test-only runs.

5.1 Example – ACDC segmentation-only model (night split)
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

5.2 Example – Final WAS+TAS model on ACDC (all conditions)
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


The evaluation prints mean IoU (mIoU) and per-class IoUs using the standard 19-class Cityscapes taxonomy.
Optional configurations in the script also allow 10-class remapping for the subset used in the thesis.






6. Assets (figures used in the thesis)

The folder mask2former_/assets/ contains the final figures that appear in the thesis:

barchart_comparison_deeplabv3plus_vs_mask2former_on_cityscapes.png

barchart_comparison_deeplabv3plus_vs_mask2former_on_acdc.png

figure_qualitative_comparison_between_baselines_and_final_proposed_method.png

table_consolidated_ablation_results_stage_1_to_7_miou_cityscapes_acdc.png

They are not required to run the code, but they document the final performance comparisons and ablation results.

7. Reproducing the main thesis experiments

High-level summary of the main setups:

Cityscapes-only baseline (Mask2Former)

Train on Cityscapes train (85%) / internal val (15%).

Evaluate on official Cityscapes val (treated as test set in the thesis).

Segmentation-only CS+ACDC variants

Use scripts under Scripts_Extra/ (e.g. main_mask2former_segonly_acdcfirst.py).

No WAS/TAS supervision, only segmentation loss.

Final CS+ACDC alternate-batch model with WAS+TAS (main contribution)

Alternate-batch schedule CS / ACDC.

Selective encoder freezing (low-level blocks frozen on CS batches).

WAS/TAS heads trained with small loss weights as auxiliary tasks.

Evaluation on:

Cityscapes val (clear weather)

ACDC val (rain, fog, snow, night, overall)

All hyperparameters (iterations, LR, crop size, batch size, etc.) are defined in the scripts and configurable via command-line flags.









8. Acknowledgements

Training strategy and the idea of weather-/time-aware supervision are inspired by:

Kerim et al., “Semantic Segmentation under Adverse Conditions: A Weather- and Nighttime-aware Synthetic Data-based Approach.”

and their official DeepLabV3+ implementation:
https://github.com/lsmcolab/Semantic-Segmentation-under-Adverse-Conditions

Mask2Former and the Swin-L backbone build upon the implementations by the original authors and the Hugging Face ecosystem.
