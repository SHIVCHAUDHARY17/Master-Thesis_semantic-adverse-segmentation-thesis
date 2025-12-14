# Mask2Former under Adverse Weather Conditions

This repository contains the code and experimental setup for my Master's thesis on
**transformer-based semantic segmentation under adverse weather conditions**.

The work is inspired by and conceptually related to the DeepLabV3+-based pipeline in  
“Semantic Segmentation under Adverse Conditions: A Weather and Nighttime-aware Synthetic
Data-based Approach” by Kerim et al. :contentReference[oaicite:0]{index=0}  
Here, DeepLabV3+ is replaced with **Mask2Former** (Swin-L backbone), and the focus is on
**real-world adverse conditions** (Cityscapes + ACDC) instead of synthetic AWSS data.

The key ideas are:

- Use a **single segmentation model** that performs robustly on both clear-weather
  (Cityscapes) and adverse-weather (ACDC) scenes.
- **Alternate-batch training** on Cityscapes and ACDC to avoid maintaining separate models.
- Optional **Weather-Aware Supervisor (WAS)** and **Time-Aware Supervisor (TAS)** heads
  for multi-task learning (weather/time classification + segmentation).
- Keep the training/evaluation pipeline reproducible and self-contained.

---

## Repository Structure

```text
Master-Thesis_semantic-adverse-segmentation-thesis/
├── LICENSE
├── README.md              # This file
└── mask2former_/          # Main code for thesis experiments
    ├── Scripts_Extra/     # Additional experiment scripts (variants / ablations)
    ├── artifacts/
    │   └── splits/        # Train/val split definitions (e.g. ACDC 85/15)
    ├── assets/            # Figures used in the thesis (bar charts, qualitative examples, etc.)
    ├── checkpoints/       # EXPECTED location for pretrained .pth files (not tracked in git)
    ├── datasets/          # Dataset wrappers for Cityscapes + ACDC
    ├── metrics/           # StreamSegMetrics (mIoU etc.)
    ├── models/            # Auxiliary heads (WAS/TAS)
    ├── network/           # Mask2Former model and wrapper modules
    ├── utils/             # Training utils, transforms, schedulers, visualizer
    └── main_mask2former_WAS_TAS_ON_1.0.py   # Main training/eval script for final model

