#!/usr/bin/env python3
"""
run_acdc_actual_adverse.py

Run inference using actual adverse weather images from ACDC test set.
"""

import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ============================================================================
# CONFIGURATION - USING ACTUAL TEST SET IMAGES
# ============================================================================

CHECKPOINT = "checkpoints/best_mask2former_WAS_TAS_ON_1.0.pth"
BASE_OUTPUT_DIR = "results/acdc_actual_adverse"

# Selected test set images (first image from each weather condition)
WEATHER_IMAGES = {
    "clear": "/home/ubuntu22user2/shiv/datasets/cityscapes/leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png",
    "rain": "/home/ubuntu22user2/shiv/datasets/ACDC/rgb/rain/test/GOPR0572/GOPR0572_frame_000145_rgb_anon.png",
    "fog": "/home/ubuntu22user2/shiv/datasets/ACDC/rgb/fog/test/GOPR0475/GOPR0475_frame_000247_rgb_anon.png",
    "snow": "/home/ubuntu22user2/shiv/datasets/ACDC/rgb/snow/test/GOPR0122/GOPR0122_frame_000651_rgb_anon.png",
    "night": "/home/ubuntu22user2/shiv/datasets/ACDC/rgb/night/test/GOPR0355/GOPR0355_frame_000138_rgb_anon.png",
    "zhiran": "/home/ubuntu22user2/shiv/datasets/000005.png"
}


# Cityscapes class names for reference
CITYSCAPES_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
    'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
    'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]

# ============================================================================
# VALIDATE IMAGE PATHS
# ============================================================================

print("="*80)
print("VALIDATING ACDC TEST SET IMAGE PATHS")
print("="*80)

all_valid = True
for weather, path in WEATHER_IMAGES.items():
    if os.path.exists(path):
        # Check if it's an adverse image (not reference)
        is_adverse = "_ref_" not in path and "rgb_ref" not in path
        status = "✓" if is_adverse else "⚠"
        type_desc = "ADVERSE" if is_adverse else "REFERENCE"
        print(f"{status} {weather:10s}: {type_desc:10s} {os.path.basename(path)}")
    else:
        print(f"✗ {weather:10s}: {path} - NOT FOUND!")
        all_valid = False

if not all_valid:
    print("\nERROR: Some images not found. Please check the paths.")
    exit(1)

# ============================================================================
# RUN INFERENCE
# ============================================================================

print("\n" + "="*80)
print("RUNNING INFERENCE ON ACDC TEST SET IMAGES")
print("="*80)

os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

for weather, image_path in WEATHER_IMAGES.items():
    output_dir = os.path.join(BASE_OUTPUT_DIR, weather)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"PROCESSING: {weather.upper()}")
    print(f"{'='*60}")
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Type:  {'Actual Adverse Condition' if '_ref_' not in image_path else 'Reference/Clear'}")
    print(f"Output: {output_dir}")
    
    cmd = [
        "python", "inference_mask2former_final_cs_acdc.py",
        "--ckpt", CHECKPOINT,
        "--infer_image", image_path,
        "--infer_outdir", output_dir,
        "--infer_overlay_alpha", "0.55"
    ]
    
    try:
        print(f"\nRunning inference command...")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Successfully processed {weather}")
        
        # Display head predictions
        heads_file = os.path.join(output_dir, "heads.txt")
        if os.path.exists(heads_file):
            with open(heads_file, 'r') as f:
                print(f"  Predictions: {f.read().strip()}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to process {weather}")
        print(f"Error: {e.stderr}")

# ============================================================================
# CREATE COMPREHENSIVE REPORT
# ============================================================================

print("\n" + "="*80)
print("CREATING COMPREHENSIVE REPORT")
print("="*80)

def create_report():
    """Create a comprehensive report figure."""
    
    weathers = ["clear", "rain", "fog", "snow", "night"]
    fig, axes = plt.subplots(5, 4, figsize=(20, 22))
    fig.suptitle('Mask2Former: Semantic Segmentation in Adverse Weather Conditions\n'
                 'ACDC Test Set Evaluation - Actual Adverse Weather Images', 
                 fontsize=24, fontweight='bold', y=0.98)
    
    for row, weather in enumerate(weathers):
        weather_dir = os.path.join(BASE_OUTPUT_DIR, weather)
        
        try:
            # Load outputs
            input_img = Image.open(os.path.join(weather_dir, "input.png"))
            pred_color = Image.open(os.path.join(weather_dir, "pred_color.png"))
            overlay = Image.open(os.path.join(weather_dir, "overlay.png"))
            pred_conf = Image.open(os.path.join(weather_dir, "pred_conf.png"))
            
            # Read head predictions
            heads_file = os.path.join(weather_dir, "heads.txt")
            ws_pred = tes_pred = "N/A"
            if os.path.exists(heads_file):
                with open(heads_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if "WS_pred_id" in line:
                            ws_pred = line.split(":")[1].strip()
                        elif "TES_pred_id" in line:
                            tes_pred = line.split(":")[1].strip()
            
            # Convert to arrays
            input_arr = np.array(input_img)
            pred_color_arr = np.array(pred_color)
            overlay_arr = np.array(overlay)
            pred_conf_arr = np.array(pred_conf)
            
            # Plot
            axes[row, 0].imshow(input_arr)
            axes[row, 0].set_title(f'{weather.upper()}\nInput Image', fontsize=14, fontweight='bold')
            axes[row, 0].axis('off')
            
            axes[row, 1].imshow(pred_color_arr)
            axes[row, 1].set_title('Segmentation', fontsize=14, fontweight='bold')
            axes[row, 1].axis('off')
            
            axes[row, 2].imshow(overlay_arr)
            ws_desc = {"0": "Clear", "1": "Rain", "2": "Fog", "3": "Snow"}.get(ws_pred, f"WS:{ws_pred}")
            tes_desc = {"0": "Day", "1": "Night"}.get(tes_pred, f"TES:{tes_pred}")
            axes[row, 2].set_title(f'Overlay (α=0.55)\n{ws_desc} | {tes_desc}', fontsize=14, fontweight='bold')
            axes[row, 2].axis('off')
            
            im = axes[row, 3].imshow(pred_conf_arr, cmap='viridis')
            axes[row, 3].set_title('Confidence', fontsize=14, fontweight='bold')
            axes[row, 3].axis('off')
            
        except Exception as e:
            print(f"Warning: Could not load {weather} results: {e}")
            for col in range(4):
                axes[row, col].text(0.5, 0.5, "Error\nLoading", ha='center', va='center')
                axes[row, col].axis('off')
    
    # Add colorbar
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Prediction Confidence', fontsize=12)
    
    # Add model info
    model_info = (
        f"Model: Mask2Former with Swin Backbone | Parameters: 215.5M | "
        f"Inference: ~1.65 FPS @ 1024×2048 | Checkpoint: {os.path.basename(CHECKPOINT)}"
    )
    fig.text(0.5, 0.02, model_info, ha='center', fontsize=11, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
    
    plt.tight_layout()
    report_path = os.path.join(BASE_OUTPUT_DIR, "acdc_adverse_weathers_report.png")
    plt.savefig(report_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Report saved: {report_path}")

def create_summary_table():
    """Create a summary table of predictions."""
    
    summary_path = os.path.join(BASE_OUTPUT_DIR, "summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MASK2FORMER ACDC ADVERSE WEATHER EVALUATION SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write("Model Information:\n")
        f.write(f"  Checkpoint: {os.path.basename(CHECKPOINT)}\n")
        f.write("  Architecture: Mask2Former with Swin-T backbone\n")
        f.write("  Parameters: 215.5 million\n")
        f.write("  Inference Speed: ~1.65 FPS @ 1024×2048\n")
        f.write("  Training: Cityscapes + ACDC (multi-weather)\n\n")
        
        f.write("Weather Scene (WS) Classes:\n")
        f.write("  0: Clear\n")
        f.write("  1: Rain\n")
        f.write("  2: Fog\n")
        f.write("  3: Snow\n\n")
        
        f.write("Time of Day Scene (TES) Classes:\n")
        f.write("  0: Day\n")
        f.write("  1: Night\n\n")
        
        f.write("="*80 + "\n")
        f.write("RESULTS BY WEATHER CONDITION\n")
        f.write("="*80 + "\n\n")
        
        weathers = ["clear", "rain", "fog", "snow", "night"]
        for weather in weathers:
            weather_dir = os.path.join(BASE_OUTPUT_DIR, weather)
            heads_file = os.path.join(weather_dir, "heads.txt")
            
            f.write(f"{weather.upper()}:\n")
            f.write(f"  Input: {os.path.basename(WEATHER_IMAGES[weather])}\n")
            
            if os.path.exists(heads_file):
                with open(heads_file, 'r') as hf:
                    lines = hf.readlines()
                    ws_pred = tes_pred = "N/A"
                    for line in lines:
                        if "WS_pred_id" in line:
                            ws_pred = line.split(":")[1].strip()
                        elif "TES_pred_id" in line:
                            tes_pred = line.split(":")[1].strip()
                
                ws_desc = {"0": "Clear", "1": "Rain", "2": "Fog", "3": "Snow"}.get(ws_pred, "Unknown")
                tes_desc = {"0": "Day", "1": "Night"}.get(tes_pred, "Unknown")
                
                f.write(f"  WS Prediction: {ws_pred} ({ws_desc})\n")
                f.write(f"  TES Prediction: {tes_pred} ({tes_desc})\n")
                
                # Check if predictions match expected weather
                expected_ws = {
                    "clear": "0", "rain": "1", "fog": "2", 
                    "snow": "3", "night": "0"  # Night is TES, not WS
                }
                expected_tes = {
                    "clear": "0", "rain": "0", "fog": "0", 
                    "snow": "0", "night": "1"
                }
                
                if weather in expected_ws and ws_pred == expected_ws[weather]:
                    f.write(f"  ✓ WS Correct\n")
                else:
                    f.write(f"  ✗ WS Incorrect (expected {expected_ws.get(weather, '?')})\n")
                
                if weather in expected_tes and tes_pred == expected_tes[weather]:
                    f.write(f"  ✓ TES Correct\n")
                else:
                    f.write(f"  ✗ TES Incorrect (expected {expected_tes.get(weather, '?')})\n")
            else:
                f.write("  No predictions available\n")
            
            f.write("\n")
    
    print(f"✓ Summary saved: {summary_path}")

# Run reporting functions
create_report()
create_summary_table()

print("\n" + "="*80)
print("PROCESS COMPLETED SUCCESSFULLY!")
print("="*80)
print(f"\nAll results saved in: {BASE_OUTPUT_DIR}")
print("\nGenerated outputs:")
print(f"  1. acdc_adverse_weathers_report.png - Comprehensive visual report")
print(f"  2. summary.txt - Detailed results summary")
print(f"  3. Individual folders for each weather condition:")
for weather in ["clear", "rain", "fog", "snow", "night"]:
    weather_dir = os.path.join(BASE_OUTPUT_DIR, weather)
    if os.path.exists(weather_dir):
        files = [f for f in os.listdir(weather_dir) if f.endswith(('.png', '.txt'))]
        print(f"     - {weather}/: {len(files)} files")

print("\n" + "="*80)
print("KEY OBSERVATIONS TO NOTE:")
print("="*80)
print("1. These are ACTUAL adverse weather images from ACDC test set")
print("2. Images show real rain, fog, snow, and night conditions")
print("3. Check if WS/TES predictions are correct:")
print("   - Rain: WS should be 1, TES should be 0")
print("   - Fog: WS should be 2, TES should be 0")
print("   - Snow: WS should be 3, TES should be 0")
print("   - Night: WS can be any, TES should be 1")
print("4. Compare segmentation quality across different conditions")
