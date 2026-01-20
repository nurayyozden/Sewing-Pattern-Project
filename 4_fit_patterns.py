import json
import os
from PIL import Image

# Path to saved garment area and scale results
results_path = "garment_area_results.json"
patterns_dir = "sewing_patterns"  # your folder with pattern PNGs
output_dir = "scaled_patterns"
os.makedirs(output_dir, exist_ok=True)

# Load scale data from previous step
with open(results_path, 'r') as f:
    scale_data = json.load(f)

# Choose which garment image to fit patterns to (key in JSON)
garment_key = list(scale_data.keys())[0]  # for example, just take first

pixels_per_inch = scale_data[garment_key]['pixels_per_inch']
print(f"Using scale for {garment_key}: {pixels_per_inch:.2f} pixels/inch")

# Example: Load all pattern pieces, resize to desired physical size (in inches)
# Here you must specify each pattern piece's target real-world size in inches,
# for example by measuring the original design or based on garment size.

# Example dictionary of pattern piece widths in inches (you can extend this)
pattern_sizes_inch = {
    "rbsleeve": 8,  # say 8 inches wide
    "rfsleeve": 7,
    "rback": 10,
    "rfront": 10,
    "up_back": 6,
    "up_front": 6,
    "lback": 10,
    "lfront": 10,
    "lbsleeve": 7,
    "lfsleeve": 8
}

for pattern_name, width_in in pattern_sizes_inch.items():
    pattern_path = os.path.join(patterns_dir, pattern_name + ".png")
    if not os.path.exists(pattern_path):
        print(f"Pattern image not found: {pattern_path}")
        continue
    
    img = Image.open(pattern_path).convert("RGBA")
    orig_width_px, orig_height_px = img.size
    
    # Calculate target width in pixels based on pixels_per_inch scale
    target_width_px = int(width_in * pixels_per_inch)
    
    # Calculate scale factor and new height to preserve aspect ratio
    scale_factor = target_width_px / orig_width_px
    target_height_px = int(orig_height_px * scale_factor)
    
    # Resize pattern piece
    resized_img = img.resize((target_width_px, target_height_px), Image.LANCZOS)
    
    # Save scaled pattern piece
    save_path = os.path.join(output_dir, pattern_name + "_scaled.png")
    resized_img.save(save_path)
    print(f"Saved scaled pattern {pattern_name} to {save_path}")

print("Pattern scaling complete.")
