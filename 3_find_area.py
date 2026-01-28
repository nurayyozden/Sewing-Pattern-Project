# calculates the area of the masked image in masked_images/masks

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import math
import json

# Folder containing the binary masks
mask_folder = "masked_images/masks"

# Debug: check working directory and mask folder
print("Current working directory:", os.getcwd())
print("Mask folder exists:", os.path.exists(mask_folder))

# Known paper size (US Letter)
paper_width_in = 8.5
paper_height_in = 11
paper_area_in2 = paper_width_in * paper_height_in  # 93.5 in²

# Get all mask files (allow multiple formats)
mask_paths = []
mask_paths += glob.glob(os.path.join(mask_folder, "*.png"))
mask_paths += glob.glob(os.path.join(mask_folder, "*.jpg"))
mask_paths += glob.glob(os.path.join(mask_folder, "*.jpeg"))

print(f"Found {len(mask_paths)} mask images in {mask_folder}")

results = {}

for mask_path in mask_paths:
    print(f"\nProcessing {os.path.basename(mask_path)}")

    # Load mask as grayscale
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Failed to load image at {mask_path}")
        continue

    # Threshold to ensure binary mask
    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Find external contours
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter out tiny contours (noise)
    contours = [c for c in contours if cv2.contourArea(c) > 100]

    if len(contours) < 2:
        print("Not enough objects detected.")
        continue

    # Sort contours by x-position (leftmost first)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    # Assign objects
    paper = contours[0]  # leftmost object
    garment = max(contours[1:], key=cv2.contourArea)

    # ---------------- DEBUG CHECK ----------------
    paper_area_px = cv2.contourArea(paper)
    garment_area_px = cv2.contourArea(garment)

    print(
        f"DEBUG | paper area px: {paper_area_px:.0f} | "
        f"garment area px: {garment_area_px:.0f}"
    )

    if paper_area_px < garment_area_px:
        print("⚠️  WARNING: paper area is smaller than garment area — check mask!")
    # ---------------------------------------------

    # Pixel-to-inch² scale
    pixels_per_inch2 = paper_area_px / paper_area_in2
    pixels_per_inch = math.sqrt(pixels_per_inch2)

    # Garment area in inch² and ft²
    garment_area_in2 = garment_area_px / pixels_per_inch2
    garment_area_ft2 = garment_area_in2 / 144

    print(f"Pixels per inch: {pixels_per_inch:.2f}")
    print(f"Garment area: {garment_area_in2:.1f} in² ({garment_area_ft2:.3f} ft²)")

    # Save results keyed by filename (without extension)
    key = os.path.splitext(os.path.basename(mask_path))[0]
    results[key] = {
        "pixels_per_inch": pixels_per_inch,
        "garment_area_in2": garment_area_in2,
        "garment_area_ft2": garment_area_ft2,
    }

    # Visualize paper and garment contours
    output = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(output, [paper], -1, (0, 255, 0), 2)    # Green = paper
    cv2.drawContours(output, [garment], -1, (0, 0, 255), 2)  # Red = garment

    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.title(f"{os.path.basename(mask_path)}\nPaper (green) & Garment (red)")
    plt.axis("off")
    plt.show()

# Save results to JSON
with open("garment_area_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("All results saved to garment_area_results.json")