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

# Known paper size (US Letter)
paper_width_in = 8.5
paper_height_in = 11
paper_area_in2 = paper_width_in * paper_height_in  # 93.5 in²

# Get all mask files
mask_paths = glob.glob(os.path.join(mask_folder, "*.png"))
print(f"Found {len(mask_paths)} mask images in {mask_folder}")

results = {}

for mask_path in mask_paths:
    print(f"\nProcessing {os.path.basename(mask_path)}")

    # Load mask as grayscale
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Failed to load image at {mask_path}")
        continue

    # Threshold to ensure binary mask (0 or 255)
    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Find external contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print("No objects found in mask")
        continue

    # Filter out tiny contours (noise)
    contours = [c for c in contours if cv2.contourArea(c) > 100]

    # Identify paper by leftmost contour with aspect ratio ~0.77
    paper = None
    garment = None
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / h
        if 0.7 <= aspect_ratio <= 0.85 and x < mask.shape[1] / 2:
            paper = c
            break

    if paper is None:
        print("Paper not detected. Skipping this mask.")
        continue

    remaining_contours = [c for c in contours if not np.array_equal(c, paper)]
    if not remaining_contours:
        print("Garment not detected. Skipping this mask.")
        continue

    garment = max(remaining_contours, key=cv2.contourArea)

    # Calculate areas in pixels
    paper_area_pixels = cv2.contourArea(paper)
    garment_area_pixels = cv2.contourArea(garment)

    # Pixel-to-inch² scale
    pixels_per_inch2 = paper_area_pixels / paper_area_in2
    pixels_per_inch = math.sqrt(pixels_per_inch2)

    # Garment area in inch² and ft²
    garment_area_in2 = garment_area_pixels / pixels_per_inch2
    garment_area_ft2 = garment_area_in2 / 144

    print(f"Pixels per inch: {pixels_per_inch:.2f}")
    print(f"Garment area: {garment_area_in2:.1f} in² ({garment_area_ft2:.3f} ft²)")

    # Save results keyed by filename (without extension)
    key = os.path.splitext(os.path.basename(mask_path))[0]
    results[key] = {
        'pixels_per_inch': pixels_per_inch,
        'garment_area_in2': garment_area_in2,
        'garment_area_ft2': garment_area_ft2
    }

    # Visualize paper and garment contours
    output = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(output, [paper], -1, (0, 255, 0), 2)  # Green = paper
    cv2.drawContours(output, [garment], -1, (0, 0, 255), 2)  # Red = garment

    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.title(f"{os.path.basename(mask_path)}\nPaper (green) & Garment (red)")
    plt.axis('off')
    plt.show()

# Optionally, save all results to a JSON file for easy loading later
with open('garment_area_results.json', 'w') as f:
    json.dump(results, f, indent=4)

print("All results saved to garment_area_results.json")
