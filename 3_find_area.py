import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# 1. Load binary mask image

# Path to the mask image (change filename as needed) - CHANGE TO TEST
mask_path = '/masked_images/your_mask_image.png'

# Load image in grayscale mode - maybe don't need
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

if mask is None:
    print(f"Failed to load image at {mask_path}")
else:
    # Show the mask image
    plt.imshow(mask, cmap='gray')
    plt.title("Binary Mask")
    plt.axis('off')
    plt.show()


# 2. Identify which is the paper, which is the garment


# Threshold to make sure mask is binary (0 or 255)
_, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

# Find contours (external only)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Found {len(contours)} objects")

# Calculate centroids and bounding boxes
objects = []
for cnt in contours:
    M = cv2.moments(cnt)
    if M['m00'] == 0:
        continue
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    x, y, w, h = cv2.boundingRect(cnt)
    objects.append({'contour': cnt, 'centroid': (cx, cy), 'bbox': (x, y, w, h)})

# Sort objects by centroid x-coordinate (left to right)
objects = sorted(objects, key=lambda obj: obj['centroid'][0])

# Assume leftmost is paper
paper = objects[0]
garment = objects[1] if len(objects) > 1 else None

print("Paper centroid:", paper['centroid'])
if garment:
    print("Garment centroid:", garment['centroid'])

# Visualize for sanity check
output = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
cv2.drawContours(output, [paper['contour']], -1, (0,255,0), 2)  # Green for paper
if garment:
    cv2.drawContours(output, [garment['contour']], -1, (0,0,255), 2)  # Red for garment

plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title('Paper (green) and Garment (red)')
plt.axis('off')
plt.show()

# 3. Calculate area of garment

# Calculate areas
paper_area_pixels = cv2.contourArea(paper['contour'])
garment_area_pixels = cv2.contourArea(garment['contour'])

# Known paper size (US Letter)
paper_width_in = 8.5
paper_height_in = 11
paper_area_in2 = paper_width_in * paper_height_in  # 93.5 in²

# Pixel to inch^2 scale
pixels_per_inch2 = paper_area_pixels / paper_area_in2

# Garment area in real units
garment_area_in2 = garment_area_pixels / pixels_per_inch2
garment_area_ft2 = garment_area_in2 / 144

print(f"Garment area: {garment_area_ft2:.3f} square feet")
