import cv2
import numpy as np

#image import

# background removal using U²-Net


# Load the binary mask (from segmentation step)
mask = cv2.imread("fabric_mask.png", cv2.IMREAD_GRAYSCALE)

# Threshold the imagepip
_, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on original image
image = cv2.imread("flatlay.jpg")
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

# Save or display result
cv2.imwrite("outlined_fabric.png", image)
