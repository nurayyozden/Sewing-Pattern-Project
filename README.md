# Sewing Pattern Project

A web app that helps you reuse unworn clothing fabric by fitting sewing pattern pieces onto your garment as efficiently as possible — minimising waste.

---

## What this project does

1. **Upload a photo** of a garment you've cut open and laid flat, with a US Letter sheet of paper beside it as a size reference.
2. **Choose a sewing pattern** from a library of 1,000+ garment designs (dresses, jumpsuits, skirts, jackets, tees).
3. **The app fits the pattern pieces** onto your fabric using an optimal nesting algorithm and shows you exactly where to cut.

The goal is zero-waste sewing: instead of buying new fabric, you repurpose something you already own.

---

## How it works

### Step 1 — Background removal (`2_remove_background.py`)

The uploaded photo is passed through **U²-Net**, a deep learning model for salient object detection. U²-Net produces a binary mask separating the garment and paper from the background. The result is a transparent-background PNG.

### Step 2 — Garment area measurement (`3_find_area.py`)

OpenCV finds the contours of the two objects in the mask. The **US Letter sheet of paper** (known area: 8.5 × 11 = 93.5 in²) acts as a calibration reference:

$$\text{pixels per inch} = \sqrt{\frac{\text{paper area in pixels}}{93.5}}$$

The garment's pixel area is then converted to real-world units:

$$\text{garment area (in}^2) = \frac{\text{garment area (px}^2)}{\text{pixels per inch}^2}$$

The garment's pixel contour is stored for later use as the nesting boundary.

### Step 3 — Pattern scaling (`4_fit_patterns.py`)

Each sewing pattern in the library comes with three files:

| File | Contents |
|------|----------|
| `*_pattern.svg` | 2D flat pattern pieces with named labels |
| `*_scan_imitation.obj` | 3D mesh of the finished garment (coordinates in cm) |
| `*_scan_imitation_segmentation.txt` | Maps each mesh vertex to its pattern piece name |

The SVG pattern pieces are in arbitrary pixel units. To recover their real-world size, the 3D mesh surface area of each piece (in cm²) is compared to its SVG polygon area (in px²):

$$\text{scale} = \text{median over all pieces}\left(\sqrt{\frac{\text{mesh area (cm}^2)}{\text{SVG area (px}^2)}}\right)$$

This gives a reliable cm-per-pixel scale factor that is then applied to every polygon in the SVG.

### Step 4 — NFP-based nesting (`4_fit_patterns.py`)

Fitting pattern pieces into an irregular garment shape is a variant of the **2D irregular bin-packing problem**, which is NP-hard in general. This project uses **No-Fit Polygons (NFP)** — the industry-standard approach used in professional CAD cutting software.

#### Inner Fit Polygon (IFP)

The IFP of the container $C$ and a piece $P$ is the set of all positions where $P$'s reference point can be placed such that $P$ lies entirely inside $C$:

$$\text{IFP}(C, P) = \{p \mid P + p \subseteq C\}$$

This is the **Minkowski difference** $C \ominus P$, computed here as the intersection of $C$ shifted by $-v$ for every vertex $v$ of $P$:

$$\text{IFP}(C, P) = \bigcap_{v \in P}(C - v)$$

#### No-Fit Polygon (NFP)

The NFP of two pieces $A$ and $B$ is the set of positions for $B$'s reference point that would cause $B$ to overlap $A$:

$$\text{NFP}(A, B) = A \oplus (-B)$$

where $\oplus$ is the **Minkowski sum** and $-B$ is $B$ reflected through the origin. This is computed using [pyclipper](https://github.com/fonttools/pyclipper), which handles non-convex polygons correctly.

#### Placement algorithm

Pieces are placed largest-first. For each piece, 24 rotation angles are tried (every 15°). For each rotation:

1. Compute the IFP — all valid positions inside the garment.
2. Subtract the NFP for every already-placed piece — the forbidden overlap zones.
3. The remaining **valid region** = IFP $\setminus \bigcup_i \text{NFP}(A_i, P)$.
4. Find the point in the valid region **nearest to the bottom-left corner** of the garment using Shapely's `nearest_points`. This implements a gravity-fill strategy that packs pieces tightly into the corner, minimising wasted fabric.
5. Place the piece there and move on to the next.

---

## Project structure

```
app.py                        Flask web application
pipeline.py                   Web-facing wrappers for the processing steps
2_remove_background.py        U²-Net background removal
3_find_area.py                Garment area calculation from mask
4_fit_patterns.py             Pattern scaling + NFP nesting
templates/                    HTML templates (upload, select, result)
static/style.css              Stylesheet
patterns/                     Sewing pattern library (not tracked in git)
  {category}_{id}/
    *_pattern.svg
    *_scan_imitation.obj
    *_scan_imitation_segmentation.txt
    *_camera_front.png        (optional — used as thumbnail in UI)
u2net/                        U²-Net model code and weights
```

---

## Setup

```bash
# Create and activate the virtual environment
python3 -m venv env
source env/bin/activate

# Install dependencies
pip install torch torchvision flask shapely pyclipper opencv-python pillow matplotlib numpy

# Download U²-Net weights and place at:
# u2net/saved_models/u2net/u2net.pth

# Add pattern data to patterns/ following the folder structure above

# Run the app
python app.py
# Open http://127.0.0.1:5000
```

---

## Usage

1. Cut your garment open along the seams so it lays completely flat as one piece of fabric.
2. Place a US Letter (8.5 × 11 in) sheet of paper to the left of the fabric on a flat surface.
3. Photograph from directly above — both the paper and the full garment must be visible.
4. Upload the photo at `http://127.0.0.1:5000`.
5. Choose a pattern category (Dress, Jumpsuit, Skirt, etc.) and select a specific design.
6. The app fits the pattern pieces and shows an overlay of where to cut.

---

## Citation

The sewing pattern library uses data from:

> Korosteleva, M., Kemper, F., & Wenninger, S. (2024). *GarmentCodeData (v2): A Dataset of 3D Made-to-Measure Garments With Sewing Patterns* (v2) [Dataset]. ETH Zurich. https://doi.org/10.3929/ethz-b-000690432

Project lead: Olga Sorkine-Hornung (IGL, ETH Zurich).
License: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
Project page: https://igl.ethz.ch/projects/GarmentCodeData/

> Note: Claude Code assisted in writing the code for this project.