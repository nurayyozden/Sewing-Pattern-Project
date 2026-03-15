"""
pipeline.py

Callable wrappers around the processing scripts for use in the web app.
"""

import os
import sys
import math
import json
import shutil

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for web use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import importlib.util

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'u2net', 'saved_models', 'u2net', 'u2net.pth')
PATTERNS_DIR = os.path.join(BASE_DIR, 'patterns')


# ─── Load remove_background module (filename starts with digit) ───────────────

def _load_remove_bg():
    spec = importlib.util.spec_from_file_location(
        'remove_bg', os.path.join(BASE_DIR, '2_remove_background.py')
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.main


# ─── Step 1: background removal ───────────────────────────────────────────────

def remove_background(image_path, session_dir):
    """
    Run U²-Net on a single image.
    Saves mask to  session_dir/masks/upload.jpeg
    Saves rgba to  session_dir/transparent/upload.png
    Returns the mask path.
    """
    input_dir = os.path.join(session_dir, 'input')
    os.makedirs(input_dir, exist_ok=True)

    # Copy to input dir (app.py already normalised to JPEG)
    jpeg_path = os.path.join(input_dir, 'upload.jpeg')
    if image_path != jpeg_path:
        shutil.copy(image_path, jpeg_path)

    remove_bg_fn = _load_remove_bg()
    remove_bg_fn(input_dir, session_dir, model_path=MODEL_PATH)

    mask_path = os.path.join(session_dir, 'masks', 'upload.jpeg')
    return mask_path


# ─── Step 2: compute garment area from mask ───────────────────────────────────

def compute_garment_area(mask_path):
    """
    Detect paper (leftmost object) and garment in the mask.
    Returns a garment-data dict compatible with garment_area_results.json,
    or raises ValueError if detection fails.
    """
    PAPER_AREA_IN2 = 8.5 * 11  # US Letter

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f'Could not load mask: {mask_path}')

    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 100]

    if len(contours) < 2:
        raise ValueError(
            'Could not detect both the paper and the garment. '
            'Make sure a US Letter sheet of paper is visible next to the garment.'
        )

    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    paper = contours[0]
    garment = max(contours[1:], key=cv2.contourArea)

    paper_area_px = cv2.contourArea(paper)
    garment_area_px = cv2.contourArea(garment)

    pixels_per_inch2 = paper_area_px / PAPER_AREA_IN2
    pixels_per_inch = math.sqrt(pixels_per_inch2)
    garment_area_in2 = garment_area_px / pixels_per_inch2

    return {
        'pixels_per_inch': pixels_per_inch,
        'garment_area_in2': garment_area_in2,
        'garment_area_ft2': garment_area_in2 / 144,
        'paper_contour': paper.squeeze(axis=1).tolist(),
        'garment_contour': garment.squeeze(axis=1).tolist(),
    }


# ─── Step 3: fit pattern and render overlay ───────────────────────────────────

_FP = None

def _fp_module():
    global _FP
    if _FP is None:
        spec = importlib.util.spec_from_file_location(
            'fit_patterns', os.path.join(BASE_DIR, '4_fit_patterns.py')
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _FP = mod
    return _FP


COLORS = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabebe',
]


def fit_and_render(pattern_id, garment_data, output_path):
    """
    Run the full nesting pipeline for one pattern against garment_data.
    Saves overlay image to output_path.
    Returns (placed, unplaced, garment_area_cm2, placed_area_cm2).
    """
    fp = _fp_module()

    svg_path, obj_path, seg_path = fp.pattern_files(PATTERNS_DIR, pattern_id)

    svg_pieces = fp.load_svg_pieces(svg_path)
    vertices, faces, v_labels = fp.load_obj_with_segmentation(obj_path, seg_path)
    mesh_areas = fp.compute_mesh_surface_areas(vertices, faces, v_labels)
    scale = fp.compute_scale_factor(svg_pieces, mesh_areas)
    cm_pieces = fp.svg_pieces_to_cm(svg_pieces, scale)

    # Wrap garment_data in the expected JSON structure
    garment_json_data = {'garment': garment_data}
    tmp_json = output_path + '_garment_tmp.json'
    with open(tmp_json, 'w') as f:
        json.dump(garment_json_data, f)

    garment_poly = fp.load_garment_contour(tmp_json, key='garment')
    os.remove(tmp_json)

    placed, unplaced = fp.nest_pieces(cm_pieces, garment_poly)

    # Render overlay (non-interactive)
    fig, ax = plt.subplots(figsize=(10, 10))
    gx, gy = garment_poly.exterior.xy
    ax.fill(gx, gy, alpha=0.08, color='gray')
    ax.plot(gx, gy, color='black', linewidth=2)

    legend_patches = []
    for i, (name, poly) in enumerate(placed):
        color = COLORS[i % len(COLORS)]
        px, py = poly.exterior.xy
        ax.fill(px, py, alpha=0.55, color=color)
        ax.plot(px, py, color=color, linewidth=1)
        cx, cy = poly.centroid.x, poly.centroid.y
        ax.text(cx, cy, name, fontsize=7, ha='center', va='center', fontweight='bold')
        legend_patches.append(mpatches.Patch(color=color, label=name))

    placed_area = sum(p.area for _, p in placed)
    waste_area = garment_poly.area - placed_area
    title = (
        f'{pattern_id}\n'
        f'Garment: {garment_poly.area:.0f} cm²  |  '
        f'Placed: {placed_area:.0f} cm²  |  '
        f'Waste: {waste_area:.0f} cm²'
    )
    if unplaced:
        title += f'\nDid not fit: {", ".join(unplaced)}'

    ax.set_title(title, fontsize=10)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xlabel('cm')
    ax.set_ylabel('cm')
    if legend_patches:
        ax.legend(handles=legend_patches, loc='upper right', fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    return placed, unplaced, garment_poly.area, placed_area


# ─── Pattern discovery ────────────────────────────────────────────────────────

CATEGORY_DISPLAY_NAMES = {
    'dress':                   'Dress',
    'jacket_hood_sleeveless':  'Hooded Sleeveless Jacket',
    'jacket_sleeveless':       'Sleeveless Jacket',
    'jumpsuit':                'Jumpsuit',
    'skirt_waistband':         'Skirt',
    'tee_hood':                'Hooded Tee',
    'wb_jumpsuit_sleeveless':  'Sleeveless Jumpsuit',
}


def get_available_patterns():
    """Returns dict: category -> list of pattern_ids (sorted)."""
    fp = _fp_module()
    return fp.discover_patterns(PATTERNS_DIR)


def category_display_name(category):
    return CATEGORY_DISPLAY_NAMES.get(category, category.replace('_', ' ').title())
