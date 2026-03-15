"""
4_fit_patterns.py

Given:
  - A patterns/ directory with one subfolder per garment, each containing:
      {id}_pattern.svg
      {id}_scan_imitation.obj
      {id}_scan_imitation_segmentation.txt
  - A garment contour from garment_area_results.json (pixels with known px/inch)

Steps:
  1. Discover all available patterns grouped by category
  2. Parse SVG → extract named pattern piece polygons (in px)
  3. Parse .obj + segmentation → compute real surface area (cm²) per piece
  4. Derive px→cm scale by comparing SVG polygon areas to mesh surface areas
  5. Scale SVG pieces to cm
  6. Load garment contour, convert to cm
  7. Nest (greedily place) pattern pieces inside the garment contour
  8. Visualize the result as an overlay
"""

import xml.etree.ElementTree as ET
import re
import json
import os
import numpy as np
import pyclipper
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.affinity import rotate as shapely_rotate, translate
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

CLIPPER_SCALE = 1000  # pyclipper requires integer coordinates


# ─── 0. Pattern discovery ─────────────────────────────────────────────────────

def discover_patterns(patterns_dir):
    """
    Scan patterns/ for valid subfolders (those with all 3 required files).
    Returns a dict: category -> list of pattern_ids, e.g.:
      { 'dress': ['dress_0XAVEH5G53', ...], 'jumpsuit': [...], ... }
    """
    available = {}
    for name in sorted(os.listdir(patterns_dir)):
        folder = os.path.join(patterns_dir, name)
        if not os.path.isdir(folder):
            continue
        svg = os.path.join(folder, f'{name}_pattern.svg')
        obj = os.path.join(folder, f'{name}_scan_imitation.obj')
        seg = os.path.join(folder, f'{name}_scan_imitation_segmentation.txt')
        if not (os.path.exists(svg) and os.path.exists(obj) and os.path.exists(seg)):
            continue
        # Category is everything before the last underscore+ID segment
        # e.g. 'dress_0XAVEH5G53' -> 'dress'
        #      'jacket_hood_sleeveless_XXXX' -> 'jacket_hood_sleeveless'
        parts = name.rsplit('_', 1)
        category = parts[0] if len(parts) == 2 else name
        available.setdefault(category, []).append(name)
    return available


def pattern_files(patterns_dir, pattern_id):
    """Return (svg_path, obj_path, seg_path) for a given pattern_id."""
    folder = os.path.join(patterns_dir, pattern_id)
    return (
        os.path.join(folder, f'{pattern_id}_pattern.svg'),
        os.path.join(folder, f'{pattern_id}_scan_imitation.obj'),
        os.path.join(folder, f'{pattern_id}_scan_imitation_segmentation.txt'),
    )


# ─── 1. Parse SVG pattern pieces ──────────────────────────────────────────────

def tokenize_path(d):
    """Split SVG path data into command letters and numeric tokens."""
    return re.findall(
        r'[MmLlHhVvCcSsQqTtAaZz]|[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', d
    )


def parse_svg_path(d):
    """
    Parse a simple SVG path (M, L, Q, z) into a list of (x, y) points.
    Quadratic bezier curves (Q) are sampled at 12 points.
    """
    tokens = tokenize_path(d)
    points = []
    current = (0.0, 0.0)
    i = 0

    while i < len(tokens):
        cmd = tokens[i]
        i += 1

        if cmd == 'M':
            x, y = float(tokens[i]), float(tokens[i + 1])
            i += 2
            current = (x, y)
            points.append(current)

        elif cmd == 'L':
            x, y = float(tokens[i]), float(tokens[i + 1])
            i += 2
            current = (x, y)
            points.append(current)

        elif cmd == 'Q':
            cx, cy = float(tokens[i]), float(tokens[i + 1])
            i += 2
            x, y = float(tokens[i]), float(tokens[i + 1])
            i += 2
            # Sample bezier (skip t=0, it's already in points as current)
            for t in np.linspace(0, 1, 13)[1:]:
                bx = (1 - t) ** 2 * current[0] + 2 * (1 - t) * t * cx + t ** 2 * x
                by = (1 - t) ** 2 * current[1] + 2 * (1 - t) * t * cy + t ** 2 * y
                points.append((bx, by))
            current = (x, y)

        elif cmd in ('z', 'Z'):
            pass  # polygon closed; don't duplicate the first point

        else:
            i += 1  # skip unknown

    return points


def load_svg_pieces(svg_path):
    """
    Returns dict: piece_name -> list of (x, y) in SVG pixels.
    Paths and text labels alternate in the SVG, so they are paired by index.
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()

    # Strip namespace prefix if present
    def strip_ns(tag):
        return re.sub(r'\{[^}]+\}', '', tag)

    paths = [el for el in root.iter() if strip_ns(el.tag) == 'path']
    texts = [el for el in root.iter() if strip_ns(el.tag) == 'text']

    pieces = {}
    for idx, path_el in enumerate(paths):
        d = path_el.get('d', '')
        pts = parse_svg_path(d)
        name = texts[idx].text.strip() if idx < len(texts) else f'piece_{idx}'
        pieces[name] = pts

    return pieces


# ─── 2. Parse .obj + segmentation, compute surface areas ──────────────────────

def load_obj_with_segmentation(obj_path, seg_path):
    """
    Returns:
      vertices  – np.array (N, 3) in cm
      faces     – list of [v0, v1, v2] (0-indexed)
      v_labels  – list of piece name strings, one per vertex
    """
    vertices = []
    faces = []

    with open(obj_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v':
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'f':
                idx = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                if len(idx) == 3:
                    faces.append(idx)

    with open(seg_path) as f:
        v_labels = [line.strip() for line in f]

    return np.array(vertices), faces, v_labels


def compute_mesh_surface_areas(vertices, faces, v_labels):
    """
    Returns dict: piece_name -> surface area in cm².
    Each face is assigned to the piece of its first vertex.
    """
    areas = {}
    for v0, v1, v2 in faces:
        label = v_labels[v0]
        a = vertices[v1] - vertices[v0]
        b = vertices[v2] - vertices[v0]
        area = 0.5 * np.linalg.norm(np.cross(a, b))
        areas[label] = areas.get(label, 0.0) + area
    return areas


# ─── 3. Compute px→cm scale factor ────────────────────────────────────────────

def polygon_area_px(pts):
    """Shoelace formula for a list of (x, y) points."""
    n = len(pts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]
    return abs(area) / 2.0


def compute_scale_factor(svg_pieces, mesh_areas, skip_labels=('stitch',)):
    """
    For each piece present in both SVG and mesh, compute sqrt(cm² / px²).
    Returns the median across all pieces as the scale factor (cm per px).
    """
    ratios = []
    print("  Piece                SVG area (px²)   Mesh area (cm²)   cm/px")
    print("  " + "-" * 62)
    for name, pts in svg_pieces.items():
        if name in skip_labels or name not in mesh_areas:
            continue
        svg_area = polygon_area_px(pts)
        mesh_area = mesh_areas[name]
        if svg_area > 0 and mesh_area > 0:
            ratio = np.sqrt(mesh_area / svg_area)
            ratios.append(ratio)
            print(f"  {name:<20} {svg_area:>14.1f}   {mesh_area:>15.1f}   {ratio:.4f}")

    scale = float(np.median(ratios))
    print(f"\n  Median scale: {scale:.4f} cm/px")
    return scale


# ─── 4. Convert SVG pieces to Shapely polygons in cm ──────────────────────────

def svg_pieces_to_cm(svg_pieces, scale, skip_labels=('stitch',)):
    """
    Returns dict: piece_name -> Shapely Polygon in cm.
    Skips any degenerate pieces.
    """
    result = {}
    for name, pts in svg_pieces.items():
        if name in skip_labels:
            continue
        cm_pts = [(x * scale, y * scale) for x, y in pts]
        poly = Polygon(cm_pts)
        if not poly.is_valid:
            poly = poly.buffer(0)  # fix self-intersections
        if poly.area > 0:
            result[name] = poly
    return result


# ─── 5. Load garment contour from JSON ────────────────────────────────────────

def load_garment_contour(json_path, key=None):
    """
    Returns a Shapely Polygon of the garment outline in cm,
    normalised so its bounding box starts at (0, 0).
    """
    with open(json_path) as f:
        data = json.load(f)

    key = key or list(data.keys())[0]
    entry = data[key]
    px_per_inch = entry['pixels_per_inch']
    cm_per_px = 2.54 / px_per_inch

    pts_cm = [(x * cm_per_px, y * cm_per_px) for x, y in entry['garment_contour']]
    poly = Polygon(pts_cm)
    if not poly.is_valid:
        poly = poly.buffer(0)

    minx, miny = poly.bounds[0], poly.bounds[1]
    poly = Polygon([(x - minx, y - miny) for x, y in poly.exterior.coords])
    return poly


# ─── 6. NFP-based nesting ─────────────────────────────────────────────────────

def normalise_to_origin(poly):
    """Translate a polygon so its bounding box starts at (0, 0)."""
    minx, miny = poly.bounds[0], poly.bounds[1]
    return translate(poly, -minx, -miny)


def _to_clipper(poly):
    """Shapely Polygon → pyclipper integer path (loses closing duplicate point)."""
    coords = list(poly.exterior.coords)[:-1]
    return [
        (int(round(x * CLIPPER_SCALE)), int(round(y * CLIPPER_SCALE)))
        for x, y in coords
    ]


def _from_clipper(paths):
    """pyclipper integer paths → Shapely Polygon / MultiPolygon, or None."""
    polys = []
    for path in paths:
        if len(path) < 3:
            continue
        coords = [(x / CLIPPER_SCALE, y / CLIPPER_SCALE) for x, y in path]
        try:
            p = Polygon(coords)
            if not p.is_valid:
                p = p.buffer(0)
            if p.area > 0:
                polys.append(p)
        except Exception:
            pass
    if not polys:
        return None
    result = unary_union(polys)
    return None if result.is_empty else result


def compute_ifp(container, piece):
    """
    Inner Fit Polygon: all valid positions for piece's reference point (0,0)
    such that piece fits entirely inside container.

    Computed as the intersection of (container shifted by -v) for every vertex v
    of the piece — the exact Minkowski difference.
    """
    valid = container
    for x, y in list(piece.exterior.coords)[:-1]:
        valid = valid.intersection(translate(container, -x, -y))
        if valid.is_empty:
            return None
    return None if valid.is_empty else valid


def compute_nfp(placed_poly, new_piece):
    """
    No-Fit Polygon: all positions of new_piece's reference point (0,0) that
    would cause it to overlap placed_poly.

    NFP(A, B) = Minkowski sum of A with the reflection of B = A ⊕ (−B).
    Computed via pyclipper for correctness on non-convex polygons.
    """
    a_path = _to_clipper(placed_poly)
    b_neg  = [(-x, -y) for x, y in _to_clipper(new_piece)]
    try:
        result = pyclipper.MinkowskiSum(a_path, b_neg, True)
        return _from_clipper(result)
    except Exception:
        return None


def _bottomleft(region):
    """
    Return the bottommost-then-leftmost vertex of a (Multi)Polygon.
    In our coordinate system y increases downward, so 'bottom' = largest y.
    """
    best = None
    geoms = list(region.geoms) if hasattr(region, 'geoms') else [region]
    for geom in geoms:
        if not hasattr(geom, 'exterior'):
            continue
        for x, y in geom.exterior.coords:
            if best is None or y > best[1] or (abs(y - best[1]) < 1e-9 and x < best[0]):
                best = (x, y)
    return best


def nest_pieces(cm_pieces, garment, rotation_step=15):
    """
    NFP-based nesting with gravity fill.

    For each piece (largest first), tries every `rotation_step` degrees (0–360).
    For each rotation:
      1. Computes the Inner Fit Polygon (IFP) — valid reference-point positions
         for the piece inside the container.
      2. Subtracts the No-Fit Polygon (NFP) for every already-placed piece —
         the forbidden zones that would cause overlap.
      3. Picks the bottommost-leftmost point of the remaining valid region.
    Returns (placed, unplaced) where placed is a list of (name, Polygon).
    """
    angles = range(0, 360, rotation_step)
    sorted_pieces = sorted(cm_pieces.items(), key=lambda x: x[1].area, reverse=True)
    placed = []
    unplaced = []
    total = len(sorted_pieces)

    for idx, (name, piece) in enumerate(sorted_pieces):
        print(f'  [{idx + 1}/{total}] {name} ...', flush=True)
        best_poly  = None
        best_score = float('inf')

        for angle in angles:
            rotated = normalise_to_origin(
                shapely_rotate(piece, float(angle), origin='centroid')
            )

            # --- valid region starts as the full IFP ---
            ifp = compute_ifp(garment, rotated)
            if ifp is None or ifp.is_empty:
                continue

            valid = ifp
            for _, placed_poly in placed:
                nfp = compute_nfp(placed_poly, rotated)
                if nfp is not None and not nfp.is_empty:
                    valid = valid.difference(nfp)
                if valid is None or valid.is_empty:
                    break

            if valid is None or valid.is_empty:
                continue

            pt = _bottomleft(valid)
            if pt is None:
                continue

            dx, dy = pt
            # maximise y (pack toward bottom), then minimise x
            score = -dy * 1e9 + dx
            if score < best_score:
                best_score = score
                best_poly  = translate(rotated, dx, dy)

        if best_poly is not None:
            placed.append((name, best_poly))
            print(f'      ✓ placed', flush=True)
        else:
            unplaced.append(name)
            print(f'      ✗ no fit', flush=True)

    return placed, unplaced


# ─── 7. Visualise ─────────────────────────────────────────────────────────────

COLORS = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabebe',
]


def visualise(garment, placed, unplaced, output_path='pattern_overlay.png'):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Garment outline
    gx, gy = garment.exterior.xy
    ax.fill(gx, gy, alpha=0.08, color='gray')
    ax.plot(gx, gy, color='black', linewidth=2, label='garment')

    # Placed pieces
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
    waste_area = garment.area - placed_area
    title = (
        f'Pattern nesting result\n'
        f'Garment: {garment.area:.0f} cm²  |  '
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
    ax.legend(handles=legend_patches, loc='upper right', fontsize=7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f'Saved overlay → {output_path}')
    plt.show()


# ─── Main ─────────────────────────────────────────────────────────────────────

def run(pattern_id, garment_json, patterns_dir, output_path='pattern_overlay.png'):
    """
    Full pipeline for one pattern against one garment.
    Can be called from a UI or directly from the command line.
    """
    svg_path, obj_path, seg_path = pattern_files(patterns_dir, pattern_id)

    print(f'=== Pattern: {pattern_id} ===\n')

    # 1. Load SVG
    print('--- 1. Loading SVG pattern pieces ---')
    svg_pieces = load_svg_pieces(svg_path)
    print(f'Found {len(svg_pieces)} pieces: {list(svg_pieces.keys())}')

    # 2. Load 3D mesh
    print('\n--- 2. Loading 3D mesh ---')
    vertices, faces, v_labels = load_obj_with_segmentation(obj_path, seg_path)
    print(f'Loaded {len(vertices)} vertices, {len(faces)} faces')
    mesh_areas = compute_mesh_surface_areas(vertices, faces, v_labels)

    # 3. Derive px→cm scale
    print('\n--- 3. Deriving scale factor ---')
    scale = compute_scale_factor(svg_pieces, mesh_areas)

    # 4. Scale pieces to cm
    print('\n--- 4. Scaling SVG pieces to cm ---')
    cm_pieces = svg_pieces_to_cm(svg_pieces, scale)
    total_pattern_area = sum(p.area for p in cm_pieces.values())
    for name, poly in cm_pieces.items():
        print(f'  {name:<20} {poly.area:.1f} cm²')
    print(f'  {"TOTAL":<20} {total_pattern_area:.1f} cm²')

    # 5. Load garment contour
    print('\n--- 5. Loading garment contour ---')
    garment = load_garment_contour(garment_json)
    print(f'Garment area: {garment.area:.1f} cm²')
    print(f'Pattern total: {total_pattern_area:.1f} cm²')
    if garment.area < total_pattern_area:
        print('⚠  Garment is smaller than total pattern area — not all pieces will fit.')

    # 6. Nest pieces
    print('\n--- 6. Nesting pattern pieces ---')
    placed, unplaced = nest_pieces(cm_pieces, garment)
    print(f'\nResult: {len(placed)}/{len(cm_pieces)} pieces placed')
    if unplaced:
        print(f'Unplaced: {unplaced}')

    # 7. Visualise
    print('\n--- 7. Generating overlay ---')
    visualise(garment, placed, unplaced, output_path=output_path)

    return placed, unplaced


if __name__ == '__main__':
    import sys

    base = os.path.dirname(os.path.abspath(__file__))
    patterns_dir = os.path.join(base, 'patterns')
    garment_json = os.path.join(base, 'garment_area_results.json')

    # Discover and display all available patterns
    available = discover_patterns(patterns_dir)
    print('Available patterns by category:')
    for category, ids in available.items():
        print(f'  {category} ({len(ids)} patterns)')

    # Accept a pattern ID as a command-line argument, or prompt
    if len(sys.argv) > 1:
        pattern_id = sys.argv[1]
    else:
        print()
        pattern_id = input('Enter a pattern ID to run (e.g. dress_0XAVEH5G53): ').strip()

    # Validate
    all_ids = [pid for ids in available.values() for pid in ids]
    if pattern_id not in all_ids:
        print(f'Error: "{pattern_id}" not found in {patterns_dir}')
        sys.exit(1)

    output_path = os.path.join(base, f'pattern_overlay_{pattern_id}.png')
    run(pattern_id, garment_json, patterns_dir, output_path=output_path)
