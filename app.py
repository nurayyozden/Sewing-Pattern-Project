"""
app.py — Flask web app for the Sewing Pattern Project.

Flow:
  GET  /                          → upload page
  POST /upload                    → process image → redirect to /select/<sid>
  GET  /select/<sid>              → pattern picker
  POST /select/<sid>              → run nesting → redirect to /result/<sid>/<pattern_id>
  GET  /result/<sid>/<pattern_id> → show overlay
"""

import os
import uuid
import json
import traceback
from io import BytesIO
from PIL import Image

from flask import (
    Flask, render_template, request, redirect, url_for, jsonify, abort
)
import pipeline

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB upload limit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SESSIONS_DIR = os.path.join(BASE_DIR, 'static', 'sessions')
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}


def session_dir(sid):
    return os.path.join(SESSIONS_DIR, sid)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ─── Upload ───────────────────────────────────────────────────────────────────

@app.route('/', methods=['GET'])
def index():
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('image')
    if not file or file.filename == '':
        return render_template('upload.html', error='Please select an image to upload.')
    if not allowed_file(file.filename):
        return render_template('upload.html', error='Only JPG and PNG images are supported.')

    sid = uuid.uuid4().hex
    sdir = session_dir(sid)
    os.makedirs(sdir, exist_ok=True)

    # Read into PIL immediately — handles HEIC-named-as-PNG, format mismatches, etc.
    try:
        img = Image.open(BytesIO(file.read())).convert('RGB')
    except Exception:
        return render_template('upload.html', error='Could not read the image. Please upload a JPG or PNG file.')

    image_path = os.path.join(sdir, 'upload.jpeg')
    img.save(image_path, 'JPEG', quality=95)

    try:
        # Background removal
        mask_path = pipeline.remove_background(image_path, sdir)

        # Garment area computation
        garment_data = pipeline.compute_garment_area(mask_path)

        # Save garment data for later use
        with open(os.path.join(sdir, 'garment_data.json'), 'w') as f:
            json.dump(garment_data, f)

        # Save upload filename for display
        with open(os.path.join(sdir, 'meta.json'), 'w') as f:
            json.dump({'upload_ext': 'jpeg'}, f)

    except ValueError as e:
        return render_template('upload.html', error=str(e))
    except Exception as e:
        traceback.print_exc()
        return render_template('upload.html', error=f'Error: {e}')

    return redirect(url_for('select_pattern', sid=sid))


# ─── Pattern selection ────────────────────────────────────────────────────────

@app.route('/select/<sid>', methods=['GET'])
def select_pattern(sid):
    sdir = session_dir(sid)
    if not os.path.isdir(sdir):
        abort(404)

    with open(os.path.join(sdir, 'garment_data.json')) as f:
        garment_data = json.load(f)

    with open(os.path.join(sdir, 'meta.json')) as f:
        meta = json.load(f)

    # Find transparent image for display
    stem = f"upload"
    transparent_path = os.path.join(sdir, 'transparent', f'{stem}.png')
    if os.path.exists(transparent_path):
        garment_img_url = url_for('static', filename=f'sessions/{sid}/transparent/{stem}.png')
    else:
        garment_img_url = url_for('static', filename=f'sessions/{sid}/upload.{meta["upload_ext"]}')

    available = pipeline.get_available_patterns()
    categories = [
        {
            'id': cat,
            'display': pipeline.category_display_name(cat),
            'count': len(ids),
            'first_id': ids[0],
        }
        for cat, ids in available.items()
    ]

    return render_template(
        'select.html',
        sid=sid,
        garment_img_url=garment_img_url,
        garment_area_in2=round(garment_data['garment_area_in2'], 1),
        categories=categories,
    )


@app.route('/select/<sid>', methods=['POST'])
def fit_pattern(sid):
    sdir = session_dir(sid)
    if not os.path.isdir(sdir):
        abort(404)

    pattern_id = request.form.get('pattern_id', '').strip()
    if not pattern_id:
        return redirect(url_for('select_pattern', sid=sid))

    with open(os.path.join(sdir, 'garment_data.json')) as f:
        garment_data = json.load(f)

    output_path = os.path.join(sdir, f'overlay_{pattern_id}.png')

    try:
        placed, unplaced, garment_area_cm2, placed_area_cm2 = pipeline.fit_and_render(
            pattern_id, garment_data, output_path
        )
    except Exception:
        traceback.print_exc()
        return redirect(url_for('select_pattern', sid=sid))

    # Save result summary
    result = {
        'pattern_id': pattern_id,
        'placed': [name for name, _ in placed],
        'unplaced': unplaced,
        'garment_area_cm2': round(garment_area_cm2, 1),
        'placed_area_cm2': round(placed_area_cm2, 1),
        'waste_cm2': round(garment_area_cm2 - placed_area_cm2, 1),
    }
    with open(os.path.join(sdir, f'result_{pattern_id}.json'), 'w') as f:
        json.dump(result, f)

    return redirect(url_for('show_result', sid=sid, pattern_id=pattern_id))


# ─── Result ───────────────────────────────────────────────────────────────────

@app.route('/result/<sid>/<pattern_id>')
def show_result(sid, pattern_id):
    sdir = session_dir(sid)
    result_json = os.path.join(sdir, f'result_{pattern_id}.json')
    overlay_png = os.path.join(sdir, f'overlay_{pattern_id}.png')

    if not os.path.exists(result_json) or not os.path.exists(overlay_png):
        abort(404)

    with open(result_json) as f:
        result = json.load(f)

    overlay_url = url_for('static', filename=f'sessions/{sid}/overlay_{pattern_id}.png')

    available = pipeline.get_available_patterns()
    categories = [
        {'id': cat, 'display': pipeline.category_display_name(cat), 'first_id': ids[0]}
        for cat, ids in available.items()
    ]

    return render_template(
        'result.html',
        sid=sid,
        result=result,
        overlay_url=overlay_url,
        categories=categories,
    )


# ─── API: patterns for a category (used by JS on select page) ─────────────────

@app.route('/api/patterns/<category>')
def api_patterns(category):
    available = pipeline.get_available_patterns()
    ids = available.get(category, [])
    return jsonify(ids)


# ─── Serve pattern preview images ─────────────────────────────────────────────

@app.route('/pattern-preview/<pattern_id>')
def pattern_preview(pattern_id):
    from flask import send_file
    patterns_dir = pipeline.PATTERNS_DIR
    # Prefer camera_front photo if available, fall back to SVG
    for filename in (f'{pattern_id}_camera_front.png', f'{pattern_id}_camera_front.jpg',
                     f'{pattern_id}_pattern.svg'):
        path = os.path.join(patterns_dir, pattern_id, filename)
        if os.path.exists(path):
            return send_file(path)
    abort(404)


if __name__ == '__main__':
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    app.run(debug=True, port=5000)
