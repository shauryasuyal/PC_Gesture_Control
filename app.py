"""
FG Gesture Control — Flask Web Server
Serves the dashboard UI, REST API, and real-time WebSocket events.

The app runs the existing MotionControl modes (Cursor, Piano, Drawing, Sandbox)
as the primary gesture system, with custom ML gestures layered on top.

Usage: python app.py
Then open http://localhost:5000
"""

import os
import json
import time
import threading
import numpy as np
from datetime import datetime

from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO

from gesture_engine import GestureEngine, GESTURE_SHAPES, BUILTIN_SHAPES, FREEFORM_GESTURES
from gesture_classifier import train_model, SAMPLES_DIR

# ──────────────────────────────────────────────
# APP SETUP
# ──────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = 'mk-gesture-control-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GESTURES_FILE = os.path.join(BASE_DIR, "data", "gestures.json")

# Initialize engine
engine = GestureEngine(socketio=socketio)


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def load_gestures():
    """Load gestures config from JSON file."""
    with open(GESTURES_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_gestures(config):
    """Save gestures config to JSON file."""
    # Update sample counts
    for g in config.get("gestures", []):
        sample_path = os.path.join(SAMPLES_DIR, f"{g['id']}.npy")
        if os.path.exists(sample_path):
            samples = np.load(sample_path)
            g["samples_count"] = len(samples)
        else:
            g["samples_count"] = 0

    with open(GESTURES_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def refresh_engine_config():
    """Reload config into the engine."""
    config = load_gestures()
    engine.load_config(config)


# ──────────────────────────────────────────────
# PAGE ROUTES
# ──────────────────────────────────────────────
@app.route('/api/model/reload', methods=['POST'])
def reload_model():
    """Hot-reload the ML model into the running gesture engine."""
    success = engine.reload_model()
    return jsonify({"success": success, "model_loaded": engine.predictor.loaded})

@app.route('/')
def index():
    return render_template('index.html')


# ──────────────────────────────────────────────
# VIDEO FEED
# ──────────────────────────────────────────────
@app.route('/video_feed')
def video_feed():
    return Response(
        engine.generate_mjpeg(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# ──────────────────────────────────────────────
# API: GESTURES (custom ML gestures)
# ──────────────────────────────────────────────
@app.route('/api/gestures', methods=['GET'])
def get_gestures():
    config = load_gestures()
    return jsonify(config)


@app.route('/api/gesture_shapes', methods=['GET'])
def get_gesture_shapes():
    """Return the gesture shape catalog for the frontend dropdown.
    Merges named emoji shapes + 10 free-form custom slots."""
    combined = dict(GESTURE_SHAPES)
    combined.update(FREEFORM_GESTURES)
    return jsonify(combined)



@app.route('/api/gestures', methods=['POST'])
def add_gesture():
    data = request.get_json()
    if not data or not data.get('name'):
        return jsonify({"error": "Name is required"}), 400

    config = load_gestures()

    # Generate ID from name
    gid = data['name'].lower().replace(' ', '_').replace('-', '_')
    gid = ''.join(c for c in gid if c.isalnum() or c == '_')

    # Check for duplicates
    existing_ids = [g['id'] for g in config['gestures']]
    if gid in existing_ids:
        return jsonify({"error": "A gesture with this name already exists"}), 409

    gesture = {
        "id": gid,
        "name": data['name'],
        "icon": (
            GESTURE_SHAPES.get(data.get('gesture_shape'), {}).get('icon') or
            FREEFORM_GESTURES.get(data.get('gesture_shape'), {}).get('icon') or
            '👋'
        ),
        "gesture_shape": data.get('gesture_shape', ''),
        "action_type": data.get('action_type', 'shortcut'),
        "action_value": data.get('action_value', ''),
        "action_label": data.get('action_label', ''),
        "samples_count": 0,
        "created_at": datetime.now().isoformat()
    }

    config['gestures'].append(gesture)
    save_gestures(config)
    refresh_engine_config()

    return jsonify(gesture), 201


@app.route('/api/gestures/<gesture_id>', methods=['DELETE'])
def delete_gesture(gesture_id):
    config = load_gestures()
    original_len = len(config['gestures'])
    config['gestures'] = [g for g in config['gestures'] if g['id'] != gesture_id]

    if len(config['gestures']) == original_len:
        return jsonify({"error": "Gesture not found"}), 404

    # Delete samples file
    sample_path = os.path.join(SAMPLES_DIR, f"{gesture_id}.npy")
    if os.path.exists(sample_path):
        os.remove(sample_path)

    save_gestures(config)
    refresh_engine_config()

    return jsonify({"success": True, "id": gesture_id})


@app.route('/api/gestures/<gesture_id>/action', methods=['PUT'])
def update_action(gesture_id):
    data = request.get_json()
    if not data:
        return jsonify({"error": "Data required"}), 400

    config = load_gestures()
    found = False
    for g in config['gestures']:
        if g['id'] == gesture_id:
            if 'action_type' in data:
                g['action_type'] = data['action_type']
            if 'action_value' in data:
                g['action_value'] = data['action_value']
            if 'action_label' in data:
                g['action_label'] = data['action_label']
            found = True
            break

    if not found:
        return jsonify({"error": "Gesture not found"}), 404

    save_gestures(config)
    refresh_engine_config()

    return jsonify({"success": True, "id": gesture_id})


# ──────────────────────────────────────────────
# API: RECORDING
# ──────────────────────────────────────────────
@app.route('/api/gestures/<gesture_id>/record', methods=['POST'])
def start_recording(gesture_id):
    config = load_gestures()
    found = any(g['id'] == gesture_id for g in config['gestures'])
    if not found:
        return jsonify({"error": "Gesture not found"}), 404

    engine.start_recording(gesture_id)
    return jsonify({"success": True, "gesture_id": gesture_id, "status": "recording"})


@app.route('/api/gestures/<gesture_id>/stop', methods=['POST'])
def stop_recording(gesture_id):
    saved_id, samples = engine.stop_recording()

    if not samples:
        return jsonify({"error": "No samples recorded"}), 400

    # Save samples to disk
    features = np.array(samples)
    sample_path = os.path.join(SAMPLES_DIR, f"{gesture_id}.npy")

    # Append to existing if any
    if os.path.exists(sample_path):
        existing = np.load(sample_path)
        features = np.vstack([existing, features])

    os.makedirs(SAMPLES_DIR, exist_ok=True)
    np.save(sample_path, features)

    # Update sample count in config
    config = load_gestures()
    for g in config['gestures']:
        if g['id'] == gesture_id:
            g['samples_count'] = len(features)
            break
    save_gestures(config)

    return jsonify({
        "success": True,
        "gesture_id": gesture_id,
        "new_samples": len(samples),
        "total_samples": len(features)
    })


# ──────────────────────────────────────────────
# API: TRAINING
# ──────────────────────────────────────────────
@app.route('/api/train', methods=['POST'])
def train():
    config = load_gestures()

    def progress_cb(stage, percent, message):
        socketio.emit("training_progress", {
            "stage": stage,
            "percent": percent,
            "message": message
        })

    # Run training in a thread to not block
    def do_train():
        result = train_model(config, progress_callback=progress_cb)
        engine.reload_model()
        refresh_engine_config()
        socketio.emit("training_complete", result)

    thread = threading.Thread(target=do_train, daemon=True)
    thread.start()

    return jsonify({"success": True, "status": "training_started"})


# ──────────────────────────────────────────────
# API: STATUS & MODE CONTROL
# ──────────────────────────────────────────────
@app.route('/api/status', methods=['GET'])
def get_status():
    status = engine.get_status()
    config = load_gestures()
    status['total_custom_gestures'] = len(config.get('gestures', []))
    trained_count = sum(1 for g in config.get('gestures', []) if g.get('samples_count', 0) > 0)
    status['trained_gestures'] = trained_count
    return jsonify(status)


@app.route('/api/mode', methods=['POST'])
def set_mode():
    data = request.get_json()
    mode = data.get('mode', '')
    engine.set_mode(mode)
    return jsonify({"success": True, "mode": engine.current_mode})


@app.route('/api/engine/toggle', methods=['POST'])
def toggle_recognition():
    engine.custom_gesture_enabled = not engine.custom_gesture_enabled
    return jsonify({"custom_gesture_enabled": engine.custom_gesture_enabled})


@app.route('/api/engine/power', methods=['POST'])
def toggle_power():
    """Toggle engine on/off (pause/resume camera processing)."""
    if engine.paused:
        engine.resume()
    else:
        engine.pause()
    return jsonify({"paused": engine.paused, "running": engine.running})


# ──────────────────────────────────────────────
# SOCKET.IO EVENTS
# ──────────────────────────────────────────────
@socketio.on('connect')
def handle_connect():
    print("Client connected")
    socketio.emit("status_update", engine.get_status())


@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected")


# ──────────────────────────────────────────────
# STARTUP
# ──────────────────────────────────────────────
if __name__ == '__main__':
    # Load config and start engine
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

    config = load_gestures()
    engine.load_config(config)
    engine.start()

    print("\n" + "=" * 50)
    print("  FG Gesture Control Dashboard")
    print("  Open http://localhost:5000")
    print("=" * 50 + "\n")

    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)