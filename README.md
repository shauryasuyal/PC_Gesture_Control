<div align="center">

# FG Gesture Control

**Control your desktop with nothing but your hand.**

A real-time hand gesture recognition system powered by MediaPipe and scikit-learn — with a live web dashboard to manage, record, and train your own custom gestures.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-SocketIO-black?logo=flask)](https://flask-socketio.readthedocs.io/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-green?logo=google)](https://mediapipe.dev)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange?logo=scikit-learn)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-purple)](LICENSE)

SCREENSHOTS - https://drive.google.com/drive/folders/1Psc7fawUpmTPLp4yR0qQ7gwrdIuPqYZS?usp=sharing

</div>

---

## What Is This?

FG Gesture Control lets you operate your Windows desktop entirely through hand gestures picked up by a webcam. It ships with a set of **built-in gestures** (cursor movement, clicking, scrolling, drawing, and more) and a full **machine-learning pipeline** that lets you define, record, and train your own custom gestures — all through a slick browser-based dashboard.

No special hardware. No cloud. Everything runs locally, in real time.

---

## Demo

> Open `http://localhost:5000` after launching — you'll see your camera feed, live gesture detection, action logs, and all controls in one place.

<div align="center">

| Dashboard | Gestures | Training |
|-----------|----------|----------|
| Live camera feed + action log | Create & record custom gestures | Train the ML model with one click |

</div>

---

## How It Works

The system runs two gesture layers simultaneously:

### Layer 1 — Built-in Gestures (Rule-Based)
These are hardcoded, deterministic, and always active. They use MediaPipe's 21 hand landmark positions to detect specific finger configurations in real time.

| Mode | Gesture | Action |
|------|---------|--------|
| **Cursor** | Open palm | Move cursor |
| **Cursor** | Pinch (thumb + index) | Left click |
| **Cursor** | Double pinch | Double click |
| **Cursor** | Ring finger down | Right click |
| **Cursor** | Index + pinky up | Scroll |
| **Cursor** | Fist rotate | Volume control |
| **Cursor** | Four fingers up | Task View |
| **Cursor** | Peace sign | Voice dictation |
| **Drawing** | Index out | Draw |
| **Drawing** | Rotate fist | Change color |
| **Drawing** | Fist | Clear canvas |

### Layer 2 — Custom Gestures (ML-Based)
You define these yourself. The app collects hand landmark samples, trains a **Random Forest classifier**, and then runs predictions alongside Layer 1 every few frames. Custom gesture predictions go through **temporal smoothing** (a sliding-window majority vote) to eliminate single-frame jitter before any action is triggered.

---

## Architecture

```
FG Gesture Control
│
├── app.py                  ← Flask + SocketIO server, REST API
├── gesture_engine.py       ← Core processing loop: MediaPipe → modes → ML
├── gesture_classifier.py   ← ML pipeline: feature extraction, training, prediction
├── MotionControl.py        ← Built-in gesture modes (Cursor, Drawing)
│
├── static/
│   ├── app.js              ← Navigation, SocketIO client, toasts, modals
│   ├── dashboard.js        ← Live camera feed, action log, power controls
│   ├── gestures.js         ← Gesture management UI (add, record, delete)
│   ├── training.js         ← Training page with progress ring
│   └── tutorial.js         ← Interactive tutorial overlays
│
├── templates/
│   └── index.html          ← Single-page app shell
│
├── data/
│   ├── gestures.json       ← Gesture config & metadata
│   └── samples/            ← .npy files of recorded landmark vectors
│
└── models/
    ├── gesture_model.pkl   ← Trained RandomForest
    └── label_encoder.pkl   ← sklearn LabelEncoder
```

---

## Key Technical Details

### Feature Extraction
Each hand frame produces 21 MediaPipe landmarks (x, y, z). These are:
1. **Translated** relative to the wrist (translation invariance)
2. **Scaled** by the wrist-to-middle-MCP distance (scale invariance)
3. **Flattened** to a 63-element feature vector

### Temporal Smoothing
Raw predictions run every 5th frame. A `GestureSmoother` maintains a sliding deque of the last 8 predictions and requires **≥ 60% vote share** before reporting a gesture as active. This prevents jittery false triggers.

### Gesture Validation
Before any ML prediction is acted on, the engine cross-checks the predicted gesture's expected finger configuration against the live landmark state. If the fingers don't match the trained shape, the prediction is suppressed — adding a second layer of robustness.

### Action Execution Cooldown
After a custom gesture fires an action, a per-gesture cooldown prevents repeated triggering while the pose is held. App-launch gestures use a **hold-duration** model (trigger once on hold, not on every frame).

---

## Getting Started

### Prerequisites

- Python **3.9+**
- Windows (uses `win32api`, `winsound`, and `ctypes` for desktop control)
- A working **webcam**

### Installation

```bash
# 1. Clone the repository

# 2. Create a virtual environment (recommended)

# 3. Install dependencies

```

<details>
<summary> Core dependencies</summary>

```
flask
flask-socketio
mediapipe
opencv-python
scikit-learn
numpy
pywin32
mouse
keyboard
```

</details>

### Running the App

```bash
python app.py
```

Then open your browser to **http://localhost:5000**

---

## Creating Your First Custom Gesture

1. **Go to the Gestures page** in the dashboard
2. Click **"Add Custom Gesture"**
3. Give it a name, choose a hand shape from the dropdown, and assign an action (keyboard shortcut, app launch, or terminal command)
4. Click **"Record"** — hold the gesture in front of your webcam for ~10–30 seconds
5. Head to the **Training page** and click **"Train Model"**
6. Done! Your new gesture is live immediately after training completes

> **Tip:** Record at least 150–300 samples per gesture for reliable recognition. More variety in your recording (slightly different angles, distances) improves robustness.

---

## Gesture Shapes Available

The system ships with a library of named hand shapes you can use as a starting point, each with validation logic to guard against false positives:

| Shape | Icon | Description |
|-------|------|-------------|
| Fist | ✊ | All fingers closed |
| Open Palm | 🖐️ | All 5 fingers extended |
| Thumbs Up | 👍 | Thumb up, fingers curled |
| Peace Sign | ✌️ | Index + middle up |
| Pointing | 👆 | Index only |
| Shaka | 🤙 | Thumb + pinky out |
| Rock On | 🤘 | Index + pinky up |
| Three Fingers | 3️⃣ | Index, middle, ring up |
| Four Fingers | 4️⃣ | All fingers, thumb tucked |
| + 10 free-form slots | 🔮 | Train any shape you invent |

---

## 📡 REST API Reference

The Flask server exposes a simple REST API if you want to integrate or script against it:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/status` | Engine status, FPS, current gesture, mode |
| `GET` | `/api/gestures` | All gesture configs and sample counts |
| `POST` | `/api/gestures` | Create a new custom gesture |
| `DELETE` | `/api/gestures/:id` | Delete a gesture and its samples |
| `POST` | `/api/gestures/:id/record` | Start recording samples |
| `POST` | `/api/gestures/:id/stop` | Stop recording and save |
| `POST` | `/api/train` | Kick off model training (async via SocketIO) |
| `POST` | `/api/model/reload` | Hot-reload trained model into live engine |
| `POST` | `/api/mode` | Switch between CURSOR / DRAWING modes |
| `POST` | `/api/engine/toggle` | Enable/disable custom gesture recognition |
| `POST` | `/api/engine/power` | Pause/resume camera processing |

### SocketIO Events

| Event | Direction | Payload |
|-------|-----------|---------|
| `status_update` | Server → Client | Engine status object |
| `action_executed` | Server → Client | `{gesture, action, confidence, time}` |
| `recording_progress` | Server → Client | `{gesture_id, count}` |
| `training_progress` | Server → Client | `{stage, percent, message}` |
| `training_complete` | Server → Client | Training result with accuracy |

---

## Project Structure Details

### `gesture_engine.py`
The heart of the system. Runs a background thread that:
- Reads webcam frames via OpenCV
- Runs MediaPipe hand detection
- Passes landmarks to the active mode (Cursor/Drawing)
- Every 5th frame: runs ML prediction + smoothing + validation
- Executes actions and emits WebSocket events
- Streams MJPEG frames to the browser

### `gesture_classifier.py`
Self-contained ML module with:
- `extract_features()` — deterministic, translation+scale invariant landmark vectorizer
- `train_model()` — full training pipeline with cross-validation and per-class confusion warnings
- `GesturePredictor` — loads the saved model and serves real-time predictions
- Automatic **background noise class injection** when only 1 gesture exists, to prevent overconfident false positives

### `MotionControl.py`
The original standalone gesture controller, now imported as a library. Contains `CursorMode`, `DrawingMode`, HUD rendering helpers, and all the deterministic gesture logic.

---

## Configuration

Gesture metadata is stored in `data/gestures.json`. You can edit this manually, but the dashboard handles it automatically.

The ML model lives in `models/` and is hot-reloaded after training — no restart needed.

---

## Training Tips

- **Minimum samples:** ~100 per gesture (more = better)
- **Variety helps:** Record at slightly different distances and angles
- **Distinct shapes:** The more visually different your gestures are from each other, the higher your accuracy
- **Warning system:** After training, the app surfaces specific confusion warnings (e.g., *"'thumbs_up' is being confused with 'shaka'"*) to guide re-recording
- **One gesture only?** The trainer automatically injects synthetic background noise data to prevent the model from predicting that gesture for literally everything


## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

Built with 🖐️, OpenCV, MediaPipe, Flask, and a lot of patience.

**If this project helped you, give it a ⭐ — it means a lot!**

</div>
