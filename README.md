# FG Gesture Control

A **gesture-based desktop control system** with a customizable AI and a polished web dashboard.  
Control your desktop with hand gestures — add new gestures, train a model, and map them to actions.

## Two Ways to Run

| Mode | Command | Description |
|------|---------|-------------|
| **Web Dashboard** | `python app.py` | Full web UI at `http://localhost:5000` with gesture management, training, and live recognition |
| **Standalone** | `python MotionControl.py` | Original multi-mode hand tracker (Cursor, Piano, Drawing, Sandbox) |

---

## 🌐 Web Dashboard (`python app.py`)

Open `http://localhost:5000` in your browser to access the dashboard.

### Dashboard Page
- Live camera feed with hand skeleton overlay
- Current detected gesture + confidence level
- FPS counter and model status
- Real-time action log

### Gesture Library Page
- View all defined gestures as cards
- **Add** new gestures with name, icon, and action
- **Record** samples by holding a gesture in front of the camera
- **Change action mapping** via dropdown
- **Delete** gestures with confirmation

### Training Page
- One-click **Train Model** button
- Animated progress ring with status updates
- Accuracy display after training
- Per-gesture sample count overview

### Settings Page
- Toggle recognition on/off

### Available Desktop Actions
| Action | Keyboard Shortcut |
|--------|-------------------|
| Play / Pause Media | `Space` |
| Next Tab | `Ctrl+Tab` |
| Previous Tab | `Ctrl+Shift+Tab` |
| Volume Up / Down | Volume keys |
| Switch Window | `Alt+Tab` |
| Task View | `Win+Tab` |
| Close Window | `Alt+F4` |
| Screenshot | `Win+Shift+S` |
| Browser Back / Forward | `Alt+Left/Right` |
| Minimize All | `Win+M` |
| Lock Screen | `Win+L` |
| Mute / Unmute | Volume Mute |
| Next / Prev Track | Media keys |

---

## 🎮 Standalone Mode (`python MotionControl.py`)

### Cursor Mode (`C`) — Air Mouse
| Gesture | Action |
|---------|--------|
| Open hand | Move cursor |
| Pinch index + thumb | Left click |
| Ring finger down | Right click |
| Thumb crosses index knuckle | Scroll up |
| Pinky down | Scroll down |
| Thumb + pinky tips together | Task View |
| ✌️ Peace sign + move hand up/down | Volume control |
| All fingers closed except index | Voice dictation |

### Piano Mode (`P`) — Virtual Piano (C4–C5)
### Drawing Mode (`X`) — Finger Painting
| Gesture | Action |
|---------|--------|
| 1 finger (index only) | Draw |
| 2 fingers (index + middle) | Set color: Red |
| 3 fingers (index + middle + ring) | Set color: Blue |
| 4 fingers (index + middle + ring + pinky) | Set color: Green |
| Open palm (all 5 up) | Erase (large eraser circle) |
| Fist (no fingers up) | Clear entire canvas |
### Sandbox Mode (`Z`) — Landmark Debug View

Press `Q` to quit.

---

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies
- `opencv-python` — Camera and image processing
- `mediapipe` — Hand tracking
- `numpy` — Numeric operations
- `pywin32`, `mouse`, `keyboard` — Desktop control (Windows)
- `flask`, `flask-socketio` — Web server + real-time
- `scikit-learn` — Gesture classifier (Random Forest)

---

## How It Works

1. **MediaPipe** detects 21 hand landmarks in real-time
2. Landmarks are normalized (relative to wrist, scaled by hand size) into 63 features
3. A **Random Forest classifier** predicts which gesture is being shown
4. The mapped **desktop action** is executed via keyboard shortcuts
5. Everything is managed through the **web dashboard**

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the dashboard
python app.py

# 3. Open browser
# Go to http://localhost:5000

# 4. Add gestures, record samples, train, and use!
```
