"""
Gesture Engine — integrates the existing MotionControl modes (Cursor, Drawing)
and layers custom ML-classified gestures on top.

The original hardcoded gestures (click, scroll, dictation, task view, volume, etc.) are
preserved as the PRIMARY system.  Custom gestures trained via the web UI are a SECONDARY
system that runs alongside.
"""

import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import os
import math
import subprocess
import ctypes
import keyboard as kb_module

from gesture_classifier import extract_features, GesturePredictor

# Import existing mode classes and helpers from MotionControl.py
from MotionControl import (
    HandData, CursorMode, DrawingMode,
    draw_hud, draw_mode_flash, draw_loading_animation, draw_orbit_ellipses,
    MODE_CURSOR, MODE_DRAWING,
    clamp
)

# ── Temporal smoothing for ML gesture predictions ────────────────────────────
from collections import deque

class GestureSmoother:
    """
    Sliding-window majority vote over the last N gesture predictions.

    Every time a raw (gesture_id, confidence) pair arrives from the classifier,
    it is pushed into a fixed-length deque.  The reported gesture is whichever
    label appears most often in the window — but only if it clears a minimum
    vote-share threshold (default 60 %).  If no label clears that bar the
    smoother returns (None, 0.0), preventing jittery single-frame spikes from
    ever reaching the action layer.

    Args:
        window  – number of prediction slots kept (default 8 ≈ 40 ms @ 5-frame stride)
        threshold – fraction of window that must agree (default 0.60)
    """

    def __init__(self, window: int = 8, threshold: float = 0.60):
        self._window    = window
        self._threshold = threshold
        self._buf: deque = deque(maxlen=window)  # stores (gesture_id | None, confidence)

    # ------------------------------------------------------------------
    def update(self, gesture_id, confidence: float):
        """Push one raw prediction; return the smoothed (gesture_id, confidence)."""
        self._buf.append((gesture_id, confidence))
        return self.current()

    # ------------------------------------------------------------------
    def current(self):
        """Return best smoothed (gesture_id, avg_confidence) without a new sample."""
        if not self._buf:
            return None, 0.0

        # Count votes per label (None = no-detection)
        votes: dict = {}
        conf_sum: dict = {}
        for gid, conf in self._buf:
            votes[gid]    = votes.get(gid, 0) + 1
            conf_sum[gid] = conf_sum.get(gid, 0.0) + conf

        best_gid  = max(votes, key=votes.get)
        best_vote = votes[best_gid]

        if best_gid is None or best_vote < self._threshold * self._window:
            return None, 0.0

        avg_conf = conf_sum[best_gid] / best_vote
        return best_gid, avg_conf

    # ------------------------------------------------------------------
    def reset(self):
        """Clear the buffer (call when hand disappears or mode changes)."""
        self._buf.clear()


# ── Custom gesture action types ──
ACTION_TYPE_SHORTCUT = "shortcut"
ACTION_TYPE_APP = "app"
ACTION_TYPE_COMMAND = "command"

# ── Gesture shape catalog ──
# fingers: [thumb, index, middle, ring, pinky]
#   True  = must be UP/extended
#   False = must be DOWN/curled
#   None  = don't care (ignored in validation — used for ambiguous fingers)
# extended_count: exact count of True fingers (None = skip count check)
# curl_threshold: how far tip must be below MCP to count as curled (higher = stricter)
# spread_check: if True, fingers must be spread apart (for spider/claw gestures)
GESTURE_SHAPES = {
    # ── All fingers fully closed, thumb tucked ──
    "fist": {
        "name": "Fist", "icon": "✊",
        "desc": "All fingers fully closed into a fist",
        "fingers": [False, False, False, False, False],
        "extended_count": 0,
        "strict": True,   # all 5 fingers must match exactly
    },

    # ── All 5 fully extended and spread ──
    "open_palm": {
        "name": "Open Palm", "icon": "🖐️",
        "desc": "All 5 fingers fully extended and spread open",
        "fingers": [True, True, True, True, True],
        "extended_count": 5,
        "strict": True,
    },

    # ── Only thumb sticks up, all fingers curled ──
    "thumbs_up": {
        "name": "Thumbs Up", "icon": "👍",
        "desc": "Thumb pointing up, all 4 fingers curled into fist",
        "fingers": [True, False, False, False, False],
        "extended_count": 1,
        "strict": True,
    },

    # ── Index + middle up, ring + pinky + thumb down ──
    "peace_sign": {
        "name": "Peace Sign", "icon": "✌️",
        "desc": "Index and middle fingers up, others curled",
        "fingers": [False, True, True, False, False],
        "extended_count": 2,
        "strict": True,
    },

    # ── Only index finger extended ──
    "pointing": {
        "name": "Pointing", "icon": "👆",
        "desc": "Only index finger extended, others curled",
        "fingers": [False, True, False, False, False],
        "extended_count": 1,
        "strict": True,
    },

    # ── Thumb + pinky out, middle 3 curled ──
    "shaka": {
        "name": "Shaka", "icon": "🤙",
        "desc": "Thumb and pinky extended, middle 3 fingers curled",
        "fingers": [True, False, False, False, True],
        "extended_count": 2,
        "strict": True,
    },

    # ── Index + pinky up (horns), middle + ring + thumb down ──
    "rock_on": {
        "name": "Rock On / Horns", "icon": "🤘",
        "desc": "Index and pinky up, middle and ring curled, thumb tucked",
        "fingers": [False, True, False, False, True],
        "extended_count": 2,
        "strict": True,
    },

    # ── Index + middle + ring up, thumb + pinky down ──
    "three_fingers": {
        "name": "Three Fingers", "icon": "3️⃣",
        "desc": "Index, middle, and ring extended; thumb and pinky curled",
        "fingers": [False, True, True, True, False],
        "extended_count": 3,
        "strict": True,
    },

    # ── All 4 fingers up, thumb tucked ──
    "four_fingers": {
        "name": "Four Fingers", "icon": "4️⃣",
        "desc": "Index, middle, ring, pinky extended; thumb tucked",
        "fingers": [False, True, True, True, True],
        "extended_count": 4,
        "strict": True,
    },

    # ── Thumb + index pinched, middle + ring + pinky extended ──
    "ok_sign": {
        "name": "OK Sign", "icon": "👌",
        "desc": "Middle, ring, pinky extended; thumb and index form a circle",
        "fingers": [False, False, True, True, True],
        "extended_count": 3,
        "strict": True,
    },

    # ── Only middle finger up ──
    "middle_finger": {
        "name": "Middle Finger Up", "icon": "🖕",
        "desc": "Only middle finger extended, all others curled",
        "fingers": [False, False, True, False, False],
        "extended_count": 1,
        "strict": True,
    },

    # ── Thumb + index extended, middle + ring + pinky curled ──
    "finger_gun": {
        "name": "Finger Gun", "icon": "👉",
        "desc": "Thumb and index extended like a gun, others curled",
        "fingers": [True, True, False, False, False],
        "extended_count": 2,
        "strict": True,
    },

    # ── Spider / claw: all 5 fingers spread but bent/curved ──
    # Finger tips will be BELOW their PIP joints (curved), NOT above like open_palm.
    # We use a special "spider" validator that checks curvature, not just up/down.
    "spider": {
        "name": "Love you", "icon": "🤟",
        "desc": "Thumb, index finger and pink extended, others curled",
        "fingers": [True, True, None, None, True],  # custom validator handles this
        "extended_count": 3,                      # skip count check
        "strict": True,  
    },

    # ── Additional hand signs ──

    "crossed_fingers": {
        "name": "Crossed Fingers", "icon": "🤞",
        "desc": "Index and middle crossed, others curled",
        "fingers": [False, True, True, False, False],
        "extended_count": 2,
        "strict": True,
    },


    "palm_down": {
        "name": "Palm Down", "icon": "🫳",
        "desc": "Hand extended face down, all fingers out",
        "fingers": [True, True, True, True, True],
        "extended_count": 5,
        "strict": True,
    },

    "palm_up": {
        "name": "Palm Up", "icon": "🫴",
        "desc": "Hand extended face up, all fingers out",
        "fingers": [True, True, True, True, True],
        "extended_count": 5,
        "strict": True,
    },

    "vulcan_salute": {
        "name": "Vulcan Salute", "icon": "🖖",
        "desc": "4 fingers up split in middle, thumb extended",
        "fingers": [True, True, True, True, True],
        "extended_count": 5,
        "strict": True,
    },

    "raised_back": {
        "name": "Raised Back of Hand", "icon": "🤚",
        "desc": "All 5 fingers extended — back of hand shown",
        "fingers": [True, True, True, True, True],
        "extended_count": 5,
        "strict": True,
    },

}

# Shape IDs used by built-in gestures (excluded from custom gesture dropdown)
# Shapes that conflict with built-in cursor/drawing gestures — excluded from
# the custom gesture dropdown so users can't accidentally clash with them.
# Built-in cursor gestures use: open palm (move), pointing (voice), fist (volume),
# peace_sign-like states, pinch states, etc.
# Built-in drawing gestures use: pointing (draw), peace_sign (red), three_fingers (green),
# four_fingers (blue), open_palm (erase), fist (clear).
BUILTIN_SHAPES = {
    "fist",           # cursor: volume control; drawing: clear
    "open_palm",      # cursor: move cursor;    drawing: erase
    "pointing",       # cursor: voice dictation; drawing: draw
    "peace_sign",     # drawing: red colour
    "three_fingers",  # drawing: green colour
    "four_fingers",   # drawing: blue colour
    "index_pointing_up",  # same as pointing
    "thumbs_up",      # commonly misdetected as cursor gestures
}

FREEFORM_GESTURES = {
    f"custom_{i+1:02d}": {
        "name": f"Custom Gesture {i+1}",
        "icon": "",
        "desc": f"Custom gesture slot {i+1} — record any hand shape you like. "
                "Avoid gestures already used by built-in functions (open palm, fist, "
                "pointing index finger, peace sign, 3/4 fingers up).",
        "fingers": None,   # None = no shape constraint; pure ML decides
        "extended_count": None,
        "freeform": True,
    }
    for i in range(10)
}


def compute_finger_states(landmarks):
    """
    Compute per-finger extended/curled state from MediaPipe landmarks.
    Reads each landmark attribute exactly once and uses inline comparisons
    to avoid repeated attribute lookups.
    """
    lm = landmarks

    # Cache y-coords for the 4 fingers (tip / pip / mcp) — 12 attribute reads
    i_ty, i_py, i_my = lm[8].y,  lm[6].y,  lm[5].y
    m_ty, m_py, m_my = lm[12].y, lm[10].y, lm[9].y
    r_ty, r_py, r_my = lm[16].y, lm[14].y, lm[13].y
    p_ty, p_py, p_my = lm[20].y, lm[18].y, lm[17].y

    # Thumb: lateral distance from wrist (x-axis)
    th_tx = lm[4].x; th_ix = lm[3].x; w_x = lm[0].x
    thumb_dist_tip = abs(th_tx - w_x)
    thumb_dist_ip  = abs(th_ix - w_x)

    return {
        "thumb":  thumb_dist_tip > thumb_dist_ip * 1.1,
        "index":  i_ty < i_py or i_ty < i_my,
        "middle": m_ty < m_py or m_ty < m_my,
        "ring":   r_ty < r_py or r_ty < r_my,
        "pinky":  p_ty < p_py or p_ty < p_my,
    }


def check_spider_gesture(landmarks):
    """
    Detect a spider/claw hand: all 5 fingers spread and CURVED (bent at middle joints).
    Key difference from open_palm: in spider, fingertips are BELOW their PIP joints
    even though fingers are spread out — they curve downward like claws.

    Returns True if the hand matches a spider/claw shape.
    """
    lm = landmarks

    # For each finger, tip should be BELOW (y >) its PIP joint — meaning curled/bent
    # but spread away from the palm (tip not close to palm center)
    finger_tips  = [4,  8,  12, 16, 20]
    finger_pips  = [3,  6,  10, 14, 18]   # thumb IP, others PIP
    finger_mcps  = [2,  5,   9, 13, 17]

    curved_count = 0
    spread_count = 0

    for tip_i, pip_i, mcp_i in zip(finger_tips, finger_pips, finger_mcps):
        tip = lm[tip_i]
        pip = lm[pip_i]
        mcp = lm[mcp_i]

        # Curved: tip is BELOW pip (y inverted — tip.y > pip.y means tip hangs down)
        if tip.y > pip.y:
            curved_count += 1

        # Spread: tip is further from palm center than MCP (finger is extended outward)
        # Use distance from wrist (landmark 0) as proxy
        wrist = lm[0]
        dist_tip = ((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2) ** 0.5
        dist_mcp = ((mcp.x - wrist.x)**2 + (mcp.y - wrist.y)**2) ** 0.5
        if dist_tip > dist_mcp * 1.15:
            spread_count += 1

    # Spider requires at least 4 of 5 fingers curved AND spread
    return curved_count >= 4 and spread_count >= 4


def validate_finger_state(landmarks, shape_id):
    """
    Validate whether current hand landmarks match a given gesture shape.
    - For spider: uses curvature-based check (check_spider_gesture)
    - For all others: uses per-finger up/down matching + extended count
    - None in fingers[] = don't care (skip that finger)
    - Returns True if hand matches shape, False if not, True if shape unknown.
    """
    shape = GESTURE_SHAPES.get(shape_id)
    if not shape:
        return True  # unknown shape = no constraint = always pass

    # ── Spider/claw: special curvature validator ──
    if shape.get("spider_check"):
        return check_spider_gesture(landmarks)

    # ── All other shapes: finger up/down matching ──
    finger_states = compute_finger_states(landmarks)
    expected = shape["fingers"]
    keys = ["thumb", "index", "middle", "ring", "pinky"]

    for key, expected_val in zip(keys, expected):
        if expected_val is None:
            continue  # don't care — skip this finger
        if finger_states[key] != expected_val:
            return False

    # Extended count check (skipped if None)
    expected_count = shape.get("extended_count")
    if expected_count is not None:
        actual_count = sum(1 for k in keys if finger_states[k])
        if actual_count != expected_count:
            return False

    return True



class GestureEngine:
    """
    Core engine that runs the existing MotionControl modes AND custom ML gestures.
    Provides MJPEG video feed and Socket.IO status for the web dashboard.
    """

    def __init__(self, socketio=None):
        self.socketio = socketio
        self.predictor = GesturePredictor()
        self.predictor.load()

        # ── Existing mode state (from MotionControl main()) ──
        self.current_mode = MODE_CURSOR
        self.hand_counter = 0
        self.lost_hand_count = 0
        self.active_hand_count = 0
        self.hand_shown = False

        # Two-palm mode toggle state
        self.two_palm_start = None   # when both palms first detected side-by-side
        self.two_palm_toggled = False  # prevents re-toggle until hands separate

        # Mode handlers
        self.cursor_mode = CursorMode()
        self.drawing_mode = None  # initialized after first frame

        # Screen dimensions
        user32 = ctypes.windll.user32
        self.screen_w = user32.GetSystemMetrics(0)
        self.screen_h = user32.GetSystemMetrics(1)

        # ── Custom gesture state ──
        self.running = False
        self.paused = False
        self.recording = False
        self.recording_gesture_id = None
        self.recorded_samples = []
        self.current_custom_gesture = None
        self.current_custom_confidence = 0.0
        self.custom_gesture_enabled = True
        self.action_cooldown_time = 0.0
        self.action_log = []
        self.gesture_last_seen_time = 0.0  # decay timer for stale gesture display
        self.current_finger_states = {}  # live finger up/down state for tutorial
        
        # ── Temporal smoothing ──
        # 8-frame window, 60 % agreement required before a gesture is reported.
        # Keeps single-frame classifier noise from ever triggering an action.
        self.smoother = GestureSmoother(window=8, threshold=0.60)

        # ── App launch tracking (for app-type gestures) ──
        self.app_gesture_hold_start = {}  # gesture_id -> timestamp when gesture started being held
        self.launched_apps = {}  # action_value -> timestamp when app was last launched
        self.app_hold_threshold = 4.0  # seconds to hold gesture before reopening an app

        # ── FPS ──
        self.fps = 0

        # ── Camera ──
        self.cap = None
        self.frame_lock = threading.Lock()
        self.output_frame = None

        # ── MediaPipe ──
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        # ── Config ──
        self.gestures_config = None
        self.custom_actions_map = {}

    def load_config(self, gestures_config):
        """Load gesture config and build custom action lookup."""
        self.gestures_config = gestures_config
        self.custom_actions_map = {}
        self._gesture_name_cache = {}
        # gesture_shape is purely cosmetic (icon only) — not a recognition filter.
        self._configured_shapes = {}   # always empty — no shape gating
        self._has_unconstrained = True  # always True — ML runs every frame
        for g in gestures_config.get("gestures", []):
            gid = g["id"]
            self.custom_actions_map[gid] = {
                "name": g["name"],
                "action_type": g.get("action_type", ACTION_TYPE_SHORTCUT),
                "action_value": g.get("action_value", ""),
                "action_label": g.get("action_label", g.get("name", "")),
                "gesture_shape": g.get("gesture_shape"),
            }
            self._gesture_name_cache[gid] = g.get("name", gid)

    def reload_model(self):
        """Reload the ML classifier from disk."""
        success = self.predictor.load()
        if success and self.socketio:
            self.socketio.emit("model_reloaded", {"loaded": True})
        return success

    def set_mode(self, mode):
        """Switch the active mode (called from web UI)."""
        if mode in (MODE_CURSOR, MODE_DRAWING):
            if mode != self.current_mode:
                self.current_mode = mode
                self.smoother.reset()   # stale window from old mode is irrelevant
                if mode != MODE_CURSOR:
                    self.cursor_mode.reset_trail()

    def start(self):
        """Start the engine in a background thread."""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the engine."""
        self.running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def pause(self):
        """Pause the engine (camera stays open but processing stops)."""
        self.paused = True
        # Release any held click when pausing
        if hasattr(self, 'cursor_mode') and self.cursor_mode.left_click_down:
            import win32api, win32con
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            self.cursor_mode.left_click_down = False

    def resume(self):
        """Resume the engine."""
        self.paused = False
        # Clear frozen paused frame so the live feed picks up immediately
        with self.frame_lock:
            self.output_frame = None

    def start_recording(self, gesture_id):
        """Begin recording ML samples for a custom gesture."""
        self.recording = True
        self.recording_gesture_id = gesture_id
        self.recorded_samples = []

    def stop_recording(self):
        """Stop recording and return samples — app.py handles saving to disk."""
        self.recording = False
        samples = self.recorded_samples.copy()
        gesture_id = self.recording_gesture_id
        self.recording_gesture_id = None
        self.recorded_samples = []
        # Do NOT save here — app.py stop route is the sole writer to avoid doubles.
        return gesture_id, samples

    def get_status(self):
        """Return current engine status as dict."""
        return {
            "running": self.running,
            "paused": self.paused,
            "hand_detected": self.hand_shown,
            "current_mode": self.current_mode,
            "current_custom_gesture": self.current_custom_gesture,
            "current_custom_confidence": round(self.current_custom_confidence * 100, 1),
            "fps": self.fps,
            "model_loaded": self.predictor.loaded,
            "recording": self.recording,
            "recording_gesture_id": self.recording_gesture_id,
            "recorded_count": len(self.recorded_samples),
            "custom_gesture_enabled": self.custom_gesture_enabled,
            "action_log": self.action_log[-10:]
        }

    def _get_gesture_shape(self, gesture_id):
        """Look up the gesture shape constraint for a gesture ID."""
        mapping = self.custom_actions_map.get(gesture_id)
        if mapping:
            return mapping.get("gesture_shape")
        return None

    # ────────────────────────────────────────────
    # CUSTOM GESTURE ACTION EXECUTION
    # ────────────────────────────────────────────
    # ── Windows VK codes for media / special keys ──────────────────────────────
    _VK_MEDIA_NEXT    = 0xB0
    _VK_MEDIA_PREV    = 0xB1
    _VK_MEDIA_STOP    = 0xB2
    _VK_MEDIA_PLAY    = 0xB3
    _VK_VOLUME_MUTE   = 0xAD
    _VK_VOLUME_DOWN   = 0xAE
    _VK_VOLUME_UP     = 0xAF
    _VK_LWIN          = 0x5B
    _VK_SHIFT         = 0x10
    _VK_TAB           = 0x09
    _VK_M             = 0x4D
    _VK_D             = 0x44
    _VK_E             = 0x45
    _VK_I             = 0x49
    _VK_S             = 0x53
    _KEYEVENTF_KEYUP  = 0x0002

    def _vk_tap(self, *vk_codes):
        """Press then release a sequence of VK codes via keybd_event."""
        ke = ctypes.windll.user32.keybd_event
        for vk in vk_codes:
            ke(vk, 0, 0, 0)
        for vk in reversed(vk_codes):
            ke(vk, 0, self._KEYEVENTF_KEYUP, 0)

    def _send_shortcut(self, value):
        """
        Send any keyboard shortcut robustly.
        Routes media keys and Windows-key combos through ctypes (which work
        reliably) and everything else through keyboard.send().
        Normalises the value so spaces vs underscores don't matter.
        """
        v = value.lower().replace(" ", "").replace("-", "")

        # Lock screen — only LockWorkStation() works (Win+L is kernel-intercepted)
        if v in ("windows+l", "win+l"):
            ctypes.windll.user32.LockWorkStation()
            return

        # Media / volume keys — must use VK codes, keyboard.send() can't do these
        media_vk = {
            "volumeup":          self._VK_VOLUME_UP,
            "volume_up":         self._VK_VOLUME_UP,
            "volumedown":        self._VK_VOLUME_DOWN,
            "volume_down":       self._VK_VOLUME_DOWN,
            "volumemute":        self._VK_VOLUME_MUTE,
            "volume_mute":       self._VK_VOLUME_MUTE,
            "mute":              self._VK_VOLUME_MUTE,
            "nexttrack":         self._VK_MEDIA_NEXT,
            "next_track":        self._VK_MEDIA_NEXT,
            "previoustrack":     self._VK_MEDIA_PREV,
            "previous_track":    self._VK_MEDIA_PREV,
            "prev_track":        self._VK_MEDIA_PREV,
            "mediaplaypause":    self._VK_MEDIA_PLAY,
            "media_play_pause":  self._VK_MEDIA_PLAY,
            "mediastop":         self._VK_MEDIA_STOP,
        }
        # Strip all separators for matching
        v_clean = v.replace("+", "").replace("_", "")
        for key, vk in media_vk.items():
            if v_clean == key.replace("_", ""):
                self._vk_tap(vk)
                return

        # Windows-key combos — keyboard lib is unreliable for these
        win_combos = {
            "windows+tab":      (self._VK_LWIN, self._VK_TAB),
            "win+tab":          (self._VK_LWIN, self._VK_TAB),
            "windows+shift+s":  (self._VK_LWIN, self._VK_SHIFT, self._VK_S),
            "win+shift+s":      (self._VK_LWIN, self._VK_SHIFT, self._VK_S),
            "windows+m":        (self._VK_LWIN, self._VK_M),
            "win+m":            (self._VK_LWIN, self._VK_M),
            "windows+d":        (self._VK_LWIN, self._VK_D),
            "win+d":            (self._VK_LWIN, self._VK_D),
            "windows+e":        (self._VK_LWIN, self._VK_E),
            "win+e":            (self._VK_LWIN, self._VK_E),
            "windows+i":        (self._VK_LWIN, self._VK_I),
            "win+i":            (self._VK_LWIN, self._VK_I),
        }
        if v in win_combos:
            self._vk_tap(*win_combos[v])
            return

        # Everything else — standard keyboard.send()
        kb_module.send(value)

    def _execute_custom_action(self, gesture_id, confidence, hold_duration=0.0):
        """Execute the action mapped to a custom ML gesture."""
        now = time.time()
        if now - self.action_cooldown_time < 1.5:
            return
        if confidence < 0.65:
            return

        mapping = self.custom_actions_map.get(gesture_id)
        if not mapping or not mapping["action_value"]:
            return

        action_type = mapping["action_type"]
        action_value = mapping["action_value"]
        action_label = mapping["action_label"]

        try:
            if action_type in (ACTION_TYPE_SHORTCUT, "preset"):
                self._send_shortcut(action_value)
                self.action_cooldown_time = now

            elif action_type == ACTION_TYPE_APP:
                # Always set cooldown so the gesture can't rapid-fire even
                # if we decide not to launch (e.g. already running).
                self.action_cooldown_time = now
                last_launch = self.launched_apps.get(action_value, 0)
                if (now - last_launch) >= 2.0 or hold_duration >= self.app_hold_threshold:
                    # Quote paths that contain spaces so the shell handles them correctly
                    launch_cmd = action_value
                    if ' ' in action_value and not action_value.startswith('"'):
                        launch_cmd = f'"{action_value}"'
                    subprocess.Popen(launch_cmd, shell=True)
                    self.launched_apps[action_value] = now

            elif action_type == ACTION_TYPE_COMMAND:
                subprocess.Popen(action_value, shell=True)
                self.action_cooldown_time = now

            log_entry = {
                "gesture": mapping["name"],
                "action": action_label,
                "type": action_type,
                "confidence": round(confidence * 100, 1),
                "time": time.strftime("%H:%M:%S")
            }
            self.action_log.append(log_entry)
            if len(self.action_log) > 50:
                self.action_log = self.action_log[-50:]
            if self.socketio:
                self.socketio.emit("action_executed", log_entry)
        except Exception as e:
            print(f"Custom action execution error: {e}")


    # ────────────────────────────────────────────
    # MAIN CAMERA + RECOGNITION LOOP
    # ────────────────────────────────────────────
    def _run_loop(self):
        """Main loop: camera → MediaPipe → existing modes + custom ML gestures."""
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        # Only buffer 1 frame — we always want the LATEST frame, not a stale queue
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # ── Dedicated capture thread ──────────────────────────────────────────
        # cap.read() blocks ~33ms waiting for the next camera frame.
        # Running it inline stalls the whole pipeline each cycle.
        # A background thread drains the camera continuously so the processing
        # loop always grabs the freshest frame with zero blocking wait.
        _cap_lock   = threading.Lock()
        _latest_raw = [None]
        _cap_alive  = [True]
        _frame_seq  = [0]          # incremented each time camera produces a new frame

        def _capture_loop():
            while _cap_alive[0] and self.running:
                ok, frm = self.cap.read()
                if ok:
                    with _cap_lock:
                        _latest_raw[0] = frm
                        _frame_seq[0] += 1
        _cap_thread = threading.Thread(target=_capture_loop, daemon=True)
        _cap_thread.start()

        # ── One-time setup ────────────────────────────────────────────────────
        frame_count       = 0
        total_frame_count = 0
        fps_start         = time.time()
        fps_str           = "FPS: 0"          # cache formatted string
        custom_stable_count  = 0
        last_custom_gesture  = None
        predict_frame_counter = 0

        # Pre-built DrawingSpec — avoids Python object allocation per frame
        _lm_style  = self.mp_drawing.DrawingSpec(thickness=2, circle_radius=1, color=(0, 127, 255))
        _con_style = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=0, color=(0, 80, 160))

        # MediaPipe runs on a DOWNSCALED frame (320×240 = 4× fewer pixels).
        # Landmark coords are normalized 0–1 so they're resolution-independent.
        # The full 640×480 frame is used only for display / drawing.
        MP_W, MP_H = 320, 240
        _small_buf = np.empty((MP_H, MP_W, 3), dtype=np.uint8)  # BGR reusable
        _rgb_small = np.empty((MP_H, MP_W, 3), dtype=np.uint8)  # RGB reusable

        # Non-blocking socketio emit queue
        import queue as _queue
        _emit_q = _queue.Queue(maxsize=16)

        def _emit_worker():
            while self.running or not _emit_q.empty():
                try:
                    event, data = _emit_q.get(timeout=1.0)
                    if self.socketio:
                        self.socketio.emit(event, data)
                    _emit_q.task_done()
                except _queue.Empty:
                    pass

        _emit_thread = threading.Thread(target=_emit_worker, daemon=True)
        _emit_thread.start()

        def _safe_emit(event, data):
            try:
                _emit_q.put_nowait((event, data))
            except _queue.Full:
                pass

        # Wait for the first frame before entering MediaPipe context
        while _latest_raw[0] is None and self.running:
            time.sleep(0.01)

        image_h, image_w = 480, 640  # initialise; updated on first real frame

        with self.mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            max_num_hands=1,
        ) as hands:

            _last_seq = -1

            while self.running:
                # ── Grab latest frame (non-blocking) ──────────────────────────
                with _cap_lock:
                    image    = _latest_raw[0]
                    this_seq = _frame_seq[0]

                if image is None:
                    time.sleep(0.004)
                    continue

                # Camera hasn't produced a new frame yet — yield CPU, don't spin
                if this_seq == _last_seq:
                    time.sleep(0.002)
                    continue
                _last_seq = this_seq

                image_h, image_w = image.shape[:2]

                # Init drawing canvas once
                if self.drawing_mode is None:
                    self.drawing_mode = DrawingMode(image_w, image_h)

                # ── Paused ────────────────────────────────────────────────────
                if self.paused:
                    if self.output_frame is None:
                        img_p = cv2.flip(image, 1)
                        ov = img_p.copy()
                        cv2.rectangle(ov, (0, 0), (image_w, image_h), (0, 0, 0), -1)
                        cv2.addWeighted(ov, 0.7, img_p, 0.3, 0, img_p)
                        (tw, th), _ = cv2.getTextSize("PAUSED", cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
                        cv2.putText(img_p, "PAUSED",
                                    ((image_w - tw) // 2, (image_h + th) // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 100), 3)
                        with self.frame_lock:
                            self.output_frame = img_p
                    time.sleep(0.05)
                    continue

                # ── MediaPipe on downscaled frame ─────────────────────────────
                # Resize into pre-allocated buffer; convert BGR→RGB in-place
                cv2.resize(image, (MP_W, MP_H), dst=_small_buf,
                           interpolation=cv2.INTER_LINEAR)
                cv2.cvtColor(_small_buf, cv2.COLOR_BGR2RGB, dst=_rgb_small)
                _rgb_small.flags.writeable = False
                results = hands.process(_rgb_small)
                _rgb_small.flags.writeable = True

                # ── Hand detected ─────────────────────────────────────────────
                now = time.time()   # single time.time() call per frame
                if results.multi_hand_landmarks:
                    self.lost_hand_count  = 0
                    self.active_hand_count += 1
                    self.hand_shown = True

                    is_left_hand = False
                    for i in results.multi_handedness:
                        if i.classification[0].label == 'Right':
                            is_left_hand = True

                    hand_landmarks = results.multi_hand_landmarks[0]
                    landmarks      = hand_landmarks.landmark

                    # Build HandData (works on normalized coords — resolution-independent)
                    hd = HandData(hand_landmarks, self.screen_w, self.screen_h, image_w, image_h)

                    # Compute finger states once; reuse everywhere
                    self.current_finger_states = compute_finger_states(landmarks)

                    # ── Recording ────────────────────────────────────────────
                    if self.recording:
                        features = extract_features(landmarks)
                        self.recorded_samples.append(features)
                        count   = len(self.recorded_samples)
                        prog    = min(count / 400, 1.0)
                        bar_w   = int(200 * prog)
                        cv2.circle(image, (30, 30), 12, (0, 0, 255), -1)
                        cv2.putText(image, f"REC: {count} samples",
                                    (50, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.rectangle(image, (50, 50), (250, 65), (60, 60, 60), -1)
                        cv2.rectangle(image, (50, 50), (50 + bar_w, 65), (0, 200, 255), -1)
                        if total_frame_count % 15 == 0:
                            _safe_emit("recording_progress", {
                                "gesture_id": self.recording_gesture_id,
                                "count": count
                            })

                    # ── Normal processing ─────────────────────────────────────
                    else:
                        self.hand_counter = min(self.hand_counter + 2, 60)

                        if self.current_mode == MODE_CURSOR:
                            image = self.cursor_mode.process(
                                image, hd, self.screen_w, self.screen_h,
                                image_w, image_h, is_left_hand)
                        elif self.current_mode == MODE_DRAWING:
                            image = self.drawing_mode.process(
                                image, hd, self.screen_w, self.screen_h,
                                image_w, image_h)

                        # ── Custom ML (every 5th frame) ───────────────────────
                        predict_frame_counter += 1
                        if (self.custom_gesture_enabled and
                                self.predictor.loaded and
                                predict_frame_counter % 5 == 0):

                            has_any_match = self._has_unconstrained or not self._configured_shapes
                            if not has_any_match:
                                for shape_id in self._configured_shapes.values():
                                    if validate_finger_state(landmarks, shape_id):
                                        has_any_match = True
                                        break

                            # ── Raw prediction ───────────────────────────────
                            raw_gid, raw_conf = None, 0.0
                            if has_any_match:
                                raw_gid, raw_conf = self.predictor.predict(landmarks)

                            if raw_gid and raw_gid not in self.custom_actions_map:
                                raw_gid, raw_conf = None, 0.0

                            if raw_gid and raw_conf > 0.70:
                                shape = self._configured_shapes.get(raw_gid)
                                if shape and not validate_finger_state(landmarks, shape):
                                    raw_gid, raw_conf = None, 0.0

                            # ── Temporal smoothing (sliding-window majority vote) ──
                            gid, conf = self.smoother.update(raw_gid, raw_conf)

                            if gid and conf > 0.70:
                                self.current_custom_gesture    = gid
                                self.current_custom_confidence = conf
                                self.gesture_last_seen_time    = now

                                if gid == last_custom_gesture:
                                    custom_stable_count += 1
                                    if gid not in self.app_gesture_hold_start:
                                        self.app_gesture_hold_start[gid] = now
                                    hold_duration = now - self.app_gesture_hold_start[gid]
                                else:
                                    custom_stable_count = 1
                                    last_custom_gesture = gid
                                    self.app_gesture_hold_start[gid] = now
                                    hold_duration = 0.0

                                if custom_stable_count >= 4:
                                    self._execute_custom_action(gid, conf, hold_duration)

                                cv2.putText(image,
                                            f"Custom: {self._gesture_name_cache.get(gid, gid)} ({int(conf*100)}%)",
                                            (10, image_h - 20),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 150), 2)
                            else:
                                if last_custom_gesture:
                                    self.app_gesture_hold_start.pop(last_custom_gesture, None)
                                self.current_custom_gesture    = None
                                self.current_custom_confidence = 0.0
                                custom_stable_count  = 0
                                last_custom_gesture  = None
                        else:
                            # Decay stale gesture label
                            if (self.current_custom_gesture and
                                    now - self.gesture_last_seen_time > 0.5):
                                if last_custom_gesture:
                                    self.app_gesture_hold_start.pop(last_custom_gesture, None)
                                self.current_custom_gesture    = None
                                self.current_custom_confidence = 0.0
                                custom_stable_count  = 0
                                last_custom_gesture  = None

                    # Skeleton overlay (pre-built specs — no allocation)
                    self.mp_drawing.draw_landmarks(
                        image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        _lm_style, _con_style)

                else:
                    # No hand
                    self.lost_hand_count += 1
                    self.cursor_mode.reset_trail()
                    self.cursor_mode.x_arr.clear()
                    self.cursor_mode.y_arr.clear()
                    if self.lost_hand_count >= 20:
                        self.hand_counter      = 0
                        self.active_hand_count = 0
                        self.hand_shown        = False
                    if last_custom_gesture:
                        self.app_gesture_hold_start.pop(last_custom_gesture, None)
                    self.current_custom_gesture    = None
                    self.current_custom_confidence = 0.0
                    custom_stable_count  = 0
                    last_custom_gesture  = None
                    self.smoother.reset()   # clear stale window when hand leaves frame

                # ── FPS (update string only when value changes) ───────────────
                frame_count       += 1
                total_frame_count += 1
                elapsed = now - fps_start
                if elapsed >= 1.0:
                    new_fps = int(frame_count / elapsed)
                    if new_fps != self.fps:
                        self.fps = new_fps
                        fps_str  = f"FPS: {new_fps}"
                    frame_count = 0
                    fps_start   = now

                # ── Mirror + HUD ──────────────────────────────────────────────
                image = cv2.flip(image, 1)
                cv2.putText(image, fps_str, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)

                # Push to MJPEG stream (no copy — reference swap)
                with self.frame_lock:
                    self.output_frame = image

                # Periodic websocket emits (off the hot path via queue)
                if total_frame_count % 30 == 0:
                    _safe_emit("status_update", self.get_status())
                if self.current_finger_states and total_frame_count % 20 == 0:
                    _safe_emit("finger_state", self.current_finger_states)
                if results.multi_hand_landmarks and total_frame_count % 20 == 0:
                    try:
                        lm = landmarks
                        tt = lm[4]; it = lm[8]; pt = lm[20]
                        wr = lm[0]; mm = lm[9]; im = lm[5]; pm = lm[17]
                        _safe_emit("tutorial_metrics", {
                            "pinch_thumb_index": ((tt.x-it.x)**2+(tt.y-it.y)**2)**0.5,
                            "pinch_thumb_pinky": ((tt.x-pt.x)**2+(tt.y-pt.y)**2)**0.5,
                            "palm_span":          abs(pm.x - im.x),
                            "thumb_cross_index":  tt.x > im.x,
                            "rotation_deg":       math.degrees(math.atan2(mm.y-wr.y, mm.x-wr.x))+90,
                        })
                    except Exception:
                        pass

        _cap_alive[0] = False
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def generate_mjpeg(self):
        """Generator that yields MJPEG frames for video streaming."""
        encode_params   = [cv2.IMWRITE_JPEG_QUALITY, 45]
        target_interval = 1.0 / 12.0   # 12 FPS display — smooth enough, low CPU
        last_frame_id   = -1

        while self.running:
            t0 = time.time()

            with self.frame_lock:
                frame_ref = self.output_frame

            if frame_ref is None:
                time.sleep(0.04)
                continue

            fid = id(frame_ref)
            if fid == last_frame_id:
                time.sleep(0.008)
                continue
            last_frame_id = fid

            frame = frame_ref.copy()          # copy outside the lock
            ret, buf = cv2.imencode('.jpg', frame, encode_params)
            if not ret:
                continue

            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                   buf.tobytes() + b'\r\n')

            wait = target_interval - (time.time() - t0)
            if wait > 0:
                time.sleep(wait)