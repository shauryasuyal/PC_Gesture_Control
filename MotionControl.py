"""
FG Motion Control — Multi-Mode Hand Tracker
Modes: Cursor (C) | Piano (P) | Drawing (X) | Sandbox (Z)
Press Q to quit.
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import time
import random
import ctypes
import winsound
import threading

# Windows-specific imports
import win32api
import win32con
import mouse
import keyboard

# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────
MODE_CURSOR  = "CURSOR"
MODE_PIANO   = "PIANO"
MODE_DRAWING = "DRAWING"
MODE_SANDBOX = "SANDBOX"

# Performance / rendering toggles
PERFORMANCE_MODE = True     # global performance switch
DRAW_SKELETON    = False    # set True if you want the hand skeleton overlay

if PERFORMANCE_MODE:
    MAX_HANDS   = 1
    CAM_WIDTH   = 480
    CAM_HEIGHT  = 360
else:
    MAX_HANDS   = 2
    CAM_WIDTH   = 640
    CAM_HEIGHT  = 480

# Piano note frequencies (C4 to C5 white keys)
PIANO_NOTES = {
    0: ("C4",  262),
    1: ("D4",  294),
    2: ("E4",  330),
    3: ("F4",  349),
    4: ("G4",  392),
    5: ("A4",  440),
    6: ("B4",  494),
    7: ("C5",  523),
}

# Drawing color palette mapped to rotation angle ranges
DRAW_COLORS = [
    (0,   0,   255),   # Red
    (0,   127, 255),   # Orange
    (0,   255, 255),   # Yellow
    (0,   255, 0),     # Green
    (255, 255, 0),     # Cyan
    (255, 0,   0),     # Blue
    (255, 0,   127),   # Purple
    (255, 255, 255),   # White
]

# Landmark names for Sandbox mode
LANDMARK_NAMES = [
    "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]


# ──────────────────────────────────────────────
# UTILITY HELPERS
# ──────────────────────────────────────────────
def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def play_tone_async(frequency, duration_ms=150):
    """Play a tone in a background thread so it doesn't block the frame loop."""
    threading.Thread(target=winsound.Beep, args=(frequency, duration_ms), daemon=True).start()


def draw_rounded_rect(img, pt1, pt2, color, thickness, radius=10):
    """Draw a rectangle with rounded corners."""
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)


# ──────────────────────────────────────────────
# LANDMARK EXTRACTOR
# ──────────────────────────────────────────────
class HandData:
    """Extracts and stores all useful hand landmarks and derived metrics in one place."""

    def __init__(self, hand_landmarks, screen_w, screen_h, image_w, image_h):
        lm = hand_landmarks.landmark
        self.raw = lm  # keep raw access

        # Screen-mapped coordinates (mirrored X for natural control)
        def sx(l): return int(screen_w - l.x * screen_w)
        def sy(l): return int(l.y * screen_h)

        # ── Tips ──
        self.index_tip      = (sx(lm[8]),  sy(lm[8]))
        self.middle_tip     = (sx(lm[12]), sy(lm[12]))
        self.ring_tip       = (sx(lm[16]), sy(lm[16]))
        self.pinky_tip      = (sx(lm[20]), sy(lm[20]))
        self.thumb_tip      = (sx(lm[4]),  sy(lm[4]))

        # ── Knuckles (MCP) ──
        self.index_mcp      = (sx(lm[5]),  sy(lm[5]))
        self.middle_mcp     = (sx(lm[9]),  sy(lm[9]))
        self.ring_mcp       = (sx(lm[13]), sy(lm[13]))
        self.pinky_mcp      = (sx(lm[17]), sy(lm[17]))
        self.thumb_cmc      = (sx(lm[1]),  sy(lm[1]))

        # ── DIP / PIP ──
        self.index_pip      = (sx(lm[6]),  sy(lm[6]))
        self.index_dip      = (sx(lm[7]),  sy(lm[7]))

        # ── Wrist ──
        self.wrist          = (sx(lm[0]),  sy(lm[0]))

        # ── Raw normalized coords (for image-space drawing) ──
        self.raw_middle_tip     = (lm[12].x, lm[12].y)
        self.raw_middle_mcp     = (lm[9].x,  lm[9].y)
        self.raw_thumb_tip      = (lm[4].x,  lm[4].y)
        self.raw_pinky_tip      = (lm[20].x, lm[20].y)
        self.raw_wrist          = (lm[0].x,  lm[0].y)
        self.raw_index_tip      = (lm[8].x,  lm[8].y)

        # ── Image-space center (for ellipse drawing) ──
        self.img_center = (int(lm[9].x * image_w), int(lm[9].y * image_h))

        # ── Finger states (DOWN = tip below knuckle and above wrist) ──
        self.index_down  = (self.index_mcp[1] <= self.index_tip[1] < self.wrist[1])
        self.middle_down = (self.middle_mcp[1] <= self.middle_tip[1] < self.wrist[1])
        self.ring_down   = (self.ring_mcp[1] <= self.ring_tip[1] < self.wrist[1])
        self.pinky_down  = (self.pinky_mcp[1] <= self.pinky_tip[1] < self.wrist[1])

        # Thumb: compare x instead (thumb bends sideways)
        self.thumb_down  = abs(self.thumb_tip[0] - self.thumb_cmc[0]) < abs(self.index_mcp[0] - self.pinky_mcp[0]) * 0.3

        # ── Derived metrics ──
        # Use a single isqrt-free path: compute squared distance, then one sqrt
        # hand_width: thumb tip → pinky tip span
        _wdx = (lm[4].x - lm[20].x) * image_w
        _wdy = (lm[4].y - lm[20].y) * image_h
        self.hand_width  = max(int(math.sqrt(_wdx*_wdx + _wdy*_wdy)), 1)

        # hand_height: middle tip → middle MCP
        _hdx = (lm[12].x - lm[9].x) * image_w
        _hdy = (lm[12].y - lm[9].y) * image_h
        self.hand_height = max(int(math.sqrt(_hdx*_hdx + _hdy*_hdy)), 1)

        self.dist_from_screen = max(int(image_h / self.hand_height), 6)

        # Rotation — single atan2 (already optimal)
        self.rotation = math.degrees(math.atan2(lm[12].y - lm[0].y, lm[12].x - lm[0].x)) + 90

        # Pinch distances (still need actual distances for ratio comparisons)
        _pidx = (lm[4].x - lm[8].x) * screen_w
        _pidy = (lm[4].y - lm[8].y) * screen_h
        self.thumb_index_dist = math.sqrt(_pidx*_pidx + _pidy*_pidy)

        _pkdx = self.thumb_tip[0] - self.pinky_tip[0]
        _pkdy = self.thumb_tip[1] - self.pinky_tip[1]
        self.thumb_pinky_dist = math.sqrt(_pkdx*_pkdx + _pkdy*_pkdy)


# ──────────────────────────────────────────────
# HUD OVERLAY
# ──────────────────────────────────────────────
def draw_hud(image, mode, fps, image_h, image_w):
    """Draw a semi-transparent top bar with mode info and FPS."""
    bar_h = 40
    # Use ROI-based blend instead of full image copy for performance
    roi = image[0:bar_h, 0:image_w]
    dark = np.full_like(roi, (20, 20, 20), dtype=np.uint8)
    cv2.addWeighted(dark, 0.7, roi, 0.3, 0, roi)
    image[0:bar_h, 0:image_w] = roi

    # Mode label
    mode_colors = {
        MODE_CURSOR:  (0, 127, 255),
        MODE_PIANO:   (0, 255, 180),
        MODE_DRAWING: (180, 0, 255),
        MODE_SANDBOX: (255, 255, 0),
    }
    color = mode_colors.get(mode, (255, 255, 255))
    cv2.putText(image, f"MODE: {mode}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # FPS
    cv2.putText(image, f"FPS: {fps}", (image_w - 120, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Controls hint
    hint = "C:Cursor  P:Piano  X:Draw  Z:Sandbox  Q:Quit"
    cv2.putText(image, hint, (image_w // 2 - 200, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    return image


def draw_mode_flash(image, mode, flash_alpha, image_h, image_w):
    """Brief full-screen color flash when mode changes."""
    if flash_alpha <= 0:
        return image
    mode_colors = {
        MODE_CURSOR:  (0, 80, 160),
        MODE_PIANO:   (0, 160, 100),
        MODE_DRAWING: (120, 0, 160),
        MODE_SANDBOX: (160, 160, 0),
    }
    color = mode_colors.get(mode, (80, 80, 80))
    alpha = clamp(flash_alpha, 0.0, 0.4)
    color_layer = np.full_like(image, color, dtype=np.uint8)
    cv2.addWeighted(color_layer, alpha, image, 1.0 - alpha, 0, image)

    # Big mode name in center
    text = mode
    font_scale = 2.0
    thickness = 4
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cx = (image_w - tw) // 2
    cy = (image_h + th) // 2
    cv2.putText(image, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    return image


# ──────────────────────────────────────────────
# IRON MAN LOADING ANIMATION (from original)
# ──────────────────────────────────────────────
def draw_loading_animation(image, hd, hand_counter, active_count, image_w, image_h):
    """The original Iron Man-style loading rings shown while hand is first detected."""
    cx, cy = hd.img_center
    hw, hh = hd.hand_width, hd.hand_height
    rot = hd.rotation
    dist = hd.dist_from_screen
    t = hand_counter  # shorthand

    if hd.raw_middle_mcp[1] * image_h > hd.raw_middle_tip[1] * image_h:
        return  # hand inverted, skip

    if t * 6 > 360:
        return  # loading done

    clr_val = clamp(int(t * 4.25), 0, 255)
    clr = (clr_val, clr_val, clr_val)
    clr_orange = (0, clamp(int(t * 2.116), 0, 255), clr_val)

    # Ring thickness progression
    if t * 6 <= 120:
        ring_thick = 3
    elif t * 6 <= 240:
        ring_thick = 5
    else:
        ring_thick = 7

    # Phase 1: first ring
    if t * 6 <= 120:
        cv2.ellipse(image, (cx, cy), (int(hw * 1.65), int(hh * 0.1)), 45, 0,
                     int(t * t / 2) + 90, clr, 4)
    # Phase 2: two rings
    elif t * 6 <= 240:
        cv2.ellipse(image, (cx, cy), (int(hw * 1.65), int(hh * 0.1)), 45, 0, 360, clr, 4)
        cv2.ellipse(image, (cx, cy), (int(hw * 1.65), int(hh * 0.1)), 315, 90,
                     int(t * t / 3) + 90, clr, 4)
    # Phase 3: three rings
    else:
        cv2.ellipse(image, (cx, cy), (int(hw * 1.65), int(hh * 0.1)), 45, 0, 360, clr, 4)
        cv2.ellipse(image, (cx, cy), (int(hw * 1.65), int(hh * 0.1)), 315, 0, 360, clr, 4)
        cv2.ellipse(image, (cx, cy), (int(hw * 1.65), int(hh * 0.1)), 90, 0,
                     int(t * 5) + 60, clr, 4)

    thick = max(int(60 / dist / 2), 1)
    cv2.ellipse(image, (cx, cy), (hw, hh), rot, 0, int(t * 6), clr_orange, 5)
    cv2.ellipse(image, (cx, cy), (int(hw * 1.1), int(hh * 1.1)), rot,
                 90 + abs(359 - t * 6), 90 + 360, clr_orange, 4)
    cv2.ellipse(image, (cx, cy), (int(hw * 1.2), int(hh * 1.2)), rot,
                 180, 180 + abs(t * 6), clr_orange, 3)

    scale = clamp(t * 0.0132, 0.01, 2.0)
    cv2.ellipse(image, (cx, cy), (int(hw * scale), int(hh * scale)), rot,
                 int(t * t / 2), int(t * t / 2 + 60), (255, 255, 255), ring_thick)
    cv2.ellipse(image, (cx, cy), (int(hw * scale), int(hh * scale)), rot,
                 180 + int(t * t / 2), 180 + int(t * t / 2 + 60), (255, 255, 255), ring_thick)
    cv2.ellipse(image, (cx, cy), (int(hw * 1.675), int(hh * 1.675)), rot,
                 0, -int(t * 6), clr, max(ring_thick - 2, 1))


def draw_orbit_ellipses(image, hd, active_count):
    """The spinning orbit ellipses shown when hand is fully active."""
    cx, cy = hd.img_center
    hw, hh = hd.hand_width, hd.hand_height
    rot = hd.rotation
    dist = hd.dist_from_screen
    thick = max(int(60 / dist / 2), 1)
    t = active_count

    # Inner rotating segments
    for offset in [0, 90, 180, 270]:
        cv2.ellipse(image, (cx, cy), (int(hw * 0.8), int(hh * 0.8)), rot,
                     offset + t * 3, offset + t * 3 + 45, (255, 255, 255), thick + 1)

    # Main ring
    cv2.ellipse(image, (cx, cy), (hw, hh), rot, 0, 360, (0, 127, 255), thick)

    # Outer glow rings
    cv2.ellipse(image, (cx, cy), (int(hw * 1.1), int(hh * 1.1)), rot,
                 90 + abs(int(t * 4)), 90 + abs(int(t * 4)) + 120, (242, 255, 255), thick)
    cv2.ellipse(image, (cx, cy), (int(hw * 1.2), int(hh * 1.2)), rot,
                 -abs(int(t * 6)), -abs(int(t * 6)) + 120, (0, 127, 255), thick)


# ──────────────────────────────────────────────
# MODE: CURSOR (C)
# ──────────────────────────────────────────────
class CursorMode:
    def __init__(self):
        self.x_arr = []
        self.y_arr = []
        self.x_coord = 1
        self.y_coord = 1
        self.left_click_down = False
        self.pinch_count = 0  # stability counter for pinch-click
        self.up_count = 0
        self.down_count = 0
        self.vol_level = 50
        self.vol_cooldown = 0.0
        self.voice_count = 0  # stability counter for voice dictation gesture
        self.rclick_count = 0  # stability counter for right-click gesture
        self.dclick_count = 0  # stability counter for double-click gesture
        self.taskview_count = 0  # debounce latch for task-view gesture
        # Non-blocking cooldowns (avoid time.sleep() in frame loop)
        self.taskview_last_fire = 0.0
        self.voice_last_fire = 0.0
        self.dclick_last_fire = 0.0
        self.rclick_last_fire = 0.0
        # Fist-rotation volume control state
        self.fist_prev_angle = None      # wrist angle on previous frame
        self.fist_angle_accum = 0.0     # accumulated rotation degrees
        self.fist_vol_cooldown = 0.0

    def reset_trail(self):
        self.x_arr.clear()
        self.y_arr.clear()

    def process(self, image, hd, screen_w, screen_h, image_w, image_h, is_left_hand):
        dist = hd.dist_from_screen
        now = time.time()

        # ── Gestures (check BEFORE cursor movement to suppress it when needed) ──
        lm = hd.raw

        # Voice dictation gesture: index UP, all other fingers DOWN (strict check)
        # Checked FIRST — must take priority over task view, which can misfire when
        # only the index is raised (thumb near palm looks like thumb+pinky close).
        # Requires 20 stable frames (~0.67s) to prevent accidental triggers.
        is_voice_gesture = (
            not hd.index_down and  # index UP
            hd.middle_down and     # middle DOWN
            hd.ring_down and       # ring DOWN
            hd.pinky_down and      # pinky DOWN
            hd.thumb_down          # thumb DOWN (curled in)
        )
        if is_voice_gesture:
            self.voice_count += 1
            self.taskview_count = 0  # prevent task-view from firing while in voice pose
            if self.voice_count >= 20:
                if (now - self.voice_last_fire) > 2.0:
                    keyboard.send('windows+h')
                    self.voice_last_fire = now
                    self.voice_count = 0
                    return image
        else:
            self.voice_count = 0

        # Task View: thumb tip touches pinky tip while hand is spread open.
        # Only blocked when the voice-dictation pose is active (index alone raised),
        # which is the one pose that was causing false Task View triggers.
        thumb_x, thumb_y = lm[4].x * image_w, lm[4].y * image_h
        pinky_x, pinky_y = lm[20].x * image_w, lm[20].y * image_h
        index_mcp_x = lm[5].x * image_w
        pinky_mcp_x = lm[17].x * image_w
        thumb_pinky_dist_px = distance(thumb_x, thumb_y, pinky_x, pinky_y)
        hand_spread_px = abs(pinky_mcp_x - index_mcp_x)
        # Scale thresholds relative to image width so they work at any camera resolution
        scale = image_w / 1280.0
        is_taskview = (
            thumb_pinky_dist_px < 70 * scale * 2
            and hand_spread_px > 80 * scale
            and not is_voice_gesture   # only guard needed: block when index alone is raised
        )
        if is_taskview:
            # Add a small cooldown so it doesn't spam while held
            if self.taskview_count == 0 and (now - self.taskview_last_fire) > 1.5:
                keyboard.send('windows+tab')
                self.taskview_count = 1
                self.taskview_last_fire = now
                return image
        else:
            self.taskview_count = 0

        # ── Double left click: detect middle finger STARTING to curl, pause cursor immediately ──
        # Check if middle finger tip is below its PIP joint (starting to curl down)
        lm = hd.raw
        middle_tip_y = lm[12].y  # normalized y (0-1, increases downward)
        middle_pip_y = lm[10].y  # PIP joint y
        middle_mcp_y = lm[9].y   # MCP joint y
        
        # Middle finger is starting to curl if tip is below PIP but not fully down yet
        middle_starting_curl = middle_tip_y > middle_pip_y and middle_tip_y < middle_mcp_y
        
        # Middle finger is fully down (double-click gesture)
        is_double_click_gesture = (
            hd.middle_down and      # middle DOWN
            not hd.index_down and   # index UP
            not hd.ring_down and    # ring UP
            not hd.pinky_down       # pinky UP
        )
        
        # If middle finger is starting to curl OR fully down, suppress cursor movement
        if middle_starting_curl or is_double_click_gesture:
            if is_double_click_gesture:
                self.dclick_count += 1
                if self.dclick_count >= 3:
                    if (now - self.dclick_last_fire) > 0.6:
                        # Release any held left click first
                        if self.left_click_down:
                            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                            self.left_click_down = False
                        # Double click: two rapid click events (no blocking sleep)
                        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                        self.dclick_last_fire = now
                        self.dclick_count = 0
            else:
                self.dclick_count = 0
            # Suppress cursor movement when middle finger is curling or down
            return image
        else:
            self.dclick_count = 0

        # ── Cursor movement (direct full-range mapping) ──
        alpha = 0.55  # EMA smoothing factor

        target_x = hd.middle_tip[0]
        target_y = hd.middle_tip[1]

        # Calculate movement delta
        x_delta = target_x - self.x_coord
        y_delta = target_y - self.y_coord

        # Check if movement exceeds threshold
        if abs(x_delta) >= 1 or abs(y_delta) >= 1:
            # Apply EMA smoothing
            self.x_coord = int(self.x_coord + x_delta * alpha)
            self.y_coord = int(self.y_coord + y_delta * alpha)
            win32api.SetCursorPos((
                clamp(self.x_coord, 1, screen_w - 1),
                clamp(self.y_coord, 1, screen_h - 1)
            ))

        # ── Volume control: FIST + rotate wrist left/right ──
        # Detect fist: all 4 fingers down (thumb ignored — hard to detect reliably)
        is_fist = hd.index_down and hd.middle_down and hd.ring_down and hd.pinky_down

        if is_fist:
            self.rclick_count = 0  # reset right-click counter during fist
            self.pinch_count = 0
            # Release left click if held — index curls before other fingers,
            # so left-click-down may have fired during the closing transition
            if self.left_click_down:
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                self.left_click_down = False
            current_angle = hd.rotation  # degrees, from HandData

            if self.fist_prev_angle is not None:
                delta = current_angle - self.fist_prev_angle
                # Wrap delta to [-180, 180] to handle angle wraparound
                if delta > 180:
                    delta -= 360
                elif delta < -180:
                    delta += 360

                # Accumulate small movements to reduce jitter
                self.fist_angle_accum -= delta  # negate: raw coords are mirrored from user's perspective

                now = time.time()
                if abs(self.fist_angle_accum) >= 8 and now - self.fist_vol_cooldown > 0.08:
                    if self.fist_angle_accum > 0:
                        # Clockwise rotation = volume up
                        keyboard.send('volume up')
                        self.vol_level = min(self.vol_level + 2, 100)
                    else:
                        # Counter-clockwise = volume down
                        keyboard.send('volume down')
                        self.vol_level = max(self.vol_level - 2, 0)
                    self.fist_vol_cooldown = now
                    self.fist_angle_accum = 0.0  # reset after firing

            self.fist_prev_angle = current_angle

            # Draw volume HUD
            vol_pct = self.vol_level
            bar_x, bar_y, bar_w, bar_h = 20, image_h - 80, 160, 12
            cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
            cv2.rectangle(image, (bar_x, bar_y),
                          (bar_x + int(bar_w * vol_pct / 100), bar_y + bar_h), (0, 200, 255), -1)
            cv2.putText(image, f"VOL: {vol_pct}%",
                        (bar_x, bar_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
            cv2.putText(image, "Rotate fist L/R",
                        (bar_x, bar_y + bar_h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)
            return image
        else:
            # Reset fist state when hand opens
            self.fist_prev_angle = None
            self.fist_angle_accum = 0.0


        # ── Near-fist guard: suppress gestures during fist transition ──
        # When 2+ fingers are down (but not all 4 = full fist), the hand is
        # transitioning to/from a fist.  Skip all individual gesture checks
        # to prevent scroll, click, and right-click from firing mid-close.
        fingers_down = sum([hd.index_down, hd.middle_down, hd.ring_down, hd.pinky_down])
        if fingers_down >= 2:
            if self.left_click_down:
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                self.left_click_down = False
            self.pinch_count = 0
            self.rclick_count = 0
            self.up_count = 0
            self.down_count = 0
            return image

        # Right click: ring finger down, others up
        # Requires 3 stable frames to avoid firing during fist transition
        if (hd.ring_down and not hd.pinky_down and not hd.index_down and
            not hd.middle_down):
            self.rclick_count += 1
            if self.rclick_count >= 3:
                if (now - self.rclick_last_fire) > 0.5:
                    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
                    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
                    self.rclick_last_fire = now
                self.rclick_count = 0

        # Left click: pinch index + thumb (stable for 2 frames)
        # Uses a ratio relative to hand width to stay consistent across distances.
        # (Reverted) No Y-proximity gating.
        elif (
            self.left_click_down or
            ((hd.thumb_index_dist / max(hd.hand_width, 1)) < 0.36 and
             not hd.index_down and not hd.pinky_down and not hd.middle_down)
        ):
            pinch_ratio = hd.thumb_index_dist / max(hd.hand_width, 1)
            pinch_down = (pinch_ratio < 0.36 and not hd.index_down and not hd.pinky_down and not hd.middle_down)
            pinch_release = pinch_ratio > 0.46

            if pinch_down and not self.left_click_down:
                self.pinch_count += 1
                if self.pinch_count >= 2:
                    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                    self.left_click_down = True
                    self.pinch_count = 0
            elif not pinch_down:
                self.pinch_count = 0

            if self.left_click_down and pinch_release:
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                self.left_click_down = False

            return image

        # Scroll up (right hand): thumb crosses past index knuckle
        elif (hd.thumb_tip[0] > hd.index_mcp[0] and not is_left_hand and
              not hd.middle_down and not hd.pinky_down and
              hd.thumb_tip[0] < hd.pinky_tip[0]):
            self.up_count += 1
            self.down_count = 0
            if self.up_count >= 30:
                mouse.wheel(4)
            elif self.up_count >= 20:
                mouse.wheel(2)
            else:
                mouse.wheel(1)

        # Scroll up (left hand)
        elif (hd.thumb_tip[0] < hd.index_mcp[0] and is_left_hand and
              not hd.middle_down and not hd.pinky_down and
              hd.thumb_tip[0] > hd.pinky_tip[0]):
            self.up_count += 1
            self.down_count = 0
            if self.up_count >= 30:
                mouse.wheel(4)
            elif self.up_count >= 20:
                mouse.wheel(2)
            else:
                mouse.wheel(1)

        # Scroll down: pinky down
        elif (hd.pinky_down and not hd.middle_down and
              hd.index_tip[1] < hd.wrist[1]):
            self.down_count += 1
            self.up_count = 0
            if self.down_count >= 30:
                mouse.wheel(-4)
            elif self.down_count >= 20:
                mouse.wheel(-2)
            else:
                mouse.wheel(-1)
        else:
            self.up_count = 0
            self.down_count = 0

        return image


# ──────────────────────────────────────────────
# MODE: PIANO (P)
# ──────────────────────────────────────────────
class PianoMode:
    def __init__(self):
        self.key_states = [False] * 8  # whether each key is currently pressed
        self.key_glow = [0.0] * 8     # glow animation timer per key
        self.last_play_time = [0.0] * 8

    def process(self, image, hd, screen_w, screen_h, image_w, image_h):
        num_keys = 8
        key_w = image_w // num_keys
        key_h = int(image_h * 0.30)
        key_y_start = image_h - key_h

        # Collect all fingertips that are extended (not bent)
        active_tips = []
        if not hd.index_down:
            active_tips.append(hd.raw_index_tip)
        if not hd.middle_down:
            active_tips.append(hd.raw_middle_tip)
        if not hd.ring_down:
            raw_ring = (hd.raw[16].x, hd.raw[16].y)
            active_tips.append(raw_ring)
        if not hd.pinky_down:
            raw_pinky = (hd.raw[20].x, hd.raw[20].y)
            active_tips.append(raw_pinky)
        if not hd.thumb_down:
            active_tips.append(hd.raw_thumb_tip)

        # Check which keys are hit
        current_states = [False] * num_keys
        for tip in active_tips:
            # Convert normalized coords to image coords (mirrored)
            tx = int(tip[0] * image_w)
            ty = int(tip[1] * image_h)

            if ty >= key_y_start:
                key_idx = clamp(tx // key_w, 0, num_keys - 1)
                current_states[key_idx] = True

        # Trigger sounds on new presses
        now = time.time()
        for i in range(num_keys):
            if current_states[i] and not self.key_states[i]:
                if now - self.last_play_time[i] > 0.15:
                    freq = PIANO_NOTES[i][1]
                    play_tone_async(freq, 200)
                    self.last_play_time[i] = now
                    self.key_glow[i] = 1.0
            # Decay glow
            self.key_glow[i] = max(0.0, self.key_glow[i] - 0.05)

        self.key_states = current_states

        # ── Draw piano keys ──
        for i in range(num_keys):
            x1 = i * key_w
            x2 = (i + 1) * key_w - 2
            y1 = key_y_start
            y2 = image_h - 2

            # Key color
            if self.key_states[i]:
                # Pressed: bright teal
                key_color = (200, 255, 200)
                border_color = (0, 255, 180)
            elif self.key_glow[i] > 0:
                # Glowing after release
                g = int(self.key_glow[i] * 180)
                key_color = (g // 2, g, g // 2)
                border_color = (0, 200, 150)
            else:
                # Normal
                key_color = (40, 40, 45)
                border_color = (80, 80, 90)

            # Draw key body — use ROI blend instead of full image copy
            alpha = 0.6 if not self.key_states[i] else 0.85
            roi = image[y1:y2, x1:x2]
            color_block = np.full_like(roi, key_color, dtype=np.uint8)
            cv2.addWeighted(color_block, alpha, roi, 1 - alpha, 0, roi)
            image[y1:y2, x1:x2] = roi

            # Border
            cv2.rectangle(image, (x1, y1), (x2, y2), border_color, 2)

            # Note label
            note_name = PIANO_NOTES[i][0]
            text_size = cv2.getTextSize(note_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            text_x = x1 + (key_w - text_size[0]) // 2
            text_y = y2 - 15
            cv2.putText(image, note_name, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Glow effect on press — ROI-based for performance
            if self.key_glow[i] > 0.3:
                g_intensity = int(self.key_glow[i] * 255)
                glow_roi = image[y1:y2, x1:x2]
                glow_color = np.full_like(glow_roi, (0, g_intensity, int(g_intensity * 0.7)), dtype=np.uint8)
                cv2.addWeighted(glow_color, 0.3, glow_roi, 0.7, 0, glow_roi)
                image[y1:y2, x1:x2] = glow_roi

        # Draw separator line
        cv2.line(image, (0, key_y_start - 1), (image_w, key_y_start - 1), (0, 255, 180), 2)

        # Instructions
        cv2.putText(image, "Touch keys with fingertips to play!", (10, key_y_start - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 180), 1)

        return image


# ──────────────────────────────────────────────
# MODE: DRAWING (X)
# Only active when MS Paint is the foreground window.
# Uses real mouse events + keyboard to draw/change colours inside Paint.
#
# Gesture map (fingers = non-thumb fingers up):
#   Index only        → draw   (left-button held while moving)
#   Index + middle    → RED
#   3 fingers         → GREEN
#   4 fingers         → BLUE
#   Open palm (all 5) → Eraser
#   Fist (0 fingers)  → Clear canvas (Ctrl+A → Delete)
# ──────────────────────────────────────────────
class DrawingMode:

    # BGR colours for the HUD overlay
    _COLOR_BGR = {"red": (0,0,255), "green": (0,255,0), "blue": (255,0,0)}

    def __init__(self, image_w, image_h):
        self.image_w = image_w
        self.image_h = image_h

        self.color_name = "red"
        self.color_bgr  = (0, 0, 255)

        # Live mouse state
        self.is_drawing = False
        self.is_erasing = False

        # EMA smoothing for fingertip position
        self.smooth_x = None
        self.smooth_y = None
        self.SMOOTH   = 0.45   # lower = smoother but more lag; 0.45 is a good balance

        # Debounce counters
        self.last_gesture       = None
        self.gesture_hold_count = 0
        self.DRAW_THRESHOLD     = 2    # frames before draw/erase activates (fast)
        self.ACTION_THRESHOLD   = 10   # frames before colour/clear fires (slow = no accidents)

        # One-shot flag so colour/clear only fires once per gesture hold
        self.action_fired     = False
        self.action_last_time = 0.0
        self.ACTION_COOLDOWN  = 2.0   # seconds between repeated colour actions

        # Cached screen metrics (avoids ctypes call every frame)
        import ctypes
        self._screen_w = ctypes.windll.user32.GetSystemMetrics(0)
        self._screen_h = ctypes.windll.user32.GetSystemMetrics(1)

        # Cached Paint hwnd — invalidated when window list changes
        self._paint_hwnd      = None
        self._hwnd_cache_tick = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Gesture classification
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _classify(lm):
        i_up = lm[8].y  < lm[6].y
        m_up = lm[12].y < lm[10].y
        r_up = lm[16].y < lm[14].y
        p_up = lm[20].y < lm[18].y
        t_up = abs(lm[4].x - lm[0].x) > abs(lm[3].x - lm[0].x) * 1.1

        n = t_up + i_up + m_up + r_up + p_up

        if n == 0:                                          return "clear"
        if n == 5:                                          return "erase"
        if i_up and not m_up and not r_up and not p_up:    return "draw"
        if i_up and m_up and not r_up and not p_up:        return "color_red"
        if i_up and m_up and r_up and not p_up:            return "color_green"
        if i_up and m_up and r_up and p_up:                return "color_blue"
        return "idle"

    # ─────────────────────────────────────────────────────────────────────────
    # Paint window detection — NO focus stealing, NO ShowWindow
    # ─────────────────────────────────────────────────────────────────────────
    def _find_paint_hwnd(self):
        """Find classic MS Paint hwnd. Cached for 90 frames. Never focuses it."""
        self._hwnd_cache_tick += 1
        if self._hwnd_cache_tick < 90 and self._paint_hwnd:
            try:
                import win32gui
                if win32gui.IsWindow(self._paint_hwnd) and win32gui.IsWindowVisible(self._paint_hwnd):
                    return self._paint_hwnd
            except Exception:
                pass
        self._hwnd_cache_tick = 0
        try:
            import win32gui
            found = []
            def _cb(hwnd, _):
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd).lower()
                    # Match "paint" but not "paint 3d"
                    if "paint" in title and "3d" not in title:
                        found.append(hwnd)
            win32gui.EnumWindows(_cb, None)
            self._paint_hwnd = found[0] if found else None
        except Exception:
            self._paint_hwnd = None
        return self._paint_hwnd

    def _paint_is_foreground(self):
        """Return True if Paint is currently the foreground (active) window."""
        try:
            import win32gui
            hwnd = self._find_paint_hwnd()
            if not hwnd:
                return False
            fg = win32gui.GetForegroundWindow()
            # Also accept child windows (e.g. Paint's canvas child)
            while fg:
                if fg == hwnd:
                    return True
                fg = win32gui.GetParent(fg)
            return False
        except Exception:
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # Focus helper — briefly bring Paint to foreground, run action, restore focus
    # This is the ONLY reliable way to send keys to modern MS Paint (ribbon UI).
    # WM_KEYDOWN PostMessage / SendMessage are ignored by the ribbon framework.
    # ─────────────────────────────────────────────────────────────────────────
    def _with_paint_focus(self, action_fn):
        """
        Temporarily set Paint as the foreground window, run action_fn(), then
        restore the previous foreground window. Safe to call from a daemon thread.
        """
        import win32gui, win32con, ctypes
        hwnd = self._find_paint_hwnd()
        if not hwnd:
            return

        try:
            prev_fg = win32gui.GetForegroundWindow()

            # Windows requires the calling thread to be attached to the foreground
            # thread to steal focus. Use AllowSetForegroundWindow + SetForegroundWindow.
            cur_tid  = ctypes.windll.kernel32.GetCurrentThreadId()
            fg_tid   = ctypes.windll.user32.GetWindowThreadProcessId(prev_fg, None)
            paint_tid = ctypes.windll.user32.GetWindowThreadProcessId(hwnd, None)

            ctypes.windll.user32.AttachThreadInput(cur_tid, fg_tid, True)
            ctypes.windll.user32.AttachThreadInput(cur_tid, paint_tid, True)

            win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
            ctypes.windll.user32.SetForegroundWindow(hwnd)
            time.sleep(0.08)   # let Paint become active

            action_fn()        # do the actual work

            time.sleep(0.05)

            # Restore previous window
            if prev_fg and prev_fg != hwnd:
                ctypes.windll.user32.SetForegroundWindow(prev_fg)

            ctypes.windll.user32.AttachThreadInput(cur_tid, fg_tid, False)
            ctypes.windll.user32.AttachThreadInput(cur_tid, paint_tid, False)

        except Exception:
            try:
                # Simple fallback — just try SetForegroundWindow directly
                import win32gui, ctypes
                ctypes.windll.user32.SetForegroundWindow(hwnd)
                time.sleep(0.1)
                action_fn()
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────────────────────
    # Colour selection — bring Paint forward, click the swatch by scanning
    # the screen for the target pixel colour, then restore focus.
    # ─────────────────────────────────────────────────────────────────────────
    def _select_color_in_paint(self, color_name):
        import win32gui, win32api, win32con, ctypes
        import keyboard as kb

        TARGET_RGB = {
            "red":   (255, 0,   0),
            "green": (0,   128, 0),
            "blue":  (0,   0,   255),
        }
        if color_name not in TARGET_RGB:
            return
        tr, tg, tb = TARGET_RGB[color_name]

        def do_select():
            hwnd = self._find_paint_hwnd()
            if not hwnd:
                return

            cli_x, cli_y = win32gui.ClientToScreen(hwnd, (0, 0))
            dpi   = ctypes.windll.user32.GetDpiForWindow(hwnd)
            scale = dpi / 96.0

            # Scan the swatch row for the target colour
            scan_y_start = int(cli_y + 38 * scale)
            scan_y_end   = int(cli_y + 82 * scale)
            scan_x_start = int(cli_x + 50 * scale)
            scan_x_end   = int(cli_x + 560 * scale)

            hdc = ctypes.windll.user32.GetDC(0)
            best_x, best_y, best_dist = None, None, float('inf')
            for py in range(scan_y_start, scan_y_end, 2):
                for px in range(scan_x_start, scan_x_end, 3):
                    raw = ctypes.windll.gdi32.GetPixel(hdc, px, py)
                    if raw == 0xFFFFFFFF or raw < 0:
                        continue
                    r, g, b = raw & 0xFF, (raw >> 8) & 0xFF, (raw >> 16) & 0xFF
                    d = (r-tr)**2 + (g-tg)**2 + (b-tb)**2
                    if d < best_dist:
                        best_dist, best_x, best_y = d, px, py
            ctypes.windll.user32.ReleaseDC(0, hdc)

            if best_x is not None and best_dist < 12000:
                # Move real cursor to swatch and click
                old_pos = win32api.GetCursorPos()
                win32api.SetCursorPos((best_x, best_y))
                time.sleep(0.05)
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                time.sleep(0.04)
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                time.sleep(0.04)
                win32api.SetCursorPos(old_pos)
            else:
                # Fallback: use keyboard shortcut (Paint must be in focus — we ensured that above)
                # Alt+H opens Home tab, then navigate color picker
                # Simpler: use hardcoded swatch index positions
                idx = {"red": 4, "green": 9, "blue": 7}[color_name]
                fx = int(cli_x + (86 + idx * 20) * scale)
                fy = int(cli_y + 52 * scale)
                old_pos = win32api.GetCursorPos()
                win32api.SetCursorPos((fx, fy))
                time.sleep(0.05)
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                time.sleep(0.04)
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                time.sleep(0.04)
                win32api.SetCursorPos(old_pos)

            # Restore pencil tool — now Paint is in focus so keyboard.send works
            kb.send('p')
            time.sleep(0.02)

        self._with_paint_focus(do_select)

    def _paint_eraser_tool(self):
        """Switch Paint to the Eraser tool. Paint must be in focus for keyboard shortcuts."""
        def do_erase():
            import keyboard as kb
            kb.send('e')
            time.sleep(0.03)
        self._with_paint_focus(do_erase)

    def _paint_pencil_tool(self):
        """Switch Paint back to the Pencil tool."""
        def do_pencil():
            import keyboard as kb
            kb.send('p')
            time.sleep(0.03)
        self._with_paint_focus(do_pencil)

    def _paint_clear_all(self):
        """Select all and delete canvas content, then restore pencil."""
        def do_clear():
            import keyboard as kb
            kb.send('ctrl+a')
            time.sleep(0.08)
            kb.send('delete')
            time.sleep(0.05)
            kb.send('p')
            time.sleep(0.02)
        self._with_paint_focus(do_clear)

    # ─────────────────────────────────────────────────────────────────────────
    # Mouse drawing helpers — MOUSEEVENTF_ABSOLUTE + MOVE generates WM_MOUSEMOVE
    # ─────────────────────────────────────────────────────────────────────────
    def _abs(self, sx, sy):
        """Convert screen px → absolute mouse coords (0-65535)."""
        ax = int(sx * 65535 / max(self._screen_w - 1, 1))
        ay = int(sy * 65535 / max(self._screen_h - 1, 1))
        return ax, ay

    def _move_and_draw(self, sx, sy):
        ax, ay = self._abs(sx, sy)
        if not self.is_drawing:
            # Move to position FIRST (no button held) to avoid stray stroke
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE | win32con.MOUSEEVENTF_ABSOLUTE, ax, ay, 0, 0)
            time.sleep(0.006)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            self.is_drawing = True
        else:
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE | win32con.MOUSEEVENTF_ABSOLUTE, ax, ay, 0, 0)

    def _move_and_erase(self, sx, sy):
        ax, ay = self._abs(sx, sy)
        if not self.is_erasing:
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE | win32con.MOUSEEVENTF_ABSOLUTE, ax, ay, 0, 0)
            time.sleep(0.006)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            self.is_erasing = True
        else:
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE | win32con.MOUSEEVENTF_ABSOLUTE, ax, ay, 0, 0)

    def _release_mouse(self):
        if self.is_drawing or self.is_erasing:
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            self.is_drawing = False
            self.is_erasing = False

    # ─────────────────────────────────────────────────────────────────────────
    # Main process — called every frame
    # ─────────────────────────────────────────────────────────────────────────
    def process(self, image, hd, screen_w, screen_h, image_w, image_h):
        lm      = hd.raw
        gesture = self._classify(lm)
        now     = time.time()

        # ── Check if Paint is the active window ───────────────────────────────
        paint_active = self._paint_is_foreground()

        # ── Debounce ──────────────────────────────────────────────────────────
        prev_gesture = self.last_gesture
        if gesture == self.last_gesture:
            self.gesture_hold_count = min(self.gesture_hold_count + 1, 120)
        else:
            self._release_mouse()          # always release on gesture change
            self.last_gesture       = gesture
            self.gesture_hold_count = 0
            self.action_fired       = False
            self.smooth_x           = None
            self.smooth_y           = None

        thresh    = self.DRAW_THRESHOLD if gesture in ("draw","erase") else self.ACTION_THRESHOLD
        confirmed = self.gesture_hold_count >= thresh

        # ── Smooth fingertip position ─────────────────────────────────────────
        raw_x = (1.0 - lm[8].x) * screen_w
        raw_y = lm[8].y * screen_h
        if self.smooth_x is None:
            self.smooth_x, self.smooth_y = raw_x, raw_y
        else:
            a = self.SMOOTH
            self.smooth_x = self.smooth_x*(1-a) + raw_x*a
            self.smooth_y = self.smooth_y*(1-a) + raw_y*a
        sx = int(max(0, min(screen_w-1, self.smooth_x)))
        sy = int(max(0, min(screen_h-1, self.smooth_y)))

        # ── Paint must exist; drawing also requires it to be in focus ──────────
        paint_exists = self._find_paint_hwnd() is not None

        if confirmed and paint_exists:
            if gesture == "draw":
                # Drawing requires Paint to be the active window (mouse events go to cursor pos)
                if paint_active:
                    self._move_and_draw(sx, sy)
                else:
                    self._release_mouse()

            elif gesture == "erase":
                # Erasing: switch Paint to Eraser tool once on first confirmed frame,
                # then drag mouse with left-button held. Requires Paint in focus for mouse.
                if paint_active:
                    if not self.action_fired:
                        self.action_fired = True
                        # Switch to eraser tool — runs in thread, brings Paint to focus
                        threading.Thread(target=self._paint_eraser_tool, daemon=True).start()
                    self._move_and_erase(sx, sy)
                else:
                    self._release_mouse()

            elif gesture in ("color_red", "color_green", "color_blue"):
                # Colour change: PostMessage-based, works without Paint in focus
                self._release_mouse()
                if not self.action_fired and now - self.action_last_time > self.ACTION_COOLDOWN:
                    name = gesture[6:]
                    self.color_name       = name
                    self.color_bgr        = self._COLOR_BGR[name]
                    self.action_fired     = True
                    self.action_last_time = now
                    threading.Thread(target=self._select_color_in_paint,
                                     args=(name,), daemon=True).start()

            elif gesture == "clear":
                # Clear: PostMessage-based, works without Paint in focus
                self._release_mouse()
                if not self.action_fired and now - self.action_last_time > self.ACTION_COOLDOWN:
                    self.action_fired     = True
                    self.action_last_time = now
                    threading.Thread(target=self._paint_clear_all, daemon=True).start()

            else:
                self._release_mouse()

            # When erase gesture ends, switch Paint back to pencil tool
            if prev_gesture == "erase" and gesture != "erase":
                threading.Thread(target=self._paint_pencil_tool, daemon=True).start()

        else:
            # No Paint window found, or gesture not yet confirmed
            self._release_mouse()

        # ── HUD overlay ───────────────────────────────────────────────────────
        h, w = image.shape[:2]

        # Paint status banner
        if not self._find_paint_hwnd():
            cv2.rectangle(image, (0, 0), (w, 36), (0,0,120), -1)
            cv2.putText(image, "Open MS Paint to start drawing",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80,180,255), 1, cv2.LINE_AA)
        elif not paint_active:
            cv2.rectangle(image, (0, 0), (w, 36), (0,60,0), -1)
            cv2.putText(image, "Click MS Paint window to activate drawing",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80,255,120), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(image, (0, 0), (w, 36), (0,90,0), -1)
            cv2.putText(image, "MS Paint ACTIVE — drawing enabled",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120,255,120), 1, cv2.LINE_AA)

        # Colour swatches
        PALETTE = [("RED",(0,0,255),"2 fin"), ("GREEN",(0,255,0),"3 fin"), ("BLUE",(255,0,0),"4 fin")]
        sw = 82
        for i, (lbl, col, hint) in enumerate(PALETTE):
            x1, x2 = 10+i*(sw+4), 10+i*(sw+4)+sw
            cv2.rectangle(image, (x1,42), (x2,74), col, -1)
            if lbl.lower() == self.color_name:
                cv2.rectangle(image, (x1-2,40), (x2+2,76), (255,255,255), 3)
            cv2.putText(image, lbl,  (x1+4,60), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(image, hint, (x1+4,72), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (220,220,220), 1, cv2.LINE_AA)

        # Gesture status
        G_LABELS = {"draw":"DRAWING","color_red":"RED","color_green":"GREEN",
                    "color_blue":"BLUE","erase":"ERASE","clear":"CLEAR","idle":"—"}
        lbl_col = (0,255,100) if (confirmed and paint_active) else (100,100,100)
        cv2.putText(image, G_LABELS.get(gesture,"?"), (10,h-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, lbl_col, 2, cv2.LINE_AA)

        # Fingertip dot
        fx = int(lm[8].x * w)
        fy = int(lm[8].y * h)
        dot_col = self.color_bgr if (gesture=="draw" and confirmed and paint_active) else (160,160,160)
        cv2.circle(image, (w-fx, fy), 10, dot_col, -1)
        cv2.circle(image, (w-fx, fy), 12, (255,255,255), 1)

        # Confirmation ring (grows as gesture stabilises)
        if 0 < self.gesture_hold_count < thresh:
            r = int(14 * self.gesture_hold_count / thresh)
            cv2.circle(image, (w-fx, fy), r, (255,255,200), 1)

        # Instructions
        info = ["1 finger=draw","2 fingers=RED","3 fingers=GREEN",
                "4 fingers=BLUE","Palm=Erase","Fist=Clear"]
        for j, ln in enumerate(info):
            cv2.putText(image, ln, (w-178, h-len(info)*16+j*16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (140,140,140), 1, cv2.LINE_AA)

        return image
# ──────────────────────────────────────────────
# MODE: SANDBOX (Z)
# ──────────────────────────────────────────────
class SandboxMode:
    def process(self, image, hand_landmarks, hd, mp_hands, image_w, image_h):
        lm = hand_landmarks.landmark

        # ── Draw connections with gradient colors ──
        connections = mp_hands.HAND_CONNECTIONS
        for conn in connections:
            p1 = conn[0]
            p2 = conn[1]
            x1 = int(lm[p1].x * image_w)
            y1 = int(lm[p1].y * image_h)
            x2 = int(lm[p2].x * image_w)
            y2 = int(lm[p2].y * image_h)
            cv2.line(image, (x1, y1), (x2, y2), (100, 100, 100), 2, lineType=cv2.LINE_AA)

        # ── Draw landmarks with color coding ──
        tip_ids = [4, 8, 12, 16, 20]
        mcp_ids = [1, 2, 5, 9, 13, 17]
        pip_dip_ids = [3, 6, 7, 10, 11, 14, 15, 18, 19]

        for i, l in enumerate(lm):
            x = int(l.x * image_w)
            y = int(l.y * image_h)

            if i == 0:  # Wrist
                color = (0, 0, 255)  # Red
                radius = 8
            elif i in tip_ids:
                color = (0, 255, 0)  # Green
                radius = 7
            elif i in mcp_ids:
                color = (255, 150, 0)  # Blue-ish
                radius = 6
            else:
                color = (255, 255, 0)  # Cyan
                radius = 4

            cv2.circle(image, (x, y), radius, color, -1)
            cv2.circle(image, (x, y), radius + 1, (255, 255, 255), 1)

            # Label
            label = f"{i}"
            cv2.putText(image, label, (x + 8, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        return image


# ──────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────
def main():
    # ── MediaPipe setup ──
    mp_drawing = mp.solutions.drawing_utils
    mp_hands_module = mp.solutions.hands

    # ── Screen dimensions ──
    user32 = ctypes.windll.user32
    screen_w = user32.GetSystemMetrics(0)
    screen_h = user32.GetSystemMetrics(1)

    # ── State ──
    current_mode = MODE_CURSOR
    hand_counter = 0
    lost_hand_count = 0
    active_hand_count = 0
    hand_shown = False
    image_size_mod = 5

    # FPS tracking
    fps = 0
    frame_count = 0
    fps_start = time.time()

    # Mode handlers
    cursor_mode = CursorMode()
    piano_mode = PianoMode()
    drawing_mode = None  # initialized after first frame (need image dimensions)
    sandbox_mode = SandboxMode()

    # Two-palms-together toggle (Cursor <-> Drawing)
    palms_together_start = None
    last_palms_toggle_time = 0.0

    # ── Camera (resolution chosen for performance) ──
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    with mp_hands_module.Hands(
        max_num_hands=MAX_HANDS,
        model_complexity=0,           # 0 = lite model, ~2x faster than 1
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5   # higher = more tracking, less re-detection
    ) as hands:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image_h, image_w, _ = image.shape

            # Initialize drawing mode canvas on first frame
            if drawing_mode is None:
                drawing_mode = DrawingMode(image_w, image_h)

            # ── Mode switching via keyboard ──
            key = cv2.waitKey(1) & 0xFF
            new_mode = None
            if key == ord('c'):
                new_mode = MODE_CURSOR
            elif key == ord('p'):
                new_mode = MODE_PIANO
            elif key == ord('x'):
                new_mode = MODE_DRAWING
            elif key == ord('z'):
                new_mode = MODE_SANDBOX
            elif key == ord('q'):
                break

            if new_mode and new_mode != current_mode:
                current_mode = new_mode
                flash_alpha = 0.4
                flash_mode = new_mode
                # Reset cursor trail when leaving cursor mode
                if current_mode != MODE_CURSOR:
                    cursor_mode.reset_trail()

            # ── Process frame through MediaPipe ──
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            image.flags.writeable = True
            now = time.time()

            # ── If hand detected ──
            if results.multi_hand_landmarks:
                lost_hand_count = 0
                active_hand_count += 1
                hand_shown = True

                # Determine handedness
                is_left_hand = False
                for i in results.multi_handedness:
                    if str(i.classification[0].label) == 'Right':
                        is_left_hand = True  # mirrored

                # Build list of HandData for this frame (so we can reuse it)
                hd_list = []

                for hand_landmarks in results.multi_hand_landmarks:
                    # Extract hand data
                    hd = HandData(hand_landmarks, screen_w, screen_h, image_w, image_h)
                    hd_list.append(hd)

                    # ── Loading counter (tracks hand presence, no drawing) ──
                    hand_counter = min(hand_counter + 2, 60)

                    # ── Mode-specific processing ──
                    if current_mode == MODE_CURSOR:
                        image = cursor_mode.process(image, hd, screen_w, screen_h, image_w, image_h, is_left_hand)
                    elif current_mode == MODE_PIANO:
                        image = piano_mode.process(image, hd, screen_w, screen_h, image_w, image_h)
                    elif current_mode == MODE_DRAWING:
                        image = drawing_mode.process(image, hd, screen_w, screen_h, image_w, image_h)
                    elif current_mode == MODE_SANDBOX:
                        image = sandbox_mode.process(image, hand_landmarks, hd, mp_hands_module, image_w, image_h)

                    # ── Optional hand skeleton (purely visual, can be disabled) ──
                    if DRAW_SKELETON and current_mode != MODE_SANDBOX:
                        mp_drawing.draw_landmarks(
                            image, hand_landmarks, mp_hands_module.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(thickness=2, circle_radius=1, color=(0, 127, 255))
                        )

                # ── Two open palms together for 3s: toggle Cursor <-> Drawing ──
                # Disabled automatically in PERFORMANCE_MODE (MAX_HANDS == 1)
                if MAX_HANDS >= 2 and len(hd_list) >= 2:
                    hd1, hd2 = hd_list[0], hd_list[1]
                    open1 = (not hd1.index_down and not hd1.middle_down and not hd1.ring_down and not hd1.pinky_down and not hd1.thumb_down)
                    open2 = (not hd2.index_down and not hd2.middle_down and not hd2.ring_down and not hd2.pinky_down and not hd2.thumb_down)

                    dx = hd1.raw_wrist[0] - hd2.raw_wrist[0]
                    dy = hd1.raw_wrist[1] - hd2.raw_wrist[1]
                    wrist_dist = math.sqrt(dx * dx + dy * dy)

                    palms_together = open1 and open2 and wrist_dist < 0.10
                    if palms_together:
                        if palms_together_start is None:
                            palms_together_start = now
                        elif (now - palms_together_start) >= 3.0 and (now - last_palms_toggle_time) > 2.0:
                            if current_mode == MODE_CURSOR:
                                current_mode = MODE_DRAWING
                            elif current_mode == MODE_DRAWING:
                                current_mode = MODE_CURSOR
                            last_palms_toggle_time = now
                            palms_together_start = None
                    else:
                        palms_together_start = None
                else:
                    palms_together_start = None

            else:
                # No hand detected
                lost_hand_count += 1
                cursor_mode.reset_trail()
                cursor_mode.x_arr.clear()
                cursor_mode.y_arr.clear()
                if lost_hand_count >= 20:
                    hand_counter = 0
                    active_hand_count = 0
                    hand_shown = False

            # ── FPS counter ──
            frame_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps = int(frame_count / elapsed)
                frame_count = 0
                fps_start = time.time()

            # ── Flip image for mirror view ──
            image = cv2.flip(image, 1)

            # ── Minimal FPS overlay ──
            cv2.putText(image, f"FPS: {fps}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)

            # ── Window sizing (arrow keys when no hand) ──
            if not hand_shown:
                if key == 82 and image_size_mod > 1:
                    image_size_mod -= 1
                elif key == 84 and image_size_mod < 7:
                    image_size_mod += 1

            # ── Display ──
            display_w = int(round(screen_w / image_size_mod))
            display_h = int(round(screen_h / image_size_mod))
            display_img = cv2.resize(image, (display_w, display_h))

            cv2.imshow('FG Motion Control', display_img)
            cv2.setWindowProperty('FG Motion Control', cv2.WND_PROP_TOPMOST, 1)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()