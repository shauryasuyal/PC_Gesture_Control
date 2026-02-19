# ──────────────────────────────────────────────
# utils.py — helpers, HUD overlay, Iron Man animations
# ──────────────────────────────────────────────

import cv2
import math
import threading
import winsound
import numpy as np

from constants import (
    MODE_CURSOR, MODE_CNN, MODE_DRAWING, MODE_SANDBOX,
)


# ── General helpers ───────────────────────────

def clamp(value, lo, hi):
    return max(lo, min(hi, value))

def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def play_tone_async(frequency, duration_ms=150):
    threading.Thread(
        target=winsound.Beep, args=(frequency, duration_ms), daemon=True
    ).start()

def draw_rounded_rect(img, pt1, pt2, color, thickness, radius=10):
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90,  color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90,  color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90,  0, 90,  color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0,   0, 90,  color, thickness)


# ── HUD bar ───────────────────────────────────

def draw_hud(image, mode, fps, image_h, image_w):
    bar_h = 40
    roi  = image[0:bar_h, 0:image_w]
    dark = np.full_like(roi, (20, 20, 20), dtype=np.uint8)
    cv2.addWeighted(dark, 0.7, roi, 0.3, 0, roi)
    image[0:bar_h, 0:image_w] = roi

    mode_colors = {
        MODE_CURSOR:  (0, 127, 255),
        MODE_CNN:     (50, 255, 100),
        MODE_DRAWING: (180, 0, 255),
        MODE_SANDBOX: (255, 255, 0),
    }
    color = mode_colors.get(mode, (255, 255, 255))
    cv2.putText(image, f"MODE: {mode}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(image, f"FPS: {fps}", (image_w - 120, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    hint = "C:Cursor  P:CNN  X:Draw  Z:Sandbox  Q:Quit"
    cv2.putText(image, hint, (image_w // 2 - 190, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    return image


def draw_mode_flash(image, flash_mode, flash_alpha, image_w, image_h):
    """Coloured flash + big mode name when switching modes."""
    mode_flash_colors = {
        MODE_CURSOR:  (0, 80, 160),
        MODE_CNN:     (0, 130, 60),
        MODE_DRAWING: (120, 0, 160),
        MODE_SANDBOX: (160, 160, 0),
    }
    color       = mode_flash_colors.get(flash_mode, (80, 80, 80))
    alpha       = clamp(flash_alpha, 0.0, 0.4)
    color_layer = np.full_like(image, color, dtype=np.uint8)
    cv2.addWeighted(color_layer, alpha, image, 1.0 - alpha, 0, image)

    font_scale, thickness = 2.0, 4
    (tw, th), _ = cv2.getTextSize(flash_mode, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.putText(image, flash_mode,
                ((image_w - tw) // 2, (image_h + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    return image


# ── Iron Man loading animation ────────────────

def draw_loading_animation(image, hd, hand_counter, image_w, image_h):
    cx, cy = hd.img_center
    hw, hh = hd.hand_width, hd.hand_height
    rot = hd.rotation
    t   = hand_counter

    if hd.raw_middle_mcp[1] * image_h > hd.raw_middle_tip[1] * image_h:
        return
    if t * 6 > 360:
        return

    clr_val    = clamp(int(t * 4.25), 0, 255)
    clr        = (clr_val, clr_val, clr_val)
    clr_orange = (0, clamp(int(t * 2.116), 0, 255), clr_val)
    ring_thick = 3 if t * 6 <= 120 else (5 if t * 6 <= 240 else 7)

    if t * 6 <= 120:
        cv2.ellipse(image, (cx, cy), (int(hw * 1.65), int(hh * 0.1)), 45, 0, int(t * t / 2) + 90, clr, 4)
    elif t * 6 <= 240:
        cv2.ellipse(image, (cx, cy), (int(hw * 1.65), int(hh * 0.1)), 45,  0, 360, clr, 4)
        cv2.ellipse(image, (cx, cy), (int(hw * 1.65), int(hh * 0.1)), 315, 90, int(t * t / 3) + 90, clr, 4)
    else:
        cv2.ellipse(image, (cx, cy), (int(hw * 1.65), int(hh * 0.1)), 45,  0, 360, clr, 4)
        cv2.ellipse(image, (cx, cy), (int(hw * 1.65), int(hh * 0.1)), 315, 0, 360, clr, 4)
        cv2.ellipse(image, (cx, cy), (int(hw * 1.65), int(hh * 0.1)), 90,  0, int(t * 5) + 60, clr, 4)

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
    cx, cy = hd.img_center
    hw, hh = hd.hand_width, hd.hand_height
    rot    = hd.rotation
    dist   = hd.dist_from_screen
    thick  = max(int(60 / dist / 2), 1)
    t      = active_count

    for offset in [0, 90, 180, 270]:
        cv2.ellipse(image, (cx, cy), (int(hw * 0.8), int(hh * 0.8)), rot,
                    offset + t * 3, offset + t * 3 + 45, (255, 255, 255), thick + 1)
    cv2.ellipse(image, (cx, cy), (hw, hh), rot, 0, 360, (0, 127, 255), thick)
    cv2.ellipse(image, (cx, cy), (int(hw * 1.1), int(hh * 1.1)), rot,
                90 + abs(int(t * 4)), 90 + abs(int(t * 4)) + 120, (242, 255, 255), thick)
    cv2.ellipse(image, (cx, cy), (int(hw * 1.2), int(hh * 1.2)), rot,
                -abs(int(t * 6)), -abs(int(t * 6)) + 120, (0, 127, 255), thick)