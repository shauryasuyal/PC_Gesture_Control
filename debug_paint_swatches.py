"""
debug_paint_swatches.py
-----------------------
Run this WITH MS Paint open to see exactly where the colour swatches are
on YOUR screen. It prints the best-match coordinates for red, green, blue
and saves a screenshot annotated with the scan area and hits.

Usage:
    1. Open MS Paint (classic, NOT Paint 3D)
    2. Run:  python debug_paint_swatches.py
    3. Look at the console output and the saved file: swatch_debug.png
"""

import ctypes
import time
import win32gui
import win32api
import win32con

try:
    import numpy as np
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# ── Find Paint ────────────────────────────────────────────────────────────────
found = []
def _cb(hwnd, _):
    if win32gui.IsWindowVisible(hwnd):
        title = win32gui.GetWindowText(hwnd).lower()
        if "paint" in title and "3d" not in title:
            found.append(hwnd)
win32gui.EnumWindows(_cb, None)

if not found:
    print("ERROR: MS Paint not found. Open classic MS Paint first.")
    input("Press Enter to exit.")
    exit(1)

hwnd = found[0]
title = win32gui.GetWindowText(hwnd)
print(f"Found Paint window: '{title}'  hwnd={hwnd}")

# ── Get rects ─────────────────────────────────────────────────────────────────
win_rect = win32gui.GetWindowRect(hwnd)     # includes title bar + ribbon
cli_rect = win32gui.GetClientRect(hwnd)
cli_origin = win32gui.ClientToScreen(hwnd, (0, 0))

dpi   = ctypes.windll.user32.GetDpiForWindow(hwnd)
scale = dpi / 96.0

print(f"\nWindow rect  (screen): {win_rect}")
print(f"Client rect  (local) : {cli_rect}")
print(f"Client origin(screen): {cli_origin}")
print(f"DPI={dpi}  scale={scale:.2f}")

win_x, win_y = win_rect[0], win_rect[1]
win_w = win_rect[2] - win_rect[0]
win_h = win_rect[3] - win_rect[1]

# ── Scan for swatches ─────────────────────────────────────────────────────────
TARGETS = {
    "red":   (255, 0,   0),
    "green": (0,   128, 0),
    "blue":  (0,   0,   255),
}

scan_y_start = int(win_y + 48 * scale)
scan_y_end   = int(win_y + 110 * scale)
scan_x_start = int(win_x + 260 * scale)
scan_x_end   = int(win_x + min(win_w - 5, 780 * scale))

print(f"\nScan region: x=[{scan_x_start},{scan_x_end}]  y=[{scan_y_start},{scan_y_end}]")
print("Scanning... (this takes ~1 second)")

hdc = ctypes.windll.user32.GetDC(0)

# Collect every pixel in scan region for analysis
pixels = []
for py in range(scan_y_start, scan_y_end, 1):
    for px in range(scan_x_start, scan_x_end, 1):
        raw = ctypes.windll.gdi32.GetPixel(hdc, px, py)
        if raw < 0 or raw == 0xFFFFFFFF:
            continue
        r, g, b = raw & 0xFF, (raw >> 8) & 0xFF, (raw >> 16) & 0xFF
        pixels.append((px, py, r, g, b))

ctypes.windll.user32.ReleaseDC(0, hdc)
print(f"Total valid pixels scanned: {len(pixels)}")

# Find best match for each colour
results = {}
for name, (tr, tg, tb) in TARGETS.items():
    best_px, best_py, best_dist = None, None, float('inf')
    for px, py, r, g, b in pixels:
        d = (r-tr)**2 + (g-tg)**2 + (b-tb)**2
        if d < best_dist:
            best_dist, best_px, best_py = d, px, py
    results[name] = (best_px, best_py, best_dist)
    found_rgb = next(((r,g,b) for px,py,r,g,b in pixels if px==best_px and py==best_py), None)
    print(f"\n  {name.upper():6s}: best match at screen ({best_px},{best_py})  dist={best_dist}  actual_rgb={found_rgb}")
    if best_dist < 5000:
        print(f"           ✅ GOOD — will be used for click")
    elif best_dist < 15000:
        print(f"           ⚠️  MARGINAL — may work but colour may be slightly off")
    else:
        print(f"           ❌ BAD — swatch not found in scan region. Paint may need to be repositioned or scan bounds updated.")

# ── Also print what the most pure red/green/blue pixels found were ────────────
print("\n── Most saturated pixels found in scan region ──")
def saturation_score(r, g, b):
    mx = max(r, g, b)
    mn = min(r, g, b)
    return mx - mn  # simple chroma

top = sorted(pixels, key=lambda p: saturation_score(p[2],p[3],p[4]), reverse=True)[:20]
for px, py, r, g, b in top:
    print(f"  screen({px},{py})  rgb=({r},{g},{b})  chroma={saturation_score(r,g,b)}")

# ── Save annotated screenshot if cv2 available ────────────────────────────────
if HAS_CV2:
    import subprocess
    # Take screenshot using PIL if available, else skip
    try:
        from PIL import ImageGrab
        screen = ImageGrab.grab()
        img = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)

        # Draw scan rectangle
        cv2.rectangle(img, (scan_x_start, scan_y_start), (scan_x_end, scan_y_end), (0, 255, 255), 2)

        # Draw hit circles
        colors_bgr = {"red": (0,0,255), "green": (0,255,0), "blue": (255,0,0)}
        for name, (bx, by, dist) in results.items():
            if bx is not None:
                col = colors_bgr[name]
                cv2.circle(img, (bx, by), 12, col, 3)
                cv2.putText(img, f"{name} d={dist:.0f}", (bx+14, by+5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

        out_path = "swatch_debug.png"
        cv2.imwrite(out_path, img)
        print(f"\nAnnotated screenshot saved to: {out_path}")
    except ImportError:
        print("\n(PIL not available — skipping screenshot. Install Pillow to enable.)")

print("\nDone. Share the output above so the scan bounds can be corrected if needed.")
input("Press Enter to exit.")