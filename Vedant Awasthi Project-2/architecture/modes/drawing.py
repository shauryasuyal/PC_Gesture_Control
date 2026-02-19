# ──────────────────────────────────────────────
# modes/drawing.py — DrawingMode
# Persistent canvas drawn with finger gestures.
# Optionally driven by a trained CNN engine.
# ──────────────────────────────────────────────

import sys, os
# Add the architecture/ folder to path so 'utils', 'constants' etc. resolve
# whether this file is imported via main.py OR run directly by an IDE.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == '__main__':
    print("drawing.py is a module — run main.py instead.")
    sys.exit(0)

import cv2
import time
import numpy as np

from utils     import clamp
from constants import DRAW_COLORS, ACTION_CATALOGUE


class DrawingMode:
    def __init__(self, image_w, image_h):
        self.canvas        = np.zeros((image_h, image_w, 3), dtype=np.uint8)
        self.prev_point    = None
        self.current_color = (0, 127, 255)
        self.brush_size    = 4
        self.fist_start    = None
        self.erasing       = False
        self.image_w       = image_w
        self.image_h       = image_h

    # ── main per-frame call ───────────────────────
    def process(self, image, hd, screen_w, screen_h, image_w, image_h,
                cnn_engine=None):
        if self.canvas.shape[:2] != (image_h, image_w):
            self.canvas  = np.zeros((image_h, image_w, 3), dtype=np.uint8)
            self.image_w = image_w
            self.image_h = image_h

        # ── Resolve CNN action for this frame ──
        cnn_action, cnn_conf = None, 0.0
        if cnn_engine is not None:
            action, conf, _ = cnn_engine.get_active_action(hd.raw)
            if action and action != 'none':
                cat = ACTION_CATALOGUE.get(action, ('', 'none', 0))[1]
                if cat in ('draw', 'both'):
                    cnn_action, cnn_conf = action, conf

        # ── Colour from hand rotation, brush from pinch ──
        color_idx = clamp(int(hd.rotation / 180 * len(DRAW_COLORS)), 0, len(DRAW_COLORS) - 1)
        self.current_color = DRAW_COLORS[color_idx]
        self.brush_size    = clamp(int(hd.thumb_index_dist / 30), 1, 20)

        # ── Fist → clear canvas (finger-state only, not CNN-overridable) ──
        if hd.index_down and hd.middle_down and hd.ring_down and hd.pinky_down:
            if cnn_action not in ('draw', 'erase'):
                if self.fist_start is None:
                    self.fist_start = time.time()
                elapsed  = time.time() - self.fist_start
                progress = clamp(elapsed / 1.0, 0, 1)
                bar_w    = int(200 * progress)
                cx = image_w // 2
                cy = image_h // 2
                cv2.rectangle(image, (cx - 100, cy - 10), (cx - 100 + bar_w, cy + 10), (0, 0, 255), -1)
                cv2.putText(image, "CLEARING...", (cx - 60, cy - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                if elapsed >= 1.0:
                    self.canvas     = np.zeros((image_h, image_w, 3), dtype=np.uint8)
                    self.fist_start = None
                    self.prev_point = None
                self.prev_point = None
                self._composite_and_wheel(image, image_w, image_h)
                return image
        else:
            self.fist_start = None

        # ── CNN: instant clear ──
        if cnn_action == 'clear' and cnn_engine and cnn_engine.can_fire('clear'):
            self.canvas     = np.zeros((image_h, image_w, 3), dtype=np.uint8)
            self.prev_point = None
            cnn_engine.mark_fired('clear')
            cv2.putText(image, "CNN: Clear Canvas",
                        (image_w // 2 - 80, image_h - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 255), 2)
            self._composite_and_wheel(image, image_w, image_h)
            return image

        # ── CNN: next colour ──
        if cnn_action == 'color_next' and cnn_engine and cnn_engine.can_fire('color_next'):
            idx = (DRAW_COLORS.index(self.current_color) + 1) % len(DRAW_COLORS)
            self.current_color = DRAW_COLORS[idx]
            cnn_engine.mark_fired('color_next')

        # ── CNN: brush size ──
        if cnn_action == 'brush_bigger' and cnn_engine and cnn_engine.can_fire('brush_bigger'):
            self.brush_size = min(self.brush_size + 1, 20)
            cnn_engine.mark_fired('brush_bigger')
        if cnn_action == 'brush_smaller' and cnn_engine and cnn_engine.can_fire('brush_smaller'):
            self.brush_size = max(self.brush_size - 1, 1)
            cnn_engine.mark_fired('brush_smaller')

        # ── Determine draw / erase source ──
        use_draw  = (cnn_action == 'draw') or (
            cnn_action is None and not hd.index_down and not hd.middle_down)
        use_erase = (cnn_action == 'erase') or (
            cnn_action is None and hd.pinky_down and hd.ring_down and
            not hd.index_down and not hd.middle_down)

        ix = int(hd.raw_index_tip[0] * image_w)
        iy = int(hd.raw_index_tip[1] * image_h)

        if use_erase:
            self.erasing = True
            r = self.brush_size * 3
            cv2.circle(self.canvas, (ix, iy), r, (0, 0, 0), -1)
            cv2.circle(image,       (ix, iy), r, (100, 100, 100), 2)
            label = f"CNN: Erase  {int(cnn_conf*100)}%" if cnn_action == 'erase' else "ERASER"
            cv2.putText(image, label, (ix + 15, iy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
            self.prev_point = None
        elif use_draw:
            self.erasing = False
            if self.prev_point is not None:
                cv2.line(self.canvas, self.prev_point, (ix, iy),
                         self.current_color, self.brush_size, lineType=cv2.LINE_AA)
            self.prev_point = (ix, iy)
            cv2.circle(image, (ix, iy), self.brush_size, self.current_color, 2)
            if cnn_action == 'draw':
                cv2.putText(image, f"CNN: Draw  {int(cnn_conf*100)}%",
                            (image_w // 2 - 60, image_h - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 0, 255), 1)
        else:
            self.prev_point = None

        self._composite_and_wheel(image, image_w, image_h)
        return image

    # ── Internal helpers ──────────────────────────
    def _composite_and_wheel(self, image, image_w, image_h):
        """Blend canvas onto live image and draw the colour wheel."""
        mask = np.any(self.canvas > 0, axis=2)
        image[mask] = cv2.addWeighted(image, 0.3, self.canvas, 0.7, 0)[mask]

        cx, cy, r = image_w - 50, 80, 30
        for i, col in enumerate(DRAW_COLORS):
            a0 = int(i * 360 / len(DRAW_COLORS))
            a1 = int((i + 1) * 360 / len(DRAW_COLORS))
            cv2.ellipse(image, (cx, cy), (r, r), 0, a0, a1, col,
                        3 if col == self.current_color else 2)
        cv2.circle(image, (cx, cy), 12, self.current_color, -1)
        cv2.circle(image, (cx, cy), 12, (255, 255, 255), 1)
        cv2.putText(image, f"Size:{self.brush_size}", (cx - 28, cy + r + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
        cv2.putText(image,
                    "Draw:point | Rotate:color | Pinch:size | Fist:clear | Ring+Pinky:erase",
                    (10, image_h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (180, 0, 255), 1)