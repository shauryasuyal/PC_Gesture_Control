# ──────────────────────────────────────────────
# hand_data.py — HandData class
# ──────────────────────────────────────────────

import math
from utils import distance


class HandData:
    """
    Wraps one MediaPipe hand_landmarks result for easy access.

    Screen-space coords are mirrored (moving hand right → cursor right).
    Finger-down booleans: True when that finger is curled toward the palm.
    """

    def __init__(self, hand_landmarks, screen_w, screen_h, image_w, image_h):
        lm      = hand_landmarks.landmark
        self.raw = lm

        def sx(l): return int(screen_w - l.x * screen_w)
        def sy(l): return int(l.y * screen_h)

        # ── Fingertips ────────────────────────────────
        self.index_tip  = (sx(lm[8]),  sy(lm[8]))
        self.middle_tip = (sx(lm[12]), sy(lm[12]))
        self.ring_tip   = (sx(lm[16]), sy(lm[16]))
        self.pinky_tip  = (sx(lm[20]), sy(lm[20]))
        self.thumb_tip  = (sx(lm[4]),  sy(lm[4]))

        # ── Knuckles (MCP) ────────────────────────────
        self.index_mcp  = (sx(lm[5]),  sy(lm[5]))
        self.middle_mcp = (sx(lm[9]),  sy(lm[9]))
        self.ring_mcp   = (sx(lm[13]), sy(lm[13]))
        self.pinky_mcp  = (sx(lm[17]), sy(lm[17]))
        self.thumb_cmc  = (sx(lm[1]),  sy(lm[1]))

        # ── PIP / DIP / Wrist ─────────────────────────
        self.index_pip  = (sx(lm[6]),  sy(lm[6]))
        self.index_dip  = (sx(lm[7]),  sy(lm[7]))
        self.wrist      = (sx(lm[0]),  sy(lm[0]))

        # ── Raw normalised coords (for CNN / distance math) ──
        self.raw_middle_tip = (lm[12].x, lm[12].y)
        self.raw_middle_mcp = (lm[9].x,  lm[9].y)
        self.raw_thumb_tip  = (lm[4].x,  lm[4].y)
        self.raw_pinky_tip  = (lm[20].x, lm[20].y)
        self.raw_wrist      = (lm[0].x,  lm[0].y)
        self.raw_index_tip  = (lm[8].x,  lm[8].y)

        # ── Image-space centre (for animation overlay) ──
        self.img_center = (int(lm[9].x * image_w), int(lm[9].y * image_h))

        # ── Finger-down booleans ──────────────────────
        # A finger is "down" when its tip is below MCP but above the wrist
        self.index_down  = self.index_mcp[1]  <= self.index_tip[1]  < self.wrist[1]
        self.middle_down = self.middle_mcp[1] <= self.middle_tip[1] < self.wrist[1]
        self.ring_down   = self.ring_mcp[1]   <= self.ring_tip[1]   < self.wrist[1]
        self.pinky_down  = self.pinky_mcp[1]  <= self.pinky_tip[1]  < self.wrist[1]
        # Thumb bends sideways — compare X instead of Y
        self.thumb_down  = (
            abs(self.thumb_tip[0] - self.thumb_cmc[0]) <
            abs(self.index_mcp[0] - self.pinky_mcp[0]) * 0.3
        )

        # ── Size / depth estimates ────────────────────
        self.hand_width  = max(int(math.sqrt(
            ((lm[4].x - lm[20].x) ** 2) * image_w * 280 +
            ((lm[4].y - lm[20].y) ** 2) * image_h * 280)), 1)
        self.hand_height = max(int(math.sqrt(
            ((lm[12].x - lm[9].x) ** 2) * image_w * 900 +
            ((lm[12].y - lm[9].y) ** 2) * image_h * 900)), 1)

        self.dist_from_screen = max(int(image_h / self.hand_height), 6)
        self.rotation = (
            math.degrees(math.atan2(lm[12].y - lm[0].y, lm[12].x - lm[0].x)) + 90
        )

        # ── Pinch distances ───────────────────────────
        self.thumb_index_dist = distance(
            lm[4].x * screen_w, lm[4].y * screen_h,
            lm[8].x * screen_w, lm[8].y * screen_h)
        self.thumb_pinky_dist = distance(
            self.thumb_tip[0], self.thumb_tip[1],
            self.pinky_tip[0], self.pinky_tip[1])


if __name__ == '__main__':
    print('hand_data.py is a module — run main.py instead.')