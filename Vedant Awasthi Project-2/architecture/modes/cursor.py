# ──────────────────────────────────────────────
# modes/cursor.py — CursorMode
# Controls the mouse cursor with hand gestures.
# Optionally driven by a trained CNN engine.
# ──────────────────────────────────────────────

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == '__main__':
    print("cursor.py is a module — run main.py instead.")
    sys.exit(0)

import cv2
import math
import time

import win32api
import win32con
import mouse
import keyboard

from utils     import clamp
from constants import ACTION_CATALOGUE

class CursorMode:
    def __init__(self):
        self.x_arr   = []
        self.y_arr   = []
        self.x_coord = 1
        self.y_coord = 1
        self.left_click_down = False
        self.up_count   = 0
        self.down_count = 0
        self.vol_level   = 50
        self.vol_cooldown = 0.0

    def reset_trail(self):
        self.x_arr.clear()
        self.y_arr.clear()

    # ── main per-frame call ───────────────────────
    def process(self, image, hd, screen_w, screen_h, image_w, image_h,
                is_left_hand, cnn_engine=None):
        dist = hd.dist_from_screen

        # ── CNN-driven actions (override hardcoded gestures when active) ──
        if cnn_engine is not None:
            action, conf, _ = cnn_engine.get_active_action(hd.raw)
            if action and action != 'none':
                cat = ACTION_CATALOGUE.get(action, ('', 'none', 0))[1]
                if cat in ('cursor', 'both') and cnn_engine.can_fire(action):
                    fired = self._dispatch_action(action)
                    if fired:
                        cnn_engine.mark_fired(action)
                        label = ACTION_CATALOGUE[action][0]
                        cv2.putText(image, f"CNN: {label}  {int(conf*100)}%",
                                    (image_w // 2 - 80, image_h - 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 180), 2)
                        if action != 'move_cursor':
                            return image

        # ── Cursor movement (smoothed moving average) ──
        x_diff = abs((self.x_coord - hd.middle_tip[0]) / max(self.x_coord, 1)) * 100
        y_diff = abs((self.y_coord - hd.middle_tip[1]) / max(self.y_coord, 1)) * 100
        if x_diff >= 0.4 or y_diff >= 0.4:
            buf_len = clamp(int(dist / 1.5), 1, 4)
            self.x_arr.insert(0, hd.middle_tip[0])
            self.y_arr.insert(0, hd.middle_tip[1])
            self.x_arr = self.x_arr[:buf_len]
            self.y_arr = self.y_arr[:buf_len]
            self.x_coord = int(sum(self.x_arr) / len(self.x_arr))
            self.y_coord = int(sum(self.y_arr) / len(self.y_arr))
            win32api.SetCursorPos((clamp(self.x_coord, 1, screen_w - 1),
                                   clamp(self.y_coord, 1, screen_h - 1)))

        # ── Hand-fills-frame → voice dictation ──
        hand_area = math.pi * hd.hand_width * hd.hand_height
        if image_w * image_h * 0.90 < hand_area:
            keyboard.send('windows+h')
            time.sleep(1.5)

        # ── Pinch thumb+pinky → task view ──
        if (abs(hd.pinky_tip[0] - hd.thumb_tip[0]) < 150 / dist and
                abs(hd.pinky_tip[1] - hd.thumb_tip[1]) < 150 / dist and
                abs(hd.pinky_mcp[0] - hd.index_mcp[0]) > 150 / dist):
            keyboard.send('windows+tab')
            time.sleep(0.8)
            return image

        # ── Index up only → voice dictation ──
        if (not hd.index_down and hd.middle_down and
                hd.ring_down and hd.pinky_down and hd.thumb_down):
            keyboard.send('windows+h')
            time.sleep(1.0)
            return image

        # ── Peace sign → volume control ──
        if (not hd.index_down and not hd.middle_down and
                hd.ring_down and hd.pinky_down and hd.thumb_down):
            vol_target = clamp(int((1.0 - hd.middle_tip[1] / screen_h) * 100), 0, 100)
            now = time.time()
            if now - self.vol_cooldown > 0.08:
                if vol_target > self.vol_level + 3:
                    keyboard.send('volume up')
                    self.vol_level = min(self.vol_level + 2, 100)
                    self.vol_cooldown = now
                elif vol_target < self.vol_level - 3:
                    keyboard.send('volume down')
                    self.vol_level = max(self.vol_level - 2, 0)
                    self.vol_cooldown = now
            return image

        # ── Remaining gestures: right-click, scroll, left-click ──
        if (hd.ring_down and not hd.pinky_down and
                not hd.index_down and not hd.middle_down):
            win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
            win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP,   0, 0, 0, 0)
            time.sleep(0.25)
        elif (hd.thumb_tip[0] > hd.index_mcp[0] and not is_left_hand and
              not hd.middle_down and not hd.pinky_down and
              hd.thumb_tip[0] < hd.pinky_tip[0]):
            self.up_count += 1; self.down_count = 0
            mouse.wheel(4 if self.up_count >= 30 else (2 if self.up_count >= 20 else 1))
        elif (hd.thumb_tip[0] < hd.index_mcp[0] and is_left_hand and
              not hd.middle_down and not hd.pinky_down and
              hd.thumb_tip[0] > hd.pinky_tip[0]):
            self.up_count += 1; self.down_count = 0
            mouse.wheel(4 if self.up_count >= 30 else (2 if self.up_count >= 20 else 1))
        elif (hd.index_tip[1] > hd.index_dip[1] and not self.left_click_down and
              hd.index_tip[1] < hd.wrist[1] and
              not hd.pinky_down and not hd.middle_down):
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            self.left_click_down = True
        elif (hd.index_tip[1] <= hd.index_dip[1] and self.left_click_down and
              hd.index_tip[1] < hd.wrist[1] and not hd.middle_down):
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            self.left_click_down = False
        elif (hd.pinky_down and not hd.middle_down and
              hd.index_tip[1] < hd.wrist[1]):
            self.down_count += 1; self.up_count = 0
            mouse.wheel(-4 if self.down_count >= 30 else (-2 if self.down_count >= 20 else -1))
        else:
            self.up_count = 0; self.down_count = 0

        return image

    # ── CNN action dispatcher ─────────────────────
    def _dispatch_action(self, action):
        """Fire the named cursor action. Returns True if something was sent."""
        if action == 'move_cursor':
            return False   # handled by main movement block
        elif action == 'left_click':
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,  0, 0, 0, 0)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,    0, 0, 0, 0)
        elif action == 'right_click':
            win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
            win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP,   0, 0, 0, 0)
        elif action == 'scroll_up':   mouse.wheel(2)
        elif action == 'scroll_down': mouse.wheel(-2)
        elif action == 'vol_up':      keyboard.send('volume up')
        elif action == 'vol_down':    keyboard.send('volume down')
        elif action == 'task_view':   keyboard.send('windows+tab')
        else:
            return False
        return True