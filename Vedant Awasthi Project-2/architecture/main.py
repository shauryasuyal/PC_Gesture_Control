"""
MK Motion Control — Multi-Mode Hand Tracker
============================================
Modes:  C = Cursor  |  P = CNN Gesture  |  X = Drawing  |  Z = Sandbox
Press Q to quit.

Run from inside the mk_motion/ directory:
    python main.py
"""

import sys
import os

# Ensure the folder containing main.py is always on the path,
# regardless of how the script is invoked (IDE, shell, etc.)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import ctypes
import mediapipe as mp
import time

from constants import MODE_CURSOR, MODE_CNN, MODE_DRAWING, MODE_SANDBOX
from hand_data import HandData
from utils     import draw_hud, draw_mode_flash, clamp
from modes     import CursorMode, CNNMode, DrawingMode, SandboxMode


def main():
    # ── Screen dimensions ─────────────────────────
    user32   = ctypes.windll.user32
    screen_w = user32.GetSystemMetrics(0)
    screen_h = user32.GetSystemMetrics(1)

    # ── MediaPipe setup ───────────────────────────
    mp_drawing      = mp.solutions.drawing_utils
    mp_hands_module = mp.solutions.hands

    # ── State ─────────────────────────────────────
    current_mode    = MODE_CURSOR
    flash_alpha     = 0.0
    flash_mode      = MODE_CURSOR
    hand_counter    = 0
    lost_hand_count = 0
    active_count    = 0
    hand_shown      = False
    image_size_mod  = 5
    fps = frame_count = 0
    fps_start = time.time()

    # ── Mode instances ────────────────────────────
    cursor_mode  = CursorMode()
    cnn_mode     = CNNMode()
    drawing_mode = None      # initialised after first frame (need image size)
    sandbox_mode = SandboxMode()

    # ── Camera ────────────────────────────────────
    print("Opening camera...")
    cap = None
    for cam_index in [0, 1, 2]:
        _cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)   # CAP_DSHOW = faster on Windows
        if _cap.isOpened():
            ok, _test = _cap.read()
            if ok:
                cap = _cap
                print(f"Camera opened on index {cam_index}")
                break
            _cap.release()
        else:
            _cap.release()

    if cap is None:
        print("ERROR: Could not open any camera (tried indices 0-2).")
        print("  - Make sure no other app (Teams, Zoom, etc.) is using the camera.")
        print("  - Try unplugging and replugging the camera.")
        input("Press Enter to exit.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print("Camera ready. Starting main loop — press Q in the window to quit.")

    # ── Create window at 85% of screen ───────────
    WIN_NAME = 'MK Motion Control'
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    win_w = int(screen_w * 0.85)
    win_h = int(screen_h * 0.85)
    cv2.resizeWindow(WIN_NAME, win_w, win_h)
    cv2.moveWindow(WIN_NAME, (screen_w - win_w) // 2, (screen_h - win_h) // 2)
    cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_TOPMOST, 1)
    print(f"Window: {win_w}x{win_h}  (Up/Down arrows to resize)")

    with mp_hands_module.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as hands:

        _bad_frames = 0
        while cap.isOpened():
            ok, image = cap.read()
            if not ok:
                _bad_frames += 1
                print(f"Bad frame #{_bad_frames}")
                if _bad_frames > 30:
                    print("Too many bad frames — camera may have disconnected.")
                    break
                continue
            _bad_frames = 0

            image_h, image_w, _ = image.shape

            # Late-init drawing mode (needs frame size)
            if drawing_mode is None:
                drawing_mode = DrawingMode(image_w, image_h)

            # ── Keyboard input ────────────────────
            key = cv2.waitKey(1) & 0xFF
            new_mode = {
                ord('c'): MODE_CURSOR,
                ord('p'): MODE_CNN,
                ord('x'): MODE_DRAWING,
                ord('z'): MODE_SANDBOX,
            }.get(key)

            if key == ord('q'):
                break

            if new_mode and new_mode != current_mode:
                current_mode = new_mode
                flash_alpha  = 0.4
                flash_mode   = new_mode
                cursor_mode.reset_trail()

            if flash_alpha > 0:
                flash_alpha = max(0.0, flash_alpha - 0.03)

            # ── MediaPipe inference ───────────────
            image.flags.writeable = False
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            image.flags.writeable = True

            if results.multi_hand_landmarks:
                lost_hand_count = 0
                active_count   += 1
                hand_shown      = True

                hand_pairs = list(zip(results.multi_hand_landmarks,
                                      results.multi_handedness))

                for hand_landmarks, handedness in hand_pairs:
                    # MediaPipe 'Right' = mirror-right = user's left hand
                    is_left = handedness.classification[0].label == 'Right'
                    hand_label = "L" if is_left else "R"
                    hd = HandData(hand_landmarks, screen_w, screen_h, image_w, image_h)
                    hand_counter = min(hand_counter + 2, 60)

                    if current_mode == MODE_CURSOR:
                        image = cursor_mode.process(
                            image, hd, screen_w, screen_h, image_w, image_h,
                            is_left, cnn_engine=cnn_mode)
                    elif current_mode == MODE_CNN:
                        image = cnn_mode.process(
                            image, hand_landmarks, hd, image_w, image_h,
                            key, hand_label)
                    elif current_mode == MODE_DRAWING:
                        image = drawing_mode.process(
                            image, hd, screen_w, screen_h, image_w, image_h,
                            cnn_engine=cnn_mode)
                    elif current_mode == MODE_SANDBOX:
                        image = sandbox_mode.process(
                            image, hand_landmarks, hd, mp_hands_module, image_w, image_h)

                    if current_mode != MODE_SANDBOX:
                        mp_drawing.draw_landmarks(
                            image, hand_landmarks, mp_hands_module.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(thickness=2, circle_radius=1,
                                                   color=(0, 127, 255)))
            else:
                lost_hand_count += 1
                cursor_mode.reset_trail()
                if lost_hand_count >= 20:
                    hand_counter = active_count = 0
                    hand_shown   = False

            # ── FPS counter ───────────────────────
            frame_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps         = int(frame_count / elapsed)
                frame_count = 0
                fps_start   = time.time()

            # ── Overlays ──────────────────────────
            image = draw_hud(image, current_mode, fps, image_h, image_w)
            if flash_alpha > 0:
                image = draw_mode_flash(image, flash_mode, flash_alpha, image_w, image_h)

            # ── Mirror + display ──────────────────
            image = cv2.flip(image, 1)

            # Arrow keys resize window live
            if key == 82:   # up arrow → bigger
                win_w = min(int(win_w * 1.1), screen_w)
                win_h = int(win_w / (image_w / image_h))
                cv2.resizeWindow(WIN_NAME, win_w, win_h)
            elif key == 84:  # down arrow → smaller
                win_w = max(int(win_w * 0.9), 320)
                win_h = int(win_w / (image_w / image_h))
                cv2.resizeWindow(WIN_NAME, win_w, win_h)

            cv2.imshow(WIN_NAME, image)

    cap.release()
    cv2.destroyAllWindows()
    print("Exited cleanly.")


if __name__ == '__main__':
    main()