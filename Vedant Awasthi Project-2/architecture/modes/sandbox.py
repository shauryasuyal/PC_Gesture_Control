# ──────────────────────────────────────────────
# modes/sandbox.py — SandboxMode
# Visualises raw MediaPipe landmarks with colour-
# coded joints and index numbers for debugging.
# ──────────────────────────────────────────────

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == '__main__':
    print("sandbox.py is a module — run main.py instead.")
    sys.exit(0)

import cv2


class SandboxMode:
    def process(self, image, hand_landmarks, hd, mp_hands, image_w, image_h):
        lm = hand_landmarks.landmark

        # Draw skeleton bones
        for conn in mp_hands.HAND_CONNECTIONS:
            p1, p2 = conn[0], conn[1]
            x1, y1 = int(lm[p1].x * image_w), int(lm[p1].y * image_h)
            x2, y2 = int(lm[p2].x * image_w), int(lm[p2].y * image_h)
            cv2.line(image, (x1, y1), (x2, y2), (100, 100, 100), 2, lineType=cv2.LINE_AA)

        # Draw joints with colour coding
        tip_ids = {4, 8, 12, 16, 20}
        mcp_ids = {1, 2, 5, 9, 13, 17}
        for i, l in enumerate(lm):
            x, y = int(l.x * image_w), int(l.y * image_h)
            if i == 0:           color, radius = (0, 0, 255),    8  # wrist
            elif i in tip_ids:   color, radius = (0, 255, 0),    7  # fingertips
            elif i in mcp_ids:   color, radius = (255, 150, 0),  6  # knuckles
            else:                color, radius = (255, 255, 0),  4  # pip / dip

            cv2.circle(image, (x, y), radius, color, -1)
            cv2.circle(image, (x, y), radius + 1, (255, 255, 255), 1)
            cv2.putText(image, str(i), (x + 8, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        return image