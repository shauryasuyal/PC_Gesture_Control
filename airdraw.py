import cv2
import mediapipe as mp
import numpy as np

# ── MediaPipe setup ──────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(max_num_hands=1,
                          min_detection_confidence=0.75,
                          min_tracking_confidence=0.75)

# ── State ────────────────────────────────────────────────────────────────────
COLORS = {
    "red":   (0,   0,   255),
    "blue":  (255, 0,   0),
    "green": (0,   255, 0),
    "white": (255, 255, 255),   # eraser colour (drawn on canvas)
}

draw_color   = COLORS["red"]
brush_size   = 8
eraser_size  = 40
prev_x, prev_y = None, None
canvas       = None          # created once we know frame size

# ── Finger-tip & pip landmark IDs ───────────────────────────────────────────
TIPS = [4, 8, 12, 16, 20]   # thumb, index, middle, ring, pinky
PIPS = [3, 6, 10, 14, 18]

def fingers_up(lm, w, h):
    """Return list of 5 bools [thumb, index, middle, ring, pinky]."""
    pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in range(21)]

    up = []
    # Thumb: compare x (for right hand facing camera, tip to the left of pip)
    up.append(pts[TIPS[0]][0] < pts[PIPS[0]][0])
    # Other four: tip y < pip y  (tip higher on screen)
    for i in range(1, 5):
        up.append(pts[TIPS[i]][1] < pts[PIPS[i]][1])
    return up, pts

def classify_gesture(up):
    """Map finger states to a gesture string."""
    thumb, index, middle, ring, pinky = up

    raised = sum(up)

    # Fist  – no fingers up
    if raised == 0:
        return "clear"

    # Open palm – all five fingers up
    if raised == 5:
        return "erase"

    # Draw  – only index up
    if index and not middle and not ring and not pinky:
        return "draw"

    # 2 fingers (index + middle) → red
    if index and middle and not ring and not pinky:
        return "color_red"

    # 3 fingers (index + middle + ring) → blue
    if index and middle and ring and not pinky:
        return "color_blue"

    # 4 fingers (index + middle + ring + pinky) → green
    if index and middle and ring and pinky:
        return "color_green"

    return "idle"

# ── UI helpers ───────────────────────────────────────────────────────────────
PALETTE = [
    ("RED  (2)",   COLORS["red"]),
    ("BLUE (3)",   COLORS["blue"]),
    ("GREEN(4)",   COLORS["green"]),
]

def draw_ui(frame, gesture, colour):
    h, w = frame.shape[:2]

    # Colour swatches at the top
    swatch_w = 110
    for i, (label, col) in enumerate(PALETTE):
        x1 = 10 + i * (swatch_w + 8)
        x2 = x1 + swatch_w
        cv2.rectangle(frame, (x1, 8), (x2, 48), col, -1)
        # Highlight active colour
        if col == colour:
            cv2.rectangle(frame, (x1, 8), (x2, 48), (255,255,255), 3)
        cv2.putText(frame, label, (x1+4, 38), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255,255,255), 1, cv2.LINE_AA)

    # Gesture status bottom-left
    cv2.putText(frame, f"Gesture: {gesture}", (10, h-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200,200,200), 2, cv2.LINE_AA)

    # Instructions bottom-right
    info = [
        "1 finger = draw",
        "2 fingers = RED",
        "3 fingers = BLUE",
        "4 fingers = GREEN",
        "Open palm = erase",
        "Fist = clear all",
        "Q = quit",
    ]
    for j, line in enumerate(info):
        cv2.putText(frame, line, (w-210, h - len(info)*20 + j*20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180,180,180), 1, cv2.LINE_AA)

# ── Main loop ────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

gesture = "idle"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)          # mirror for natural drawing
    h, w  = frame.shape[:2]

    # Create blank canvas on first frame
    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        lm     = result.multi_hand_landmarks[0].landmark
        up, pts = fingers_up(lm, w, h)
        gesture = classify_gesture(up)
        tip_x, tip_y = pts[8]           # index fingertip

        # ── Gesture actions ──────────────────────────────────────────────
        if gesture == "clear":
            canvas[:] = 0
            prev_x, prev_y = None, None

        elif gesture == "color_red":
            draw_color = COLORS["red"]
            prev_x, prev_y = None, None

        elif gesture == "color_blue":
            draw_color = COLORS["blue"]
            prev_x, prev_y = None, None

        elif gesture == "color_green":
            draw_color = COLORS["green"]
            prev_x, prev_y = None, None

        elif gesture == "draw":
            if prev_x is not None and prev_y is not None:
                cv2.line(canvas, (prev_x, prev_y), (tip_x, tip_y),
                         draw_color, brush_size)
            prev_x, prev_y = tip_x, tip_y

        elif gesture == "erase":
            cv2.circle(canvas, (tip_x, tip_y), eraser_size, (0, 0, 0), -1)
            prev_x, prev_y = None, None

        else:
            prev_x, prev_y = None, None   # idle / unrecognised

        # Draw hand skeleton
        mp_draw.draw_landmarks(frame,
                               result.multi_hand_landmarks[0],
                               mp_hands.HAND_CONNECTIONS)

        # Show eraser circle on frame
        if gesture == "erase":
            cv2.circle(frame, (tip_x, tip_y), eraser_size, (0,0,0), 2)

    else:
        gesture = "idle"
        prev_x, prev_y = None, None

    # ── Merge canvas onto frame ──────────────────────────────────────────
    # Wherever canvas has colour, overlay it on the camera feed
    mask        = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask     = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    mask_inv    = cv2.bitwise_not(mask)
    frame_bg    = cv2.bitwise_and(frame, frame, mask=mask_inv)
    canvas_fg   = cv2.bitwise_and(canvas, canvas, mask=mask)
    combined    = cv2.add(frame_bg, canvas_fg)

    draw_ui(combined, gesture, draw_color)

    cv2.imshow("✋ Air Canvas  |  Q to quit", combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()