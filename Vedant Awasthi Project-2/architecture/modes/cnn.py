# ──────────────────────────────────────────────
# modes/cnn.py — CNNMode
#
# Three logical sections kept in one file because
# they share private state (model, data, device):
#   1. GestureCNN   – PyTorch architecture
#   2. Skeleton     – landmark → 64×64 image
#   3. CNNMode      – capture / train / predict / bind / UI
# ──────────────────────────────────────────────

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == '__main__':
    print("cnn.py is a module — run main.py instead.")
    sys.exit(0)

import cv2
import os
import time
import pickle
import threading

import numpy as np

import keyboard as kb

from constants import (
    CNN_IMG_SIZE, MIN_SAMPLES_PER_CLASS, MAX_SAMPLES_PER_CLASS,
    DIVERSITY_THRESHOLD, CNN_CONFIDENCE_THRESHOLD,
    MP_CONNECTIONS, ACTION_CATALOGUE, BINDING_KEYS,
)
from utils import play_tone_async

# ── PyTorch (optional) ────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True

    if torch.cuda.is_available():
        _DEVICE = torch.device("cuda")
        print(f"[CNN] GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        _DEVICE = torch.device("mps")
        print("[CNN] Apple Silicon GPU (MPS)")
    else:
        _DEVICE = torch.device("cpu")
        print("[CNN] CPU  (install CUDA PyTorch for faster training)")

except ImportError:
    TORCH_AVAILABLE = False
    _DEVICE = None
    print("[WARNING] PyTorch not found — CNN mode disabled. pip install torch")


# ══════════════════════════════════════════════
# 1. MODEL ARCHITECTURE
# ══════════════════════════════════════════════

class GestureCNN(nn.Module):
    """
    3-block CNN for 64×64 hand skeleton images.
    Activation progression: ReLU → LeakyReLU → Tanh
      Block 1 (ReLU)      — sharp early feature detection
      Block 2 (LeakyReLU) — keeps negative gradients alive in mid-level features
      Block 3 (Tanh)      — smooth bounded activations before classification
    Input:  (N, 1, 64, 64)  float32
    Output: (N, num_classes) logits
    """
    def __init__(self, num_classes):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),          # → 32×32
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),          # → 16×16
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.MaxPool2d(2),          # → 8×8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)


# ══════════════════════════════════════════════
# 2. SKELETON IMAGE CONVERSION
# ══════════════════════════════════════════════

def landmarks_to_skeleton_image(lm, size=CNN_IMG_SIZE):
    """
    Convert 21 MediaPipe landmarks → normalised skeleton image.
    - Centred on wrist, scale-normalised (rotation/distance invariant).
    - Returns float32 array shape (1, size, size) in [0, 1].
    """
    pts = np.array([[l.x, l.y] for l in lm], dtype=np.float32)
    pts -= pts[0]                                # centre on wrist
    scale = np.max(np.abs(pts))
    pts  /= (scale if scale > 1e-6 else 1.0)    # normalise scale

    margin = int(size * 0.12)
    pts = (pts + 1.0) / 2.0 * (size - 2 * margin) + margin
    pts = pts.astype(np.int32)

    canvas = np.zeros((size, size), dtype=np.float32)
    for a, b in MP_CONNECTIONS:
        x1, y1 = np.clip(pts[a], 0, size - 1)
        x2, y2 = np.clip(pts[b], 0, size - 1)
        cv2.line(canvas, (x1, y1), (x2, y2), 0.6, 1, lineType=cv2.LINE_AA)

    tip_ids = {4, 8, 12, 16, 20}
    for i, p in enumerate(pts):
        x, y = np.clip(p, 0, size - 1)
        cv2.circle(canvas, (x, y), 3 if i in tip_ids else 2, 1.0, -1)

    return canvas[np.newaxis, :, :]   # (1, size, size)


# ══════════════════════════════════════════════
# 3. CNN MODE
# ══════════════════════════════════════════════

class CNNMode:
    """
    Self-training gesture recogniser.

    Workflow:
      1. Press 1-9  → select gesture slot
      2. Hold SPACE → capture skeleton frames
      3. Press T    → train CNN in background
      4. Auto       → live prediction once trained
      5. Press B    → open binding panel (assign actions)
      6. Press S/L  → save / load dataset + model
    """

    IDLE       = "IDLE"
    CAPTURING  = "CAPTURING"
    TRAINING   = "TRAINING"
    PREDICTING = "PREDICTING"
    BINDING    = "BINDING"

    SLOT_COLORS = [
        (255, 80,  80),  (255, 165, 0),  (255, 255, 0),
        (80,  255, 80),  (0,   255, 255),(80,  80,  255),
        (200, 0,   255), (255, 100, 200),(200, 200, 200),
    ]

    def __init__(self):
        self.data:        dict[int, list] = {}
        self.label_names: dict[int, str]  = {}
        self.selected_slot = 1
        self.sub_state     = self.IDLE

        self.model       = None
        self.device      = _DEVICE
        self.num_classes = 0
        self.class_order: list[int] = []

        self.last_pred_label = ""
        self.last_pred_conf  = 0.0
        self.conf_history:   list[float] = []

        self._train_thread = None
        self._train_status = ""
        self._train_done   = False

        self._capture_cooldown  = 0.0
        self._capture_interval  = 0.05
        self._last_captured_vec: dict[int, np.ndarray] = {}

        self.bindings: dict[int, str]   = {}
        self._action_last_fired: dict[str, float] = {}
        self._prev_binding_state = self.IDLE

        self._preview_img = None

    # ── Slot helpers ──────────────────────────────
    def _slot_color(self, s):
        return self.SLOT_COLORS[(s - 1) % len(self.SLOT_COLORS)]

    def _sample_count(self, s):
        return len(self.data.get(s, []))

    def _ready_to_train(self):
        return (TORCH_AVAILABLE and len(self.data) >= 2 and
                all(len(v) >= MIN_SAMPLES_PER_CLASS for v in self.data.values()))

    # ── Capture ───────────────────────────────────
    def _landmark_vector(self, lm):
        pts = np.array([[l.x, l.y] for l in lm], dtype=np.float32)
        pts -= pts[0]
        pts /= (np.max(np.abs(pts)) + 1e-6)
        return pts.flatten()

    def _is_diverse(self, lm, slot):
        vec  = self._landmark_vector(lm)
        prev = self._last_captured_vec.get(slot)
        return prev is None or float(np.mean(np.abs(vec - prev))) > DIVERSITY_THRESHOLD

    def _capture_frame(self, lm):
        s = self.selected_slot
        self.data.setdefault(s, [])
        self.label_names.setdefault(s, f"Gesture {s}")
        if len(self.data[s]) < MAX_SAMPLES_PER_CLASS and self._is_diverse(lm, s):
            self.data[s].append(landmarks_to_skeleton_image(lm))
            self._last_captured_vec[s] = self._landmark_vector(lm)
            return True
        return False

    # ── Augmentation ─────────────────────────────
    @staticmethod
    def _augment(imgs: np.ndarray) -> np.ndarray:
        N, C, H, W = imgs.shape
        out = imgs.copy()
        for i in range(N):
            img = out[i, 0]
            if np.random.rand() < 0.5:
                img = np.fliplr(img).copy()
            for M in [
                cv2.getRotationMatrix2D((W/2, H/2), np.random.uniform(-20, 20), 1.0),
                cv2.getRotationMatrix2D((W/2, H/2), 0, np.random.uniform(0.85, 1.15)),
                np.float32([[1, 0, np.random.uniform(-6, 6)],
                             [0, 1, np.random.uniform(-6, 6)]]),
            ]:
                img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            img = np.clip(img + np.random.normal(0, 0.03, img.shape).astype(np.float32), 0, 1)
            out[i, 0] = img
        return out

    # ── Training ─────────────────────────────────
    def _train_worker(self):
        try:
            self._train_status = "Preparing data..."
            self.class_order   = sorted(self.data.keys())
            X_list, y_list = [], []
            for idx, slot in enumerate(self.class_order):
                for img in self.data[slot]:
                    X_list.append(img); y_list.append(idx)

            X_np = np.array(X_list, dtype=np.float32)
            y_np = np.array(y_list, dtype=np.int64)

            # 5× augmentation passes for better generalisation
            self._train_status = "Augmenting (5×)..."
            X_aug = np.concatenate([X_np] + [self._augment(X_np) for _ in range(5)])
            y_aug = np.tile(y_np, 6)
            perm  = np.random.permutation(len(X_aug))
            X_aug, y_aug = X_aug[perm], y_aug[perm]

            # 90/10 train/val split
            split     = int(len(X_aug) * 0.9)
            X_tr, X_val = X_aug[:split], X_aug[split:]
            y_tr, y_val = y_aug[:split], y_aug[split:]

            self.num_classes = len(self.class_order)
            model     = GestureCNN(self.num_classes).to(self.device)
            optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
            # Warm up for 10 epochs then cosine decay
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=3e-3,
                steps_per_epoch=max(1, len(X_tr) // 32),
                epochs=60, pct_start=0.15)
            # Label smoothing reduces overconfidence on small datasets
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

            tr_loader = DataLoader(
                TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
                batch_size=32, shuffle=True, num_workers=0,
                pin_memory=(str(self.device) == 'cuda'))

            dev = str(self.device).upper()
            best_val_acc = 0.0
            best_state   = None

            for ep in range(1, 61):
                # ── train ──
                model.train(); total_loss = 0.0
                for xb, yb in tr_loader:
                    xb = xb.to(self.device, non_blocking=True)
                    yb = yb.to(self.device, non_blocking=True)
                    optimizer.zero_grad()
                    loss = criterion(model(xb), yb)
                    loss.backward(); optimizer.step(); scheduler.step()
                    total_loss += loss.item()

                # ── validate every 5 epochs ──
                if ep % 5 == 0 or ep == 60:
                    model.eval()
                    with torch.no_grad():
                        xv = torch.tensor(X_val).to(self.device)
                        yv = torch.tensor(y_val).to(self.device)
                        preds = model(xv).argmax(dim=1)
                        val_acc = (preds == yv).float().mean().item() * 100
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_state   = {k: v.clone() for k, v in model.state_dict().items()}
                    self._train_status = (
                        f"[{dev}] Ep {ep}/60  "
                        f"loss={total_loss/len(tr_loader):.3f}  "
                        f"val={val_acc:.0f}%  best={best_val_acc:.0f}%")
                else:
                    self._train_status = (
                        f"[{dev}] Epoch {ep}/60  "
                        f"loss={total_loss/len(tr_loader):.3f}")

            # Load best checkpoint
            if best_state:
                model.load_state_dict(best_state)
            model.eval()
            self.model        = model
            self._train_status = (
                f"Done!  val acc={best_val_acc:.0f}%  "
                f"({len(X_aug)} samples, {self.num_classes} gestures)")
            self._train_done   = True

        except Exception as e:
            self._train_status = f"Error: {e}"
            self._train_done   = True

    def start_training(self):
        if not self._ready_to_train(): return
        self.sub_state   = self.TRAINING
        self._train_done = False
        threading.Thread(target=self._train_worker, daemon=True).start()

    # ── Inference ────────────────────────────────
    def _predict(self, lm):
        img    = landmarks_to_skeleton_image(lm)
        tensor = torch.tensor(img[np.newaxis], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1).cpu().numpy()[0]
        idx  = int(np.argmax(probs))
        slot = self.class_order[idx]
        return self.label_names.get(slot, f"Gesture {slot}"), float(probs[idx])

    # ── Action binding API (called by other modes) ──
    def get_active_action(self, lm):
        """
        Returns (action_id, confidence, slot) when model is confident.
        Returns (None, 0.0, None) otherwise.
        """
        if self.model is None or self.sub_state not in (self.PREDICTING, self.BINDING):
            return None, 0.0, None
        img    = landmarks_to_skeleton_image(lm)
        tensor = torch.tensor(img[np.newaxis], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1).cpu().numpy()[0]
        idx    = int(np.argmax(probs))
        conf   = float(probs[idx])
        if conf < CNN_CONFIDENCE_THRESHOLD:
            return None, conf, None
        slot   = self.class_order[idx]
        action = self.bindings.get(slot, 'none')
        return (None, conf, slot) if action == 'none' else (action, conf, slot)

    def can_fire(self, action_id):
        _, _, debounce = ACTION_CATALOGUE.get(action_id, ('', 'none', 0.3))
        return (time.time() - self._action_last_fired.get(action_id, 0.0)) >= debounce

    def mark_fired(self, action_id):
        self._action_last_fired[action_id] = time.time()

    # ── Save / Load ───────────────────────────────
    def save(self, path="gesture_data.pkl"):
        with open(path, "wb") as f:
            pickle.dump({
                "data": self.data, "label_names": self.label_names,
                "class_order": self.class_order, "num_classes": self.num_classes,
                "model_state": self.model.state_dict() if self.model else None,
                "bindings": self.bindings,
            }, f)

    def load(self, path="gesture_data.pkl"):
        if not os.path.exists(path): return False
        with open(path, "rb") as f:
            p = pickle.load(f)
        self.data, self.label_names = p["data"], p["label_names"]
        self.class_order, self.num_classes = p["class_order"], p["num_classes"]
        self.bindings = p.get("bindings", {})
        if p["model_state"] and TORCH_AVAILABLE:
            self.model = GestureCNN(self.num_classes).to(self.device)
            self.model.load_state_dict(p["model_state"])
            self.model.eval()
            self.sub_state = self.PREDICTING
        return True

    # ── Main process call ─────────────────────────
    def process(self, image, hand_landmarks, hd, image_w, image_h, key, hand_label="R"):
        lm = hand_landmarks.landmark
        self._handle_keys(key)

        if self.sub_state == self.TRAINING and self._train_done:
            self.sub_state = self.PREDICTING

        # Capture
        if kb.is_pressed('space') and self.sub_state != self.TRAINING:
            self.sub_state = self.CAPTURING
            now = time.time()
            if now - self._capture_cooldown >= self._capture_interval:
                accepted = self._capture_frame(lm)
                self._capture_cooldown = now
                play_tone_async(880 if accepted else 440, 20)
        elif self.sub_state == self.CAPTURING and not kb.is_pressed('space'):
            self.sub_state = self.IDLE

        # Predict
        if self.sub_state == self.PREDICTING and self.model is not None:
            label, conf = self._predict(lm)
            self.conf_history.append(conf)
            if len(self.conf_history) > 5: self.conf_history.pop(0)
            self.last_pred_label = f"[{hand_label}] {label}"
            self.last_pred_conf  = float(np.mean(self.conf_history))

        # Skeleton preview
        skel = landmarks_to_skeleton_image(lm, size=96)
        self._preview_img = (skel[0] * 255).astype(np.uint8)

        self._draw_ui(image, image_w, image_h)
        return image

    def _handle_keys(self, key):
        if key == 255: return
        if self.sub_state == self.BINDING:
            if key in BINDING_KEYS:
                action_id = BINDING_KEYS[key]
                self.bindings[self.selected_slot] = action_id
                self.label_names[self.selected_slot] = ACTION_CATALOGUE[action_id][0]
            elif key in (27, ord('b'), ord('B')):
                self.sub_state = self._prev_binding_state
            return

        if ord('1') <= key <= ord('9'):
            self.selected_slot = key - ord('0')
            if self.sub_state == self.PREDICTING: self.sub_state = self.IDLE
        elif key in (ord('b'), ord('B')) and self.model:
            self._prev_binding_state = self.sub_state
            self.sub_state = self.BINDING
        elif key in (ord('t'), ord('T')) and self._ready_to_train():
            self.start_training()
        elif key in (ord('r'), ord('R')):
            self.data.clear(); self.label_names.clear(); self.bindings.clear()
            self.model = None; self.sub_state = self.IDLE
            self._train_status = ""; self._train_done = False
        elif key in (ord('s'), ord('S')): self.save()
        elif key in (ord('l'), ord('L')): self.load()

    # ── UI drawing ────────────────────────────────
    def _draw_ui(self, image, image_w, image_h):
        self._draw_skeleton_preview(image, image_w, image_h)
        panel_y = self._draw_slot_list(image, image_w, image_h, y=50)
        panel_y = self._draw_state_badge(image, panel_y)
        panel_y = self._draw_train_status(image, panel_y)
        self._draw_prediction(image, panel_y, image_w)
        if self.sub_state == self.BINDING:
            self._draw_binding_panel(image, image_w, image_h)
        self._draw_bottom_hints(image, image_w, image_h)

    def _draw_skeleton_preview(self, image, image_w, image_h):
        if self._preview_img is None: return
        pad, sz = 8, 96
        x1, y1 = image_w - sz - pad, 50
        x2, y2 = image_w - pad, y1 + sz
        if y2 < image_h and x1 > 0:
            bgr = cv2.cvtColor(self._preview_img, cv2.COLOR_GRAY2BGR)
            cv2.addWeighted(bgr, 0.85, image[y1:y2, x1:x2], 0.15, 0, image[y1:y2, x1:x2])
            cv2.rectangle(image, (x1, y1), (x2, y2), (80, 80, 80), 1)
            cv2.putText(image, "skeleton", (x1 + 4, y2 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)

    def _draw_slot_list(self, image, image_w, image_h, y):
        px, lh = 8, 20
        cv2.putText(image, "GESTURE SLOTS", (px, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        y += lh
        shown = min(9, max(len(self.data) + 1, self.selected_slot))
        for s in range(1, shown + 1):
            count  = self._sample_count(s)
            name   = self.label_names.get(s, f"Gesture {s}")
            color  = self._slot_color(s)
            prefix = "► " if s == self.selected_slot else "  "
            bx, by = px + 125, y - 10
            # Progress bar
            cv2.rectangle(image, (bx, by), (bx + 55, by + 10), (40, 40, 40), -1)
            fw = int(55 * min(count, MIN_SAMPLES_PER_CLASS) / MIN_SAMPLES_PER_CLASS)
            if fw > 0: cv2.rectangle(image, (bx, by), (bx + fw, by + 10), color, -1)
            cv2.rectangle(image, (bx, by), (bx + 55, by + 10), (80, 80, 80), 1)
            # Binding badge
            aid = self.bindings.get(s, 'none')
            alabel, acat, _ = ACTION_CATALOGUE.get(aid, ('—', 'none', 0))
            cat_col = {'cursor': (0, 200, 255), 'draw': (180, 0, 255)}.get(acat, (70, 70, 70))
            cv2.putText(image, alabel[:8] if aid != 'none' else '—',
                        (bx + 58, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, cat_col, 1)
            # Slot label
            col = color if s == self.selected_slot else (150, 150, 150)
            cv2.putText(image, f"{prefix}[{s}] {name[:9]:9s} {count:3d}",
                        (px, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
            y += lh
        return y + 4

    def _draw_state_badge(self, image, y):
        colors = {
            self.IDLE: (120,120,120), self.CAPTURING: (0,220,100),
            self.TRAINING: (0,180,255), self.PREDICTING: (255,200,0),
            self.BINDING: (255,100,0),
        }
        cv2.putText(image, f"[ {self.sub_state} ]", (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors.get(self.sub_state, (200,200,200)), 2)
        return y + 24

    def _draw_train_status(self, image, y):
        if self._train_status:
            cv2.putText(image, self._train_status, (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 200, 255), 1)
            return y + 20
        return y

    def _draw_prediction(self, image, y, image_w):
        if self.sub_state != self.PREDICTING or not self.last_pred_label: return
        px = 8
        conf_pct = int(self.last_pred_conf * 100)
        cv2.putText(image, self.last_pred_label, (px, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 200, 0), 2)
        y += 26
        bw    = 160
        fill  = int(bw * self.last_pred_conf)
        bcolor= (0,255,100) if conf_pct>70 else ((0,200,255) if conf_pct>40 else (0,100,255))
        cv2.rectangle(image, (px, y), (px + bw, y + 12), (40, 40, 40), -1)
        if fill: cv2.rectangle(image, (px, y), (px + fill, y + 12), bcolor, -1)
        cv2.rectangle(image, (px, y), (px + bw, y + 12), (80, 80, 80), 1)
        cv2.putText(image, f"{conf_pct}%", (px + bw + 6, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, bcolor, 1)

    def _draw_bottom_hints(self, image, image_w, image_h):
        px = 8
        if self.sub_state != self.BINDING:
            for i, txt in enumerate([
                "SPACE:capture  T:train  B:bind  R:reset  S:save  L:load",
                "1-9:select slot  (CNN drives actions in other modes)",
            ]):
                cv2.putText(image, txt, (px, image_h - 30 + i * 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.30, (100, 200, 100), 1)

        if not self._ready_to_train() and self.sub_state == self.IDLE:
            needed = max(0, 2 - len(self.data))
            slots_ok = sum(1 for v in self.data.values() if len(v) >= MIN_SAMPLES_PER_CLASS)
            msg = (f"Need {needed} more slot(s) with ≥{MIN_SAMPLES_PER_CLASS} samples"
                   if needed else f"Press T to train  ({slots_ok} slots ready)")
            cv2.putText(image, msg, (px, image_h - 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 160, 60), 1)

        dev_str = str(self.device).upper() if self.device else "N/A"
        dev_col = (0,255,80) if dev_str=="CUDA" else (180,180,0) if dev_str=="MPS" else (120,120,120)
        if not TORCH_AVAILABLE:
            cv2.putText(image, "PyTorch not installed!  pip install torch",
                        (px, image_h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,0,255), 2)
        else:
            cv2.putText(image, f"Device: {dev_str}", (px, image_h - 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, dev_col, 1)

    def _draw_binding_panel(self, image, image_w, image_h):
        slot  = self.selected_slot
        name  = self.label_names.get(slot, f"Gesture {slot}")
        color = self._slot_color(slot)
        px, py, pw, ph = image_w - 260, 45, 255, image_h - 55

        roi = image[py:py+ph, px:px+pw]
        cv2.addWeighted(np.full_like(roi, (18,18,28), np.uint8), 0.88, roi, 0.12, 0, roi)
        image[py:py+ph, px:px+pw] = roi
        cv2.rectangle(image, (px, py), (px+pw, py+ph), color, 1)

        cx, cy, lh = px + 8, py + 18, 17
        cv2.putText(image, f"BIND SLOT [{slot}]: {name[:12]}", (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
        cy += lh
        cv2.line(image, (cx, cy), (px+pw-8, cy), (80,80,80), 1)
        cy += lh

        cursor_items = [
            ('[1] Move Cursor','move_cursor'), ('[2] Left Click','left_click'),
            ('[3] Right Click','right_click'), ('[4] Scroll Up','scroll_up'),
            ('[5] Scroll Down','scroll_down'), ('[6] Vol Up','vol_up'),
            ('[7] Vol Down','vol_down'),       ('[8] Task View','task_view'),
        ]
        draw_items = [
            ('[A] Draw','draw'),          ('[D] Erase','erase'),
            ('[F] Clear Canvas','clear'), ('[G] Next Color','color_next'),
            ('[H] Brush +','brush_bigger'),('[J] Brush -','brush_smaller'),
        ]

        cv2.putText(image, "── CURSOR ──", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,200,255), 1)
        cy += lh
        for label, aid in cursor_items:
            active = self.bindings.get(slot) == aid
            cv2.putText(image, ("✓ " if active else "  ") + label, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                        (0,255,150) if active else (180,220,230), 1)
            cy += lh

        cy += 4
        cv2.putText(image, "── DRAWING ──", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180,0,255), 1)
        cy += lh
        for label, aid in draw_items:
            active = self.bindings.get(slot) == aid
            cv2.putText(image, ("✓ " if active else "  ") + label, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                        (200,100,255) if active else (210,180,220), 1)
            cy += lh

        cy += 4
        cv2.putText(image, "[0] Unbind  |  [B]/[Esc] Close", (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.31, (140,140,140), 1)