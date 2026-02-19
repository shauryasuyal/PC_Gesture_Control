"""
Gesture Classifier — ML pipeline for hand gesture recognition.
Uses scikit-learn RandomForest trained on MediaPipe hand landmark features.
"""

import os
import json
import pickle
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "gesture_model.pkl")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "models", "label_encoder.pkl")
SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "data", "samples")


def extract_features(landmarks):
    """
    Convert 21 MediaPipe landmarks to a normalized 63-feature vector.
    Features are relative to wrist (translation invariant) and scaled by hand size.
    
    Args:
        landmarks: list of 21 landmark objects with .x, .y, .z or a flat array
    
    Returns:
        numpy array of 63 features
    """
    if hasattr(landmarks[0], 'x'):
        # MediaPipe landmark objects
        coords = np.array([[l.x, l.y, l.z] for l in landmarks])
    else:
        # Already a numpy array or list of lists
        coords = np.array(landmarks).reshape(21, 3)

    # Translate relative to wrist (landmark 0)
    wrist = coords[0].copy()
    coords = coords - wrist

    # Scale by hand size (distance from wrist to middle finger MCP)
    hand_scale = np.linalg.norm(coords[9])  # middle finger MCP
    if hand_scale > 0:
        coords = coords / hand_scale

    return coords.flatten()


def load_training_data(gestures_config):
    """
    Load all recorded samples and labels from disk.
    
    Returns:
        X: numpy array of features (n_samples, 63)
        y: numpy array of labels (n_samples,)
        label_names: list of gesture IDs
    """
    X_list = []
    y_list = []

    for gesture in gestures_config.get("gestures", []):
        gid = gesture["id"]
        sample_path = os.path.join(SAMPLES_DIR, f"{gid}.npy")
        if os.path.exists(sample_path):
            samples = np.load(sample_path)
            if len(samples) > 0:
                X_list.append(samples)
                y_list.extend([gid] * len(samples))

    if not X_list:
        return None, None, []

    X = np.vstack(X_list)
    y = np.array(y_list)
    label_names = sorted(list(set(y_list)))
    return X, y, label_names


def train_model(gestures_config, progress_callback=None):
    """
    Train (or retrain) the gesture classifier.
    
    Args:
        gestures_config: dict from gestures.json
        progress_callback: optional fn(stage, percent, message)
    
    Returns:
        dict with training results {accuracy, num_samples, num_gestures, status}
    """
    if progress_callback:
        progress_callback("loading", 10, "Loading training data...")

    X, y, label_names = load_training_data(gestures_config)

    if X is None or len(label_names) < 1:
        return {
            "status": "error",
            "message": "Need at least 1 gesture with recorded samples to train.",
            "accuracy": 0,
            "num_samples": 0,
            "num_gestures": 0
        }

    # If only 1 gesture class, add dummy noise data as a "Background" class
    # This prevents the classifier from predicting the single class with 100% confidence for everything
    if len(label_names) == 1:
        n_noise = max(50, len(X) // 2)
        noise_X = np.random.rand(n_noise, X.shape[1])
        noise_y = np.array(["_background_noise"] * n_noise)
        X = np.vstack([X, noise_X])
        y = np.concatenate([y, noise_y])
        label_names.append("_background_noise")  # MUST be in label_names so encoder is consistent


    if progress_callback:
        progress_callback("encoding", 20, "Encoding labels...")

    # Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    if progress_callback:
        progress_callback("training", 40, "Training Random Forest classifier...")

    # Train classifier (50 trees — fast inference with minimal accuracy loss)
    clf = RandomForestClassifier(
        n_estimators=50,
        max_depth=12,
        min_samples_split=3,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X, y_encoded)

    if progress_callback:
        progress_callback("evaluating", 70, "Evaluating accuracy...")

    # Cross-validation accuracy
    accuracy = 0.0
    if len(X) >= 10:
        n_splits = min(5, len(label_names))
        try:
            scores = cross_val_score(clf, X, y_encoded, cv=n_splits, scoring='accuracy')
            accuracy = float(np.mean(scores))
        except Exception:
            accuracy = clf.score(X, y_encoded)
    else:
        accuracy = clf.score(X, y_encoded)

    if progress_callback:
        progress_callback("saving", 90, "Saving model...")

    # Save model and encoder
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(clf, f)
    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump(encoder, f)

    if progress_callback:
        progress_callback("complete", 100, "Training complete!")

    # Report real counts — exclude the synthetic _background_noise class
    real_label_names = [l for l in label_names if l != "_background_noise"]
    real_mask = np.isin(y, real_label_names)
    real_sample_count = int(np.sum(real_mask))
    real_accuracy = accuracy

    # ── Confusion detection: check per-class accuracy ──
    warnings = []
    if len(real_label_names) >= 2 and np.sum(real_mask) >= 5:
        try:
            X_real = X[real_mask]
            y_real_raw = y[real_mask]
            y_real_enc = encoder.transform(y_real_raw)
            preds = clf.predict(X_real)
            # Check per-gesture accuracy
            for label in real_label_names:
                label_idx = encoder.transform([label])[0]
                mask = y_real_enc == label_idx
                if np.sum(mask) < 2:
                    continue
                label_preds = preds[mask]
                correct = np.sum(label_preds == label_idx)
                acc = correct / np.sum(mask)
                if acc < 0.80:
                    # Find what it's confused with
                    wrong = label_preds[label_preds != label_idx]
                    if len(wrong) > 0:
                        confused_counts = Counter(wrong)
                        top_confused_idx = confused_counts.most_common(1)[0][0]
                        confused_name = encoder.inverse_transform([top_confused_idx])[0]
                        if confused_name == '_background_noise':
                            warnings.append({
                                'gesture': label,
                                'accuracy': round(acc * 100),
                                'message': f'\"{label}\" is not distinctive enough — try holding the pose more clearly and recording more samples.'
                            })
                        else:
                            warnings.append({
                                'gesture': label,
                                'confused_with': confused_name,
                                'accuracy': round(acc * 100),
                                'message': f'\"{label}\" is being confused with \"{confused_name}\" — try making these two gestures more distinct from each other.'
                            })
        except Exception:
            pass  # non-critical, skip if anything goes wrong

    return {
        "success": True,
        "status": "success",
        "accuracy": round(real_accuracy * 100, 1),
        "num_samples": real_sample_count,
        "num_gestures": len(real_label_names),
        "num_classes": len(real_label_names),
        "total_samples": real_sample_count,
        "gestures_trained": real_label_names,
        "warnings": warnings
    }


class GesturePredictor:
    """Loads a trained model and provides real-time predictions."""

    def __init__(self):
        self.clf = None
        self.encoder = None
        self.loaded = False

    def load(self):
        """Load model from disk. Returns True if successful."""
        try:
            if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
                with open(MODEL_PATH, 'rb') as f:
                    self.clf = pickle.load(f)
                with open(ENCODER_PATH, 'rb') as f:
                    self.encoder = pickle.load(f)
                self.loaded = True
                return True
        except Exception as e:
            print(f"Error loading model: {e}")
        self.loaded = False
        return False

    def predict(self, landmarks):
        """
        Predict gesture from landmarks.
        
        Args:
            landmarks: 21 MediaPipe landmarks
            
        Returns:
            (gesture_id, confidence) or (None, 0.0)
        """
        if not self.loaded or self.clf is None:
            return None, 0.0

        try:
            features = extract_features(landmarks)
            features = features.reshape(1, -1)
            proba = self.clf.predict_proba(features)[0]
            pred_idx = np.argmax(proba)
            confidence = float(proba[pred_idx])
            gesture_id = self.encoder.inverse_transform([pred_idx])[0]
            # Suppress background noise class — it should never be returned as a real gesture
            if gesture_id == "_background_noise":
                return None, 0.0
            return gesture_id, confidence
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0.0

    def get_all_probabilities(self, landmarks):
        """Get probabilities for all classes."""
        if not self.loaded or self.clf is None:
            return {}

        try:
            features = extract_features(landmarks)
            features = features.reshape(1, -1)
            proba = self.clf.predict_proba(features)[0]
            classes = self.encoder.inverse_transform(range(len(proba)))
            return {cls: float(p) for cls, p in zip(classes, proba)
                    if cls != "_background_noise"}
        except Exception:
            return {}