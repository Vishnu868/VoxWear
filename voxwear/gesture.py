"""MediaPipe hand landmarks -> 86-dimensional feature vector -> XGBoost gesture classifier."""
import itertools

import cv2
import joblib
import mediapipe as mp
import numpy as np

FINGERTIPS = [4, 8, 12, 16, 20]
# (MCP, PIP, TIP) triplets for the 5 finger bend angles
FINGER_JOINTS = [(1, 2, 4), (5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]

INTENTS = {
    "hello": "greeting", "thanks": "gratitude", "help": "assistance_request", "yes": "affirm",
    "no": "deny", "water": "drink_request", "food": "food_request", "medicine": "medical_request",
    "pain": "medical_alert", "stop": "stop", "come": "call", "go": "dismiss",
    "what": "question", "where": "question", "happy": "emotion", "sad": "emotion",
}


def landmark_features(lm: np.ndarray) -> np.ndarray:
    """lm: (21, 3) normalised landmarks -> 86 features."""
    pts = lm - lm[0]                                           # wrist-centred
    scale = np.linalg.norm(pts[9]) + 1e-8                      # wrist -> middle MCP
    pts = pts / scale                                          # scale invariant
    coords = pts.ravel()                                       # 63

    dists = [np.linalg.norm(pts[a] - pts[b])                   # 10 tip-tip + 5 tip-wrist = 15
             for a, b in itertools.combinations(FINGERTIPS, 2)]
    dists += [np.linalg.norm(pts[t]) for t in FINGERTIPS]

    angles = []                                                # 5 bend angles
    for a, b, c in FINGER_JOINTS:
        v1, v2 = pts[a] - pts[b], pts[c] - pts[b]
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        angles.append(np.arccos(np.clip(cos, -1, 1)))

    normal = np.cross(pts[5], pts[17])                         # 3 palm orientation
    normal = normal / (np.linalg.norm(normal) + 1e-8)
    return np.concatenate([coords, dists, angles, normal]).astype(np.float32)   # 86


class HandTracker:
    def __init__(self, max_hands: int = 1, min_conf: float = 0.6):
        self.hands = mp.solutions.hands.Hands(max_num_hands=max_hands,
                                              min_detection_confidence=min_conf,
                                              min_tracking_confidence=0.5)

    def landmarks(self, bgr: np.ndarray):
        res = self.hands.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_hand_landmarks:
            return None, None
        hand = res.multi_hand_landmarks[0]
        lm = np.array([[p.x, p.y, p.z] for p in hand.landmark])
        handed = res.multi_handedness[0].classification[0].label
        return lm, handed

    def draw(self, bgr, lm):
        h, w = bgr.shape[:2]
        for x, y, _ in lm:
            cv2.circle(bgr, (int(x * w), int(y * h)), 3, (0, 255, 0), -1)


class GestureRecognizer:
    def __init__(self, model_path: str = "models/gesture_xgb.joblib", threshold: float = 0.7):
        bundle = joblib.load(model_path)
        self.model, self.labels = bundle["model"], bundle["labels"]
        self.threshold = threshold
        self.tracker = HandTracker()

    def predict(self, bgr: np.ndarray):
        lm, _ = self.tracker.landmarks(bgr)
        if lm is None:
            return None
        probs = self.model.predict_proba(landmark_features(lm)[None])[0]
        i = int(np.argmax(probs))
        if probs[i] < self.threshold:
            return None
        g = self.labels[i]
        return {"gesture": g, "confidence": float(probs[i]), "intent": INTENTS.get(g, g),
                "landmarks": lm}
