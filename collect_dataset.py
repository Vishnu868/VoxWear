"""VoxWear-FPV dataset recorder.

For each captured frame it saves: the RGB image, a night-vision (negative) image,
the 21 MediaPipe landmarks, and metadata.  SPACE = capture, A = auto-capture (0.5 s), Q = quit.
"""
import argparse
import csv
import json
import os
import time
from datetime import datetime

import cv2
import numpy as np

from voxwear.gesture import HandTracker


def night_vision(bgr: np.ndarray) -> np.ndarray:
    neg = cv2.bitwise_not(bgr)
    gray = cv2.cvtColor(neg, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)   # contrast boost
    tinted = np.zeros_like(bgr)
    tinted[:, :, 1] = gray                                                   # green tint
    tinted[:, :, 0] = tinted[:, :, 2] = (gray * 0.25).astype(np.uint8)
    return tinted


def lighting_condition(bgr: np.ndarray) -> str:
    v = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 2].mean()
    return "bright" if v > 170 else "normal" if v > 100 else "dim" if v > 50 else "dark"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gesture", required=True)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    for sub in ("normal", "negative"):
        os.makedirs(os.path.join(args.out, sub, args.gesture), exist_ok=True)
    meta_path = os.path.join(args.out, "metadata.csv")
    new_file = not os.path.exists(meta_path)
    meta = open(meta_path, "a", newline="")
    writer = csv.writer(meta)
    if new_file:
        writer.writerow(["file", "gesture", "timestamp", "lighting", "handedness", "landmarks"])

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    tracker = HandTracker()
    saved, auto, last, failed = 0, False, 0.0, 0

    while saved < args.samples:
        ok, frame = cap.read()
        if not ok:
            break
        lm, handed = tracker.landmarks(frame)
        view = frame.copy()
        if lm is not None:
            tracker.draw(view, lm)
        cv2.putText(view, f"{args.gesture} {saved}/{args.samples} {'AUTO' if auto else ''}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("VoxWear FPV recorder", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("a"):
            auto = not auto
        if key == ord(" ") or (auto and time.time() - last > 0.5):
            if lm is None:
                failed += 1
                continue
            name = f"{args.gesture}_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
            cv2.imwrite(os.path.join(args.out, "normal", args.gesture, name), frame)
            cv2.imwrite(os.path.join(args.out, "negative", args.gesture, name), night_vision(frame))
            writer.writerow([name, args.gesture, datetime.now().isoformat(),
                             lighting_condition(frame), handed, json.dumps(lm.round(5).tolist())])
            saved, last = saved + 1, time.time()

    meta.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {saved} samples for '{args.gesture}' ({failed} failed detections)")


if __name__ == "__main__":
    main()
