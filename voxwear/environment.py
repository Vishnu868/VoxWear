"""YOLOv8-nano environment detection -> natural-language scene description."""
from collections import Counter


class EnvironmentDetector:
    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.5):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf = conf

    def describe(self, bgr) -> str:
        res = self.model(bgr, conf=self.conf, verbose=False)[0]
        names = [res.names[int(c)] for c in res.boxes.cls]
        if not names:
            return "I don't see any notable objects."
        w = bgr.shape[1]
        counts = Counter(names)
        parts = []
        for name, n in counts.most_common(6):
            # rough left/right position of the first instance
            box = next(b for b, c in zip(res.boxes.xyxy, res.boxes.cls) if res.names[int(c)] == name)
            cx = float(box[0] + box[2]) / 2
            side = "on your left" if cx < w / 3 else ("on your right" if cx > 2 * w / 3 else "ahead")
            parts.append(f"{n} {name}{'s' if n > 1 else ''} {side}" if n > 1 else f"a {name} {side}")
        return "I see " + ", ".join(parts) + "."
