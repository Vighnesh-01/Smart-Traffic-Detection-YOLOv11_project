"""
utils/detector.py
────────────────────────────────────────────────────────────
Fixes applied:
  • DET-1: vehicle_classes now accepted as constructor param (MAIN-3)
  • DET-2: Python 3.8 compatible type hints (Optional, List)
"""

from typing import Optional, List
from ultralytics import YOLO


class TrafficDetector:
    def __init__(self,
                 vehicle_model_path: str = "weights/yolo11n.pt",
                 helmet_model_path:  Optional[str] = None,
                 confidence:         float = 0.4,
                 # DET-1: accept vehicle_classes from config
                 vehicle_classes:    Optional[List[int]] = None):

        self.vehicle_model   = YOLO(vehicle_model_path)
        self.confidence      = confidence
        # DET-1: fallback to COCO car/motorcycle/bus/truck if not provided
        self.vehicle_classes = vehicle_classes or [2, 3, 5, 7]

        self.helmet_model = None
        if helmet_model_path:
            self.helmet_model = YOLO(helmet_model_path)

    def detect_vehicles(self, frame):
        """
        Uses .track() so ByteTrack assigns stable IDs across frames.
        persist=True is REQUIRED — without it a new ID is given every frame.
        confidence and vehicle_classes both come from config.yaml.
        """
        results = self.vehicle_model.track(
            frame,
            conf=self.confidence,
            classes=self.vehicle_classes,   # DET-1: from config, not hardcoded
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )
        return results