"""
utils/tracker.py
────────────────────────────────────────────────────────────
Fixes applied:
  • TRACK-1: Python 3.8 compatible type hints (no float|None, list[dict] etc.)
  • TRACK-2: get_previous_bottom_y() wired correctly — stores (cy, y2) tuples
  • TRACK-3: get_previous_y() kept as legacy but clearly marked
"""

import logging
from typing import Optional, List, Dict
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

SPEED_SMOOTH_FRAMES = 10  # Increased from 5 to 10 for better stability
MAX_SPEED_CAP = 150.0

class ObjectTracker:
    def __init__(self, cfg: dict):
        speed_cfg    = cfg.get("speed", {})
        wrongway_cfg = cfg.get("wrong_way", {})

        self.ppm           = float(speed_cfg.get("pixels_per_meter", 8.0))
        self.speed_limit   = float(speed_cfg.get("limit_kmh", 60))
        self.fps           = None   # set externally: tracker.fps = cap.get(CAP_PROP_FPS)
        self.speed_enabled = bool(speed_cfg.get("enabled", True))

        self.wrongway_enabled = bool(wrongway_cfg.get("enabled", True))
        self.ww_min_frames    = int(wrongway_cfg.get("min_frames_to_confirm", 8))

        # Stores (centre_y, bottom_y) tuples per vehicle ID
        # TRACK-1: Dict[int, deque] instead of dict[int, deque] for Python 3.8
        self._history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=30))

    # ─────────────────────────────────────────────────────────
    # TRACK-1: List[dict] instead of list[dict]
    def get_tracking_data(self, results) -> List[dict]:
        tracking_info = []

        if results[0].boxes.id is None:
            return tracking_info

        boxes   = results[0].boxes.xyxy.cpu().numpy()
        ids     = results[0].boxes.id.cpu().numpy().astype(int)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)

        for box, obj_id, cls in zip(boxes, ids, classes):
            x1, y1, x2, y2 = box
            cy = (y1 + y2) / 2

            # Store both so callers can pick the right coordinate type
            self._history[obj_id].append((cy, y2))

            speed_kmh = self._estimate_speed(obj_id)
            wrong_way = self._check_wrong_way(obj_id)

            tracking_info.append({
                "id":        obj_id,
                "bbox":      box,
                "class":     cls,
                "speed_kmh": speed_kmh,
                "wrong_way": wrong_way,
                "speeding":  speed_kmh is not None and speed_kmh > self.speed_limit,
            })

        return tracking_info

    # ─────────────────────────────────────────────────────────
    # TRACK-1: Optional[float] instead of float | None
    def _estimate_speed(self, obj_id: int) -> Optional[float]:
        """Average displacement over SPEED_SMOOTH_FRAMES with a sanity cap."""
        if not self.speed_enabled or self.fps is None:
            return None

        hist = self._history[obj_id]
        # We need enough history to calculate an average
        if len(hist) < SPEED_SMOOTH_FRAMES + 1:
            return None

        # 1. Get the vertical center positions (cy) for the window
        window = [cy for cy, _ in list(hist)[-SPEED_SMOOTH_FRAMES - 1:]]
        
        # 2. Calculate displacement between consecutive frames
        total_px = sum(abs(b - a) for a, b in zip(window, window[1:]))
        avg_px = total_px / SPEED_SMOOTH_FRAMES

        # 3. Convert Pixels/Frame to km/h
        # Formula: (Pixels / PPM) * FPS * 3.6
        speed_kmh = (avg_px / self.ppm * self.fps) * 3.6
        
        # 4. Apply the "Sanity Cap" (The 222 km/h Fix)
        if speed_kmh > MAX_SPEED_CAP:
            # If the speed is impossible, return the previous known speed 
            # or 0 to avoid false violation alerts.
            return 0.0

        return round(speed_kmh, 1)

    # ─────────────────────────────────────────────────────────
    def _check_wrong_way(self, obj_id: int) -> bool:
        """Normal traffic moves DOWN (Y increases). Flag sustained upward movement."""
        if not self.wrongway_enabled:
            return False

        hist = self._history[obj_id]
        if len(hist) < self.ww_min_frames:
            return False

        recent       = [cy for cy, _ in list(hist)[-self.ww_min_frames:]]
        upward_steps = sum(1 for a, b in zip(recent, recent[1:]) if b < a)
        return upward_steps >= (self.ww_min_frames - 1)

    # ─────────────────────────────────────────────────────────
    # TRACK-1: Optional[float] instead of float | None
    def get_previous_bottom_y(self, obj_id: int) -> Optional[float]:
        """
        TRACK-2: Returns bottom-edge Y (y2) from the previous frame.
        Use this for crossing-line detection in main.py — it must match
        the y2 coordinate used on the current frame.
        """
        hist = self._history[obj_id]
        if len(hist) < 2:
            return None
        _, bottom_y = list(hist)[-2]
        return bottom_y

    def get_previous_y(self, obj_id: int) -> Optional[float]:
        """Returns centre-Y from previous frame. Use get_previous_bottom_y() for crossing checks."""
        hist = self._history[obj_id]
        if len(hist) < 2:
            return None
        cy, _ = list(hist)[-2]
        return cy