"""
utils/ocr_engine.py
────────────────────────────────────────────────────────────
Fixes applied:
  • OCR-1: cfg param now actually READ — confidence threshold from config
  • OCR-2: Python 3.8 compatible type hint (Optional[dict])
  • Previous fixes retained:
      FIX #5  — bottom 20% crop only
      FIX #10 — auto GPU detection
"""

import cv2
import numpy as np
import torch
import easyocr
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class PlateReader:
    # OCR-2: Optional[dict] instead of dict = None for Python 3.8
    def __init__(self, cfg: Optional[dict] = None):
        use_gpu = torch.cuda.is_available()
        if not use_gpu:
            logger.info("OCR running on CPU (no GPU detected). Will be slower.")
        self.reader = easyocr.Reader(['en'], gpu=use_gpu)

        # OCR-1: read confidence threshold from config if provided
        # falls back to 0.3 if not set
        ocr_cfg = (cfg or {}).get("ocr", {})
        self.min_confidence = float(ocr_cfg.get("min_confidence", 0.3))
        logger.info("OCR min_confidence threshold: %.2f", self.min_confidence)

    def read_plate(self, frame, bbox) -> str:
        x1, y1, x2, y2 = map(int, bbox)

        # Crop bottom 20% of vehicle bbox — plate is always near the bumper
        plate_y1  = int(y2 - (y2 - y1) * 0.35)
        plate_roi = frame[plate_y1:y2, x1:x2]

        if plate_roi.size == 0:
            return "UNKNOWN"

        # 1. Convert to Grayscale
        gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)

        # 2. Increase Contrast (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # 3. Sharpening Filter
        kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)

        # 4. Upscale (3x instead of 2x)
        h, w = sharpened.shape[:2]
        final_roi = cv2.resize(sharpened, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)

        results = self.reader.readtext(plate_roi)

        text = ""
        for res in results:
            confidence = res[2]
            if confidence > self.min_confidence:   # OCR-1: uses config value
                text += res[1].upper().strip()

        return text if text else "NOT_READABLE"