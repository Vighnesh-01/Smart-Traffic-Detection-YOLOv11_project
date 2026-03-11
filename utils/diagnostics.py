"""
utils/diagnostics.py
────────────────────────────────────────────────────────────
FIX #7: No longer loads the YOLO model — that caused it to be
         loaded TWICE at startup (once here, once in TrafficDetector).
         Now just checks the file exists and is non-zero size.
"""

import torch
import cv2
import os
import logging

logger = logging.getLogger(__name__)


def run_system_check(video_path: str, weights_path: str):
    logger.info("--- SYSTEM DIAGNOSTICS ---")

    # 1. Weights check — existence + size only, no YOLO load (FIX #7)
    if os.path.exists(weights_path):
        size_mb = os.path.getsize(weights_path) / (1024 * 1024)
        if size_mb < 0.1:
            logger.error("Weights file exists but looks empty (%.2f MB): %s", size_mb, weights_path)
        else:
            logger.info("Weights found: %s  (%.1f MB)", weights_path, size_mb)
    else:
        logger.error("Weights MISSING at: %s", weights_path)

    # 2. GPU check
    if torch.cuda.is_available():
        logger.info("GPU detected: %s", torch.cuda.get_device_name(0))
    else:
        logger.warning("No GPU found — running on CPU (detection will be slow).")

    # 3. Video check
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        fps    = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info("Video OK: %s  |  %dx%d  %.1f fps  ~%.0f frames", video_path, w, h, fps, frames)
        cap.release()
    else:
        logger.error("Video not found or format not supported: %s", video_path)

    logger.info("--- DIAGNOSTICS COMPLETE ---")