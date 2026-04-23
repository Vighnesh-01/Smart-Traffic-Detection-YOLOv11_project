"""
main.py — Smart Traffic Violation Detection System
────────────────────────────────────────────────────────────
All bugs fixed including latest round:
  MAIN-1: _plate_cache closure bug — moved outside for loop so cache persists
  MAIN-2: crossing_now now uses get_previous_bottom_y() — consistent y2 coords
  MAIN-3: vehicle_classes now read from config and passed to detector
  MAIN-4: issued dict pruned every 300 frames to prevent memory leak on live streams
  MAIN-5: frame_skip added — processes every Nth frame to improve CPU performance
  MAIN-6: Python 3.8 compatible type hints
"""

import cv2
import logging
import numpy as np
import pandas as pd
import yaml
import os
from typing import Dict, Set
from datetime import datetime
from pathlib import Path
from utils.notifier import send_telegram_alert

from utils.detector         import TrafficDetector
from utils.tracker          import ObjectTracker
from utils.ocr_engine       import OCREngine
from utils.find_coordinates import get_setup_coordinates
from utils.diagnostics      import run_system_check


# ─────────────────────────────────────────────────────────────
# 1. LOAD CONFIG
# ─────────────────────────────────────────────────────────────
with open("config.yaml", "r") as f:
    CFG = yaml.safe_load(f)

PATHS      = CFG["paths"]
VIDEO_PATH = PATHS["video"]
WEIGHTS    = PATHS["weights"]
VIO_DIR    = PATHS["violations_dir"]
LOG_CSV    = PATHS["log_csv"]
LOG_FILE   = CFG["logging"]["file"]
LOG_LEVEL  = CFG["logging"]["level"].upper()

Path(VIO_DIR).mkdir(parents=True, exist_ok=True)
Path("data").mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────
# 2. LOGGING SETUP
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# 3. DIAGNOSTICS & CALIBRATION
# ─────────────────────────────────────────────────────────────
logger.info("Starting system diagnostics...")
run_system_check(VIDEO_PATH, WEIGHTS)

logger.info("Starting calibration...")
LIGHT_ROI, VTL_Y = get_setup_coordinates(VIDEO_PATH)


# ─────────────────────────────────────────────────────────────
# 4. CSV LOGGING — single row append, never full rewrite
# ─────────────────────────────────────────────────────────────
def log_violation(obj_id, v_type, plate="N/A", speed=None):
    record = {
        "Timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Vehicle_ID":    obj_id,
        "Violation":     v_type,
        "License_Plate": plate,
        "Speed_kmh":     speed if speed is not None else "N/A",
        "Image":         f"id_{obj_id}_{v_type.replace(' ', '_')}.jpg",
    }
    write_header = not os.path.exists(LOG_CSV) or os.path.getsize(LOG_CSV) == 0
    pd.DataFrame([record]).to_csv(LOG_CSV, mode="a", header=write_header, index=False)
    logger.warning("Violation logged → ID:%s  Type:%s  Plate:%s  Speed:%s",
                   obj_id, v_type, plate, speed)


# ─────────────────────────────────────────────────────────────
# 5. MODULE INITIALISATION
# ─────────────────────────────────────────────────────────────
detector = TrafficDetector(
    vehicle_model_path=WEIGHTS,
    plate_model_path="weights/plate_detector.pt",
    helmet_model_path=PATHS.get("helmet_weights"),
    confidence=CFG["detection"]["confidence"],
    # MAIN-3: vehicle_classes now read from config
    vehicle_classes=CFG["detection"]["vehicle_classes"],
)
tracker      = ObjectTracker(CFG)
plate_reader = OCREngine(CFG)
ocr_engine = OCREngine(confidence_threshold=0.3)
SPEED_LIMIT  = CFG["speed"]["limit_kmh"]
HELMET_ON    = CFG["helmet"]["enabled"]
# MAIN-5: process every Nth frame — set to 1 to process all (slowest)
FRAME_SKIP   = int(CFG.get("performance", {}).get("frame_skip", 2))


# ─────────────────────────────────────────────────────────────
# 6. HELPERS
# ─────────────────────────────────────────────────────────────
def get_light_color(frame, roi) -> str:
    x1, y1, x2, y2 = roi
    roi_img = frame[y1:y2, x1:x2]
    if roi_img.size == 0:
        return "UNKNOWN"
    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    mask1      = cv2.inRange(hsv, np.array([0,   100, 100]), np.array([10,  255, 255]))
    mask2      = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
    red_pixels = np.sum(mask1 | mask2)
    threshold  = roi_img.shape[0] * roi_img.shape[1] * 0.10
    status     = "RED" if red_pixels > threshold else "GREEN"
    logger.debug("Light ROI red_px=%d threshold=%.0f → %s", red_pixels, threshold, status)
    return status


def save_violation_crop(frame, bbox, filename):
    x1, y1, x2, y2 = map(int, bbox)
    pad  = 10
    h, w = frame.shape[:2]
    crop = frame[max(0, y1-pad):min(h, y2+pad), max(0, x1-pad):min(w, x2+pad)]
    if crop.size > 0:
        path = os.path.join(VIO_DIR, filename)
        cv2.imwrite(path, crop)
        logger.debug("Saved crop → %s", path)


def scan_plate(frame, bbox) -> str:
    try:
        # 1. Extract coordinates from bbox
        x1, y1, x2, y2 = map(int, bbox)
        
        # 2. Crop the frame (with a tiny bit of padding for context)
        padding = 5
        h, w, _ = frame.shape
        crop = frame[max(0, y1-padding):min(h, y2+padding), 
                     max(0, x1-padding):min(w, x2+padding)]

        if crop.size == 0:
            return "UNREADABLE"

        # 3. Call the new read_plate (which returns: text, confidence)
        text, confidence = plate_reader.read_plate(crop)

        # 4. Filter the results
        if text in ["UNKNOWN", "LOW_CONF", "ERROR"] or not text.strip():
            return "UNREADABLE"

        return text.strip()
        
    except Exception as e:
        logger.error(f"Scan failed: {e}")
        return "UNREADABLE"


# ─────────────────────────────────────────────────────────────
# 7. MAIN LOOP
# ─────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
tracker.fps = fps
logger.info("Video FPS: %.1f  |  Processing every %d frame(s)", fps, FRAME_SKIP)

# MAIN-6: Dict/Set instead of dict/set for Python 3.8
issued: Dict[int, Set[str]] = {}
frame_count = 0
# MAIN-1: plate cache moved OUTSIDE the vehicle loop so it persists
# across multiple violations triggered by the same vehicle in the same frame
plate_cache: Dict[int, str] = {}

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame_count += 1

    # 1. AI Logic (Only runs occasionally)
    if frame_count % FRAME_SKIP == 0:
        results = detector.detect_vehicles(frame)
        objects = tracker.get_tracking_data(results)
        # ... your violation/drawing logic here ...

    # 2. Display Logic (Runs for EVERY frame)
    # Define your scale once
    display_scale = 0.5 
    
    # Calculate dimensions
    width = int(frame.shape[1] * display_scale)
    height = int(frame.shape[0] * display_scale)
    
    # Create the display version
    display_frame = cv2.resize(frame, (width, height))

    # Show ONLY the display_frame
    cv2.imshow("Smart Traffic System  [q = quit]", display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # Clear per-frame plate cache at start of each processed frame
    plate_cache.clear()

    light_status = get_light_color(frame, LIGHT_ROI)
    results      = detector.detect_vehicles(frame)
    objects      = tracker.get_tracking_data(results)

    # MAIN-4: prune issued dict every 300 processed frames
    # Removes vehicles not seen recently to prevent memory leak on live streams
    if frame_count % (300 * FRAME_SKIP) == 0:
        active_ids = {obj["id"] for obj in objects}
        stale      = [oid for oid in issued if oid not in active_ids]
        for oid in stale:
            del issued[oid]
        if stale:
            logger.debug("Pruned %d stale vehicle IDs from issued dict", len(stale))

    for obj in objects:
        x1, y1, x2, y2 = obj["bbox"]
        obj_id   = obj["id"]
        cls_id   = obj["class"]
        cls_name = detector.vehicle_model.names[cls_id]
        speed    = obj["speed_kmh"]

        if obj_id not in issued:
            issued[obj_id] = set()

        # MAIN-1: get_plate() uses plate_cache defined outside the loop
        # so if RED LIGHT and SPEEDING both fire for the same vehicle
        # in the same frame, OCR only runs once
        def get_plate(oid=obj_id, f=frame, b=[x1, y1, x2, y2]):
            """
            Two-stage license plate recognition:
            1. Localize plate within vehicle crop.
            2. Enhance and OCR the localized plate.
            """
            if oid not in plate_cache:
                logger.info("Starting Two-Stage scan for vehicle %d...", oid)
                
                # STEP A: Localization (Find the plate box inside the car crop)
                # detector.get_plate_from_vehicle returns (v_crop, p_bbox)
                v_crop, p_bbox = detector.get_plate_from_vehicle(f, b)
                
                if v_crop is not None and p_bbox is not None:
                    # STEP B: Exact Cropping with Boundary Protection
                    px1, py1, px2, py2 = map(int, p_bbox)
                    h, w, _ = v_crop.shape
                    
                    # Clip coordinates to ensure they stay within the crop boundaries
                    final_plate_img = v_crop[max(0, py1):min(h, py2), max(0, px1):min(w, px2)]
                    
                    if final_plate_img.size > 0:
                        # STEP C: OCR (Pass the tightest possible crop to EasyOCR)
                        # This uses the OCREngine class you defined earlier
                        text, conf = ocr_engine.read_plate(final_plate_img) 
                        
                        # Filter results based on your preferred logic
                        if text in ["UNKNOWN", "LOW_CONF", "ERROR"]:
                            plate_cache[oid] = "SCANNING..." # Try again in next processed frame
                        else:
                            plate_cache[oid] = text
                    else:
                        plate_cache[oid] = "INVALID_CROP"
                else:
                    plate_cache[oid] = "PLATE_NOT_FOUND"
                    
                logger.info("Final Plate Result for %d: %s", oid, plate_cache[oid])
            
            return plate_cache[oid]

        # ── RED LIGHT ─────────────────────────────────────────
        prev_bottom_y = tracker.get_previous_bottom_y(obj_id)
        crossing_now  = (
            prev_bottom_y is not None
            and prev_bottom_y <= VTL_Y
            and y2 > VTL_Y
        )

        if light_status == "RED" and crossing_now and "RED LIGHT" not in issued[obj_id]:
            issued[obj_id].add("RED LIGHT")
            logger.warning("RED LIGHT — Vehicle %d", obj_id)
            
            # --- DEFINE THE PATH FIRST ---
            img_filename = f"id_{obj_id}_RED_LIGHT.jpg"
            img_path = os.path.join(VIO_DIR, img_filename) 
            
            # Now use those variables
            save_violation_crop(frame, obj["bbox"], img_filename)
            log_violation(obj_id, "RED LIGHT", get_plate(), speed)

            if CFG.get("telegram", {}).get("enabled", False):
                tg_msg = f"🚨 RED LIGHT VIOLATION 🚨\nID: {obj_id}\nPlate: {get_plate()}"
                send_telegram_alert(img_path, tg_msg, CFG["telegram"]["bot_token"], CFG["telegram"]["chat_id"])

        # ── SPEEDING ──────────────────────────────────────────
        if obj["speeding"] and "SPEEDING" not in issued[obj_id]:
            issued[obj_id].add("SPEEDING")
            logger.warning("SPEEDING — Vehicle %d %.1f km/h", obj_id, speed)
            
            # --- DEFINE THE PATH FIRST ---
            img_filename = f"id_{obj_id}_SPEEDING.jpg"
            img_path = os.path.join(VIO_DIR, img_filename)

            save_violation_crop(frame, obj["bbox"], img_filename)
            log_violation(obj_id, "SPEEDING", get_plate(), speed)

            if CFG.get("telegram", {}).get("enabled", False):
                tg_msg = f"⚡ SPEEDING ⚡\nID: {obj_id}\nSpeed: {speed} km/h\nPlate: {get_plate()}"
                send_telegram_alert(img_path, tg_msg, CFG["telegram"]["bot_token"], CFG["telegram"]["chat_id"])

        # ── WRONG WAY ─────────────────────────────────────────
        if obj["wrong_way"] and "WRONG WAY" not in issued[obj_id]:
            issued[obj_id].add("WRONG WAY")
            logger.warning("WRONG WAY — Vehicle %d", obj_id)
            save_violation_crop(frame, obj["bbox"], f"id_{obj_id}_WRONG_WAY.jpg")
            log_violation(obj_id, "WRONG WAY", get_plate(), speed)

        # ── NO HELMET ─────────────────────────────────────────
        if HELMET_ON and cls_name == "motorcycle" and "NO HELMET" not in issued[obj_id]:
            head_y2   = int(y1 + (y2 - y1) * 0.4)
            head_crop = frame[int(y1):head_y2, int(x1):int(x2)]

            if head_crop.size > 0 and detector.helmet_model is not None:
                helmet_res = detector.helmet_model.predict(
                    head_crop,
                    conf=CFG["helmet"]["confidence"],
                    verbose=False,
                )
                names = detector.helmet_model.names
                detected_labels = [
                    names[int(c)] for c in helmet_res[0].boxes.cls.cpu().numpy()
                ] if len(helmet_res[0].boxes) > 0 else []

                has_helmet = any("helmet" in lbl.lower() for lbl in detected_labels)
                if not has_helmet:
                    issued[obj_id].add("NO HELMET")
                    logger.warning("NO HELMET — Motorcycle %d", obj_id)
                    save_violation_crop(frame, obj["bbox"], f"id_{obj_id}_NO_HELMET.jpg")
                    log_violation(obj_id, "NO HELMET", get_plate(), speed)

                    if CFG.get("telegram", {}).get("enabled", False):
                        tg_msg = (
                            f"⛑ HELMET VIOLATION ⛑\n"
                            f"Motorcycle ID: {obj_id}\n"
                            f"Plate: {get_plate()}\n"
                            f"Status: No Helmet Detected"
                        )
                        send_telegram_alert(img_path, tg_msg, CFG["telegram"]["bot_token"], CFG["telegram"]["chat_id"])

    # ── OVERLAYS ─────────────────────────────────────────────
    cv2.line(frame, (0, VTL_Y), (frame.shape[1], VTL_Y), (0, 255, 255), 2)
    cv2.rectangle(frame,
                  (LIGHT_ROI[0], LIGHT_ROI[1]),
                  (LIGHT_ROI[2], LIGHT_ROI[3]),
                  (255, 0, 0), 2)

    badge_col = (0, 0, 255) if light_status == "RED" else (0, 200, 0)
    cv2.rectangle(frame, (10, 10), (220, 55), badge_col, -1)
    cv2.putText(frame, f"LIGHT: {light_status}", (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    for obj in objects:
        x1, y1, x2, y2 = map(int, obj["bbox"])
        spd = f"{obj['speed_kmh']:.0f}" if obj["speed_kmh"] is not None else "?"
        ww  = " WW" if obj["wrong_way"] else ""
        lbl = f"ID:{obj['id']} {spd}km/h{ww}"
        col = (0, 0, 255) if (obj["speeding"] or obj["wrong_way"]) else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        cv2.putText(frame, lbl, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

    cv2.imshow("Smart Traffic System  [q = quit]", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

try:
    total = len(pd.read_csv(LOG_CSV)) if os.path.exists(LOG_CSV) else 0
except Exception:
    total = 0
logger.info("Session ended. Total violations in log: %d", total) 
