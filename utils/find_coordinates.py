"""
utils/find_coordinates.py
────────────────────────────────────────────────────────────
Fixes applied:
  • COORD-1: Escape key / 'r' key to quit or reset — no longer stuck on misclick
  • COORD-2: ROI coordinates auto-sorted (min/max) so inverted clicks still work
  • COORD-3: Uses logger instead of print() so messages appear in system.log
"""

import cv2
import logging

logger = logging.getLogger(__name__)

DEFAULT_ROI  = [100, 50, 200, 250]
DEFAULT_VTL  = 450


def get_setup_coordinates(video_path: str):
    """
    Opens the first frame and lets the user click 3 points:
      Click 1 — TOP-LEFT of the traffic light
      Click 2 — BOTTOM-RIGHT of the traffic light  (order doesn't matter — COORD-2)
      Click 3 — anywhere on the STOP LINE

    Keys:
      R — reset all clicks and start over           (COORD-1)
      ESC / Q — quit and use default coordinates    (COORD-1)

    Window closes automatically after the 3rd click.
    Returns: light_roi (list), vtl_y (int)
    """
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()

    if not success:
        logger.error("Could not open video for calibration. Using defaults.")
        return DEFAULT_ROI, DEFAULT_VTL

    clicks = []
    WINDOW = "Calibration"

    # Callback ONLY records the click — all drawing happens in the main loop
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 3:
            clicks.append((x, y))
            logger.info("Calibration click %d recorded at (%d, %d)", len(clicks), x, y)

    cv2.namedWindow(WINDOW,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 720)
    cv2.setMouseCallback(WINDOW, click_event)

    instructions = [
        "Click 1/3: TOP-LEFT of Traffic Light  |  R=reset  ESC=use defaults",
        "Click 2/3: BOTTOM-RIGHT of Traffic Light  |  R=reset  ESC=use defaults",
        "Click 3/3: Anywhere on the STOP LINE  |  R=reset  ESC=use defaults",
        "Done! Closing...",
    ]

    logger.info("Calibration started — click on the VIDEO WINDOW (not terminal)")
    logger.info("Keys: R=reset  ESC/Q=quit with defaults")

    aborted = False

    while True:
        display = frame
        step    = len(clicks)

        # Draw recorded clicks
        for i, (cx, cy) in enumerate(clicks):
            cv2.circle(display, (cx, cy), 7, (0, 255, 0), -1)
            cv2.putText(display, str(i + 1), (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Draw ROI box after 2nd click
        if len(clicks) >= 2:
            cv2.rectangle(display, clicks[0], clicks[1], (255, 120, 0), 2)

        # Draw trip line after 3rd click
        if len(clicks) == 3:
            cv2.line(display, (0, clicks[2][1]), (display.shape[1], clicks[2][1]),
                     (0, 255, 255), 2)

        # Instruction bar
        cv2.rectangle(display, (0, 0), (display.shape[1], 45), (30, 30, 30), -1)
        cv2.putText(display, instructions[step], (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 180), 2)

        cv2.imshow(WINDOW, display)

        # Auto-close after 3rd click
        if len(clicks) == 3:
            cv2.waitKey(800)
            break

        key = cv2.waitKey(30) & 0xFF

        # COORD-1: R = reset all clicks
        if key == ord('r') or key == ord('R'):
            clicks.clear()
            logger.info("Calibration reset — start clicking again")

        # COORD-1: ESC or Q = abort and use defaults
        elif key == 27 or key == ord('q') or key == ord('Q'):
            logger.warning("Calibration aborted — using default coordinates")
            aborted = True
            break

    cv2.destroyAllWindows()

    if aborted or len(clicks) < 3:
        logger.warning("Using default ROI=%s  VTL_Y=%d", DEFAULT_ROI, DEFAULT_VTL)
        return DEFAULT_ROI, DEFAULT_VTL

    # COORD-2: sort coordinates so click order doesn't matter
    # If user clicked bottom-right first, frame[y2:y1] would be empty without this
    xs = sorted([clicks[0][0], clicks[1][0]])
    ys = sorted([clicks[0][1], clicks[1][1]])
    light_roi = [xs[0], ys[0], xs[1], ys[1]]   # [x1, y1, x2, y2] always correct
    vtl_y     = clicks[2][1]

    logger.info("Calibration complete — Light ROI: %s  Trip Line Y: %d", light_roi, vtl_y)
    return light_roi, vtl_y