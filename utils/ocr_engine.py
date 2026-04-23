import cv2
import easyocr
import numpy as np
import logging

class OCREngine:
    def __init__(self, confidence_threshold=0.3):
        # Initialize EasyOCR to only look for English/Numbers
        # Using gpu=False since your logs show you are on CPU
        self.reader = easyocr.Reader(['en'], gpu=False)
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)

    def preprocess_image(self, image):
        """Enhances the license plate image for better OCR accuracy."""
        if image is None:
            return None

        # 1. Convert to Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 2. Upscale the image (OCR works better on larger text)
        # We multiply size by 2x using Cubic Interpolation
        upscaled = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

        # 3. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # This fixes plates that are too dark or have shadows
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(upscaled)

        # 4. Blur to remove 'salt and pepper' noise
        final_image = cv2.medianBlur(contrast_enhanced, 3)

        return final_image

    def read_plate(self, plate_crop):
        """Processes the crop and returns the detected text."""
        try:
            processed_img = self.preprocess_image(plate_crop)
            
            # Read text with an allowlist (ignores symbols/random noise)
            results = self.reader.readtext(
                processed_img, 
                detail=1, 
                paragraph=False,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            )

            if not results:
                return "UNKNOWN", 0.0

            # Get the result with the highest confidence
            # results format: [([[coords]], text, confidence), ...]
            best_match = max(results, key=lambda x: x[2])
            text = best_match[1].replace(" ", "").upper()
            conf = best_match[2]

            if conf >= self.confidence_threshold:
                return text, conf
            
            return "LOW_CONF", conf

        except Exception as e:
            self.logger.error(f"OCR Error: {e}")
            return "ERROR", 0.0