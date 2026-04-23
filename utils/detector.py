import cv2
from typing import Optional, List
from ultralytics import YOLO

class TrafficDetector:
    def __init__(self,
                 vehicle_model_path: str = "weights/yolo11n.pt",
                 plate_model_path:   str = "weights/plate_detector.pt", # Added this
                 helmet_model_path:  Optional[str] = None,
                 confidence:         float = 0.4,
                 vehicle_classes:    Optional[List[int]] = None):

        # Model 1: General Vehicles (with Tracking)
        self.vehicle_model = YOLO(vehicle_model_path)
        
        # Model 2: Specialized License Plates (The one you just trained!)
        self.plate_model = YOLO(plate_model_path)
        
        self.confidence = confidence
        self.vehicle_classes = vehicle_classes or [2, 3, 5, 7]

        self.helmet_model = None
        if helmet_model_path:
            self.helmet_model = YOLO(helmet_model_path)

    def detect_vehicles(self, frame):
        """
        Runs YOLOv11 with ByteTrack to maintain vehicle IDs.
        """
        results = self.vehicle_model.track(
            frame,
            conf=self.confidence,
            classes=self.vehicle_classes,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )
        return results

    def get_plate_from_vehicle(self, frame, vehicle_bbox):
        """
        STAGE 2: Finds the plate inside a vehicle's crop.
        """
        # 1. Extract coordinates
        vx1, vy1, vx2, vy2 = map(int, vehicle_bbox)
        
        # 2. Crop the vehicle
        h, w, _ = frame.shape
        vehicle_crop = frame[max(0, vy1):min(h, vy2), max(0, vx1):min(w, vx2)]
        
        if vehicle_crop.size == 0:
            return None, None

        # 3. Use the plate_model on the crop
        # We use 'self.plate_model' which was loaded in __init__
        plate_results = self.plate_model(vehicle_crop, conf=0.4, verbose=False)  
        
        for res in plate_results:
            if len(res.boxes) > 0:
                # Get the highest confidence plate bounding box
                p_bbox = res.boxes[0].xyxy[0].cpu().numpy()
                return vehicle_crop, p_bbox
                
        return None, None
    
