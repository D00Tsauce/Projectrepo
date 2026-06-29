import cv2
from ultralytics import YOLO

class ADASObjectDetector:
    def __init__(self, model_path="yolov8n.pt"):
        """
        Initializes the YOLOv8 Object Detector.
        Automatically downloads weights if not present.
        """
        # Load pre-trained nano model (lightweight, runs fast on CPUs)
        self.model = YOLO(model_path)
        
        # COCO dataset classes related to ADAS/Driving environments
        self.target_classes = {
            0: "Pedestrian",
            1: "Bicycle",
            2: "Car",
            3: "Motorcycle",
            5: "Bus",
            7: "Truck",
            9: "Traffic Light",
            11: "Stop Sign"
        }

    def detect_obstacles(self, frame):
        """
        Processes a single video frame to detect obstacles.
        """
        # Run inference (conf=0.35 ignores weak, low-confidence predictions)
        results = self.model(frame, conf=0.35, verbose=False)[0]
        
        # Loop over all detected items
        for box in results.boxes:
            class_id = int(box.cls[0])
            
            # Filter: Only annotate items relevant to driving safety
            if class_id in self.target_classes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                label = f"{self.target_classes[class_id]} {confidence:.2f}"
                
                # Draw a sleek neon bounding box around the object
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                
                # Add a semi-transparent text background block
                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
        return frame