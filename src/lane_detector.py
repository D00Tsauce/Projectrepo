import cv2
import numpy as np

class ADASLaneDetector:
    def __init__(self):
        pass

    def _region_of_interest(self, img):
        """
        Applies a triangular/trapezoidal mask to isolate only the road lane area.
        This represents DSP spatial filtering.
        """
        height, width = img.shape[:2]
        
        # Define a trapezoid covering the lower half of the view where lanes exist
        polygons = np.array([[
            (int(width * 0.1), height),
            (int(width * 0.45), int(height * 0.6)),
            (int(width * 0.55), int(height * 0.6)),
            (int(width * 0.9), height)
        ]], np.int32)
        
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, polygons, 255)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def detect_lanes(self, frame):
        """
        ECE DSP Pipeline: Grayscale -> Gaussian Blur -> Canny Edge Detection -> ROI -> Hough Transform
        """
        # 1. Convert to Grayscale (Dimensionality reduction)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Gaussian Blur (Noise filtering / Low-pass filter)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 3. Canny Edge Detection (High-pass spatial filtering for gradients)
        edges = cv2.Canny(blur, 50, 150)
        
        # 4. Isolate driving lane region
        cropped_edges = self._region_of_interest(edges)
        
        # 5. Hough Line Transform (Feature extraction to map edges into vector lines)
        lines = cv2.HoughLinesP(cropped_edges, 1, np.pi/180, 50, 
                                minLineLength=40, maxLineGap=100)
        
        # Create an overlay image to draw the detected lane markers safely
        lane_overlay = np.zeros_like(frame)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Draw bright neon green lane lines
                cv2.line(lane_overlay, (x1, y1), (x2, y2), (0, 255, 0), 5)
        
        # Merge the original video frame with our green lane marker overlay
        combined_frame = cv2.addWeighted(frame, 0.8, lane_overlay, 1.0, 0.0)
        return combined_frame