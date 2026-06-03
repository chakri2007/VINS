import cv2
import numpy as np

class FastDetector:
    def __init__(self, threshold=20, nonmax_suppression=True):
        self.fast = cv2.FastFeatureDetector_create(
            threshold=threshold, 
            nonmaxSuppression=nonmax_suppression
        )

    def detect(self, gray_frame, mask=None):
        keypoints = self.fast.detect(gray_frame, mask=mask)
        
        # Convert OpenCV Keypoint objects to clean Nx2 NumPy coordinates
        if len(keypoints) == 0:
            return np.empty((0, 2), dtype=np.float32)
            
        return np.array([kp.pt for kp in keypoints], dtype=np.float32)