import cv2
import numpy as np
from .ssc import ssc

class FastDetector:
    def __init__(self,target_num_features = 200, num_feature_tolerance = 0.1):
        self.fast = cv2.FastFeatureDetector_create( 
                        threshold=25,
                        nonmaxSuppression=True
                        )
        self.taget_num_features = target_num_features
        self.num_features_tolerance = num_feature_tolerance
    def detect(self, gray_frame, mask=None):

        img_rows, img_cols = gray_frame.shape[:2]
        keypoints = sorted(self.fast.detect(gray_frame, mask=mask), key=lambda x: x.response, reverse=True)
        if len(keypoints) == 0:
            return np.empty((0, 2), dtype=np.float32)
            
        return np.array([kp.pt for kp in keypoints], dtype=np.float32)