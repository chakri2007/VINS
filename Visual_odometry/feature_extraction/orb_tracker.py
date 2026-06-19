# orb_tracker.py
import cv2
import numpy as np

class OrbTracker:
    def __init__(self, num_features=1000):
        self.orb = cv2.ORB_create(nfeatures=num_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def detect_and_compute(self, gray_frame, mask=None):
        keypoints, descriptors = self.orb.detectAndCompute(gray_frame, mask)
        points = np.array([kp.pt for kp in keypoints], dtype=np.float32) if keypoints else np.empty((0,2))
        return points, descriptors

    def match_frames(self, prev_descriptors, curr_descriptors):
        if prev_descriptors is None or curr_descriptors is None:
            return []
        matches = self.matcher.match(prev_descriptors, curr_descriptors)
        return sorted(matches, key=lambda x: x.distance)