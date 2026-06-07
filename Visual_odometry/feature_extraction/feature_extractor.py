# feature_extractor.py
import numpy as np
import cv2

from .fast_detector import FastDetector
from .klt_tracker import KltTracker
from .orb_tracker import OrbTracker

class FeatureExtractor:
    def __init__(self, method="FAST+KLT"):
        self.method = method
    
        if "FAST" in self.method:
            self.fast_detector = FastDetector()
        if "KLT" in self.method:
            self.klt_tracker = KltTracker(max_level=3, win_size=(21, 21))
        if "ORB" in self.method:
            self.orb_tracker = OrbTracker()
            self.prev_orb_descriptors = None # ORB needs to store its last descriptors

    def detect_initial_features(self, gray_frame):
        if "FAST" in self.method:
            return self.fast_detector.detect(gray_frame)
        elif "ORB" in self.method:
            pts, descs = self.orb_tracker.detect_and_compute(gray_frame)
            self.prev_orb_descriptors = descs
            return pts

    def track_features(self, prev_frame, curr_frame, prev_points):
        if self.method == "FAST+KLT":
            return self.klt_tracker.track(prev_frame, curr_frame, prev_points)
            
        elif self.method == "ORB":
            curr_points, curr_descriptors = self.orb_tracker.detect_and_compute(curr_frame)
        
            matches = self.orb_tracker.match_frames(self.prev_orb_descriptors, curr_descriptors)
            
            status = np.zeros(len(prev_points), dtype=np.uint8)
            output_curr_points = np.zeros_like(prev_points)
        
            for match in matches:
                prev_idx = match.queryIdx
                curr_idx = match.trainIdx
               
                if prev_idx < len(prev_points):
                    output_curr_points[prev_idx] = curr_points[curr_idx]
                    status[prev_idx] = 1
                    
            self.prev_orb_descriptors = curr_descriptors
            return output_curr_points, status

    def detect_new_features(self, gray_frame, existing_points):
        mask = np.ones(gray_frame.shape, dtype=np.uint8) * 255

        for pt in existing_points:
            cv2.circle(mask, (int(pt[0]), int(pt[1])), radius=10, color=0, thickness=-1)
            
        if "FAST" in self.method:
            return self.fast_detector.detect(gray_frame, mask=mask)
        elif "ORB" in self.method:
            pts, _ = self.orb_tracker.detect_and_compute(gray_frame, mask=mask)
            return pts