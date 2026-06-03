# klt_tracker.py
import cv2
import numpy as np

class KltTracker:
    def __init__(self):
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

    def track(self, prev_frame, curr_frame, prev_points):
        if len(prev_points) == 0:
            return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.uint8)
        curr_points, status, err = cv2.calcOpticalFlowPyrLK(
            prev_frame, curr_frame, prev_points, None, **self.lk_params
        )
        return curr_points, status.flatten()
    