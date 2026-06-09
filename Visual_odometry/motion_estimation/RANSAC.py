# motion_estimation/estimator.py
import cv2
import numpy as np

class MotionEstimator:
    def __init__(self, K, ransac_threshold=1.5, ransac_confidence=0.999):
        self.K = K
        self.threshold = ransac_threshold
        self.confidence = ransac_confidence

    def estimate_motion(self, prev_pts, curr_pts):
        if len(prev_pts) < 5:  # Essential matrix calculation requires at least 5 points
            return np.eye(3), np.zeros((3, 1)), np.zeros(len(prev_pts), dtype=bool)

        # 1. RUN RANSAC via the Essential Matrix
        # This function isolates outliers using your K matrix and point coordinates
        E, mask = cv2.findEssentialMat(
            points1=prev_pts,
            points2=curr_pts,
            cameraMatrix=self.K,
            method=cv2.RANSAC,
            prob=self.confidence,
            threshold=self.threshold
        )
        
        if E is None or E.shape != (3, 3):
            return np.eye(3), np.zeros((3, 1)), np.zeros(len(prev_pts), dtype=bool)

        # Flatten mask into a 1D boolean array (1 = Inlier, 0 = Outlier)
        inlier_mask = mask.flatten() == 1

        # 2. RECOVER ROTATION AND TRANSLATION
        # Pass E and the RANSAC inlier mask to decipher the actual direction of camera movement
        _, R, t, _ = cv2.recoverPose(E, prev_pts, curr_pts, self.K, mask=mask)

        return R, t, inlier_mask