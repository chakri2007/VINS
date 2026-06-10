import numpy as np
import cv2

from queue import Queue
from threading import Thread

from feature_extraction.feature_extractor import FeatureExtractor
from feature_database.database import FeatureDatabase
from motion_estimation.RANSAC import RANSACMotionEstimator

class VisualOdometryPipeline:
    def __init__(self, calibration_data, mode="mono", frame_size=(640, 480)):
        self.mode      = mode
        self.frame_size = frame_size   # (width, height) — must match incoming frames
        self.is_initialized = False

        # ---- Calibration ----
        self.left_calib        = calibration_data['left']
        self.intrinsics        = self.left_calib['intrinsics']  # [fu, fv, cu, cv]
        self.distortion_coeffs = np.array(self.left_calib['distortion_coefficients'])
        self.T_BS              = np.array(self.left_calib['T_BS']['data']).reshape(4, 4)

        self.K = np.array([
            [self.intrinsics[0], 0,                  self.intrinsics[2]],
            [0,                  self.intrinsics[1], self.intrinsics[3]],
            [0,                  0,                  1],
        ], dtype=np.float32)
        
        self.extractor = FeatureExtractor(method="FAST+KLT", frame_size=frame_size)
        self.database  = FeatureDatabase()
        self.estimator = None  # motion estimator wired up later

        self.current_pose = np.eye(4, dtype=np.float32)  # T_world_body

        self._global_track_min = 50
        self.gridder_max_per_cell = self.extractor.gridder.min_features_per_cell * 2

        self._estimation_queue = Queue(maxsize=2)  # bounded — drop if estimator falls behind
        self._estimator_thread = Thread(
            target=self._estimation_loop,
            daemon=True
        )
        self._estimator_thread.start()

        self.ransac = RANSACMotionEstimator(self.K, self.distortion_coeffs)

    def process_frame_mono(self, cv_frame: np.ndarray, timestamp: float):
        gray_frame = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2GRAY)

        if not self.is_initialized:
            self.handle_initialization(gray_frame, timestamp)
            print("System not initialised yet.")
            return None
        
        self._drain_result_queue()

        prev_frame, prev_points, prev_ids = self.database.get_active_tracks()

        curr_points, status = self.extractor.track_features(
            prev_frame=prev_frame,
            curr_frame=gray_frame,
            prev_points=prev_points,
        )

        valid_indices = np.where(status == 1)[0]
        lost_indices  = np.where(status == 0)[0]

        tracked_ids      = prev_ids[valid_indices]
        tracked_prev_pts = prev_points[valid_indices]
        tracked_curr_pts = curr_points[valid_indices]
        lost_ids         = prev_ids[lost_indices]

        self.database.update_active_positions(tracked_ids, tracked_curr_pts)
        self.database.purge_tracks(lost_ids)

        #Estimation - input
        snapshot = self.build_estimation_snapshot(timestamp)

        if not self._estimation_queue.full():
            self._estimation_queue.put_nowait(snapshot)

        evict_ids = self.extractor.gridder.get_overcrowded_evictions(
        tracked_points=tracked_curr_pts,
        track_ids=tracked_ids,
        track_ages=self.database.ages[
                np.isin(self.database.ids, tracked_ids)
            ],
            max_features_per_cell=self.gridder_max_per_cell,   # set to 2 * min_features_per_cell
        )
        if len(evict_ids) > 0:
            self.database.purge_tracks(evict_ids)
            # recompute tracked_curr_pts for the vacancy check below
            keep_mask = ~np.isin(tracked_ids, evict_ids)
            tracked_curr_pts = tracked_curr_pts[keep_mask]
            tracked_ids = tracked_ids[keep_mask]

        new_grid_points = self.extractor.extract_features_in_empty_cells(
            gray_frame=gray_frame,
            tracked_points=tracked_curr_pts,
        )

        if len(new_grid_points) > 0:
            self.database.add_new_tracks(new_grid_points, gray_frame)

        total_tracked = len(tracked_ids) + len(new_grid_points)
        if total_tracked < self._global_track_min:
            all_current_pts = self.database.get_active_positions()  # includes new additions
            fallback_points = self.extractor.detect_new_features(
                gray_frame,
                existing_points=all_current_pts,
            )
            if len(fallback_points) > 0:
                self.database.add_new_tracks(fallback_points, gray_frame)

        # ----------------------------------------------------------
        # Geometry / Motion Estimation (wired up later)
        # ----------------------------------------------------------
        # self.estimate_trajectory_motion(tracked_prev_pts, tracked_curr_pts)

        self.database.set_reference_frame(gray_frame)

        active_feature_history_map = self.database.get_active_feature_histories()
        return active_feature_history_map, self.K, self.distortion_coeffs

    def _estimation_loop(self):
        while True:
            snapshot = self._estimation_queue.get()
            if snapshot is None:
                break
            self.ransac_estimation(snapshot)
    
    def _run_ransac_and_estimate(self, snapshot):
        result = self.ransac.estimate(snapshot['histories'])

        if result is None:
            return

        if len(result['outlier_ids']) > 0 and not self._result_queue.full():
            self._result_queue.put_nowait({
                'outlier_ids': result['outlier_ids'],
            })

        # Pose chaining — wire up when ready
        R, t = result['R'], result['t']
        # self._update_pose(R, t)
        print(f"Estimated pose update: R=\n{R}\nt={t.ravel()}\n")
    
    def _drain_result_queue(self):
        while True:
            try:
                result = self._result_queue.get_nowait()
                self.database.purge_tracks(result['outlier_ids'])
            except Empty:
                break
    
    def _build_estimation_snapshot(self, timestamp):
        return {
            'timestamp': timestamp,
            'histories': self.database.get_snapshot_for_estimator(),
        }


    def process_frame_stereo(self, cv_frame_left, cv_frame_right, timestamp):
        pass

    def handle_initialization(self, gray_frame: np.ndarray, timestamp: float):
        initial_corners = self.extractor.detect_initial_features(gray_frame)
        self.database.initialize_ledger(initial_corners, gray_frame, timestamp)
        self.is_initialized = True
        print(f"VO Pipeline initialised successfully at timestamp: {timestamp}")

    def shutdown(self):
        self._estimation_queue.put(None)   # signal estimator thread to exit
        self._estimator_thread.join(timeout=2.0)
        self.extractor.shutdown()