import numpy as np
import cv2

from queue import Queue, Empty
from threading import Thread

from feature_extraction.feature_extractor import FeatureExtractor
from feature_database.database import FeatureDatabase
from motion_estimation.RANSAC import RANSACMotionEstimator
from motion_estimation.keyframe_selector import KeyframeSelector
from motion_estimation.Triangulator import Triangulator
from feature_database.landmark_map import LandmarkMap


class VisualOdometryPipeline:

    def __init__(self, calibration_data, mode="mono", frame_size=(640, 480)):
        self.mode       = mode
        self.frame_size = frame_size
        self.is_initialized = False

        # ── Calibration ───────────────────────────────────────────────
        self.left_calib        = calibration_data['left']
        self.intrinsics        = self.left_calib['intrinsics']  # [fu,fv,cu,cv]
        self.distortion_coeffs = np.array(
            self.left_calib['distortion_coefficients']
        )
        self.T_BS = np.array(
            self.left_calib['T_BS']['data']
        ).reshape(4, 4)

        self.K = np.array([
            [self.intrinsics[0], 0,                  self.intrinsics[2]],
            [0,                  self.intrinsics[1], self.intrinsics[3]],
            [0,                  0,                  1],
        ], dtype=np.float64)

        # ── Modules ───────────────────────────────────────────────────
        self.extractor = FeatureExtractor(
            method="FAST+KLT", frame_size=frame_size
        )
        self.database = FeatureDatabase()

        self.ransac = RANSACMotionEstimator(
            K=self.K,
            dist_coeffs=self.distortion_coeffs,
        )

        self.keyframe_selector = KeyframeSelector(
            min_parallax_px  = 15.0,
            min_rotation_deg = 1.0,
            min_frames_gap   = 3,
            min_inliers      = 20,
        )

        self.triangulator = Triangulator(
            K=self.K,
            min_angle_deg=1.0,
        )

        self.landmark_map = LandmarkMap(sliding_window_size=10)

        # ── Threading ─────────────────────────────────────────────────
        # Main thread  → puts snapshots here
        self._estimation_queue = Queue(maxsize=2)
        # Background thread → puts results here
        self._result_queue     = Queue(maxsize=8)

        self._estimator_thread = Thread(
            target=self._estimation_loop,
            daemon=True,
        )
        self._estimator_thread.start()

        # ── Misc ──────────────────────────────────────────────────────
        self._global_track_min    = 50
        self.gridder_max_per_cell = (
            self.extractor.gridder.min_features_per_cell * 2
        )
        self._frame_idx = 0   # incremented every process_frame_mono call

    def process_frame_mono(self, cv_frame: np.ndarray, timestamp: float):
        gray_frame = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2GRAY)

        # ── Init on first frame ───────────────────────────────────────
        if not self.is_initialized:
            self._handle_initialization(gray_frame, timestamp)
            return None

        self._frame_idx += 1

        # ── Drain previous RANSAC results ─────────────────────────────
        # (outlier purging + keyframe/triangulation trigger)
        self._drain_result_queue()

        # ── Feature tracking ──────────────────────────────────────────
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
        self.landmark_map.prune_feat_ids(lost_ids)

        # ── Send snapshot to RANSAC thread ────────────────────────────
        snapshot = self._build_snapshot(timestamp)
        if not self._estimation_queue.full():
            self._estimation_queue.put_nowait(snapshot)

        # ── Grid management ───────────────────────────────────────────
        evict_ids = self.extractor.gridder.get_overcrowded_evictions(
            tracked_points=tracked_curr_pts,
            track_ids=tracked_ids,
            track_ages=self.database.ages[
                np.isin(self.database.ids, tracked_ids)
            ],
            max_features_per_cell=self.gridder_max_per_cell,
        )
        if len(evict_ids) > 0:
            self.database.purge_tracks(evict_ids)
            self.landmark_map.prune_feat_ids(evict_ids)
            keep_mask = ~np.isin(tracked_ids, evict_ids)
            tracked_curr_pts = tracked_curr_pts[keep_mask]
            tracked_ids      = tracked_ids[keep_mask]

        # ── Fill empty grid cells ─────────────────────────────────────
        new_grid_points = self.extractor.extract_features_in_empty_cells(
            gray_frame=gray_frame,
            tracked_points=tracked_curr_pts,
        )
        if len(new_grid_points) > 0:
            self.database.add_new_tracks(new_grid_points, gray_frame)

        # ── Fallback: detect if total track count is too low ──────────
        total_tracked = len(tracked_ids) + len(new_grid_points)
        if total_tracked < self._global_track_min:
            all_pts = self.database.get_active_positions()
            fallback = self.extractor.detect_new_features(
                gray_frame, existing_points=all_pts
            )
            if len(fallback) > 0:
                self.database.add_new_tracks(fallback, gray_frame)

        self.database.set_reference_frame(gray_frame)
        tracks = self.database.get_active_feature_histories()

        # ── Return state for external inspection ──────────────────────
        return {
            'frame_idx'     : self._frame_idx,
            'timestamp'     : timestamp,
            'n_tracked'     : total_tracked,
            'n_landmarks'   : self.landmark_map.num_landmarks(),
            'n_keyframes'   : self.keyframe_selector.num_keyframes(),
            'landmark_summary': self.landmark_map.summary(),
            'tracks': tracks,
            'K': self.K,
            'D': self.distortion_coeffs,
        }


    def _estimation_loop(self):
        while True:
            snapshot = self._estimation_queue.get()
            if snapshot is None:
                break   # shutdown signal

            histories  = snapshot['histories']
            frame_idx  = snapshot['frame_idx']
            timestamp  = snapshot['timestamp']

            ransac_result = self.ransac.estimate(histories)

            result_packet = {
                'frame_idx'    : frame_idx,
                'timestamp'    : timestamp,
                'ransac_result': ransac_result,   # may be None
            }

            if not self._result_queue.full():
                self._result_queue.put_nowait(result_packet)

    def _drain_result_queue(self):
        while True:
            try:
                packet = self._result_queue.get_nowait()
            except Empty:
                break

            ransac_result = packet['ransac_result']
            frame_idx     = packet['frame_idx']

            #print(ransac_result)

            # ── Purge outliers ────────────────────────────────────────
            if ransac_result is not None:
                outlier_ids = ransac_result.get('outlier_ids', np.array([]))
                if len(outlier_ids) > 0:
                    #self.database.purge_tracks(outlier_ids)
                    self.landmark_map.prune_feat_ids(outlier_ids)

            # ── Keyframe decision ─────────────────────────────────────
            is_kf = self.keyframe_selector.process(
                frame_idx=frame_idx,
                ransac_result=ransac_result,
            )

            # ── Triangulate if new keyframe ───────────────────────────
            if is_kf:
                print("New Key Frame detected")
                pair = self.keyframe_selector.get_last_two_keyframes()
                if pair is not None:
                    kf_prev, kf_curr = pair
                    tri_result = self.triangulator.triangulate(kf_prev, kf_curr)
                    if tri_result is not None:
                        self.landmark_map.add_triangulation_result(tri_result)
                        print(f"[Pipeline] Triangulated "
                              f"{len(tri_result['landmarks'])} new landmarks. "
                              f"Total: {self.landmark_map.num_landmarks()}")

            # ── Prune landmark map to active window ───────────────────
            active_frames = [
                kf['frame_idx']
                for kf in self.keyframe_selector.get_all_keyframes()
            ]
            if active_frames:
                self.landmark_map.prune_outside_window(active_frames)

    def _build_snapshot(self, timestamp: float) -> dict:
        return {
            'frame_idx' : self._frame_idx,
            'timestamp' : timestamp,
            'histories' : self.database.get_snapshot_for_estimator(),
        }

    def _handle_initialization(self, gray_frame: np.ndarray, timestamp: float):
        initial_corners = self.extractor.detect_initial_features(gray_frame)
        self.database.initialize_ledger(initial_corners, gray_frame, timestamp)
        self.is_initialized = True
        print(f"[Pipeline] Initialized at timestamp={timestamp:.3f}")



    def get_landmarks_for_ba(self):
        """{ lm_id: X(3,) } — up-to-scale."""
        return self.landmark_map.get_landmarks_for_ba()

    def get_observations_for_ba(self):
        """{ lm_id: [(frame_idx, u, v), ...] }"""
        return self.landmark_map.get_observations_for_ba()

    def get_keyframe_poses(self):
        return self.keyframe_selector.get_all_keyframes()

    def process_frame_stereo(self, cv_frame_left, cv_frame_right, timestamp):
        pass   # future implementation

    def shutdown(self):
        self._estimation_queue.put(None)
        self._estimator_thread.join(timeout=2.0)
        self.extractor.shutdown()