import numpy as np
import cv2

from queue import Queue, Empty
from threading import Thread
from typing import Tuple, Optional

from feature_extraction.feature_extractor import FeatureExtractor
from feature_database.database import FeatureDatabase
from motion_estimation.RANSAC import RANSACMotionEstimator
from motion_estimation.keyframe_selector import KeyframeSelector
from motion_estimation.Triangulator import Triangulator
from motion_estimation.pnp_estimator import PnPEstimator
from feature_database.landmark_map import LandmarkMap
from motion_estimation.bundle_adjustment.bundle_adjustment import BundleAdjustment

from motion_estimation.motion_estimation import MotionEstimator, Pose


class VisualOdometryPipeline:

    def __init__(
        self,
        calibration_data,
        motion_estimator: Optional[MotionEstimator] = None,
        mode: str = "mono",
        frame_size: Tuple[int, int] = (640, 480),
    ):
        self.mode       = mode
        self.frame_size = frame_size
        self.is_initialized = False

        # ── Calibration ───────────────────────────────────────────────────
        self.left_calib        = calibration_data['left']
        self.intrinsics        = self.left_calib['intrinsics']
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

        self.extractor = FeatureExtractor(
            method="FAST+KLT", frame_size=frame_size
        )
        self.database  = FeatureDatabase()

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

        self.triangulator = Triangulator(K=self.K, min_angle_deg=1.0)

        self.pnp = PnPEstimator(
            K=self.K,
            dist_coeffs=self.distortion_coeffs,
            min_inliers=12,
            reprojection_error_px=4.0,
        )

        self.landmark_map = LandmarkMap(sliding_window_size=10)

        self.ba = BundleAdjustment(
            K=self.K,
            max_iterations=50,
            min_landmarks=15,
            min_keyframes=3,
            verbose=True,
        )
        self._ba_keyframe_interval = 3
        self._kf_count_since_ba    = 0

        self._phase = 'bootstrap'
        self._min_landmarks_for_pnp = 20

        self._current_R     = np.eye(3, dtype=np.float64)
        self._current_t     = np.zeros((3, 1), dtype=np.float64)
        self._pose_from_pnp = False

        self._last_valid_R = np.eye(3, dtype=np.float64)
        self._last_valid_t = np.zeros((3, 1), dtype=np.float64)

        self._estimation_queue = Queue(maxsize=2)
        self._result_queue     = Queue(maxsize=8)

        self._estimator_thread = Thread(
            target=self._estimation_loop, daemon=True,
        )
        self._estimator_thread.start()

        self._global_track_min    = 50
        self.gridder_max_per_cell = (
            self.extractor.gridder.min_features_per_cell * 2
        )
        self._frame_idx    = 0
        self._pose_history = {}

        # ── VIO wiring ────────────────────────────────────────────────────
        self._motion_estimator: Optional[MotionEstimator] = motion_estimator

        # [Fix Issues 4 & 5]
        # set_frame_callback() registers a callback fired on EVERY camera
        # frame so the IMU pipeline cuts a chunk per frame (not per keyframe)
        self._on_frame_cb = None

    # ── Public wiring API ─────────────────────────────────────────────────

    def set_motion_estimator(self, me: MotionEstimator) -> None:
        self._motion_estimator = me

    def set_frame_callback(self, cb) -> None:
        """
        [Fix Issue 4] Renamed from set_keyframe_callback → set_frame_callback
        to match vo_subscriber.py.

        cb(timestamp: float) is called on EVERY camera frame so the IMU
        pipeline cuts one chunk per frame interval.
        [Fix Issue 5] Callback is now wired and fires every frame.
        """
        self._on_frame_cb = cb

    # kept for backwards compatibility
    def set_keyframe_callback(self, cb) -> None:
        self.set_frame_callback(cb)

    # ── Main entry point ──────────────────────────────────────────────────

    def process_frame_mono(
        self,
        cv_frame: np.ndarray,
        timestamp: float,
    ) -> Optional[dict]:

        gray_frame = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2GRAY)

        if not self.is_initialized:
            self._handle_initialization(gray_frame, timestamp)
            return None

        self._frame_idx += 1
        self._drain_result_queue()

        # [Fix Issue 5] — notify IMU pipeline EVERY frame so chunks are cut
        # at frame rate, not just at keyframe rate
        if self._on_frame_cb is not None:
            self._on_frame_cb(timestamp)

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

        self._update_phase()
        if self._phase == 'tracking':
            self._estimate_pose_pnp(tracked_ids, tracked_curr_pts)
        else:
            # Bootstrap: use keyframe selector (but ensure continuity)
            if hasattr(self.keyframe_selector, '_world_R') and self.keyframe_selector._world_R is not None:
                self._current_R = self.keyframe_selector._world_R.copy()
                self._current_t = self.keyframe_selector._world_t.copy()
            self._pose_from_pnp = False

        # NEW: Fallback propagation on failure (prevents freezing)
        if not self._pose_from_pnp and hasattr(self, '_last_valid_R'):
            # Simple constant velocity / hold last good pose (better than freeze)
            self._current_R = self._last_valid_R.copy()
            self._current_t = self._last_valid_t.copy()
        elif self._pose_from_pnp:
            self._last_valid_R = self._current_R.copy()
            self._last_valid_t = self._current_t.copy()

        # ── Compute metric pose every frame (fast path) ───────────────────
        pose: Optional[Pose] = None
        if self._motion_estimator is not None:
            pose = self._motion_estimator.compute_pose(
                R=self._current_R,
                t=self._current_t,
                timestamp=timestamp,
            )

        snapshot = self._build_snapshot(timestamp)
        if not self._estimation_queue.full():
            self._estimation_queue.put_nowait(snapshot)

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
            keep_mask        = ~np.isin(tracked_ids, evict_ids)
            tracked_curr_pts = tracked_curr_pts[keep_mask]
            tracked_ids      = tracked_ids[keep_mask]

        new_grid_points = self.extractor.extract_features_in_empty_cells(
            gray_frame=gray_frame,
            tracked_points=tracked_curr_pts,
        )
        if len(new_grid_points) > 0:
            self.database.add_new_tracks(new_grid_points, gray_frame)

        total_tracked = len(tracked_ids) + len(new_grid_points)
        if total_tracked < self._global_track_min:
            all_pts  = self.database.get_active_positions()
            fallback = self.extractor.detect_new_features(
                gray_frame, existing_points=all_pts
            )
            if len(fallback) > 0:
                self.database.add_new_tracks(fallback, gray_frame)

        self.database.set_reference_frame(gray_frame)
        tracks = self.database.get_active_feature_histories()

        return {
            'frame_idx'       : self._frame_idx,
            'timestamp'       : timestamp,
            'n_tracked'       : total_tracked,
            'n_landmarks'     : self.landmark_map.num_landmarks(),
            'n_keyframes'     : self.keyframe_selector.num_keyframes(),
            'phase'           : self._phase,
            'pose_from_pnp'   : self._pose_from_pnp,
            'R'               : self._current_R.copy(),
            't'               : self._current_t.copy(),
            'pose'            : pose,        # Pose dataclass or None
            'landmark_summary': self.landmark_map.summary(),
            'tracks'          : tracks,
            'K'               : self.K,
            'D'               : self.distortion_coeffs,
        }

    # ── Pose estimation ───────────────────────────────────────────────────

    def _estimate_pose_pnp(self, tracked_ids, tracked_curr_pts):
        pts3d_list, pts2d_list = [], []

        for feat_id, pt2d in zip(tracked_ids, tracked_curr_pts):
            lm_id = self.landmark_map.feat_to_lm.get(int(feat_id))
            if lm_id is None or lm_id not in self.landmark_map.landmarks:
                continue
            pts3d_list.append(self.landmark_map.landmarks[lm_id])
            pts2d_list.append(pt2d)

        if len(pts3d_list) < self.pnp.min_inliers:
            self._pose_from_pnp = False
            return

        pts3d  = np.array(pts3d_list, dtype=np.float64)
        pts2d  = np.array(pts2d_list, dtype=np.float64)
        result = self.pnp.estimate(pts3d=pts3d, pts2d=pts2d)

        if result is None:
            self._pose_from_pnp = False
            print(f"[PnP] Failed frame {self._frame_idx} "
                  f"({len(pts3d)} candidates) — keeping last pose")
            return

        R, t, inlier_mask   = result
        self._current_R     = R
        self._current_t     = t
        self._pose_from_pnp = True
        print(f"[PnP] Frame {self._frame_idx}: "
              f"{inlier_mask.sum()}/{len(pts3d)} inliers")

    def _update_phase(self):
        if self._phase == 'bootstrap':
            if self.landmark_map.num_landmarks() >= self._min_landmarks_for_pnp:
                self._phase = 'tracking'
                print(f"[Pipeline] bootstrap → tracking "
                      f"({self.landmark_map.num_landmarks()} landmarks)")

    # ── Background estimation thread ──────────────────────────────────────

    def _estimation_loop(self):
        while True:
            snapshot = self._estimation_queue.get()
            if snapshot is None:
                break
            histories     = snapshot['histories']
            frame_idx     = snapshot['frame_idx']
            timestamp     = snapshot['timestamp']
            ransac_result = self.ransac.estimate(histories)
            result_packet = {
                'frame_idx'    : frame_idx,
                'timestamp'    : timestamp,
                'ransac_result': ransac_result,
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
            timestamp     = packet['timestamp']

            if ransac_result is not None:
                outlier_ids = ransac_result.get('outlier_ids', np.array([]))
                if len(outlier_ids) > 0:
                    self.landmark_map.prune_feat_ids(outlier_ids)

            is_kf = self.keyframe_selector.process(
                frame_idx=frame_idx,
                ransac_result=ransac_result,
                timestamp=timestamp,
            )

            if is_kf:
                print(f"[Pipeline] New keyframe: frame {frame_idx}")

                pair = self.keyframe_selector.get_last_two_keyframes()

                if pair is not None:
                    kf_prev, kf_curr = pair

                    if self._phase == 'tracking':
                        kf_frame_idx = kf_curr['frame_idx']
                        if kf_frame_idx in self._pose_history:
                            R_at_kf, t_at_kf = self._pose_history[kf_frame_idx]
                            kf_curr['R'] = R_at_kf.copy()
                            kf_curr['t'] = t_at_kf.copy()

                    tri_result = self.triangulator.triangulate(kf_prev, kf_curr)
                    if tri_result is not None:
                        self.landmark_map.add_triangulation_result(tri_result)
                        self._register_new_observations(kf_curr)
                        print(f"[Pipeline] Triangulated "
                              f"{len(tri_result['landmarks'])} landmarks. "
                              f"Total: {self.landmark_map.num_landmarks()}")

                        # ── Notify motion estimator (slow path / VIA) ─────
                        if self._motion_estimator is not None:
                            all_kfs = self.keyframe_selector.get_all_keyframes()
                            self._motion_estimator.on_new_keyframe(
                                keyframe_poses=all_kfs,
                                timestamp=timestamp,
                            )

                    self._update_phase()

                active_frames = [
                    kf['frame_idx']
                    for kf in self.keyframe_selector.get_all_keyframes()
                ]
                if active_frames:
                    self.landmark_map.prune_outside_window(active_frames)

                self._kf_count_since_ba += 1
                if self._kf_count_since_ba >= self._ba_keyframe_interval:
                    self._kf_count_since_ba = 0
                    ran = self.ba.run(
                        landmark_map=self.landmark_map,
                        keyframe_selector=self.keyframe_selector,
                    )
                    if ran:
                        print("[Pipeline] BA completed.")
                        if self._motion_estimator is not None:
                            ba_kfs = self.keyframe_selector.get_all_keyframes()
                            ba_pts = list(self.landmark_map.landmarks.values())
                            self._motion_estimator.on_ba_updated(
                                ba_keyframe_poses=ba_kfs,
                                map_points=ba_pts,
                                timestamp=timestamp,
                            )

    # ── Misc ──────────────────────────────────────────────────────────────

    def _register_new_observations(self, kf: dict):
        for feat_id, pt in zip(kf['feat_ids'], kf['pts']):
            lm_id = self.landmark_map.feat_to_lm.get(int(feat_id))
            if lm_id is not None:
                self.landmark_map.add_observation(
                    lm_id=lm_id,
                    frame_idx=kf['frame_idx'],
                    u=float(pt[0]),
                    v=float(pt[1]),
                )

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

    def get_current_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._current_R.copy(), self._current_t.copy()

    def get_landmarks_for_ba(self):
        return self.landmark_map.get_landmarks_for_ba()

    def get_observations_for_ba(self):
        return self.landmark_map.get_observations_for_ba()

    def get_keyframe_poses(self):
        return self.keyframe_selector.get_all_keyframes()

    def process_frame_stereo(self, cv_frame_left, cv_frame_right, timestamp):
        pass

    def shutdown(self):
        self._estimation_queue.put(None)
        self._estimator_thread.join(timeout=2.0)
        self.extractor.shutdown()