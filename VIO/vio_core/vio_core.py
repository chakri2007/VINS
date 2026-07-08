import threading

import cv2
import numpy as np

from vio_core.preprocess_image import preprocess_image
from vio_core.ransac import estimate_fundamental_matrix_ransac

from feature_manager.feature_extractor import FeatureExtractor
from memory_management.view_set import ViewSet
from memory_management.sliding_window import (
    SlidingWindowState,
    update_tracks,
    update_window_membership,
    append_imu_measurement,
    extract_imu_between,
    prune_imu_before,
)
from imu.imu_measurement import IMUMeasurement
from vio_core.triangulate import find_triangulation_candidates, triangulate_candidates, add_landmarks

from vio_core.reprojection import validate_landmarks

from vio_core.pnp import find_pnp_correspondences, PnPCorrespondence, solve_pnp
from optimization.graph_builder import GraphBuilder
from optimization.ceres_bundle_adjustment import CeresBundleAdjuster as BundleAdjuster
from optimization.ceres_bundle_adjustment_motion import bundle_adjustment_motion
from optimization.state_update import update_state_from_graph
from optimization.median_depth import normalize_map
from imu.vi_alignment import (
    initialize_visual_inertial_state,
    camera_pose_to_body_pose,
    body_pose_to_camera_pose,
)
from imu.preintegration import IMUPreintegrator
from imu.prediction import predict_state

# MATLAB reference guards acceptance of the linear VI-alignment solve
# with `info.IsSolutionUsable && scale > 1e-3` (see helperVIO.m,
# Phase 2). solve_alignment()'s `success` flag only reflects that the
# linear system was solvable (full rank) -- it says nothing about
# whether the recovered scale is physically sane. A degenerate init
# window (not enough rotation/acceleration excitation) can return a
# full-rank solve with a negative or near-zero scale. This constant
# mirrors MATLAB's floor so we reject those solutions the same way.
MIN_USABLE_SCALE = 1e-3


class VisualInertialOdometry():
    def __init__(self, calib_data):

        self.left_calib        = calib_data['left']
        self.intrinsics        = self.left_calib['intrinsics']
        self.distortion_coeffs = np.array(
            self.left_calib['distortion_coefficients']
        )
        self.T_BS = np.array(
            self.left_calib['T_BS']['data']
        ).reshape(4, 4)

        #
        # IMU noise/extrinsic calibration. self.T_BS above (from the
        # camera yaml) is the camera->body extrinsic used to fold poses
        # into the body frame before VI alignment (see VI_alignment()).
        # imu_calib carries the noise densities used to construct the
        # IMUPreintegrator; falls back to IMUPreintegrator's own
        # defaults if imu.yaml wasn't supplied by the caller.
        #
        self.imu_calib = calib_data.get('imu', {})

        self.K_raw = np.array([
            [self.intrinsics[0], 0,                  self.intrinsics[2]],
            [0,                  self.intrinsics[1], self.intrinsics[3]],
            [0,                  0,                  1],
        ], dtype=np.float64)

        self.K = self.K_raw.copy()

        self.params = {
            'Equalize':         False,
            'Undistort':        True,
            'ClipLimit':        3.0 / 256,
            'NumTiles':         (8, 8),
            'F_loop':           5,
            'F_Iterations':     2000,
            'F_Confidence':     99,
            'F_Threshold':      4,
            'keyFrameParallax': 50,

            # VINS-Mono style: BA is triggered by keyframe insertion (see
            # should_run_bundle_adjustment), not by frame count / a fixed
            # frequency, so those knobs are gone. What's left is a one-shot
            # solver time budget for the vision-only window BA — VINS-Mono
            # uses 0.2s for its init BA, a looser cap than steady-state
            # (0.04s) since this only runs once per keyframe, not every
            # frame, and is shielded from the sensor stream.
            'baMaxSolverTimeSeconds': 0.2,
        }

        self.feature_extractor = FeatureExtractor(frame_size=(612, 512))
        self.view_set          = ViewSet()
        self.sw_state          = SlidingWindowState(window_size=21)
        self.prev_img_frame    = None

        self.removed_frame_ids: list = []

        self.isFirstFrame      = True
        self.isMapInitialized  = False
        self.isVIO_initialized = False
        self.isVI_aligned      = False

        # Latched once True — mirrors MATLAB's `readyToAlignCameraAndIMU`.
        # Set the first time window_state["isWindowFull"] is seen (i.e.
        # the sliding window has reached full size at least once), and
        # never reset afterward. Alignment attempts are gated on this
        # instead of the old bare `len(align_view_ids) >= 3` floor.
        self.readyToAlignCameraAndIMU = False

        self.frameID = 0

        self.graph_builder = GraphBuilder()

        self.bundle_adjustment = None

        # Guards sw_state / view_set mutations once vio_loop_frontend()
        # (ROS callback thread) and vio_loop_backend() (dedicated worker
        # thread) can run concurrently -- see vio_subscriber.py. process_imu
        # (callback thread) and the backend's IMU-buffer reads/prunes
        # (worker thread) are the most frequent point of overlap, but this
        # lock is taken generously around any shared-state mutation rather
        # than reasoned about per-field.
        self.state_lock = threading.RLock()


    def vio_loop_frontend(self, raw_img_frame, timestamp):
        """
        Frontend-only step -- intended to run on the ROS image-callback
        thread. Mirrors VINS-Fusion's Estimator::inputImage(): KLT
        tracking, RANSAC outlier rejection, grid-based eviction, and
        new-feature detection happen here, synchronously, because
        they're bounded, cheap, and -- critically -- entirely
        per-view: process_frontend() only ever reads/writes
        all_observations[frameID]/all_ids[frameID]/current_view_id, and
        never touches sliding_window_view_ids. That means frontend is
        free to run arbitrarily far ahead of backend without racing it.

        Window MEMBERSHIP (which frames sit in sliding_window_view_ids,
        keyframe status, eviction) is decided in vio_loop_backend
        instead, not here -- see update_window_membership() in
        sliding_window.py for why that split matters.

        Returns None on the first frame (nothing to hand to the backend
        yet), otherwise (frameID, timestamp) to be pushed onto the
        backend's work queue by the caller.
        """

        with self.state_lock:
            self.frameID += 1

            # self.img_frame, self.K = preprocess_image(
            #     raw_img_frame,
            #     self.distortion_coeffs,
            #     self.K_raw,
            #     self.params,
            #
            self.img_frame = raw_img_frame.copy()

            if self.isFirstFrame:
                self._init_first_frame(
                    self.img_frame,
                    self.frameID,
                    timestamp,
                )
                return None

            self.process_frontend(
                self.img_frame,
                self.frameID,
            )

            return (self.frameID, timestamp)

    def vio_loop_backend(self, frameID, timestamp):
        """
        Backend-only step -- intended to run on a dedicated worker
        thread, fed from a FIFO queue by the caller. Mirrors
        VINS-Fusion's Estimator::processMeasurements(): window
        membership, IMU preintegration, phase selection, PnP, and
        bundle adjustment all happen here, strictly in the order
        frames were queued.

        update_window_membership() is called here rather than in
        vio_loop_frontend() specifically because it's the sole mutator
        of sliding_window_view_ids, and every decision it makes depends
        on the state left by the call for the previous frame. Since
        this method only ever runs from the one backend worker thread,
        pulling frames off a FIFO queue in order, that ordering
        guarantee holds automatically -- it would not if this call sat
        in the frontend, racing ahead of however far backend has
        actually gotten (that was the KeyError bug: a frame entering
        the window before view_set had committed a pose for it).
        """

        with self.state_lock:
            removed_frame_id, window_state = update_window_membership(
                state               = self.sw_state,
                view_id             = frameID,
                key_frame_parallax  = self.params['keyFrameParallax'],
            )

            if (removed_frame_id >= 0
                    and len(self.sw_state.sliding_window_view_ids) > 0
                    and removed_frame_id > self.sw_state.sliding_window_view_ids[0]):
                self.removed_frame_ids.append(removed_frame_id)

            self._prune_imu_buffer()

            if not self.isMapInitialized:

                self.vio_initialization(
                    window_state,
                    frameID,
                    timestamp,
                )

            elif not self.isVI_aligned:

                self.VI_alignment(
                    window_state,
                    frameID,
                    timestamp,
                )

            else:

                self.visual_inertial_optimization(
                    window_state,
                    frameID,
                    timestamp,
                )

    def vio_loop(self, raw_img_frame, timestamp):
        """
        Synchronous convenience wrapper (single-threaded use, e.g.
        offline batch processing over a rosbag/dataset where there's no
        callback-thread/worker-thread split to preserve). Equivalent to
        running vio_loop_frontend() immediately followed by
        vio_loop_backend() on the same thread -- this is what every call
        did before the frontend/backend split. Do NOT call this from the
        ROS node; use vio_loop_frontend()/vio_loop_backend() there
        instead (see vio_subscriber.py).
        """

        item = self.vio_loop_frontend(raw_img_frame, timestamp)
        if item is None:
            return
        frameID, ts = item
        self.vio_loop_backend(frameID, ts)

    def process_frame_mono(self, raw_img_frame, img_frame_timestamp):
        self.vio_loop(raw_img_frame, img_frame_timestamp)
        return {
            'pose':   None,
            'tracks': self.get_active_tracks(),
            'K':      self.K,
            'D':      self.distortion_coeffs,
        }
    
    def _init_first_frame(self, img_frame, frameID, timestamp):
        features  = self.feature_extractor.detect_initial_features(img_frame)
        num_pts   = len(features)
        point_ids = np.arange(1, num_pts + 1)

        self.sw_state.all_observations[frameID]  = features
        self.sw_state.all_ids[frameID]           = np.column_stack(
            [np.full(num_pts, frameID), point_ids]
        )
        self.sw_state.all_triangulated[frameID]  = np.zeros(num_pts, dtype=bool)
        self.sw_state.is_key_frame[frameID]      = True
        for pid in point_ids:
            self.sw_state.key_point_track_count[pid] = 1

        # First-frame registration: append into sliding_window_view_ids and
        # set current_view_id. This is the one place window membership is
        # decided outside vio_loop_backend, but it's inherently safe: it
        # runs synchronously before the frontend/backend split even starts
        # (isFirstFrame short-circuits vio_loop_frontend, nothing is queued
        # for backend yet), so there's no ordering hazard here.
        update_window_membership(
            state               = self.sw_state,
            view_id             = frameID,
            key_frame_parallax  = self.params['keyFrameParallax'],
        )

        self.prev_img_frame  = img_frame
        self.first_img_frame = img_frame
        self.view_set.add_view(view_id=frameID, R=np.eye(3), t=np.zeros(3), timestamp=timestamp)
        self.isFirstFrame = False

        self._prune_imu_buffer()

    def process_frontend(self, img_frame, frameID):
        prev_stored_id = self.sw_state.current_view_id
        prev_points    = self.sw_state.all_observations[prev_stored_id]

        tracked_points, status = self.feature_extractor.track_features(
            self.prev_img_frame,
            img_frame,
            prev_points,
        )
        valid_idx = status.astype(bool)   # same length as prev_points

        # RANSAC outlier rejection + write this frame's observations/ids +
        # current_view_id continuity. Frame-local only -- does NOT decide
        # window membership (see update_window_membership, called from
        # vio_loop_backend instead).
        update_tracks(
            state               = self.sw_state,
            image_shape         = img_frame.shape,
            curr_points_tracked = tracked_points,
            valid_idx           = valid_idx,
            view_id             = frameID,
            F_loop              = self.params['F_loop'],
            F_iterations        = self.params['F_Iterations'],
            F_confidence        = self.params['F_Confidence'],
            F_threshold         = self.params['F_Threshold'],
        )

        # ── grid-based eviction on RANSAC survivors ───────────────────────
        post_pts = self.sw_state.all_observations.get(
            frameID, np.empty((0, 2), dtype=np.float32)
        )
        post_ids = (
            self.sw_state.all_ids[frameID][:, 1]
            if frameID in self.sw_state.all_ids and len(self.sw_state.all_ids[frameID]) > 0
            else np.empty(0, dtype=np.int64)
        )

        if len(post_ids) > 0:
            ages = np.array([
                self.sw_state.key_point_track_count.get(int(pid), 1)
                for pid in post_ids
            ])
            evict = self.feature_extractor.gridder.get_overcrowded_evictions(
                post_pts, post_ids, ages
            )
            if len(evict) > 0:
                keep = ~np.isin(post_ids, evict)
                post_pts = post_pts[keep]
                post_ids = post_ids[keep]
                self.sw_state.all_observations[frameID] = post_pts
                self.sw_state.all_ids[frameID] = np.column_stack(
                    [np.full(len(post_ids), frameID), post_ids]
                )
                tri = self.sw_state.all_triangulated.get(frameID)
                if tri is not None and len(tri) == len(keep):
                    self.sw_state.all_triangulated[frameID] = tri[keep]

        # ── detect new features in sparse grid cells ──────────────────────
        new_pts = self.feature_extractor.extract_features_in_empty_cells(
            img_frame, post_pts
        )
        if len(new_pts) > 0:
            num_new  = len(new_pts)
            start_id = max(self.sw_state.key_point_track_count.keys(), default=0) + 1
            new_ids  = np.arange(start_id, start_id + num_new)
            for pid in new_ids:
                self.sw_state.key_point_track_count[pid] = 1

            post_pts = np.vstack([post_pts, new_pts])
            post_ids = np.concatenate([post_ids, new_ids])

            self.sw_state.all_observations[frameID] = post_pts
            self.sw_state.all_ids[frameID] = np.column_stack(
                [np.full(len(post_ids), frameID), post_ids]
            )
            existing_tri = self.sw_state.all_triangulated.get(
                frameID, np.zeros(len(post_ids) - num_new, dtype=bool)
            )
            self.sw_state.all_triangulated[frameID] = np.concatenate([
                existing_tri, np.zeros(num_new, dtype=bool)
            ])

        self.prev_img_frame = img_frame

    def _prune_imu_buffer(self):
        """
        Drop IMU samples older than the sliding window's oldest
        surviving keyframe. Safe to call after every window update:
        does nothing if the window is empty, and extract_imu_between
        only ever needs data from the oldest kept keyframe onward.
        """

        sw_ids = self.sw_state.sliding_window_view_ids

        if len(sw_ids) == 0:
            return

        oldest_timestamp = self.view_set.get_timestamp(sw_ids[0])
        prune_imu_before(self.sw_state, oldest_timestamp)
    
    def get_active_tracks(self, max_history_length: int = 10) -> dict:
        """Build track history for every point still alive in the current frame,
        using the last `max_history_length` RAW frames (not the sparse keyframe
        list in sliding_window_view_ids), so each trail is a dense, smooth
        sequence of small per-frame steps instead of jumping keyframe-to-keyframe."""
        current_id = self.frameID
        current_ids = self.sw_state.all_ids.get(current_id)
        if current_ids is None or len(current_ids) == 0:
            return {}
        alive_ids = set(int(pid) for pid in current_ids[:, 1])

        tracks = {pid: [] for pid in alive_ids}
        start_id = max(1, current_id - max_history_length + 1)

        for view_id in range(start_id, current_id + 1):
            obs = self.sw_state.all_observations.get(view_id)
            ids = self.sw_state.all_ids.get(view_id)
            if obs is None or ids is None or len(obs) == 0:
                continue
            for point_id, (u, v) in zip(ids[:, 1], obs):
                pid = int(point_id)
                if pid in tracks:
                    tracks[pid].append((view_id, float(u), float(v)))

        return {pid: hist for pid, hist in tracks.items() if hist}

    def vio_initialization(self, window_state, frameID, timestamp):

        if window_state["isFirstFewViews"]:

            self.view_set.add_view(
                frameID,
                np.eye(3),
                np.zeros(3),
                timestamp,
            )

            return

        if not window_state["isEnoughParallax"]:
            return

        success = self._initialise_map(frameID, timestamp)

        if success:
            candidates = find_triangulation_candidates(
                self.sw_state,
                self.view_set,
            )

            triangulated = triangulate_candidates(
                candidates,
                self.view_set,
                self.K,
            )

            add_landmarks(
                triangulated,
                self.sw_state,
            )

            self.isMapInitialized = True

            print("Map initialized.")

    
    def _initialise_map(self, frameID, timestamp) -> bool:
        sw_ids = self.sw_state.sliding_window_view_ids
        if len(sw_ids) < 2:
            return False

        id1, id2 = sw_ids[-2], sw_ids[-1]
        ids1 = self.sw_state.all_ids.get(id1)
        ids2 = self.sw_state.all_ids.get(id2)
        if ids1 is None or ids2 is None or len(ids1) < 8 or len(ids2) < 8:
            return False

        _, ia, ib = np.intersect1d(ids1[:, 1], ids2[:, 1], return_indices=True)
        if len(ia) < 8:
            return False

        matches1 = self.sw_state.all_observations[id1][ia]
        matches2 = self.sw_state.all_observations[id2][ib]

        best_F, best_inliers = None, None
        for _ in range(10):
            F, inliers = estimate_fundamental_matrix_ransac(
                matches1, matches2,
                num_trials     = self.params['F_Iterations'],
                confidence     = self.params['F_Confidence'],
                dist_threshold = self.params['F_Threshold'],
            )
            if F is None:
                continue
            if best_inliers is None or np.count_nonzero(inliers) > np.count_nonzero(best_inliers):
                best_F, best_inliers = F, inliers

        if best_F is None or np.count_nonzero(best_inliers) < 8:
            return False

        R, t = self._estimate_relative_pose(
            best_F, matches1[best_inliers], matches2[best_inliers]
        )
        if R is None:
            return False

        self.view_set.add_view(view_id=frameID, R=R, t=t, timestamp=timestamp)
        return True
    
    def _estimate_relative_pose(self, F, pts1, pts2):
        E = self.K.T @ F @ self.K
        n_in, R, t, _ = cv2.recoverPose(
            E, pts1.astype(np.float64), pts2.astype(np.float64), self.K
        )
        if n_in < 8:
            return None, None
        # cv2 returns world-to-camera; invert to camera-to-world (MATLAB convention)
        R_cw = R.T
        t_cw = -(R_cw @ t.ravel())
        return R_cw, t_cw

    def VI_alignment(self, window_state, frameID, timestamp):
        # print("Current observations:", len(self.sw_state.all_ids[frameID]))
        # print("Landmarks:", len(self.sw_state.landmarks))

        success = self.run_pnp(frameID, timestamp)

        if not success:
            print("PnP failed. Skipping triangulation.")
            return

        new_points_added = self.run_triangulation()
        # print("running triangulation")

        #
        # Build factor graph from current sliding window
        #
        factor_graph = self.graph_builder.build(
            view_set=self.view_set,
            sw_state=self.sw_state,
            K=self.K,
        )

        # factor_graph.print_summary()

        self.bundle_adjustment = BundleAdjuster(
            factor_graph,
            max_solver_time_in_seconds=self.params['baMaxSolverTimeSeconds'],
        )

        if self.should_run_bundle_adjustment(window_state):

            self.fix_bundle_adjustment_poses(window_state)

            result = self.bundle_adjustment.optimize()

            if result is not None:
                update_state_from_graph(
                            factor_graph,
                            self.view_set,
                            self.sw_state,
                        )
                scale = normalize_map(
                            self.view_set,
                            self.sw_state,
                        )

                # Latch once the window has been full at least once —
                # mirrors MATLAB's
                #   if windowState.isWindowFull && ~readyToAlignCameraAndIMU
                #       readyToAlignCameraAndIMU = true;
                # Never reset back to False afterward.
                if window_state.get("isWindowFull") and not self.readyToAlignCameraAndIMU:
                    self.readyToAlignCameraAndIMU = True

                if not self.isVIO_initialized and self.readyToAlignCameraAndIMU:

                    self.try_vi_alignment()

            self.bundle_adjustment.clear_fixed_poses()

    def try_vi_alignment(self):
        """
        Attempt linear visual-inertial alignment (metric scale, gravity
        direction, per-keyframe velocities, accelerometer bias) over
        the current sliding window.

        Only called once `self.readyToAlignCameraAndIMU` has latched
        True (window has been full at least once) — see VI_alignment().
        This mirrors the MATLAB reference's use of
        swIDs = getSlidingWindowIDs(fpManager); swIDs = swIDs(1:end-1);
        i.e. every *closed* keyframe interval in the window except the
        newest, still-open one — each interval needs a completed IMU
        preintegration between two confirmed keyframe timestamps.
        """

        sw_ids = list(self.sw_state.sliding_window_view_ids)

        align_view_ids = sw_ids[:-1]

        # Should always hold once the window has been full at least
        # once, but keep as a defensive floor for solvability.
        if len(align_view_ids) < 3:
            return

        imu_preintegrations = self._build_imu_preintegrations(align_view_ids)

        if imu_preintegrations is None:
            # Missing IMU coverage for at least one interval -- can't
            # run alignment yet.
            return

        # --- Debug dump for MATLAB comparison (first attempt only) ---
        if not hasattr(self, "_saved_alignment_debug_data"):
            self._saved_alignment_debug_data = True

            import scipy.io
            from memory_management.sliding_window import extract_imu_between

            N = len(align_view_ids)

            # Camera poses (camera-to-world, same convention MATLAB uses)
            campose_R = np.zeros((3, 3, N))
            campose_t = np.zeros((N, 3))
            for k, vid in enumerate(align_view_ids):
                R, t = self.view_set.get_pose(vid)
                campose_R[:, :, k] = R
                campose_t[k, :] = t

            # Raw IMU between each consecutive pair (NOT preintegrated --
            # MATLAB's estimateGravityRotationAndPoseScale wants raw samples)
            gyroData = np.empty((1, N - 1), dtype=object)
            accelData = np.empty((1, N - 1), dtype=object)
            for k, (i, j) in enumerate(zip(align_view_ids[:-1], align_view_ids[1:])):
                t_i = self.view_set.get_timestamp(i)
                t_j = self.view_set.get_timestamp(j)
                samples = extract_imu_between(self.sw_state, t_i, t_j)
                gyroData[0, k] = np.vstack([m.gyro for m in samples])
                accelData[0, k] = np.vstack([m.accel for m in samples])

            # Camera->IMU extrinsic (T_BS)
            T_BS_R = self.T_BS[:3, :3]
            T_BS_t = self.T_BS[:3, 3]

            # IMU noise params, saved as NxN covariance-style matrices (diagonal here)
            # to match MATLAB's IMUParameters convention -- estimateGravityRotation...
            # indexes these as noiseMatrix(1,1), so a bare scalar/1x1 double would get
            # squeezed to a 0-d value on load (squeeze_me=True) and break that indexing.
            imuSampleRate     = self.imu_calib.get('rate_hz', 100)
            imuGyroNoise      = np.diag([self.imu_calib.get('gyroscope_noise_density', 1.0e-3)] * 3)
            imuGyroBiasNoise  = np.diag([self.imu_calib.get('gyroscope_random_walk', 1.0e-5)] * 3)
            imuAccelNoise     = np.diag([self.imu_calib.get('accelerometer_noise_density', 1.0e-2)] * 3)
            imuAccelBiasNoise = np.diag([self.imu_calib.get('accelerometer_random_walk', 1.0e-4)] * 3)

            scipy.io.savemat(
                "vi_alignment_debug_python.mat",
                {
                    "swIDs": np.array(align_view_ids, dtype=float).reshape(-1, 1),
                    "campose_R": campose_R,
                    "campose_t": campose_t,
                    "gyroData": gyroData,
                    "accelData": accelData,
                    "T_BS_R": T_BS_R,
                    "T_BS_t": T_BS_t,
                    "imuSampleRate": imuSampleRate,
                    "imuGyroNoise": imuGyroNoise,
                    "imuGyroBiasNoise": imuGyroBiasNoise,
                    "imuAccelNoise": imuAccelNoise,
                    "imuAccelBiasNoise": imuAccelBiasNoise,
                },
            )
            print("Saved vi_alignment_debug_python.mat for MATLAB comparison.")

        result = initialize_visual_inertial_state(
            view_set=self.view_set,
            sliding_window=self.sw_state,
            imu_preintegrations=imu_preintegrations,
            view_ids=align_view_ids,
            sensor_transform=self.T_BS,
            apply_scale_to_map=False,
        )

        # --- Stash Python's own answer for MATLAB-side diffing (first attempt only) ---
        if not hasattr(self, "_saved_python_result"):
            self._saved_python_result = True
            scipy.io.savemat(
                "vi_alignment_debug_python_result.mat",
                {
                    "python_success": bool(result.success),
                    "python_scale": float(result.scale),
                    "python_gravity": np.asarray(result.gravity).reshape(1, 3),
                    "python_accel_bias": np.asarray(result.accel_bias).reshape(1, 3),
                },
            )

        if result.success and result.scale <= MIN_USABLE_SCALE:
            # Linear system was solvable, but the recovered scale is
            # not physically usable (zero, negative, or degenerate --
            # e.g. init window didn't have enough rotation/accel
            # excitation to separate scale from gravity/bias). MATLAB
            # rejects this the same way (`scale > 1e-3` gate) instead
            # of committing it. Don't latch isVI_aligned; just try
            # again on a later frame once the window has moved on.
            print(
                f"[VIO] VI alignment scale not usable ({result.scale:.6g} "
                f"<= {MIN_USABLE_SCALE}); rejecting and retrying."
            )
            return

        if result.success:

            print("\n========== VI Alignment ==========")
            print("Scale:", result.scale)
            print("Gravity:", result.gravity)
            print("Accel Bias:", result.accel_bias)

            self.isVIO_initialized = True
            self.isVI_aligned = True

            newest_id = sw_ids[-1]
            prev_id = align_view_ids[-1]
            if prev_id in self.sw_state.velocities:
                self.sw_state.velocities[newest_id] = self.sw_state.velocities[prev_id].copy()
                
    def _build_imu_preintegrations(self, view_ids):
        """
        Preintegrate IMU data between every consecutive pair of
        `view_ids`, sourced from the continuous timestamp-ordered
        sw_state.imu_buffer (see extract_imu_between).

        Uses the sliding window's current bias estimates as the
        preintegrator's linearization point -- zero before alignment
        has ever succeeded, matching MATLAB (no bias correction is
        available yet at this stage either).

        Returns
        -------
        dict[(int,int), PreintegratedIMU], or None if any interval is
        missing IMU coverage (e.g. IMU stream hasn't caught up yet).
        """

        preintegrations = {}

        for i, j in zip(view_ids[:-1], view_ids[1:]):

            t_i = self.view_set.get_timestamp(i)
            t_j = self.view_set.get_timestamp(j)

            samples = extract_imu_between(self.sw_state, t_i, t_j)

            if len(samples) < 2:
                return None

            preintegrator = IMUPreintegrator(
                gyro_noise=self.imu_calib.get(
                    'gyroscope_noise_density', 1.0e-3
                ),
                accel_noise=self.imu_calib.get(
                    'accelerometer_noise_density', 1.0e-2
                ),
                gyro_random_walk=self.imu_calib.get(
                    'gyroscope_random_walk', 1.0e-5
                ),
                accel_random_walk=self.imu_calib.get(
                    'accelerometer_random_walk', 1.0e-4
                ),
                bias_g=self.sw_state.gyroscope_bias,
                bias_a=self.sw_state.accelerometer_bias,
            )

            preintegrations[(i, j)] = preintegrator.integrate_measurements(
                samples
            )

        return preintegrations

    def _carry_forward_previous_pose(self, frameID, timestamp):
        """
        Pre-alignment fallback for when PnP can't produce any pose at
        all. There's no IMU state (scale/gravity/velocity) to fall
        back on yet at this stage -- that's only available in Phase 3
        (see _predict_pose_from_imu) -- so the best we can do is carry
        the previous view's pose forward as a placeholder.

        This mirrors MATLAB's behaviour of never leaving a keyframe
        without *some* pose in the view set: `helperEstimateCameraPose`
        always returns a currPose that gets added via `addView`, even
        when very few/no RANSAC inliers were found. Without this, a
        frame stays in the sliding window / feature tracks but absent
        from view_set, which is exactly what caused
        `KeyError: View id ... not found in ViewSet` in
        triangulate_candidates on a later frame.

        The frame is still reported as a PnP failure to the caller, so
        triangulation/new-landmark registration is skipped for it as
        before -- only the pose gap is fixed.
        """

        if self.view_set.num_views == 0:
            # Nothing to carry forward from (shouldn't normally happen
            # once map init has already added the first views).
            return

        prev_view_id = self.view_set.view_ids[-1]
        R_prev, t_prev = self.view_set.get_pose(prev_view_id)
        self.view_set.add_view(frameID, R_prev, t_prev, timestamp)

    def run_pnp(self, frameID, timestamp):

        correspondences = find_pnp_correspondences(
            self.sw_state,
            frameID,
        )

        if len(correspondences) < 6:
            self._carry_forward_previous_pose(frameID, timestamp)
            return False

        result = solve_pnp(
            correspondences,
            self.K,
        )

        if result is None:
            self._carry_forward_previous_pose(frameID, timestamp)
            return False

        Rwc, C, inliers = result

        if inliers is None or len(inliers) == 0:
            self._carry_forward_previous_pose(frameID, timestamp)
            return False

        self.view_set.add_view(
            frameID,
            Rwc,
            C,
            timestamp,
        )

        # Register this frame's observation on every landmark that
        # survived PnP RANSAC as an inlier — this is what keeps the
        # factor graph growing frame over frame.
        for idx in inliers.flatten():
            c = correspondences[int(idx)]
            landmark = self.sw_state.landmarks[c.point_id]
            landmark.add_observation(frameID, c.uv)

        return True
    
    def run_triangulation(self):

        candidates = find_triangulation_candidates(
            self.sw_state,
            self.view_set,
        )

        triangulated = triangulate_candidates(
            candidates,
            self.view_set,
            self.K,
        )

        num_added = add_landmarks(
            triangulated,
            self.sw_state,
        )

        validate_landmarks(
            self.sw_state,
            self.view_set,
            self.K,
        )
        # print(f"Triangulation: {num_added} new landmarks added.")

        return num_added > 0

    def should_run_bundle_adjustment(self, window_state):
        """
        VINS-Mono only runs its vision-only window BA when a new keyframe
        has actually been accepted into the window (relativePose()'s
        >30-correspondence / >20px-parallax gate deciding there's enough
        motion to be worth optimizing) — not on a fixed frame-count/
        frequency schedule, and not just because a few new points got
        triangulated on an otherwise-redundant frame. `isEnoughParallax`
        is this codebase's equivalent of that keyframe-acceptance signal
        (see update_sliding_window), so it's the trigger here too.
        """
        return bool(window_state.get("isEnoughParallax", False))

    def fix_bundle_adjustment_poses(self, window_state):
        """
        VINS-Mono GlobalSFM-style minimal gauge fixing: vision-only SfM over
        a window has 7 unobservable DOF (6 gauge + 1 scale). GlobalSFM::
        construct() removes exactly those by fully fixing the reference
        frame l's pose and fixing only the newest frame's translation (its
        rotation stays free) — the distance between those two fixed camera
        centers is what pins absolute scale. Fixing a whole chunk of poses,
        as the old code did, over-constrains the problem for no benefit.
        """

        sw_ids = list(self.sw_state.sliding_window_view_ids)

        self.bundle_adjustment.clear_fixed_poses()

        if len(sw_ids) == 0:
            return

        if len(sw_ids) == 1:
            # Nothing to triangulate a baseline against yet — just anchor
            # the one pose we have.
            self.bundle_adjustment.fix_pose(sw_ids[0])
            return

        # oldest (reference) frame: fully fixed
        self.bundle_adjustment.fix_pose(sw_ids[0])
        # newest frame: translation only, rotation stays free
        self.bundle_adjustment.fix_pose_translation(sw_ids[-1])

    def process_imu(
        self,
        accel,
        gyro,
        timestamp,
    ):
        """
        Append an incoming IMU sample to the continuous, timestamp-
        ordered buffer (sw_state.imu_buffer). Preintegration between
        any two keyframes is computed on demand from this buffer (see
        _build_imu_preintegrations), by slicing on timestamp rather
        than tracking per-frame-interval chunks — this works correctly
        regardless of which raw frames end up as non-keyframes and get
        dropped from the sliding window.
        """

        with self.state_lock:
            append_imu_measurement(
                self.sw_state,
                IMUMeasurement(
                    timestamp=timestamp,
                    accel=np.asarray(accel, dtype=np.float64),
                    gyro=np.asarray(gyro, dtype=np.float64),
                ),
            )
    def _build_single_imu_preintegration(self, from_view_id, to_view_id, to_timestamp):
        """
        Preintegrate IMU samples between an existing view (from_view_id,
        already in view_set) and a not-yet-added frame (to_view_id, whose
        timestamp is supplied directly since it hasn't been added yet).

        Uses the sliding window's current bias estimates as the
        linearization point, same convention as
        _build_imu_preintegrations. Returns None if IMU coverage is
        insufficient (mirrors that function's behaviour for a single pair).
        """

        t_from = self.view_set.get_timestamp(from_view_id)

        samples = extract_imu_between(self.sw_state, t_from, to_timestamp)

        if len(samples) < 2:
            return None

        preintegrator = IMUPreintegrator(
            gyro_noise=self.imu_calib.get(
                'gyroscope_noise_density', 1.0e-3
            ),
            accel_noise=self.imu_calib.get(
                'accelerometer_noise_density', 1.0e-2
            ),
            gyro_random_walk=self.imu_calib.get(
                'gyroscope_random_walk', 1.0e-5
            ),
            accel_random_walk=self.imu_calib.get(
                'accelerometer_random_walk', 1.0e-4
            ),
            bias_g=self.sw_state.gyroscope_bias,
            bias_a=self.sw_state.accelerometer_bias,
        )

        return preintegrator.integrate_measurements(samples)

    def _predict_pose_from_imu(self, prev_view_id, prev_velocity, preint):
        """
        IMU-only pose/velocity prediction, mirroring MATLAB's
        `fIMU.predict(prevP, prevVel, prevBias)` in Phase 3.

        Used as a fallback when vision (PnP / BA_motion) fails for the
        current frame, so the frame still gets *some* pose committed
        to view_set instead of being silently skipped -- an orphaned
        frame that's still present in the sliding window / feature
        tracks but missing from view_set is exactly what causes the
        `KeyError: View id ... not found in ViewSet` crash once a
        later frame's triangulation looks it up.

        predict_state() operates in the body/IMU frame; view_set
        stores camera-to-world poses, so we convert in and back out
        via the T_BS extrinsic (same convention as vi_alignment.py).
        """

        R_bs = self.T_BS[:3, :3]
        t_bs = self.T_BS[:3, 3]

        R_wc_i, t_wc_i = self.view_set.get_pose(prev_view_id)
        R_wb_i, t_wb_i = camera_pose_to_body_pose(R_wc_i, t_wc_i, R_bs, t_bs)

        R_wb_j, t_wb_j, v_j = predict_state(
            R_wb_i, t_wb_i, prev_velocity, self.sw_state.gravity, preint,
        )

        R_wc_j, t_wc_j = body_pose_to_camera_pose(R_wb_j, t_wb_j, R_bs, t_bs)

        return R_wc_j, t_wc_j, v_j

    def visual_inertial_optimization(self, window_state, frameID, timestamp):
        """
        Phase 3 (post VI-alignment) per-frame step, up through
        helperBundleAdjustmentMotion.m ("BA_motion"):

            1. PnP pose guess for the new frame (helperEstimateCameraPose)
            2. IMU preintegration between the previous view and this frame
               (helperExtractIMUDataBetweenViews)
            3. Motion-only Ceres BA refining this frame's
               [pose, velocity, bias] against the fixed previous state
               (helperBundleAdjustmentMotion)
            4. Write the refined pose/velocity/bias back and register
               valid landmark observations on this view.

        The sliding-window FactorGraph / optimize(fg, ...) full-window
        smoothing step is intentionally NOT invoked here yet -- that is
        the next phase, built on top of this one.
        """

        if self.view_set.num_views == 0:
            return

        # Previous view already committed to the view set/graph.
        prev_view_id = self.view_set.view_ids[-1]

        prev_velocity = self.sw_state.velocities.get(prev_view_id)
        if prev_velocity is None:
            # No velocity estimate yet for the previous view -- can't
            # form the IMU factor's previous-state anchor.
            print("[VIO] No velocity estimate for previous view; skipping BA_motion.")
            return

        # ---- 1. IMU preintegration between previous view and this frame --
        # Moved ahead of the vision steps: MATLAB always predicts `pp,pv`
        # from the IMU factor before attempting BA_motion, and every
        # vision-failure branch below now falls back to that IMU-only
        # prediction instead of leaving frameID absent from view_set
        # (see _predict_pose_from_imu docstring for why that matters).
        preint = self._build_single_imu_preintegration(prev_view_id, frameID, timestamp)

        if preint is None:
            # No IMU coverage at all for this interval -- there is
            # nothing (vision or inertial) to anchor a pose on. This
            # is the one case where the frame really cannot get a
            # pose; it stays absent from view_set.
            print("[VIO] Insufficient IMU coverage; skipping BA_motion.")
            return

        def commit_imu_fallback(reason):
            print(f"[VIO] {reason}; falling back to IMU-only prediction.")
            R_pred, t_pred, v_pred = self._predict_pose_from_imu(
                prev_view_id, prev_velocity, preint,
            )
            self.view_set.add_view(frameID, R_pred, t_pred, timestamp)
            self.sw_state.velocities[frameID] = v_pred
            # Bias carries forward unchanged -- no vision/BA update
            # available this frame to refine it.

        # ---- 2. PnP pose guess for the new frame ------------------------
        correspondences = find_pnp_correspondences(self.sw_state, frameID)

        if len(correspondences) < 6:
            commit_imu_fallback("Not enough PnP correspondences")
            return

        pnp_result = solve_pnp(correspondences, self.K)

        if pnp_result is None:
            commit_imu_fallback("PnP failed")
            return

        R_guess, C_guess, inliers = pnp_result

        if inliers is None or len(inliers) == 0:
            commit_imu_fallback("PnP found no inliers")
            return

        inlier_idx = inliers.flatten()
        xyz_pts = np.array([correspondences[i].xyz for i in inlier_idx])
        uv_pts = np.array([correspondences[i].uv for i in inlier_idx])
        point_ids = [correspondences[i].point_id for i in inlier_idx]

        prev_pose = self.view_set.get_pose(prev_view_id)
        prev_bias = (self.sw_state.gyroscope_bias, self.sw_state.accelerometer_bias)

        # Constant-velocity model for the initial guess -- BA_motion
        # refines it using the IMU factor + reprojection factors.
        velocity_guess = prev_velocity.copy()

        # ---- 3. Motion-only Ceres BA --------------------------------------
        result = bundle_adjustment_motion(
            xyz_tracked_in_current_view=xyz_pts,
            current_view_correspondences=uv_pts,
            intrinsics_K=self.K,
            image_size=self.img_frame.shape,
            current_view_pose_guess=(R_guess, C_guess),
            current_view_velocity_guess=velocity_guess,
            previous_view_pose=prev_pose,
            previous_view_velocity=prev_velocity,
            previous_view_bias=prev_bias,
            preintegrated_imu=preint,
            R_bs=self.T_BS[:3, :3],
            t_bs=self.T_BS[:3, 3],
            gravity=self.sw_state.gravity,
        )

        refined_pose, vel_refined, bias_refined, valid = result

        if refined_pose is None:
            commit_imu_fallback("BA_motion did not converge")
            return

        R_refined, C_refined = refined_pose

        # ---- 4. Write refined state back ------------------------------------
        self.view_set.add_view(frameID, R_refined, C_refined, timestamp)
        self.sw_state.velocities[frameID] = vel_refined
        self.sw_state.gyroscope_bias = bias_refined[0]
        self.sw_state.accelerometer_bias = bias_refined[1]

        for k, is_valid in enumerate(valid):
            if is_valid:
                landmark = self.sw_state.landmarks[point_ids[k]]
                landmark.add_observation(frameID, uv_pts[k])

        # NOTE: sliding-window FactorGraph build + optimize(fg, ...)
        # (full window smoothing over multiple keyframes/IMU factors)
        # intentionally stops here for now -- next phase.