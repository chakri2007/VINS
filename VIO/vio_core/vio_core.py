import cv2
import numpy as np

from vio_core.preprocess_image import preprocess_image
from vio_core.ransac import estimate_fundamental_matrix_ransac

from feature_manager.feature_extractor import FeatureExtractor
from memory_management.view_set import ViewSet
from memory_management.sliding_window import (
    SlidingWindowState,
    update_sliding_window,
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
from optimization.state_update import update_state_from_graph
from optimization.median_depth import normalize_map
from imu.vi_alignment import (
    initialize_visual_inertial_state,
)
from imu.preintegration import IMUPreintegrator


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

        self.frameID = 0

        self.graph_builder = GraphBuilder()

        self.bundle_adjustment = None


    def vio_loop(self, raw_img_frame, timestamp):

        self.frameID += 1

        # self.img_frame, self.K = preprocess_image(
        #     raw_img_frame,
        #     self.distortion_coeffs,
        #     self.K_raw,
        #     self.params,
        # 
        self.img_frame = raw_img_frame.copy()

        #
        # First frame
        #
        if self.isFirstFrame:
            self._init_first_frame(
                self.img_frame,
                self.frameID,
                timestamp,
            )
            return

        #
        # Common frontend
        #
        window_state = self.process_frontend(
            self.img_frame,
            self.frameID,
        )

        #
        # Phase selection
        #
        if not self.isMapInitialized:

            self.vio_initialization(
                window_state,
                self.frameID,
                timestamp,
            )

        elif not self.isVI_aligned:

            self.VI_alignment(
                window_state,
                self.frameID,
                timestamp,
            )

        else:

            self.visual_inertial_optimization(
                window_state,
                self.frameID,
            )

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

        # update_sliding_window first-frame branch just registers the view_id
        # and sets current_view_id = frameID.
        update_sliding_window(
            state               = self.sw_state,
            image_shape         = img_frame.shape,
            curr_points_tracked = features,
            valid_idx           = np.ones(num_pts, dtype=bool),
            view_id             = frameID,
            F_loop              = self.params['F_loop'],
            F_iterations        = self.params['F_Iterations'],
            F_confidence        = self.params['F_Confidence'],
            F_threshold         = self.params['F_Threshold'],
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


        # Pass full, unfiltered arrays — sliding_window does its own
        # v1 = valid_idx & ps_idx filtering internally.
        removed_frame_id, window_state = update_sliding_window(
            state               = self.sw_state,
            image_shape         = img_frame.shape,
            curr_points_tracked = tracked_points,
            valid_idx           = valid_idx,
            view_id             = frameID,
            F_loop              = self.params['F_loop'],
            F_iterations        = self.params['F_Iterations'],
            F_confidence        = self.params['F_Confidence'],
            F_threshold         = self.params['F_Threshold'],
            key_frame_parallax  = self.params['keyFrameParallax'],
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

        # ── track removed frame ids ───────────────────────────────────────
        if (removed_frame_id >= 0
                and len(self.sw_state.sliding_window_view_ids) > 0
                and removed_frame_id > self.sw_state.sliding_window_view_ids[0]):
            self.removed_frame_ids.append(removed_frame_id)

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

        self._prune_imu_buffer()

        return window_state

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
        print("running triangulation")

        #
        # Build factor graph from current sliding window
        #
        factor_graph = self.graph_builder.build(
            view_set=self.view_set,
            sw_state=self.sw_state,
            K=self.K,
        )

        factor_graph.print_summary()

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
                if not self.isVIO_initialized:

                    self.try_vi_alignment()

            self.bundle_adjustment.clear_fixed_poses()

    def try_vi_alignment(self):
        """
        Attempt linear visual-inertial alignment (metric scale, gravity
        direction, per-keyframe velocities, accelerometer bias) over
        the current sliding window.

        Mirrors the MATLAB reference's use of
        swIDs = getSlidingWindowIDs(fpManager); swIDs = swIDs(1:end-1);
        i.e. every *closed* keyframe interval in the window except the
        newest, still-open one — each interval needs a completed IMU
        preintegration between two confirmed keyframe timestamps.
        """

        sw_ids = list(self.sw_state.sliding_window_view_ids)

        # Need at least two closed intervals (3 keyframes) for the
        # alignment system in vi_alignment.py to be solvable at all
        # (3*N+7 unknowns from N keyframes; each pair contributes 6
        # equations).
        align_view_ids = sw_ids[:-1]

        if len(align_view_ids) < 3:
            return

        imu_preintegrations = self._build_imu_preintegrations(align_view_ids)

        if imu_preintegrations is None:
            # Missing IMU coverage for at least one interval -- can't
            # run alignment yet.
            return

        result = initialize_visual_inertial_state(
            view_set=self.view_set,
            sliding_window=self.sw_state,
            imu_preintegrations=imu_preintegrations,
            view_ids=align_view_ids,
            sensor_transform=self.T_BS,
            apply_scale_to_map=False,
        )

        if result.success:

            print("\n========== VI Alignment ==========")
            print("Scale:", result.scale)
            print("Gravity:", result.gravity)
            print("Accel Bias:", result.accel_bias)

            self.isVIO_initialized = True

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

    def run_pnp(self, frameID, timestamp):

        correspondences = find_pnp_correspondences(
            self.sw_state,
            frameID,
        )

        if len(correspondences) < 6:
            return False

        result = solve_pnp(
            correspondences,
            self.K,
        )

        if result is None:
            return False

        Rwc, C, inliers = result

        if inliers is None or len(inliers) == 0:
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
        print(f"Triangulation: {num_added} new landmarks added.")

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

        append_imu_measurement(
            self.sw_state,
            IMUMeasurement(
                timestamp=timestamp,
                accel=np.asarray(accel, dtype=np.float64),
                gyro=np.asarray(gyro, dtype=np.float64),
            ),
        )
    def visual_inertial_optimization(self, window_state, frameID):
        pass