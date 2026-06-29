import cv2
import numpy as np

from vio_core.preprocess_image import preprocess_image
from vio_core.ransac import estimate_fundamental_matrix_ransac

from feature_manager.feature_extractor import FeatureExtractor
from memory_management.view_set import ViewSet
from memory_management.sliding_window import SlidingWindowState, update_sliding_window


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

        self.K = np.array([
            [self.intrinsics[0], 0,                  self.intrinsics[2]],
            [0,                  self.intrinsics[1], self.intrinsics[3]],
            [0,                  0,                  1],
        ], dtype=np.float64)

        # Processing parameters
        self.params = {
            'Equalize':          False,
            'Undistort':         True,
            'ClipLimit':         3.0 / 256,
            'NumTiles':          (8, 8),
            'F_loop':            5,
            'F_Iterations':      2000,
            'F_Confidence':      99,
            'F_Threshold':       4,
            'keyFrameParallax':  50,
        }

        self.feature_extractor = FeatureExtractor()
        self.view_set          = ViewSet()

        self.sw_state       = SlidingWindowState(window_size=21)
        self.prev_img_frame = None

        # IDs of views that were evicted from the sliding window.
        # MATLAB: removedFrameIDs — needed downstream for keyframe filtering.
        self.removed_frame_ids: list = []

        self.isFirstFrame       = True
        self.isMapInitialized   = False

        # Set True once Phase 1 completes so vio_loop advances to Phase 2/3.
        self.isVIO_initialized  = False
        self.isVI_aligned       = False

        self.frameID = 0

    # ------------------------------------------------------------------ #
    #  Public entry points                                                 #
    # ------------------------------------------------------------------ #

    def vio_loop(self, raw_img_frame, img_frame_timestamp):
        self.frameID += 1
        self.img_frame = preprocess_image(
            raw_img_frame,
            self.distortion_coeffs,
            self.K,
            self.params,
        )

        if not self.isVIO_initialized:
            self.vio_initialization(self.img_frame, img_frame_timestamp, self.frameID)
        elif not self.isVI_aligned:
            self.VI_alignment()
        else:
            self.visual_inertial_optimization()

    def process_frame_mono(self, raw_img_frame, img_frame_timestamp):
        """Entry point called by vio_subscriber's mono_image_callback.

        Returns
        -------
        dict with keys:
            'pose'   : None during Phase 1 (no usable pose until map init)
            'tracks' : dict[point_id] -> list[(frame_idx, u, v), ...]
            'K'      : (3,3) intrinsic matrix
            'D'      : distortion coefficients
        """
        self.vio_loop(raw_img_frame, img_frame_timestamp)

        return {
            'pose':   None,
            'tracks': self.get_active_tracks(),
            'K':      self.K,
            'D':      self.distortion_coeffs,
        }

    def get_active_tracks(self) -> dict:
        """Reconstruct per-point track history across the active sliding window.

        Returns
        -------
        dict[point_id] -> list of (frame_idx, u, v), oldest -> newest.
        """
        tracks = {}
        for view_id in self.sw_state.sliding_window_view_ids:
            observations = self.sw_state.all_observations.get(view_id)
            ids          = self.sw_state.all_ids.get(view_id)
            if observations is None or ids is None:
                continue
            point_ids = ids[:, 1]
            for point_id, (u, v) in zip(point_ids, observations):
                tracks.setdefault(int(point_id), []).append(
                    (view_id, float(u), float(v))
                )
        return tracks

    # ------------------------------------------------------------------ #
    #  Phase 1 — Structure from Motion (map initialisation)               #
    # ------------------------------------------------------------------ #

    def vio_initialization(self, img_frame, img_frame_timestamp, frameID):
        """MATLAB Phase 1 loop body, translated 1-to-1."""

        if self.isFirstFrame:
            self._init_first_frame(img_frame, frameID)
            return

        # ── subsequent frames ─────────────────────────────────────────

        prev_view_id  = frameID - 1
        prev_points   = self.sw_state.all_observations[prev_view_id]

        # Track features from previous frame into current frame (KLT)
        tracked_points, status = self.feature_extractor.track_features(
            self.prev_img_frame,
            img_frame,
            prev_points,
        )
        valid_idx = status.astype(bool)

        # Keep only tracker-valid points/ids before passing to sliding window.
        curr_points = tracked_points[valid_idx]
        curr_ids    = self.sw_state.all_ids[prev_view_id][valid_idx, 1]

        # Bump track-age for every survived point.
        for pid in curr_ids:
            self.sw_state.key_point_track_count[pid] = (
                self.sw_state.key_point_track_count.get(pid, 0) + 1
            )

        # Grid-based eviction of overcrowded cells.
        track_ages = np.array(
            [self.sw_state.key_point_track_count[pid] for pid in curr_ids]
        )
        evict_ids = self.feature_extractor.gridder.get_overcrowded_evictions(
            curr_points, curr_ids, track_ages
        )
        if len(evict_ids) > 0:
            keep_mask   = ~np.isin(curr_ids, evict_ids)
            curr_points = curr_points[keep_mask]
            curr_ids    = curr_ids[keep_mask]

        # RANSAC outlier rejection + sliding window bookkeeping.
        # update_sliding_window writes all_observations/all_ids[frameID].
        removed_frame_id, window_state = update_sliding_window(
            state                    = self.sw_state,
            image_shape              = img_frame.shape,
            curr_points_tracked      = curr_points,
            valid_idx                = np.ones(curr_points.shape[0], dtype=bool),
            view_id                  = frameID,
            F_loop                   = self.params['F_loop'],
            F_iterations             = self.params['F_Iterations'],
            F_confidence             = self.params['F_Confidence'],
            F_threshold              = self.params['F_Threshold'],
            key_frame_parallax       = self.params['keyFrameParallax'],
        )

        # Track evicted frame ids (used later for keyframe filtering).
        if (removed_frame_id >= 0 and
                len(self.sw_state.sliding_window_view_ids) > 0 and
                removed_frame_id > self.sw_state.sliding_window_view_ids[0]):
            self.removed_frame_ids.append(removed_frame_id)

        # Detect new features in empty grid cells and add to current view.
        post_ransac_points = self.sw_state.all_observations.get(frameID, np.empty((0, 2), dtype=np.float32))
        post_ransac_ids    = (
            self.sw_state.all_ids[frameID][:, 1]
            if frameID in self.sw_state.all_ids
            else np.empty(0, dtype=np.int64)
        )

        new_points = self.feature_extractor.extract_features_in_empty_cells(
            img_frame, post_ransac_points
        )
        if len(new_points) > 0:
            num_new  = new_points.shape[0]
            start_id = max(self.sw_state.key_point_track_count.keys(), default=0) + 1
            new_ids  = np.arange(start_id, start_id + num_new)
            for pid in new_ids:
                self.sw_state.key_point_track_count[pid] = 1

            post_ransac_points = np.vstack([post_ransac_points, new_points])
            post_ransac_ids    = np.concatenate([post_ransac_ids, new_ids])

            self.sw_state.all_observations[frameID] = post_ransac_points
            self.sw_state.all_ids[frameID] = np.column_stack(
                [np.full(post_ransac_ids.shape[0], frameID), post_ransac_ids]
            )
            self.sw_state.all_triangulated[frameID] = np.concatenate([
                self.sw_state.all_triangulated.get(frameID, np.zeros(0, dtype=bool)),
                np.zeros(num_new, dtype=bool),
            ])

        self.prev_img_frame = img_frame

        # ── branch on window state ────────────────────────────────────

        if window_state['isFirstFewViews']:
            # Add with identity pose — still building up the sliding window.
            self.view_set.add_view(view_id=frameID, R=np.eye(3), t=np.zeros(3))

        elif window_state['isEnoughParallax']:
            # Enough baseline between two keyframes: estimate relative pose
            # and finalise map initialisation.
            success = self._initialise_map(frameID)
            if success:
                self.isMapInitialized  = True
                self.isVIO_initialized = True   # advances vio_loop to Phase 2
                print(f"Map initialisation succeeded at frame {frameID}, Advancing to Phase 2.")

    # ------------------------------------------------------------------ #
    #  Phase 1 helpers                                                     #
    # ------------------------------------------------------------------ #

    def _init_first_frame(self, img_frame, frameID):
        """Initialise state for the very first frame."""
        current_features = self.feature_extractor.detect_initial_features(img_frame)
        num_pts   = current_features.shape[0]
        point_ids = np.arange(1, num_pts + 1)   # 1-based, matching MATLAB

        self.sw_state.current_view_id        = frameID
        self.sw_state.is_key_frame[frameID]  = True
        self.sw_state.all_observations[frameID] = current_features
        self.sw_state.all_ids[frameID] = np.column_stack(
            [np.full(num_pts, frameID), point_ids]
        )
        self.sw_state.all_triangulated[frameID] = np.zeros(num_pts, dtype=bool)
        for pid in point_ids:
            self.sw_state.key_point_track_count[pid] = 1

        # First call to update_sliding_window: just registers view_id,
        # returns removed_frame_id = -1 (nothing removed).
        update_sliding_window(
            state               = self.sw_state,
            image_shape         = img_frame.shape,
            curr_points_tracked = current_features,
            valid_idx           = np.ones(num_pts, dtype=bool),
            view_id             = frameID,
            F_loop              = self.params['F_loop'],
            F_iterations        = self.params['F_Iterations'],
            F_confidence        = self.params['F_Confidence'],
            F_threshold         = self.params['F_Threshold'],
            key_frame_parallax  = self.params['keyFrameParallax'],
        )

        self.prev_img_frame = img_frame
        self.first_img_frame = img_frame
        self.view_set.add_view(view_id=frameID, R=np.eye(3), t=np.zeros(3))
        self.isFirstFrame = False

    def _initialise_map(self, frameID) -> bool:
        """Estimate relative pose between last two keyframes and add the
        current view to the view set.

        Mirrors MATLAB:
            estimateFundamentalMatrix (x10 RANSAC) -> estrelpose -> addView
        Returns True on success, False if estimation failed.
        """
        sw_ids = self.sw_state.sliding_window_view_ids
        if len(sw_ids) < 2:
            return False

        id1, id2 = sw_ids[-2], sw_ids[-1]

        # Get 2-D correspondences between the two keyframes.
        ids1 = self.sw_state.all_ids.get(id1)
        ids2 = self.sw_state.all_ids.get(id2)
        if ids1 is None or ids2 is None or len(ids1) == 0 or len(ids2) == 0:
            return False

        _, ia, ib = np.intersect1d(ids1[:, 1], ids2[:, 1], return_indices=True)
        if len(ia) < 8:
            return False

        matches1 = self.sw_state.all_observations[id1][ia]
        matches2 = self.sw_state.all_observations[id2][ib]

        # Repeated RANSAC — keep the run with the most inliers.
        # MATLAB hardcodes 10 iterations here (different from F_loop=5
        # used inside updateSlidingWindow).
        best_F       = None
        best_inliers = None
        for _ in range(10):
            F, inliers = estimate_fundamental_matrix_ransac(
                matches1, matches2,
                num_trials    = self.params['F_Iterations'],
                confidence    = self.params['F_Confidence'],
                dist_threshold= self.params['F_Threshold'],
            )
            if F is None:
                continue
            if best_inliers is None or np.count_nonzero(inliers) > np.count_nonzero(best_inliers):
                best_F       = F
                best_inliers = inliers

        if best_F is None or np.count_nonzero(best_inliers) < 8:
            return False

        inlier_pts1 = matches1[best_inliers]
        inlier_pts2 = matches2[best_inliers]

        # Recover relative pose: F -> E -> (R, t) with cheirality check.
        # Mirrors MATLAB's estrelpose(F, intrinsics, pts1, pts2).
        R, t = self._estimate_relative_pose(best_F, inlier_pts1, inlier_pts2)
        if R is None:
            return False

        self.view_set.add_view(view_id=frameID, R=R, t=t)
        return True

    def _estimate_relative_pose(
        self,
        F: np.ndarray,
        pts1: np.ndarray,
        pts2: np.ndarray,
    ):
        """Decompose fundamental matrix into (R, t).

        Mirrors MATLAB's estrelpose(F, intrinsics, inlierPts1, inlierPts2):
            E = K^T F K
            recoverPose (cv2) — cheirality check selects the correct
            (R, t) from the four possible decompositions.

        Returns
        -------
        R : (3,3) rotation matrix  — camera-to-world (2nd frame w.r.t. 1st)
        t : (3,)  translation unit vector
        Both None if estimation fails.
        """
        E = self.K.T @ F @ self.K

        pts1_f = pts1.astype(np.float64)
        pts2_f = pts2.astype(np.float64)

        n_inliers, R, t, mask = cv2.recoverPose(E, pts1_f, pts2_f, self.K)

        if n_inliers < 8:
            return None, None

        # cv2.recoverPose returns world-to-camera (R_wc, t_wc).
        # MATLAB's estrelpose / rigidtform3d / AbsolutePose stores
        # camera-to-world, so we invert: R_cw = R_wc^T, t_cw = -R_wc^T t_wc.
        R_cw = R.T
        t_cw = -(R_cw @ t.ravel())

        return R_cw, t_cw

    # ------------------------------------------------------------------ #
    #  Phase 2 — Visual-Inertial alignment                                 #
    # ------------------------------------------------------------------ #

    def VI_alignment(self):
        pass

    # ------------------------------------------------------------------ #
    #  Phase 3 — Sliding-window VIO optimisation                          #
    # ------------------------------------------------------------------ #

    def visual_inertial_optimization(self):
        pass