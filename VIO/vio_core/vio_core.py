import cv2
import numpy as np

from vio_core.preprocess_image import preprocess_image
from vio_core.ransac import estimate_fundamental_matrix_ransac

from feature_manager.feature_extractor import FeatureExtractor
from memory_management.view_set import ViewSet
from memory_management.sliding_window import SlidingWindowState, update_sliding_window

from vio_core.triangulate import find_triangulation_candidates, triangulate_candidates, add_landmarks

from vio_core.reprojection import validate_landmarks

from vio_core.pnp import find_pnp_correspondences, PnPCorrespondence, solve_pnp


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
        }

        self.feature_extractor = FeatureExtractor()
        self.view_set          = ViewSet()
        self.sw_state          = SlidingWindowState(window_size=21)
        self.prev_img_frame    = None

        self.removed_frame_ids: list = []

        self.isFirstFrame      = True
        self.isMapInitialized  = False
        self.isVIO_initialized = False
        self.isVI_aligned      = False

        self.frameID = 0

    # ------------------------------------------------------------------ #
    #  Public entry points                                                 #
    # ------------------------------------------------------------------ #

    def vio_loop(self, raw_img_frame, img_frame_timestamp):
        self.frameID += 1
        self.img_frame, self.K = preprocess_image(
            raw_img_frame,
            self.distortion_coeffs,
            self.K_raw,
            self.params,
        )

        if not self.isVIO_initialized:
            self.vio_initialization(self.img_frame, img_frame_timestamp, self.frameID)
        elif not self.isVI_aligned:
            self.VI_alignment()
        else:
            self.visual_inertial_optimization()

    def process_frame_mono(self, raw_img_frame, img_frame_timestamp):
        self.vio_loop(raw_img_frame, img_frame_timestamp)
        return {
            'pose':   None,
            'tracks': self.get_active_tracks(),
            'K':      self.K,
            'D':      self.distortion_coeffs,
        }

    def get_active_tracks(self) -> dict:
        """Build track history for every point visible in the active sliding window."""
        tracks = {}
        for view_id in self.sw_state.sliding_window_view_ids:
            obs = self.sw_state.all_observations.get(view_id)
            ids = self.sw_state.all_ids.get(view_id)
            if obs is None or ids is None or len(obs) == 0:
                continue
            for point_id, (u, v) in zip(ids[:, 1], obs):
                tracks.setdefault(int(point_id), []).append(
                    (view_id, float(u), float(v))
                )
        return tracks

    # ------------------------------------------------------------------ #
    #  Phase 1 — Structure from Motion                                     #
    # ------------------------------------------------------------------ #

    def vio_initialization(self, img_frame, img_frame_timestamp, frameID):
        if self.isFirstFrame:
            self._init_first_frame(img_frame, frameID)
            return

        # ── Track from LAST STORED FRAME (not blindly frameID-1) ─────────
        # When a frame was replaced/evicted in the sliding window,
        # all_observations[frameID-1] may not exist.  Use current_view_id
        # which always points to the last frame that was actually stored.
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

        # ── branch on window state ────────────────────────────────────────
        if window_state['isFirstFewViews']:
            self.view_set.add_view(view_id=frameID, R=np.eye(3), t=np.zeros(3))

        elif window_state['isEnoughParallax']:
            success = self._initialise_map(frameID)
            if success:
                self.isMapInitialized  = True
                self.isVIO_initialized = True
                print(f"Map initialized")

    def _init_first_frame(self, img_frame, frameID):
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
        self.view_set.add_view(view_id=frameID, R=np.eye(3), t=np.zeros(3))
        self.isFirstFrame = False

    def _initialise_map(self, frameID) -> bool:
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

        self.view_set.add_view(view_id=frameID, R=R, t=t)
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



    def VI_alignment(self):
        candidates = find_triangulation_candidates(
                    self.sw_state,
                    self.view_set,
                )

        print(f"Found {len(candidates)} triangulation candidates")
        triangulated = triangulate_candidates(
            candidates,
            self.view_set,
            self.K,
        )

        print(f"Triangulated landmarks: {len(triangulated)}")

        num_added = add_landmarks(
            triangulated,
            self.sw_state,
        )

        print(
            f"Landmark database size: "
            f"{len(self.sw_state.landmarks)}"
        )

        validate_landmarks(
        self.sw_state,
        self.view_set,
        self.K,
    )
    
    correspondences = find_pnp_correspondences(
        self.sw_state,
        current_view_id=frameID,
    )

    Rwc, C, inliers = solve_pnp(
        correspondences,
        self.K,
    )


    def visual_inertial_optimization(self):
        pass