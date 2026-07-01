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
from optimization.graph_builder import GraphBuilder
from optimization.bundle_adjustment import BundleAdjuster
from optimization.state_update import update_state_from_graph
from optimization.median_depth import normalize_map


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

            'optimizationFrequency': 10,
            'initialOptimizationFrames': 250,
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

        self.graph_builder = GraphBuilder()

        self.bundle_adjustment = None


    def vio_loop(self, raw_img_frame, timestamp):

        self.frameID += 1

        self.img_frame, self.K = preprocess_image(
            raw_img_frame,
            self.distortion_coeffs,
            self.K_raw,
            self.params,
        )

        #
        # First frame
        #
        if self.isFirstFrame:
            self._init_first_frame(
                self.img_frame,
                self.frameID,
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
            )

        elif not self.isVI_aligned:

            self.VI_alignment(
                window_state,
                self.frameID,
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

        return window_state
    
    def vio_initialization(self, window_state, frameID):

        if window_state["isFirstFewViews"]:

            self.view_set.add_view(
                frameID,
                np.eye(3),
                np.zeros(3),
            )

            return

        if not window_state["isEnoughParallax"]:
            return

        success = self._initialise_map(frameID)

        if success:

            self.isMapInitialized = True

            print("Map initialized.")

    def VI_alignment(self, window_state, frameID):

        if not self.run_pnp(frameID):
            return
        

        new_points_added = self.run_triangulation()

        #
        # Build factor graph from current sliding window
        #
        factor_graph = self.graph_builder.build(
            view_set=self.view_set,
            sw_state=self.sw_state,
            K=self.K,
        )

        factor_graph.print_summary()

        self.bundle_adjustment = BundleAdjuster(factor_graph)

        if self.should_run_bundle_adjustment(frameID,new_points_added,):
            
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

            self.bundle_adjustment.clear_fixed_poses()


        # IMU alignment

    def run_pnp(self, frameID):

        correspondences = find_pnp_correspondences(
            self.sw_state,
            frameID,
        )

        if len(correspondences) < 6:
            return False

        Rwc, C, inliers = solve_pnp(
            correspondences,
            self.K,
        )

        self.view_set.add_view(
            frameID,
            Rwc,
            C,
        )

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

        return num_added > 0

    def should_run_bundle_adjustment(
        self,
        frameID,
        new_points_added,):

        if frameID < self.params["initialOptimizationFrames"]:
            return True

        if frameID % self.params["optimizationFrequency"] == 0:
            return True

        if new_points_added:
            return True

        return False
    
    def fix_bundle_adjustment_poses(self, window_state):

        sw_ids = list(self.sw_state.sliding_window_view_ids)

        self.bundle_adjustment.clear_fixed_poses()

        if window_state["isWindowFull"]:

            for view_id in sw_ids[:11]:
                self.bundle_adjustment.fix_pose(view_id)

        else:

            self.bundle_adjustment.fix_pose(sw_ids[0])

    def visual_inertial_optimization(self, window_state, frameID):
        pass