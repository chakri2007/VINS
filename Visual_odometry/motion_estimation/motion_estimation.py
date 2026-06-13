import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional


# Type aliases
FrameIdx  = int
FeatId    = int
Pose      = Tuple[np.ndarray, np.ndarray]   # (R 3x3, t 3x1)
Obs       = Tuple[FrameIdx, float, float]    # (frame_idx, u, v)


class MotionEstimator:
    def __init__(
        self,
        K: np.ndarray,                  # (3,3) camera intrinsics
        dist_coeffs: np.ndarray,        # distortion coefficients
        sliding_window_size: int = 10,
        min_init_parallax_px: float = 20.0,   # minimum median parallax for init
        min_init_inliers: int = 50,
        min_pnp_inliers: int = 20,
        min_triangulation_angle_deg: float = 1.0,
    ):
        self.K           = K.astype(np.float64)
        self.dist_coeffs = dist_coeffs.astype(np.float64)
        self.window_size = sliding_window_size

        self.min_init_parallax_px        = min_init_parallax_px
        self.min_init_inliers            = min_init_inliers
        self.min_pnp_inliers             = min_pnp_inliers
        self.min_triangulation_angle_deg = min_triangulation_angle_deg

        self.is_initialized = False

        # Core state
        self.poses:      Dict[FrameIdx, Pose]           = {}
        self.landmarks:  Dict[int, np.ndarray]          = {}    # lm_id → X (3,)
        self.observations: Dict[int, List[Obs]]         = {}    # lm_id → obs list

        # Map feature_id (from database) → landmark_id (our internal id)
        # They start as the same but landmarks can outlive features.
        self._feat_to_lm: Dict[FeatId, int]  = {}
        self._next_lm_id = 0

        # Sliding window: ordered list of frame indices currently in the window
        self._window: List[FrameIdx] = []

        # Store undistorted points per frame for triangulation
        # { frame_idx: { feat_id: (u_undist, v_undist) } }
        self._undistorted: Dict[FrameIdx, Dict[FeatId, Tuple[float, float]]] = {}

    # ------------------------------------------------------------------
    # Main entry point — called from vo_pipeline.py every frame
    # ------------------------------------------------------------------

    def process_frame(
        self,
        frame_idx: FrameIdx,
        snapshot: Dict[FeatId, List[Obs]],   # from database.get_snapshot_for_estimator()
    ) -> Optional[Pose]:
        # Build per-frame undistorted observation lookup
        self._update_undistorted_cache(frame_idx, snapshot)

        if not self.is_initialized:
            return self._try_initialize(frame_idx, snapshot)
        else:
            return self._track_frame(frame_idx, snapshot)

    # ------------------------------------------------------------------
    # Initialization — two-frame Essential matrix bootstrap
    # ------------------------------------------------------------------

    def _try_initialize(
        self,
        curr_frame_idx: FrameIdx,
        snapshot: Dict[FeatId, List[Obs]],
    ) -> Optional[Pose]:
        """
        Try to initialize from Frame 0 and the current frame.
        Succeeds when median parallax exceeds threshold and enough inliers.
        """
        # Find features observed in both frame 0 and current frame.
        # Frame 0 is the first frame that ever appeared in the snapshot.
        ref_frame_idx = self._get_reference_frame_idx(snapshot)
        if ref_frame_idx is None or ref_frame_idx == curr_frame_idx:
            return None

        pts_ref, pts_curr, common_feat_ids = self._get_common_observations(
            snapshot, ref_frame_idx, curr_frame_idx
        )

        if len(pts_ref) < 8:
            return None

        # Check parallax
        parallax = self._median_parallax(pts_ref, pts_curr)
        if parallax < self.min_init_parallax_px:
            return None  # not enough motion yet — keep waiting

        # Undistort both sets
        pts_ref_u  = self._undistort_points(pts_ref)
        pts_curr_u = self._undistort_points(pts_curr)

        # Essential matrix with RANSAC
        E, inlier_mask = cv2.findEssentialMat(
            pts_ref_u, pts_curr_u,
            self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )

        if E is None:
            return None

        inliers = inlier_mask.ravel().astype(bool)
        if inliers.sum() < self.min_init_inliers:
            return None

        # Recover relative pose
        _, R01, t01, pose_mask = cv2.recoverPose(
            E,
            pts_ref_u[inliers],
            pts_curr_u[inliers],
            self.K,
        )
        # t01 is unit length — scale is arbitrary

        # Store Frame 0 and Frame 1 poses
        R0 = np.eye(3,    dtype=np.float64)
        t0 = np.zeros((3, 1), dtype=np.float64)
        self.poses[ref_frame_idx]  = (R0,  t0)
        self.poses[curr_frame_idx] = (R01, t01)
        self._window = [ref_frame_idx, curr_frame_idx]

        # Triangulate initial landmarks
        inlier_feat_ids = [common_feat_ids[i] for i in range(len(common_feat_ids)) if inliers[i]]
        pts_ref_in  = pts_ref_u[inliers]
        pts_curr_in = pts_curr_u[inliers]

        self._triangulate_and_store(
            ref_frame_idx,  R0,  t0,  pts_ref_in,
            curr_frame_idx, R01, t01, pts_curr_in,
            inlier_feat_ids,
            snapshot,
        )

        self.is_initialized = True
        print(f"[Estimator] Initialized: frames {ref_frame_idx}↔{curr_frame_idx}, "
              f"parallax={parallax:.1f}px, "
              f"inliers={inliers.sum()}, "
              f"landmarks={len(self.landmarks)}")

        return self.poses[curr_frame_idx]

    # ------------------------------------------------------------------
    # Per-frame tracking — PnP + new landmark triangulation
    # ------------------------------------------------------------------

    def _track_frame(
        self,
        curr_frame_idx: FrameIdx,
        snapshot: Dict[FeatId, List[Obs]],
    ) -> Optional[Pose]:
        """
        Estimate pose of curr_frame_idx via PnP against known landmarks,
        then triangulate any new landmarks visible from this frame.
        """
        # --- Step 1: collect 3D-2D correspondences for PnP ---
        pts3d = []
        pts2d = []
        lm_ids_used = []

        for feat_id, obs_list in snapshot.items():
            # Check if this feature has a known 3D landmark
            lm_id = self._feat_to_lm.get(feat_id)
            if lm_id is None or lm_id not in self.landmarks:
                continue

            # Find the observation in the current frame
            curr_obs = self._get_obs_in_frame(obs_list, curr_frame_idx)
            if curr_obs is None:
                continue

            pts3d.append(self.landmarks[lm_id])
            pts2d.append([curr_obs[1], curr_obs[2]])
            lm_ids_used.append(lm_id)

        if len(pts3d) < 6:
            print(f"[Estimator] Frame {curr_frame_idx}: only {len(pts3d)} 3D-2D matches, skipping PnP")
            return None

        pts3d = np.array(pts3d, dtype=np.float64)
        pts2d = np.array(pts2d, dtype=np.float64)

        # --- Step 2: solvePnPRansac ---
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d, pts2d,
            self.K, self.dist_coeffs,
            iterationsCount=200,
            reprojectionError=4.0,
            confidence=0.999,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success or inliers is None or len(inliers) < self.min_pnp_inliers:
            print(f"[Estimator] Frame {curr_frame_idx}: PnP failed "
                  f"(success={success}, inliers={len(inliers) if inliers is not None else 0})")
            return None

        R_curr, _ = cv2.Rodrigues(rvec)
        t_curr    = tvec  # (3,1)

        self.poses[curr_frame_idx] = (R_curr, t_curr)
        self._window.append(curr_frame_idx)

        # --- Step 3: triangulate new landmarks ---
        # Use the previous frame in the window as the reference for triangulation.
        # Using the immediately prior frame maximises baseline for new features.
        if len(self._window) >= 2:
            ref_frame_idx = self._window[-2]
            R_ref, t_ref  = self.poses[ref_frame_idx]
            self._triangulate_new_features(
                ref_frame_idx, R_ref, t_ref,
                curr_frame_idx, R_curr, t_curr,
                snapshot,
            )

        # --- Step 4: slide the window ---
        self._slide_window()

        n_lm = len(self.landmarks)
        print(f"[Estimator] Frame {curr_frame_idx}: PnP inliers={len(inliers)}, landmarks={n_lm}")

        return self.poses[curr_frame_idx]

    # ------------------------------------------------------------------
    # Triangulation
    # ------------------------------------------------------------------

    def _triangulate_and_store(
        self,
        frame_a_idx: FrameIdx, Ra: np.ndarray, ta: np.ndarray, pts_a: np.ndarray,
        frame_b_idx: FrameIdx, Rb: np.ndarray, tb: np.ndarray, pts_b: np.ndarray,
        feat_ids: List[FeatId],
        snapshot: Dict[FeatId, List[Obs]],
    ):
        """
        Triangulate a batch of point correspondences and store them as landmarks.
        pts_a, pts_b are (N,2) undistorted normalized or pixel coords.
        """
        Pa = self.K @ np.hstack([Ra, ta])   # (3,4)
        Pb = self.K @ np.hstack([Rb, tb])   # (3,4)

        pts_a_T = pts_a.T.astype(np.float32)  # (2, N)
        pts_b_T = pts_b.T.astype(np.float32)

        pts4d = cv2.triangulatePoints(Pa.astype(np.float32),
                                       Pb.astype(np.float32),
                                       pts_a_T, pts_b_T)  # (4, N)

        # Convert homogeneous → 3D, filter invalid points
        w = pts4d[3, :]
        valid = np.abs(w) > 1e-6
        pts3d = (pts4d[:3, :] / w[np.newaxis, :]).T  # (N, 3)

        for i, feat_id in enumerate(feat_ids):
            if not valid[i]:
                continue

            X = pts3d[i]

            # Cheirality: point must be in front of both cameras
            X_a = Ra @ X + ta.ravel()
            X_b = Rb @ X + tb.ravel()
            if X_a[2] <= 0 or X_b[2] <= 0:
                continue

            lm_id = self._next_lm_id
            self._next_lm_id += 1

            self.landmarks[lm_id]     = X
            self._feat_to_lm[feat_id] = lm_id

            # Store all available observations for this landmark
            if feat_id in snapshot:
                self.observations[lm_id] = list(snapshot[feat_id])

    def _triangulate_new_features(
        self,
        frame_a_idx: FrameIdx, Ra: np.ndarray, ta: np.ndarray,
        frame_b_idx: FrameIdx, Rb: np.ndarray, tb: np.ndarray,
        snapshot: Dict[FeatId, List[Obs]],
    ):
        """
        Find features visible in both frame_a and frame_b but not yet
        triangulated, and triangulate them.
        """
        pts_a_list, pts_b_list, feat_ids = [], [], []

        for feat_id, obs_list in snapshot.items():
            if feat_id in self._feat_to_lm:
                continue  # already has a landmark

            obs_a = self._get_obs_in_frame(obs_list, frame_a_idx)
            obs_b = self._get_obs_in_frame(obs_list, frame_b_idx)
            if obs_a is None or obs_b is None:
                continue

            pts_a_list.append([obs_a[1], obs_a[2]])
            pts_b_list.append([obs_b[1], obs_b[2]])
            feat_ids.append(feat_id)

        if len(feat_ids) == 0:
            return

        pts_a = self._undistort_points(np.array(pts_a_list, dtype=np.float32))
        pts_b = self._undistort_points(np.array(pts_b_list, dtype=np.float32))

        self._triangulate_and_store(
            frame_a_idx, Ra, ta, pts_a,
            frame_b_idx, Rb, tb, pts_b,
            feat_ids, snapshot,
        )

    # ------------------------------------------------------------------
    # Sliding window management
    # ------------------------------------------------------------------

    def _slide_window(self):
        """
        Drop the oldest frame if window exceeds size limit.
        Landmarks only observed in the dropped frame are removed.
        """
        while len(self._window) > self.window_size:
            dropped_frame = self._window.pop(0)
            self.poses.pop(dropped_frame, None)
            self._undistorted.pop(dropped_frame, None)

            # Remove landmarks with no observations in remaining window frames
            remaining_frames = set(self._window)
            lm_ids_to_drop = []

            for lm_id, obs_list in self.observations.items():
                still_observed = any(o[0] in remaining_frames for o in obs_list)
                if not still_observed:
                    lm_ids_to_drop.append(lm_id)

            for lm_id in lm_ids_to_drop:
                self.landmarks.pop(lm_id, None)
                self.observations.pop(lm_id, None)

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------

    def _get_reference_frame_idx(self, snapshot) -> Optional[FrameIdx]:
        """Return the oldest frame index seen across all observations."""
        oldest = None
        for obs_list in snapshot.values():
            for (fidx, u, v) in obs_list:
                if oldest is None or fidx < oldest:
                    oldest = fidx
        return oldest

    def _get_common_observations(
        self,
        snapshot: Dict[FeatId, List[Obs]],
        frame_a: FrameIdx,
        frame_b: FrameIdx,
    ) -> Tuple[np.ndarray, np.ndarray, List[FeatId]]:
        """
        Return (pts_a, pts_b, feat_ids) for features observed in both frames.
        """
        pts_a, pts_b, feat_ids = [], [], []

        for feat_id, obs_list in snapshot.items():
            obs_a = self._get_obs_in_frame(obs_list, frame_a)
            obs_b = self._get_obs_in_frame(obs_list, frame_b)
            if obs_a is None or obs_b is None:
                continue
            pts_a.append([obs_a[1], obs_a[2]])
            pts_b.append([obs_b[1], obs_b[2]])
            feat_ids.append(feat_id)

        return (np.array(pts_a, dtype=np.float32),
                np.array(pts_b, dtype=np.float32),
                feat_ids)

    @staticmethod
    def _get_obs_in_frame(
        obs_list: List[Obs],
        frame_idx: FrameIdx,
    ) -> Optional[Obs]:
        """Find the observation tuple for a specific frame index."""
        for obs in obs_list:
            if obs[0] == frame_idx:
                return obs
        return None

    def _undistort_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Undistort (N,2) pixel points using camera intrinsics and distortion.
        Returns (N,2) undistorted pixel coordinates.
        """
        if len(pts) == 0:
            return pts
        pts_reshaped = pts.reshape(-1, 1, 2).astype(np.float64)
        undist = cv2.undistortPoints(pts_reshaped, self.K, self.dist_coeffs, P=self.K)
        return undist.reshape(-1, 2).astype(np.float64)

    def _update_undistorted_cache(
        self,
        frame_idx: FrameIdx,
        snapshot: Dict[FeatId, List[Obs]],
    ):
        """Cache undistorted coordinates for the current frame's observations."""
        if frame_idx in self._undistorted:
            return
        frame_obs = {}
        for feat_id, obs_list in snapshot.items():
            obs = self._get_obs_in_frame(obs_list, frame_idx)
            if obs is not None:
                pts_u = self._undistort_points(
                    np.array([[obs[1], obs[2]]], dtype=np.float32)
                )
                frame_obs[feat_id] = (pts_u[0, 0], pts_u[0, 1])
        self._undistorted[frame_idx] = frame_obs

    @staticmethod
    def _median_parallax(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
        """Median pixel displacement between two point sets."""
        if len(pts_a) == 0:
            return 0.0
        return float(np.median(np.linalg.norm(pts_b - pts_a, axis=1)))

    # ------------------------------------------------------------------
    # Query interface for IMU scale estimator
    # ------------------------------------------------------------------

    def get_visual_poses(self) -> Dict[FrameIdx, Pose]:
        """
        Return all poses in the current sliding window.
        { frame_idx: (R, t) } — t is up-to-scale.
        Feed these into solve_scale_gravity_velocity().
        """
        return {k: self.poses[k] for k in self._window if k in self.poses}

    def get_landmarks(self) -> Dict[int, np.ndarray]:
        return dict(self.landmarks)