"""
feature_point_manager.py

Python port of MATLAB's helperFeaturePointManager.

Original MATLAB helper:
    - detects new key points whenever the number of tracked points drops
      below a threshold (params['numTrackedThresh'])
    - assigns a unique id to each key point, stores key-point tracks /
      2D-2D correspondences
    - triangulates new 3D points from tracks whenever the number of
      tracked 3D points in the current frame drops below a threshold
      (triangulatedThreshold)
    - stores 3D points and 3D-2D correspondences
    - manages a sliding window of key frames

NOTE on indexing: MATLAB is 1-based; this port is 0-based throughout
(view ids, point ids, sliding-window indices, array indices). Anywhere
the MATLAB code did "+1 / -1" bookkeeping purely because of 1-based
indexing, that bookkeeping has been removed here. Logic, thresholds and
control flow are otherwise preserved as closely as possible.

External MATLAB CV functions and their Python/OpenCV equivalents used
here:
    detectMinEigenFeatures      -> cv2.goodFeaturesToTrack (Shi-Thomasi)
    estimateFundamentalMatrix   -> cv2.findFundamentalMat (RANSAC)
    triangulate + cameraProjection -> cv2.triangulatePoints with K @ [R|t]
    selectUniform                -> grid-bucketed uniform sampling (custom)
    viewSet / poses(vSet, id)    -> ViewSet helper class below
"""

from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict, Tuple


# --------------------------------------------------------------------------
# Minimal stand-in for MATLAB's imageviewset, just enough to support
# poses(vSet, viewId) -> R, t  and  cameraProjection(intrinsics, pose)
# --------------------------------------------------------------------------
class ViewSet:
    """Minimal viewset: stores absolute pose (R, t) per view id.

    Pose convention: a 3D point in world coordinates X_w is mapped to the
    camera frame by X_c = R @ X_w + t (i.e. R, t is the world-to-camera /
    "extrinsic" transform, matching MATLAB's pose2extr output).
    """

    def __init__(self):
        self._poses: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def add_view(self, view_id: int, R: np.ndarray, t: np.ndarray) -> None:
        self._poses[view_id] = (np.asarray(R, dtype=np.float64),
                                 np.asarray(t, dtype=np.float64).reshape(3, 1))

    def pose(self, view_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (R, t) world-to-camera extrinsics for the given view id."""
        return self._poses[view_id]


def camera_projection(intrinsics: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Equivalent of MATLAB's cameraProjection(intrinsics, extrinsics).

    intrinsics: 3x3 camera matrix K
    R, t: world-to-camera extrinsics
    Returns the 3x4 camera projection matrix P = K @ [R | t]
    """
    Rt = np.hstack([R, t.reshape(3, 1)])
    return intrinsics @ Rt


# --------------------------------------------------------------------------
# Default Shi-Thomasi-equivalent detector, matching
#   detectMinEigenFeatures(grayImage, "MinQuality", 0.01, "FilterSize", 3)
# --------------------------------------------------------------------------
def default_detector(gray_image: np.ndarray, max_corners: int = 0) -> np.ndarray:
    """Returns an (N, 2) array of (x, y) corner locations, float64.

    max_corners <= 0 means "no limit" (cv2 convention), matching the
    MATLAB call which did not cap the detector's own output (capping is
    done later by helper_select_new_key_points_uniformly).
    """
    corners = cv2.goodFeaturesToTrack(
        gray_image,
        maxCorners=max_corners if max_corners > 0 else 100000,
        qualityLevel=0.01,
        minDistance=3,
        useHarrisDetector=False,
    )
    if corners is None:
        return np.zeros((0, 2), dtype=np.float64)
    return corners.reshape(-1, 2).astype(np.float64)


@dataclass
class WindowState:
    isEnoughParallax: bool = False
    isWindowFull: bool = False
    isFirstFewViews: bool = False


class FeaturePointManager:
    """Python port of helperFeaturePointManager.

    All view ids, point ids and sliding-window indices are 0-based.
    """

    def __init__(self, intrinsics: np.ndarray, params: dict,
                 max_frames: int = 5000, max_landmarks: int = 5000):
        # --- core config -----------------------------------------------
        self.intrinsics = np.asarray(intrinsics, dtype=np.float64)
        self.params = dict(params)

        # --- bookkeeping (0-based equivalents of MATLAB's 1-based) -----
        self.unique_key_point_count = 0          # MATLAB: 1
        self.AllObservations: List[Optional[np.ndarray]] = [None] * max_frames
        self.AllIds: List[Optional[np.ndarray]] = [None] * max_frames
        # AllTriangulated{i} : bool column vector per frame
        self.AllTriangulated: List[np.ndarray] = [np.zeros((0,), dtype=bool)
                                                    for _ in range(max_landmarks)]
        self.isTriangulated = np.zeros((max_landmarks,), dtype=bool)

        self.current_view_id = -1                # MATLAB: -1 (sentinel, kept as-is)
        self.new_point_ids = np.zeros((0,), dtype=np.int64)
        self.xyz_points = np.zeros((0, 3), dtype=np.float64)
        self.xyz_start_view = np.zeros((0,), dtype=np.int64)
        self.last_new_point_view_id = -1

        self.current_sliding_window_index = -1    # MATLAB: 0 -> "no views yet"
        # (0-based: -1 means empty; first inserted view goes to index 0)
        self.sliding_window_view_ids = np.zeros((self.params["windowSize"],), dtype=np.int64)

        self.xyz_val_ids = np.zeros((0,), dtype=np.int64)

        self.no_movement_at_start = True
        self.initial_mapping_successful = False

        self.detector_func: Callable[[np.ndarray], np.ndarray] = default_detector

        self.triangulated_threshold = 60

        self.window_state = WindowState()

        self.key_point_track_count = np.zeros((0,), dtype=np.int64)

        self.is_key_frame = np.zeros((max_frames,), dtype=bool)

    # ----------------------------------------------------------------
    def update_sliding_window(self, I: np.ndarray, curr_points_tracked: np.ndarray,
                               valid_idx: np.ndarray, view_id: int
                               ) -> Tuple[int, WindowState]:
        """Update the sliding window after tracking the latest view.

        Returns (rm_f, window_state):
            rm_f == -1  : first frame ever, accepted unconditionally
            rm_f == -2  : frame accepted, window grew (no removal)
            rm_f == -3  : frame accepted during "first few views" phase
            rm_f >= 0   : view id that was removed from the window
        """
        window_state = WindowState(**self.window_state.__dict__)

        if self.current_sliding_window_index == -1:
            # Very first frame. Accept it right away.
            self.current_sliding_window_index += 1
            rm_f = -1
            self.sliding_window_view_ids[self.current_sliding_window_index] = view_id
            return rm_f, window_state

        # find false tracks / outlier matches to discard them
        ps_idx = helper_within_image(curr_points_tracked, I.shape[:2])
        v1 = valid_idx & ps_idx

        inl_f = np.zeros((0,), dtype=bool)
        prev_view = self.AllObservations[view_id - 1]
        for _ in range(self.params["F_loop"]):
            _, inl_ff = estimate_fundamental_matrix_ransac(
                prev_view[v1, :], curr_points_tracked[v1, :],
                num_trials=self.params["F_Iterations"],
                confidence=self.params["F_Confidence"],
                distance_threshold=self.params["F_Threshold"],
            )
            if inl_ff.sum() > inl_f.sum():
                inl_f = inl_ff

        inl_ff_full = np.zeros(v1.shape, dtype=bool)
        inl_ff_full[v1] = inl_f
        v1 = v1 & inl_ff_full

        # update feature tracks
        self.AllObservations[view_id] = curr_points_tracked[v1, :]
        prev_idx_for_tri = max(0, view_id - 1)
        self.AllTriangulated[view_id] = self.AllTriangulated[prev_idx_for_tri][v1]
        p_ids = self.AllIds[prev_idx_for_tri][v1, 1]
        self.key_point_track_count[p_ids] += 1
        self.AllIds[view_id] = np.column_stack(
            [np.full(p_ids.shape, view_id, dtype=np.int64), p_ids]
        )
        self.current_view_id = view_id

        # remove 1 frame if the sliding window is full to accommodate
        # the current frame.
        rm_f = self.sliding_window_view_ids[0]
        no_move_window = 0.5  # between 0 and 1.
        window_size = self.params["windowSize"]

        cond_a = ((self.current_sliding_window_index < int(np.floor(window_size * no_move_window)) - 1
                   and self.no_movement_at_start)
                  or (self.current_sliding_window_index < 1))
        # NOTE: see indexing remark below cond_a derivation.

        if cond_a:
            # accept a few very first frames without extra processing if
            # there is no movement at the start (helps bias estimation).
            threshold_idx = int(np.floor(window_size * no_move_window)) - 2
            is_boundary = ((self.current_sliding_window_index == threshold_idx
                             and self.no_movement_at_start)
                            or (not self.no_movement_at_start
                                and self.current_sliding_window_index == 0))
            if is_boundary:
                ids_a = self.AllIds[self.sliding_window_view_ids[self.current_sliding_window_index]][:, 1]
                ids_b = self.AllIds[view_id][:, 1]
                _, ia, ib = intersect_1d(ids_a, ids_b)
                m1 = self.AllObservations[self.sliding_window_view_ids[self.current_sliding_window_index]][ia, :]
                m2 = self.AllObservations[view_id][ib, :]
                _, is_kf = helper_quick_check_parallax(m1, m2, self.params["keyFrameParallax"])
                if is_kf:
                    self.is_key_frame[view_id] = True
                    self.current_sliding_window_index += 1
                    self.sliding_window_view_ids[self.current_sliding_window_index] = view_id
                    rm_f = -2
                    window_state.isEnoughParallax = True
                else:
                    rm_f = view_id
            else:
                self.current_sliding_window_index += 1
                self.sliding_window_view_ids[self.current_sliding_window_index] = view_id
                rm_f = -3
                window_state.isFirstFewViews = True

        elif ((int(np.floor(window_size * no_move_window)) - 1 <= self.current_sliding_window_index < window_size - 1
               and self.no_movement_at_start)
              or (self.current_sliding_window_index < window_size - 1 and not self.no_movement_at_start)):
            # accept current frame, remove last frame if not enough
            # parallax vs. the last key frame.
            ids_a = self.AllIds[self.sliding_window_view_ids[self.current_sliding_window_index - 1]][:, 1]
            ids_b = self.AllIds[self.sliding_window_view_ids[self.current_sliding_window_index]][:, 1]
            _, ia, ib = intersect_1d(ids_a, ids_b)
            m1 = self.AllObservations[self.sliding_window_view_ids[self.current_sliding_window_index - 1]][ia, :]
            m2 = self.AllObservations[self.sliding_window_view_ids[self.current_sliding_window_index]][ib, :]
            _, is_kf = helper_quick_check_parallax(m1, m2, self.params["keyFrameParallax"])

            if is_kf or self.is_key_frame[self.sliding_window_view_ids[self.current_sliding_window_index]]:
                self.is_key_frame[self.sliding_window_view_ids[self.current_sliding_window_index]] = True
                self.current_sliding_window_index += 1
                self.sliding_window_view_ids[self.current_sliding_window_index] = view_id
                rm_f = -2
                window_state.isEnoughParallax = True
            else:
                rm_f = self.sliding_window_view_ids[self.current_sliding_window_index]
                self.sliding_window_view_ids[self.current_sliding_window_index] = view_id

        else:
            # window is full
            window_state.isWindowFull = True
            ids_a = self.AllIds[self.sliding_window_view_ids[-2]][:, 1]
            ids_b = self.AllIds[self.sliding_window_view_ids[-1]][:, 1]
            _, ia, ib = intersect_1d(ids_a, ids_b)
            m1 = self.AllObservations[self.sliding_window_view_ids[-2]][ia, :]
            m2 = self.AllObservations[self.sliding_window_view_ids[-1]][ib, :]
            _, is_kf = helper_quick_check_parallax(m1, m2, self.params["keyFrameParallax"])

            if (not is_kf) or (not self.is_key_frame[self.sliding_window_view_ids[-1]]):
                rm_f = self.sliding_window_view_ids[-1]
                self.sliding_window_view_ids[-1] = view_id
            else:
                self.sliding_window_view_ids = np.append(self.sliding_window_view_ids[1:], view_id)
                window_state.isEnoughParallax = True
                self.is_key_frame[self.sliding_window_view_ids[-2]] = True

        return int(rm_f), window_state

    # ----------------------------------------------------------------
    def create_new_feature_points(self, I: np.ndarray) -> Optional[np.ndarray]:
        """Create new feature points; return all points (tracked + new)
        in the current frame. Returns None if no points exist yet and
        none were created (mirrors MATLAB leaving currPoints undefined
        in that edge case -> here returns None instead).
        """
        if self.current_view_id < 0:
            self.current_view_id = 0
            self.is_key_frame[0] = True

        curr_points_tracked = self.AllObservations[self.current_view_id]
        if curr_points_tracked is None:
            curr_points_tracked = np.zeros((0, 2), dtype=np.float64)

        ids_here = self.AllIds[self.current_view_id]
        num_tracked_triangulated = 0
        if ids_here is not None and ids_here.shape[0] > 0:
            num_tracked_triangulated = int(self.isTriangulated[ids_here[:, 1]].sum())

        need_new_points = (
            curr_points_tracked.shape[0] < self.params["numTrackedThresh"]
            or (num_tracked_triangulated < self.triangulated_threshold
                and self.initial_mapping_successful
                and (self.params["maxPointsToTrack"] - curr_points_tracked.shape[0] > 4))
        )

        curr_points = curr_points_tracked
        if need_new_points:
            cp = self.detector_func(I)
            cp = helper_select_new_key_points_uniformly(
                curr_points_tracked, cp, 30, I.shape[:2], self.params["maxPointsToTrack"]
            )

            n_new = cp.shape[0]
            new_point_unique_ids = self.unique_key_point_count + np.arange(n_new, dtype=np.int64)
            self.unique_key_point_count += n_new

            curr_points = np.vstack([curr_points_tracked, cp.astype(np.float64)]) \
                if curr_points_tracked.shape[0] > 0 else cp.astype(np.float64)

            self.AllObservations[self.current_view_id] = curr_points

            new_ids_block = np.column_stack(
                [np.full((n_new,), self.current_view_id, dtype=np.int64), new_point_unique_ids]
            )
            if ids_here is not None and ids_here.shape[0] > 0:
                self.AllIds[self.current_view_id] = np.vstack([ids_here, new_ids_block])
            else:
                self.AllIds[self.current_view_id] = new_ids_block

            new_tri_block = np.zeros((n_new,), dtype=bool)
            existing_tri = self.AllTriangulated[self.current_view_id]
            self.AllTriangulated[self.current_view_id] = (
                np.concatenate([existing_tri, new_tri_block])
                if existing_tri.shape[0] > 0 else new_tri_block
            )

            if self.new_point_ids.shape[0] == 0:
                self.last_new_point_view_id = self.current_view_id

            self.new_point_ids = np.concatenate([self.new_point_ids, new_point_unique_ids])
            self.key_point_track_count = np.concatenate(
                [self.key_point_track_count, np.ones((n_new,), dtype=np.int64)]
            )
            self.xyz_start_view = np.concatenate(
                [self.xyz_start_view, np.full((n_new,), self.current_view_id, dtype=np.int64)]
            )
            self.is_key_frame[self.current_view_id] = True

        return curr_points

    # ----------------------------------------------------------------
    def triangulate_new_3d_points(self, vset: ViewSet
                                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """Triangulate new 3D points from 2D-2D correspondences between
        the last two key frames in the sliding window.

        Returns: (new_xyz, new_xyz_unique_ids, all_new_point_views, all_new_observations)
        """
        good_n_tr_idx = self.key_point_track_count[self.new_point_ids] > 1
        all_new_point_views: List[int] = []
        all_new_observations: List[np.ndarray] = []

        if good_n_tr_idx.size == 0 or not np.any(good_n_tr_idx):
            self.new_point_ids = np.zeros((0,), dtype=np.int64)
            return (np.zeros((0, 3)), np.zeros((0,), dtype=np.int64),
                    np.array(all_new_point_views), all_new_observations)

        new_xyz_unique_ids = self.new_point_ids[good_n_tr_idx]

        last_key_frame_id1 = self.sliding_window_view_ids[self.current_sliding_window_index - 1]
        last_key_frame_id2 = self.sliding_window_view_ids[self.current_sliding_window_index]

        ids_o = self.AllIds[last_key_frame_id1][:, 1]
        tr_ids_o, p_idx_o, p_tr_idxo = intersect_1d(ids_o, new_xyz_unique_ids)
        tr_new = np.ones((new_xyz_unique_ids.shape[0],), dtype=bool)
        tr_new[p_tr_idxo] = False
        t_v = last_key_frame_id1

        ids_b = self.AllIds[last_key_frame_id2][:, 1]
        tr_ids_pco, p_idxp_o, p_idxpc_o = intersect_1d(ids_b, tr_ids_o)

        m1 = self.AllObservations[last_key_frame_id1][p_idx_o[p_idxpc_o], :]
        m2 = self.AllObservations[last_key_frame_id2][p_idxp_o, :]

        _, is_en = helper_quick_check_parallax(m1, m2, self.params["triangulateParallax"])

        cur_ids = self.AllIds[self.current_view_id]
        num_tracked_triangulated = (
            int(self.isTriangulated[cur_ids[:, 1]].sum()) if cur_ids is not None and cur_ids.shape[0] else 0
        )

        if is_en or (num_tracked_triangulated < self.triangulated_threshold
                      and self.initial_mapping_successful and m1.shape[0] > 0):
            R1, t1 = vset.pose(t_v)
            R2, t2 = vset.pose(last_key_frame_id2)
            cam_matrix1 = camera_projection(self.intrinsics, R1, t1)
            cam_matrix2 = camera_projection(self.intrinsics, R2, t2)

            new_xyz, is_in_front = triangulate_points(m1, m2, cam_matrix1, cam_matrix2)

            v_i = is_in_front
            new_xyz = new_xyz[v_i, :]

            tr_ids_pc1 = tr_ids_pco[v_i]
            self.isTriangulated[tr_ids_pc1] = True

            _, p_ic, p_ic1 = intersect_1d(cur_ids[:, 1], tr_ids_pco)
            p_idxpci = np.zeros((cur_ids.shape[0],), dtype=bool)
            p_idxpci[p_ic] = v_i[p_ic1]
            self.AllTriangulated[self.current_view_id][p_idxpci] = True
        else:
            self.new_point_ids = new_xyz_unique_ids
            return (np.zeros((0, 3)), np.zeros((0,), dtype=np.int64),
                    np.array(all_new_point_views), all_new_observations)

        # store new 3D points
        grow_by = self.new_point_ids.shape[0]
        self.xyz_points = np.vstack([self.xyz_points, np.zeros((grow_by, 3))])
        self.xyz_points[tr_ids_pc1, :] = new_xyz
        self.xyz_val_ids = np.concatenate([self.xyz_val_ids, tr_ids_pc1])

        if not self.initial_mapping_successful:
            slv_id = 0
            self.initial_mapping_successful = True
        else:
            candidates = np.where(self.sliding_window_view_ids >= self.xyz_start_view[tr_ids_pc1].min())[0]
            slv_id = int(candidates[0]) if candidates.size > 0 else 0

        all_new_point_views = []
        all_new_observations = []
        for k in range(slv_id, self.current_sliding_window_index + 1):
            k2 = self.sliding_window_view_ids[k]
            _, p_idxx, _ = intersect_1d(self.AllIds[k2][:, 1], tr_ids_pc1)
            if p_idxx.size > 0:
                all_new_point_views.append(int(k2))
                all_new_observations.append(
                    np.column_stack([self.AllIds[k2][p_idxx, :], self.AllObservations[k2][p_idxx, :]])
                )

        self.new_point_ids = new_xyz_unique_ids[tr_new]
        new_xyz_unique_ids = tr_ids_pc1

        return new_xyz, new_xyz_unique_ids, np.array(all_new_point_views), all_new_observations

    # ----------------------------------------------------------------
    def get_2d_correspondences_between_views(self, id1: int, id2: int
                                              ) -> Tuple[np.ndarray, np.ndarray]:
        """2D-2D correspondences between two views."""
        _, ia, ib = intersect_1d(self.AllIds[id1][:, 1], self.AllIds[id2][:, 1])
        matches1 = self.AllObservations[id1][ia, :]
        matches2 = self.AllObservations[id2][ib, :]
        return matches1, matches2

    def get_key_points_in_view(self, view_id: int
                                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Key points seen in a view, their unique ids, and triangulated status."""
        key_points = self.AllObservations[view_id]
        unique_point_ids = self.AllIds[view_id][:, 1]
        is_triangulated = self.isTriangulated[unique_point_ids]
        return key_points, unique_point_ids, is_triangulated

    def get_point_ids_in_views(self, view_ids) -> np.ndarray:
        all_i = np.vstack([self.AllIds[v] for v in view_ids])
        u_i = np.unique(all_i[:, 1])
        is_t = self.isTriangulated[u_i]
        return u_i[is_t]

    def set_key_point_validity_in_view(self, view_id: int, validity: np.ndarray) -> None:
        self.AllObservations[view_id] = self.AllObservations[view_id][validity, :]
        self.AllIds[view_id] = self.AllIds[view_id][validity, :]
        self.AllTriangulated[view_id] = self.AllTriangulated[view_id][validity]

    def get_xyz_points(self, ids: Optional[np.ndarray] = None
                        ) -> Tuple[np.ndarray, np.ndarray]:
        if ids is None:
            ids = self.xyz_val_ids
        xyz = self.xyz_points[ids, :]
        return xyz, self.xyz_val_ids

    def set_xyz_points(self, xyz: np.ndarray, ids: Optional[np.ndarray] = None) -> None:
        if ids is None:
            ids = self.xyz_val_ids
        self.xyz_points[ids, :] = xyz

    def get_sliding_window_ids(self) -> np.ndarray:
        return self.sliding_window_view_ids[: self.current_sliding_window_index + 1]


# --------------------------------------------------------------------------
# Local helper functions
# --------------------------------------------------------------------------
def helper_within_image(points: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    """Check that points (x, y) fall within an image of shape
    image_size = (height, width). 0-based bounds: x in [0, width-1],
    y in [0, height-1] (MATLAB used 1-based [1, width]/[1, height])."""
    h, w = image_size
    return ((points[:, 0] >= 0) & (points[:, 0] <= w - 1)
            & (points[:, 1] >= 0) & (points[:, 1] <= h - 1))


def helper_quick_check_parallax(matches1: np.ndarray, matches2: np.ndarray,
                                 parallax_threshold: float) -> Tuple[float, bool]:
    """Average Euclidean displacement between matched points across two
    views; status is True if average parallax exceeds threshold."""
    if matches1.shape[0] == 0:
        return 0.0, False
    a = matches1 - matches2
    avg = np.sqrt((a * a).sum(axis=1)).sum() / a.shape[0]
    status = avg > parallax_threshold
    return float(avg), bool(status)


def estimate_fundamental_matrix_ransac(pts1: np.ndarray, pts2: np.ndarray,
                                        num_trials: int, confidence: float,
                                        distance_threshold: float
                                        ) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Equivalent of estimateFundamentalMatrix(..., 'Method','RANSAC', ...).

    Returns (F, inlier_mask) where inlier_mask is a boolean array aligned
    with pts1/pts2 rows. If estimation fails, returns (None, all-False).
    """
    n = pts1.shape[0]
    if n < 8:
        return None, np.zeros((n,), dtype=bool)

    F, mask = cv2.findFundamentalMat(
        pts1.astype(np.float64), pts2.astype(np.float64),
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=distance_threshold,
        confidence=confidence,
        maxIters=num_trials,
    )
    if mask is None:
        return F, np.zeros((n,), dtype=bool)
    return F, mask.reshape(-1).astype(bool)


def triangulate_points(pts1: np.ndarray, pts2: np.ndarray,
                        cam_matrix1: np.ndarray, cam_matrix2: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Equivalent of MATLAB's triangulate(pts1, pts2, camMatrix1, camMatrix2).

    Returns (xyz, is_in_front) where is_in_front flags points with
    positive depth in BOTH cameras (mirrors MATLAB's third output).

    NOTE: MATLAB's cameraProjection/triangulate pair uses row-vector,
    post-multiply convention (x_2d = X_3d_row @ P). cv2.triangulatePoints
    expects the standard column-vector convention (x_2d = P @ X_3d_col)
    with 2xN point arrays. Inputs here are passed to cv2 in its native
    convention; cam_matrix1/2 as built by camera_projection() (K @ [R|t])
    already match that convention.
    """
    if pts1.shape[0] == 0:
        return np.zeros((0, 3)), np.zeros((0,), dtype=bool)

    pts1_2xn = pts1.T.astype(np.float64)
    pts2_2xn = pts2.T.astype(np.float64)

    xyz_h = cv2.triangulatePoints(cam_matrix1, cam_matrix2, pts1_2xn, pts2_2xn)
    xyz = (xyz_h[:3, :] / xyz_h[3, :]).T  # (N, 3)

    # depth check ("in front of camera") for both views
    R1, t1 = cam_matrix1[:, :3], cam_matrix1[:, 3]  # not exact R,t recovery but
    # depth sign for projection matrices P = K[R|t] can't be split from P alone
    # in general (K may not be identity), so compute depth via re-projection
    # using the projection matrices directly: depth ~ 3rd row of P @ X_h
    depth1 = (cam_matrix1 @ xyz_h)[2, :]
    depth2 = (cam_matrix2 @ xyz_h)[2, :]
    w = xyz_h[3, :]
    # correct sign of depth relative to homogeneous scale
    is_in_front = ((depth1 * w) > 0) & ((depth2 * w) > 0)

    return xyz, is_in_front


def intersect_1d(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Equivalent of MATLAB's intersect(a, b, 'legacy'): returns
    (common_values, index_into_a, index_into_b) for the FIRST occurrence
    of each common value, sorted ascending by value (legacy MATLAB
    behavior sorts ascending; 'stable' is not used here intentionally to
    match the original 'legacy' flag's sorted-output behavior).
    """
    common, ia, ib = np.intersect1d(a, b, return_indices=True)
    return common, ia, ib


def helper_select_new_key_points_uniformly(current_tracked_corners: np.ndarray,
                                            new_corners: np.ndarray,
                                            min_dist: int,
                                            image_size: Tuple[int, int],
                                            max_corner_count: int) -> np.ndarray:
    """Select new corners that are (a) at least min_dist away from any
    existing tracked corner and (b) spread out roughly uniformly over
    the image, capped at the number of corners needed to reach
    max_corner_count total.

    current_tracked_corners: (M, 2) array of (x, y), may be empty
    new_corners: (N, 2) array of (x, y) candidate corners
    image_size: (height, width)
    Returns: (K, 2) array of selected (x, y) corners, K <= max(0, max_corner_count - M)
    """
    h, w = image_size
    num_to_select = max(0, max_corner_count - current_tracked_corners.shape[0])
    if num_to_select == 0 or new_corners.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)

    # build occupancy mask marking pixels within min_dist of an existing
    # tracked corner (equivalent of MATLAB's imageMask construction)
    mask = np.zeros((h + 2 * min_dist, w + 2 * min_dist), dtype=bool)
    if current_tracked_corners.shape[0] > 0:
        yy, xx = np.meshgrid(np.arange(-min_dist, min_dist + 1),
                              np.arange(-min_dist, min_dist + 1), indexing="ij")
        keep = (xx.ravel() ** 2 + yy.ravel() ** 2) <= (min_dist + 0.75) ** 2
        dx = xx.ravel()[keep]
        dy = yy.ravel()[keep]
        cx = np.round(current_tracked_corners[:, 0]).astype(np.int64)
        cy = np.round(current_tracked_corners[:, 1]).astype(np.int64)
        # broadcast: every tracked corner x every disk offset
        mx = (cx[:, None] + dx[None, :] + min_dist).ravel()
        my = (cy[:, None] + dy[None, :] + min_dist).ravel()
        valid = (mx >= 0) & (mx < mask.shape[1]) & (my >= 0) & (my < mask.shape[0])
        mask[my[valid], mx[valid]] = True

    nx = np.round(new_corners[:, 0]).astype(np.int64) + min_dist
    ny = np.round(new_corners[:, 1]).astype(np.int64) + min_dist
    in_bounds = (nx >= 0) & (nx < mask.shape[1]) & (ny >= 0) & (ny < mask.shape[0])
    away_from_existing = np.zeros((new_corners.shape[0],), dtype=bool)
    away_from_existing[in_bounds] = ~mask[ny[in_bounds], nx[in_bounds]]

    candidates = new_corners[away_from_existing]
    if candidates.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)

    return select_uniform(candidates, num_to_select, image_size)


def select_uniform(points: np.ndarray, num_points: int,
                    image_size: Tuple[int, int]) -> np.ndarray:
    """Equivalent of MATLAB's selectUniform: pick up to num_points points
    spread out roughly uniformly over the image grid.

    Buckets the image into a roughly square grid sized so that the
    number of cells is close to num_points, then takes (at most) one
    point per cell, preferring earlier (i.e. higher quality, since
    goodFeaturesToTrack returns corners sorted strongest-first) points
    within each cell. Fills remaining slots from leftover points if
    some cells were empty.
    """
    if num_points <= 0 or points.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if points.shape[0] <= num_points:
        return points.copy()

    h, w = image_size
    aspect = w / h if h > 0 else 1.0
    n_rows = max(1, int(round(np.sqrt(num_points / aspect))))
    n_cols = max(1, int(round(num_points / n_rows)))

    cell_h = h / n_rows
    cell_w = w / n_cols

    col_idx = np.clip((points[:, 0] / cell_w).astype(np.int64), 0, n_cols - 1)
    row_idx = np.clip((points[:, 1] / cell_h).astype(np.int64), 0, n_rows - 1)
    cell_id = row_idx * n_cols + col_idx

    selected_indices: List[int] = []
    seen_cells = set()
    for i in range(points.shape[0]):
        c = int(cell_id[i])
        if c not in seen_cells:
            seen_cells.add(c)
            selected_indices.append(i)
            if len(selected_indices) >= num_points:
                break

    if len(selected_indices) < num_points:
        remaining = [i for i in range(points.shape[0]) if i not in set(selected_indices)]
        needed = num_points - len(selected_indices)
        selected_indices.extend(remaining[:needed])

    return points[np.array(selected_indices, dtype=np.int64), :]