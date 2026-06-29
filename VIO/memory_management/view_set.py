"""
ViewSet — minimal port of MATLAB's imageviewset.

Stores one absolute pose (R, t) per view id and provides the lookup
interface used by the rest of the pipeline.

Pose convention (matches MATLAB rigidtform3d / AbsolutePose):
    (R, t) is the camera-to-world transform.
    A world point p_w maps to camera coords as:
        p_cam = R.T @ (p_w - t)
    Any world-to-camera projection (e.g. for triangulation) must invert
    this — that is the caller's responsibility, not ViewSet's.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np


@dataclass
class ViewSet:
    """
    Minimal stand-in for MATLAB's imageviewset.

    Stores absolute poses and view ids. Supports:
        add_view(view_id, R, t)         — MATLAB: addView(vSet, viewID, pose)
        get_pose(view_id)               — MATLAB: poses(vSet, viewId).AbsolutePose
        get_all_poses()                 — MATLAB: poses(vSet)
        update_pose(view_id, R, t)      — MATLAB: used after bundle adjustment
        view_ids (property)             — MATLAB: vSet.Views.ViewId
        num_views (property)            — MATLAB: vSet.NumViews
    """

    # view_id -> (R, t): R is (3,3), t is (3,) — camera-to-world pose
    _poses: Dict[int, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    # insertion-order list of view ids (for iteration and NumViews)
    _view_ids: list = field(default_factory=list)

    # ------------------------------------------------------------------ #
    #  Write                                                               #
    # ------------------------------------------------------------------ #

    def add_view(self, view_id: int, R: np.ndarray, t: np.ndarray) -> None:
        """Add a new view.  Raises ValueError on duplicate view_id."""
        if view_id in self._poses:
            raise ValueError(
                f"View id {view_id} already exists in ViewSet "
                f"(MATLAB addView raises on duplicate view ids)."
            )
        R = np.asarray(R, dtype=float)
        t = np.asarray(t, dtype=float).reshape(3)
        if R.shape != (3, 3):
            raise ValueError(f"R must be (3,3), got {R.shape}")
        self._poses[view_id] = (R.copy(), t.copy())
        self._view_ids.append(view_id)

    def update_pose(self, view_id: int, R: np.ndarray, t: np.ndarray) -> None:
        """Update the pose of an existing view (used after bundle adjustment)."""
        if view_id not in self._poses:
            raise KeyError(f"View id {view_id} not found in ViewSet.")
        R = np.asarray(R, dtype=float)
        t = np.asarray(t, dtype=float).reshape(3)
        if R.shape != (3, 3):
            raise ValueError(f"R must be (3,3), got {R.shape}")
        self._poses[view_id] = (R.copy(), t.copy())

    # ------------------------------------------------------------------ #
    #  Read                                                                #
    # ------------------------------------------------------------------ #

    def get_pose(self, view_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (R, t) for view_id.  R is (3,3), t is (3,).

        Mirrors MATLAB: poses(vSet, viewId).AbsolutePose
        """
        if view_id not in self._poses:
            raise KeyError(f"View id {view_id} not found in ViewSet.")
        return self._poses[view_id]

    def get_all_poses(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Return ordered dict of all {view_id: (R, t)}.

        Mirrors MATLAB: poses(vSet)
        """
        return {vid: self._poses[vid] for vid in self._view_ids}

    def camera_projection_matrix(self, view_id: int, K: np.ndarray) -> np.ndarray:
        """Return (3,4) camera projection matrix P = K [R^T | -R^T t].

        Used by triangulation (mirrors MATLAB's cameraProjection + pose2extr).
        K must be (3,3).
        """
        R, t = self.get_pose(view_id)
        Rt = R.T                          # world-to-camera rotation
        tt = -(Rt @ t).reshape(3, 1)      # world-to-camera translation
        return K @ np.hstack([Rt, tt])    # (3,4)
    
    def get_projection_matrices(
        self,
        view1: int,
        view2: int,
        K: np.ndarray,
    ):
        """
        Convenience function for triangulation.

        Returns
        -------
        P1 : (3,4)
        P2 : (3,4)
        """

        return (
            self.camera_projection_matrix(view1, K),
            self.camera_projection_matrix(view2, K),
        )

    # ------------------------------------------------------------------ #
    #  Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def view_ids(self) -> list:
        """Ordered list of view ids.  MATLAB: vSet.Views.ViewId"""
        return list(self._view_ids)

    @property
    def num_views(self) -> int:
        """Number of views.  MATLAB: vSet.NumViews"""
        return len(self._view_ids)