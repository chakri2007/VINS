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
class View:
    """
    One camera view.
    """

    R: np.ndarray

    t: np.ndarray

    timestamp: float

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

    # view_id -> View
    _views: Dict[int, View] = field(default_factory=dict)

    # insertion-order list of view ids (for iteration and NumViews)
    _view_ids: list = field(default_factory=list)

    # ------------------------------------------------------------------ #
    #  Write                                                               #
    # ------------------------------------------------------------------ #

    def add_view(
        self,
        view_id: int,
        R: np.ndarray,
        t: np.ndarray,
        timestamp: float,
    ) -> None:
        """
        Add a new view.
        """

        if view_id in self._views:
            raise ValueError(
                f"View id {view_id} already exists."
            )

        R = np.asarray(R, dtype=float)
        t = np.asarray(t, dtype=float).reshape(3)

        if R.shape != (3, 3):
            raise ValueError(f"R must be (3,3), got {R.shape}")

        self._views[view_id] = View(
            R=R.copy(),
            t=t.copy(),
            timestamp=float(timestamp),
        )

        self._view_ids.append(view_id)

    def update_pose(
        self,
        view_id: int,
        R: np.ndarray,
        t: np.ndarray,
    ) -> None:

        if view_id not in self._views:
            raise KeyError(f"View id {view_id} not found.")

        R = np.asarray(R, dtype=float)
        t = np.asarray(t, dtype=float).reshape(3)

        if R.shape != (3, 3):
            raise ValueError(f"R must be (3,3), got {R.shape}")

        self._views[view_id].R = R.copy()
        self._views[view_id].t = t.copy()

    # ------------------------------------------------------------------ #
    #  Read                                                                #
    # ------------------------------------------------------------------ #

    def get_pose(self, view_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (R, t) for view_id.  R is (3,3), t is (3,).

        Mirrors MATLAB: poses(vSet, viewId).AbsolutePose
        """
        if view_id not in self._poses:
            raise KeyError(f"View id {view_id} not found in ViewSet.")
        view = self._views[view_id]

        return view.R, view.t
    
    def get_timestamp(
        self,
        view_id: int,
    ) -> float:
        """
        Return the timestamp associated with a view.
        """

        if view_id not in self._views:
            raise KeyError(f"View id {view_id} not found.")

        return self._views[view_id].timestamp

    def get_all_poses(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Return ordered dict of all {view_id: (R, t)}.

        Mirrors MATLAB: poses(vSet)
        """
        return {
            vid: (
                self._views[vid].R,
                self._views[vid].t,
            )
            for vid in self._view_ids
        }

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
    
    def project_point(
        self,
        view_id: int,
        K: np.ndarray,
        xyz: np.ndarray,
    ):
        """
        Project a world point into an image.

        Parameters
        ----------
        xyz : (3,)
            World coordinates.

        Returns
        -------
        uv : (2,)
            Pixel coordinates.
        """

        R, t = self.get_pose(view_id)

        # World → Camera
        pc = R.T @ (xyz - t)

        if pc[2] <= 0:
            return None

        uv = K @ pc
        uv = uv[:2] / uv[2]

        return uv
    def relative_motion(
        self,
        from_view: int,
        to_view: int,
    ):
        """
        Compute the relative motion between two camera poses.

        Pose convention
        ---------------
        ViewSet stores camera-to-world poses

            p_w = R * p_c + t

        This function returns the transform from the camera frame at
        'from_view' to the camera frame at 'to_view'.

        Returns
        -------
        dR : (3,3)
            Relative rotation.

        dt : (3,)
            Relative translation expressed in the first camera frame.
        """

        R1, t1 = self.get_pose(from_view)
        R2, t2 = self.get_pose(to_view)

        #
        # Relative rotation
        #
        dR = R1.T @ R2

        #
        # Relative translation
        #
        dt = R1.T @ (t2 - t1)

        return dR, dt

    def consecutive_relative_motion(
        self,
        view_ids,
    ):
        """
        Compute relative motions for consecutive views.

        Parameters
        ----------
        view_ids : list[int]

        Returns
        -------
        motions : list[tuple]
            [(from_view, to_view, dR, dt), ...]
        """

        motions = []

        for i in range(len(view_ids) - 1):

            from_view = view_ids[i]
            to_view = view_ids[i + 1]

            dR, dt = self.relative_motion(
                from_view,
                to_view,
            )

            motions.append(
                (
                    from_view,
                    to_view,
                    dR,
                    dt,
                )
            )

        return motions
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