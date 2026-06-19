import numpy as np
from typing import Dict, List, Tuple, Optional


# Type aliases
LmId   = int
FeatId = int
Obs    = Tuple[int, float, float]   # (frame_idx, u, v)


class LandmarkMap:

    def __init__(self, sliding_window_size: int = 10):
        self.window_size = sliding_window_size

        self.landmarks:    Dict[LmId, np.ndarray]  = {}
        self.observations: Dict[LmId, List[Obs]]   = {}
        self.feat_to_lm:   Dict[FeatId, LmId]      = {}

        # Track which frames each landmark was last seen in
        # (for window-based pruning)
        self._lm_frames: Dict[LmId, List[int]] = {}

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def add_triangulation_result(self, result: dict):
        if result is None:
            return

        for lm_id, X in result['landmarks'].items():
            self.landmarks[lm_id]    = X.copy()
            self.observations[lm_id] = list(result['observations'][lm_id])
            self._lm_frames[lm_id]   = [
                obs[0] for obs in self.observations[lm_id]
            ]

        for feat_id, lm_id in result['feat_to_lm'].items():
            self.feat_to_lm[feat_id] = lm_id

    def add_observation(self, lm_id: LmId, frame_idx: int, u: float, v: float):
        if lm_id not in self.observations:
            return
        self.observations[lm_id].append((frame_idx, u, v))
        self._lm_frames[lm_id].append(frame_idx)

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def prune_outside_window(self, active_frame_indices: List[int]):

        active_set = set(active_frame_indices)
        to_drop = []

        for lm_id, frames in self._lm_frames.items():
            if not any(f in active_set for f in frames):
                to_drop.append(lm_id)

        for lm_id in to_drop:
            self._drop_landmark(lm_id)

        if to_drop:
            print(f"[LandmarkMap] Pruned {len(to_drop)} landmarks "
                  f"outside window. Remaining: {len(self.landmarks)}")

    def prune_feat_ids(self, lost_feat_ids: np.ndarray):
        for feat_id in lost_feat_ids:
            self.feat_to_lm.pop(int(feat_id), None)

    # ------------------------------------------------------------------
    # Query interface — for Bundle Adjustment
    # ------------------------------------------------------------------

    def get_landmarks_for_ba(self) -> Dict[LmId, np.ndarray]:
        """
        Return all landmarks.
        { lm_id: X(3,) }  — up-to-scale, ready as BA initial guess.
        After IMU alignment, multiply all X by scale factor s.
        """
        return {lm_id: X.copy() for lm_id, X in self.landmarks.items()}

    def get_observations_for_ba(self) -> Dict[LmId, List[Obs]]:
        """
        Return all observations.
        { lm_id: [(frame_idx, u, v), ...] }
        BA uses these as pixel residuals.
        """
        return {lm_id: list(obs) for lm_id, obs in self.observations.items()}

    def get_points_for_pnp(
        self,
        feat_ids: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        For a set of feature IDs, return the corresponding 3D points
        and their 2D observations in the most recent frame.

        Used for PnP pose estimation (future step).

        Returns
        -------
        pts3d    : (M, 3)  3D landmark positions
        feat_ids_out : (M,)   feat IDs that have a known landmark
        lm_ids   : (M,)   corresponding landmark IDs
        """
        pts3d_list   = []
        feat_out     = []
        lm_out       = []

        for feat_id in feat_ids:
            fid = int(feat_id)
            if fid in self.feat_to_lm:
                lm_id = self.feat_to_lm[fid]
                if lm_id in self.landmarks:
                    pts3d_list.append(self.landmarks[lm_id])
                    feat_out.append(fid)
                    lm_out.append(lm_id)

        if len(pts3d_list) == 0:
            return (np.empty((0, 3), dtype=np.float64),
                    np.empty((0,),   dtype=np.int64),
                    np.empty((0,),   dtype=np.int64))

        return (np.array(pts3d_list, dtype=np.float64),
                np.array(feat_out,   dtype=np.int64),
                np.array(lm_out,     dtype=np.int64))

    def scale_all_landmarks(self, s: float):
        """
        Multiply all landmark positions by scale factor s.
        Called after IMU VIO alignment recovers metric scale.
        """
        assert s > 0.0, f"Scale must be positive, got {s}"
        for lm_id in self.landmarks:
            self.landmarks[lm_id] *= s
        print(f"[LandmarkMap] Applied scale s={s:.4f} to "
              f"{len(self.landmarks)} landmarks.")

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def num_landmarks(self) -> int:
        return len(self.landmarks)

    def num_observations(self) -> int:
        return sum(len(obs) for obs in self.observations.values())

    def summary(self) -> str:
        return (f"LandmarkMap: {self.num_landmarks()} landmarks, "
                f"{self.num_observations()} observations, "
                f"{len(self.feat_to_lm)} active feat→lm mappings")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _drop_landmark(self, lm_id: LmId):
        self.landmarks.pop(lm_id, None)
        self.observations.pop(lm_id, None)
        self._lm_frames.pop(lm_id, None)
        # feat_to_lm entries pointing to this lm are stale but harmless;
        # they will be cleaned by prune_feat_ids over time.