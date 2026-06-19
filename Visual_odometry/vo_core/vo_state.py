import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple


@dataclass
class BACorrection:
    """Translation + rotation delta produced after a BA run."""
    delta_R: np.ndarray          # (3, 3)
    delta_t: np.ndarray          # (3,)  flattened
    ref_kf_frame_idx: int        # which keyframe the delta was derived from
    valid: bool = True


class VIOState:
    def __init__(self):
        self._lock = threading.RLock()

        # ── Alignment results ─────────────────────────────────────────────
        self._scale: Optional[float]          = None   # metric / visual
        self._g_world: Optional[np.ndarray]   = None   # gravity in camera frame (3,)
        self._velocities: Dict[int, np.ndarray] = {}   # frame_idx → (3,) velocity
        self._alignment_valid: bool            = False
        self._alignment_count: int             = 0     # how many times VIA has run

        self._ba_correction: Optional[BACorrection] = None
        self._keyframe_poses: List[dict] = []
        self._map_points: List[np.ndarray] = []   # 3-D landmarks (visual scale)

    def update_alignment(
        self,
        scale: float,
        g_world: np.ndarray,
        velocities: Dict[int, np.ndarray],
    ) -> None:
        with self._lock:
            if scale <= 0:
                return
            self._scale             = float(scale)
            self._g_world           = np.array(g_world, dtype=np.float64)
            self._velocities        = {k: np.array(v) for k, v in velocities.items()}
            self._alignment_valid   = True
            self._alignment_count  += 1

    @property
    def scale(self) -> Optional[float]:
        with self._lock:
            return self._scale

    @property
    def g_world(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._g_world.copy() if self._g_world is not None else None

    @property
    def alignment_valid(self) -> bool:
        with self._lock:
            return self._alignment_valid

    @property
    def alignment_count(self) -> int:
        with self._lock:
            return self._alignment_count

    def get_velocity(self, frame_idx: int) -> Optional[np.ndarray]:
        with self._lock:
            v = self._velocities.get(frame_idx)
            return v.copy() if v is not None else None

    def update_ba_correction(
        self,
        delta_R: np.ndarray,
        delta_t: np.ndarray,
        ref_kf_frame_idx: int,
    ) -> None:
        with self._lock:
            self._ba_correction = BACorrection(
                delta_R=np.array(delta_R, dtype=np.float64),
                delta_t=np.array(delta_t, dtype=np.float64).flatten(),
                ref_kf_frame_idx=ref_kf_frame_idx,
            )

    def consume_ba_correction(self) -> Optional[BACorrection]:
        with self._lock:
            corr = self._ba_correction
            self._ba_correction = None
            return corr

    def peek_ba_correction(self) -> Optional[BACorrection]:
        with self._lock:
            return self._ba_correction

    # ─────────────────────────────────────────────────────────────────────
    # Keyframe map (VO / BA results)
    # ─────────────────────────────────────────────────────────────────────

    def update_keyframe_poses(self, keyframe_poses: List[dict]) -> None:
        with self._lock:
            self._keyframe_poses = [dict(kf) for kf in keyframe_poses]

    def get_keyframe_poses(self) -> List[dict]:
        with self._lock:
            return [dict(kf) for kf in self._keyframe_poses]

    def update_map_points(self, points: List[np.ndarray]) -> None:
        with self._lock:
            self._map_points = [np.array(p) for p in points]

    def get_map_points(self) -> List[np.ndarray]:
        with self._lock:
            return [p.copy() for p in self._map_points]

    def snapshot(self) -> dict:
        with self._lock:
            return {
                'scale'            : self._scale,
                'alignment_valid'  : self._alignment_valid,
                'alignment_count'  : self._alignment_count,
                'n_keyframes'      : len(self._keyframe_poses),
                'n_map_points'     : len(self._map_points),
                'ba_correction_pending': self._ba_correction is not None,
            }