"""
motion_estimator.py
───────────────────
Produces the final 3-D pose (x, y, z) + orientation (qx, qy, qz, qw)
at camera frame rate.

Two loops
─────────
FAST  (every frame)   compute_pose(R, t, timestamp)
        ↳ apply stored scale → metric position
        ↳ apply pending BA correction if present
        ↳ convert R → quaternion
        ↳ return Pose dataclass

SLOW  (keyframe rate) on_new_keyframe(keyframe_poses, timestamp)
        ↳ if ≥ MIN_KF keyframes:
              call via.run_between(poses, kf_timestamps)
                  VIA fetches chunk.raw_samples from chunk_db per pair
                  VIA calls preintegrate() on real raw data
                  VIA returns (s, g_c0, velocities)
              write scale / gravity / velocities to VIOState

CORRECTION (after BA) on_ba_updated(ba_keyframe_poses, map_points, timestamp)
        ↳ compute delta between pre-BA and post-BA pose of reference KF
        ↳ write BACorrection to VIOState
        ↳ next compute_pose() call automatically applies the delta
"""

import threading
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

from Visual_odometry.vo_core.vo_state    import VIOState, BACorrection
from Visual_odometry.Inertial.imu_chunk_db import IMUChunkDatabase
from Visual_odometry.Inertial.visual_inertial_alignment import VisualInertialAlignment


# ── Output type ───────────────────────────────────────────────────────────────

@dataclass
class Pose:
    """Final metric pose output, produced every camera frame."""
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float
    timestamp:    float
    scale_status: str   # 'unscaled' | 'scaled' | 'ba_fused'


# ── Quaternion helper ─────────────────────────────────────────────────────────

def _rot_to_quat(R: np.ndarray) -> Tuple[float, float, float, float]:
    """3×3 rotation matrix → (qx, qy, qz, qw). Shepperd's method."""
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s  = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2,1] - R[1,2]) * s
        qy = (R[0,2] - R[2,0]) * s
        qz = (R[1,0] - R[0,1]) * s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s  = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        qw = (R[2,1] - R[1,2]) / s
        qx = 0.25 * s
        qy = (R[0,1] + R[1,0]) / s
        qz = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s  = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        qw = (R[0,2] - R[2,0]) / s
        qx = (R[0,1] + R[1,0]) / s
        qy = 0.25 * s
        qz = (R[1,2] + R[2,1]) / s
    else:
        s  = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        qw = (R[1,0] - R[0,1]) / s
        qx = (R[0,2] + R[2,0]) / s
        qy = (R[1,2] + R[2,1]) / s
        qz = 0.25 * s
    return float(qx), float(qy), float(qz), float(qw)


# ── MotionEstimator ───────────────────────────────────────────────────────────

class MotionEstimator:
    """
    Wires together VIOState, IMUChunkDatabase and VisualInertialAlignment
    to produce frame-rate metric pose estimates.

    Parameters
    ──────────
    vio_state      : shared VIOState instance
    via            : VisualInertialAlignment (already holds chunk_db reference)
    min_kf_for_via : minimum keyframes before attempting alignment (≥ 5)
    realign_every  : re-run VIA every N new keyframes after first success
                     (0 = align once and lock scale)
    verbose        : print debug info
    """

    MIN_KF_DEFAULT = 5

    def __init__(
        self,
        vio_state:      VIOState,
        via:            VisualInertialAlignment,
        min_kf_for_via: int  = MIN_KF_DEFAULT,
        realign_every:  int  = 10,
        verbose:        bool = True,
    ):
        self._state         = vio_state
        self._via           = via
        self._min_kf        = max(min_kf_for_via, 5)
        self._realign_every = realign_every
        self._verbose       = verbose

        # Pre-BA pose snapshot for computing BA correction delta
        # {frame_idx: (p_bar np.ndarray, R np.ndarray)}
        self._pre_ba_poses: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        self._kf_since_last_via = 0
        self._via_lock = threading.Lock()   # only one VIA call at a time

    # ── FAST PATH — every camera frame ────────────────────────────────────

    def compute_pose(
        self,
        R:         np.ndarray,   # (3,3) rotation from VO
        t:         np.ndarray,   # (3,1) or (3,) translation in visual units
        timestamp: float,
    ) -> Pose:
        """
        Called directly from process_frame_mono() on every frame.
        Must be fast — no heavy computation here.
        """
        t = np.array(t, dtype=np.float64).flatten()
        R = np.array(R, dtype=np.float64)

        scale        = self._state.scale
        scale_status = 'unscaled'

        # Apply metric scale
        if scale is not None and scale > 0:
            t_metric     = scale * t
            scale_status = 'scaled'
        else:
            t_metric = t.copy()   # visual units — consistent but not metric

        # Apply pending BA correction (consumes it — fires once per BA run)
        ba_corr: Optional[BACorrection] = self._state.consume_ba_correction()
        if ba_corr is not None and ba_corr.valid:
            t_metric     = ba_corr.delta_R @ t_metric + ba_corr.delta_t
            R            = ba_corr.delta_R @ R
            scale_status = 'ba_fused'

        qx, qy, qz, qw = _rot_to_quat(R)

        return Pose(
            x=float(t_metric[0]),
            y=float(t_metric[1]),
            z=float(t_metric[2]),
            qx=qx, qy=qy, qz=qz, qw=qw,
            timestamp=timestamp,
            scale_status=scale_status,
        )

    # ── SLOW PATH — keyframe alignment ────────────────────────────────────

    def on_new_keyframe(
        self,
        keyframe_poses: List[dict],
        timestamp:      float,
    ) -> None:
        """
        Called by vo_pipeline after triangulation succeeds on a new keyframe.

        keyframe_poses: list of dicts from keyframe_selector.get_all_keyframes()
            Must contain: frame_idx (int), timestamp (float),
                          R (3,3 ndarray), and 't' or 'p_bar' (3,) ndarray.
        """
        # Save pre-BA snapshot before normalising
        self._update_pre_ba_snapshot(keyframe_poses)

        normalised = self._normalise_kf_list(keyframe_poses)
        self._state.update_keyframe_poses(normalised)

        self._kf_since_last_via += 1
        n_kf = len(normalised)

        if n_kf < self._min_kf:
            if self._verbose:
                print(f"[MotionEstimator] {n_kf}/{self._min_kf} KFs — "
                      "waiting for more before alignment.")
            return

        first_time   = not self._state.alignment_valid
        periodic_run = (self._realign_every > 0 and
                        self._kf_since_last_via >= self._realign_every)

        if first_time or periodic_run:
            self._kf_since_last_via = 0
            # Run VIA in a background thread so camera callback never blocks
            self._run_via_async(normalised)

    # ── CORRECTION PATH — after BA ────────────────────────────────────────

    def on_ba_updated(
        self,
        ba_keyframe_poses: List[dict],
        map_points:        List[np.ndarray],
        timestamp:         float,
    ) -> None:
        """
        Called by vo_pipeline after bundle adjustment completes.
        Computes the pose correction delta and stores it in VIOState.
        """
        if not ba_keyframe_poses:
            return

        normalised = self._normalise_kf_list(ba_keyframe_poses)
        self._state.update_keyframe_poses(normalised)
        self._state.update_map_points(map_points)

        delta_R, delta_t, ref_idx = self._compute_ba_delta(normalised)
        if delta_R is not None:
            self._state.update_ba_correction(delta_R, delta_t, ref_idx)
            if self._verbose:
                print(f"[MotionEstimator] BA correction stored: "
                      f"|Δt|={np.linalg.norm(delta_t):.4f} m  ref_kf={ref_idx}")

        # Refresh pre-BA snapshot to post-BA values for next BA run
        self._update_pre_ba_snapshot(normalised)

        # Re-run VIA with BA-corrected poses for better scale estimate
        n_kf = len(normalised)
        if n_kf >= self._min_kf:
            self._run_via_async(normalised)

    # ── VIA execution ─────────────────────────────────────────────────────

    def _run_via_async(self, kf_poses: List[dict]) -> None:
        """Spawn background thread so camera callback is never blocked."""
        t = threading.Thread(
            target=self._run_via,
            args=(kf_poses,),
            daemon=True,
            name='VIA-alignment',
        )
        t.start()

    def _run_via(self, kf_poses: List[dict]) -> None:
        """
        Build (p_bar, R) pose list and kf_timestamps, then call
        via.run_between() which fetches chunk.raw_samples per pair
        and runs the full alignment pipeline.
        """
        if not self._via_lock.acquire(blocking=False):
            if self._verbose:
                print("[MotionEstimator] VIA already running — skipping.")
            return

        try:
            # Build the (p_bar, R) list VIA.run() expects
            via_poses = [(kf['p_bar'], kf['R']) for kf in kf_poses]

            # Timestamps for each keyframe — used by run_between() to look
            # up the correct IMU chunk in chunk_db
            kf_timestamps = [kf['timestamp'] for kf in kf_poses]

            if self._verbose:
                print(f"[MotionEstimator] Running VIA: "
                      f"{len(via_poses)} KFs  "
                      f"t=[{kf_timestamps[0]:.2f} … {kf_timestamps[-1]:.2f}]")

            # VIA fetches chunk.raw_samples for each pair, calls
            # preintegrate() on real data, then runs full alignment
            s, g_world, velocities_list = self._via.run_between(
                via_poses, kf_timestamps
            )

            # Map velocities to frame_idx
            vel_dict = {
                kf_poses[i]['frame_idx']: velocities_list[i]
                for i in range(len(kf_poses))
            }

            self._state.update_alignment(s, g_world, vel_dict)

            if self._verbose:
                print(f"[MotionEstimator] VIA SUCCESS  "
                      f"scale={s:.4f}  |g|={np.linalg.norm(g_world):.3f} m/s²")

        except (AssertionError, RuntimeError, np.linalg.LinAlgError) as e:
            if self._verbose:
                print(f"[MotionEstimator] VIA failed: {e}")

        finally:
            self._via_lock.release()

    # ── Pre-BA snapshot ───────────────────────────────────────────────────

    def _update_pre_ba_snapshot(self, kf_poses: List[dict]) -> None:
        """Save current visual poses so we can compute BA correction delta."""
        self._pre_ba_poses = {
            kf['frame_idx']: (
                np.array(kf.get('p_bar', kf.get('t', np.zeros(3))),
                         dtype=np.float64).flatten(),
                np.array(kf['R'], dtype=np.float64),
            )
            for kf in kf_poses
        }

    def _compute_ba_delta(
        self,
        ba_poses: List[dict],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
        """
        Compute (delta_R, delta_t, ref_frame_idx) from the most recently
        updated keyframe that exists in both pre-BA and post-BA snapshots.

        Satisfies:
            post_R = delta_R @ pre_R
            post_t = delta_R @ pre_t + delta_t   (metric units)
        """
        if not self._pre_ba_poses or not ba_poses:
            return None, None, -1

        scale = self._state.scale

        for kf in reversed(ba_poses):
            idx = kf['frame_idx']
            if idx not in self._pre_ba_poses:
                continue

            pre_p, pre_R   = self._pre_ba_poses[idx]
            post_p = kf['p_bar']
            post_R = kf['R']

            delta_R = post_R @ pre_R.T
            delta_t = post_p - delta_R @ pre_p

            # Convert translation delta to metric
            if scale is not None and scale > 0:
                delta_t = scale * delta_t

            return delta_R, delta_t, idx

        return None, None, -1

    # ── Utilities ─────────────────────────────────────────────────────────

    @staticmethod
    def _normalise_kf_list(kf_list: List[dict]) -> List[dict]:
        """
        Ensure every keyframe dict has a 'p_bar' key (some code uses 't').
        Returns a new list with standardised keys.
        """
        out = []
        for kf in kf_list:
            entry = dict(kf)
            if 'p_bar' not in entry:
                t = entry.get('t', np.zeros(3))
                entry['p_bar'] = np.array(t, dtype=np.float64).flatten()
            else:
                entry['p_bar'] = np.array(entry['p_bar'], dtype=np.float64).flatten()
            entry['R'] = np.array(entry['R'], dtype=np.float64)
            out.append(entry)
        return out

    # ── Debug ─────────────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            'alignment_valid' : self._state.alignment_valid,
            'alignment_count' : self._state.alignment_count,
            'scale'           : self._state.scale,
            'kf_since_via'    : self._kf_since_last_via,
            'ba_pending'      : self._state.peek_ba_correction() is not None,
            'pre_ba_kfs'      : len(self._pre_ba_poses),
        }