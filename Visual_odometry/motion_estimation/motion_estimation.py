"""
motion_estimation.py
────────────────────
Produces metric 3-D pose (x, y, z) + orientation (qx, qy, qz, qw) at
camera frame rate.

Also provides VIOEstimationPipeline — a single object that constructs and
owns chunk_db, vio_state, via, and motion_estimator, so vo_subscriber can
do:
    self.estimation_core = VIOEstimationPipeline(calib, noise)
    self.estimation_core.chunk_db          → IMUChunkDatabase
    self.estimation_core.motion_estimator  → MotionEstimator

Fix applied [Issue 7]: VIOEstimationPipeline added.
"""

import threading
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

from vo_core.vo_state import VIOState, BACorrection
from Inertial.imu_chunk_db import IMUChunkDatabase
from Inertial.visual_inertial_alignment import VisualInertialAlignment


# ── Output type ───────────────────────────────────────────────────────────────

@dataclass
class Pose:
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
    norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    return float(qx/norm), float(qy/norm), float(qz/norm), float(qw/norm)


# ── MotionEstimator ───────────────────────────────────────────────────────────

class MotionEstimator:

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

        self._pre_ba_poses: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self._kf_since_last_via = 0
        self._via_lock = threading.Lock()

    # ── FAST PATH — every camera frame ────────────────────────────────────

    def compute_pose(self, R: np.ndarray, t: np.ndarray,
                     timestamp: float) -> Pose:
        t = np.array(t, dtype=np.float64).flatten()
        R = np.array(R, dtype=np.float64)

        scale        = self._state.scale
        scale_status = 'unscaled'

        if scale is not None and scale > 0:
            t_metric     = scale * t
            scale_status = 'scaled'
        else:
            t_metric = t.copy()

        ba_corr: Optional[BACorrection] = self._state.consume_ba_correction()
        if ba_corr is not None and ba_corr.valid:
            t_metric     = ba_corr.delta_R @ t_metric + ba_corr.delta_t
            R            = ba_corr.delta_R @ R
            scale_status = 'ba_fused'

        qx, qy, qz, qw = _rot_to_quat(R)

        return Pose(
            x=float(t_metric[0]), y=float(t_metric[1]), z=float(t_metric[2]),
            qx=qx, qy=qy, qz=qz, qw=qw,
            timestamp=timestamp,
            scale_status=scale_status,
        )

    # ── SLOW PATH — keyframe rate ─────────────────────────────────────────

    def on_new_keyframe(self, keyframe_poses: List[dict],
                        timestamp: float) -> None:
        self._update_pre_ba_snapshot(keyframe_poses)
        normalised = self._normalise_kf_list(keyframe_poses)
        self._state.update_keyframe_poses(normalised)

        self._kf_since_last_via += 1
        n_kf = len(normalised)

        if n_kf < self._min_kf:
            if self._verbose:
                print(f"[MotionEstimator] {n_kf}/{self._min_kf} KFs — "
                      "waiting before alignment.")
            return

        first_time   = not self._state.alignment_valid
        periodic_run = (self._realign_every > 0 and
                        self._kf_since_last_via >= self._realign_every)

        if first_time or periodic_run:
            self._kf_since_last_via = 0
            self._run_via_async(normalised)

    # ── CORRECTION PATH — after BA ────────────────────────────────────────

    def on_ba_updated(self, ba_keyframe_poses: List[dict],
                      map_points: List[np.ndarray],
                      timestamp: float) -> None:
        if not ba_keyframe_poses:
            return

        normalised = self._normalise_kf_list(ba_keyframe_poses)
        self._state.update_keyframe_poses(normalised)
        self._state.update_map_points(map_points)

        delta_R, delta_t, ref_idx = self._compute_ba_delta(normalised)
        if delta_R is not None:
            self._state.update_ba_correction(delta_R, delta_t, ref_idx)
            if self._verbose:
                print(f"[MotionEstimator] BA correction: "
                      f"|Δt|={np.linalg.norm(delta_t):.4f} m  ref_kf={ref_idx}")

        self._update_pre_ba_snapshot(normalised)

        if len(normalised) >= self._min_kf:
            self._run_via_async(normalised)

    # ── VIA execution ─────────────────────────────────────────────────────

    def _run_via_async(self, kf_poses: List[dict]) -> None:
        threading.Thread(
            target=self._run_via, args=(kf_poses,),
            daemon=True, name='VIA-alignment',
        ).start()

    def _run_via(self, kf_poses: List[dict]) -> None:
        if not self._via_lock.acquire(blocking=False):
            if self._verbose:
                print("[MotionEstimator] VIA already running — skipping.")
            return
        try:
            via_poses     = [(kf['p_bar'], kf['R']) for kf in kf_poses]
            kf_timestamps = [kf['timestamp'] for kf in kf_poses]

            if self._verbose:
                print(f"[MotionEstimator] VIA: {len(via_poses)} KFs  "
                      f"t=[{kf_timestamps[0]:.2f}…{kf_timestamps[-1]:.2f}]")

            s, g_world, velocities_list = self._via.run_between(
                via_poses, kf_timestamps
            )

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
        self._pre_ba_poses = {
            kf['frame_idx']: (
                np.array(kf.get('p_bar', kf.get('t', np.zeros(3))),
                         dtype=np.float64).flatten(),
                np.array(kf['R'], dtype=np.float64),
            )
            for kf in kf_poses
        }

    def _compute_ba_delta(
        self, ba_poses: List[dict],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
        if not self._pre_ba_poses or not ba_poses:
            return None, None, -1
        scale = self._state.scale
        for kf in reversed(ba_poses):
            idx = kf['frame_idx']
            if idx not in self._pre_ba_poses:
                continue
            pre_p, pre_R = self._pre_ba_poses[idx]
            post_p = np.array(kf['p_bar'], dtype=np.float64).flatten()
            post_R = np.array(kf['R'],     dtype=np.float64)
            delta_R = post_R @ pre_R.T
            delta_t = post_p - delta_R @ pre_p
            if scale is not None and scale > 0:
                delta_t = scale * delta_t
            return delta_R, delta_t, idx
        return None, None, -1

    # ── Utilities ─────────────────────────────────────────────────────────

    @staticmethod
    def _normalise_kf_list(kf_list: List[dict]) -> List[dict]:
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

    def status(self) -> dict:
        return {
            'alignment_valid' : self._state.alignment_valid,
            'alignment_count' : self._state.alignment_count,
            'scale'           : self._state.scale,
            'kf_since_via'    : self._kf_since_last_via,
            'ba_pending'      : self._state.peek_ba_correction() is not None,
            'pre_ba_kfs'      : len(self._pre_ba_poses),
        }


# ── VIOEstimationPipeline ─────────────────────────────────────────────────────

class VIOEstimationPipeline:
    """
    Convenience wrapper that constructs and owns all VIO components.
    [Fix for Issue 7]

    vo_subscriber creates one instance and accesses:
        .chunk_db          → passed to IMUPipeline
        .motion_estimator  → passed to VisualOdometryPipeline

    Parameters
    ──────────
    calibration_data : dict from load_calibration_files()
                       must contain calibration_data['left']['T_BS']['data']
    imu_noise_params : dict with sigma_a, sigma_w, sigma_ba, sigma_bw
    min_kf_for_via   : minimum keyframes before first alignment attempt
    realign_every    : re-run VIA every N keyframes (0 = once only)
    max_chunks       : IMUChunkDatabase capacity
    verbose          : print debug info
    """

    def __init__(
        self,
        calibration_data: dict,
        imu_noise_params: dict,
        min_kf_for_via:   int  = 5,
        realign_every:    int  = 10,
        max_chunks:       int  = 500,
        verbose:          bool = True,
    ):
        # Camera-to-IMU extrinsic from calibration
        T_bc = np.array(
            calibration_data['left']['T_BS']['data'],
            dtype=np.float64,
        ).reshape(4, 4)

        # Shared components
        self.chunk_db  = IMUChunkDatabase(max_chunks=max_chunks)
        self.vio_state = VIOState()
        self.via       = VisualInertialAlignment(T_bc, imu_noise_params,
                                                 self.chunk_db)
        self.motion_estimator = MotionEstimator(
            vio_state      = self.vio_state,
            via            = self.via,
            min_kf_for_via = min_kf_for_via,
            realign_every  = realign_every,
            verbose        = verbose,
        )

    def status(self) -> dict:
        return {
            'motion_estimator': self.motion_estimator.status(),
            'chunk_db'        : self.chunk_db.summary(),
            'vio_state'       : self.vio_state.snapshot(),
        }