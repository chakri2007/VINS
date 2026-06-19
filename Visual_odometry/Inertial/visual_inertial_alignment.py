"""
visual_inertial_alignment.py
────────────────────────────
Aligns a monocular VO map with IMU measurements to recover metric scale,
gravity vector, and per-keyframe velocities.

Fix applied [Issue 6]:
  run_between() now collects ALL per-frame chunks between two keyframe
  timestamps, concatenates their raw_samples into one flat list, and
  passes that single flat list to preintegrate() for a fresh integration
  over the full keyframe-to-keyframe interval.

  Previously it picked only the 'best' single chunk — since chunks are
  now cut at every frame, that discarded most of the IMU data.
"""

import numpy as np
from typing import List, Tuple, Optional

from Inertial.imu_chunk_db import IMUChunkDatabase
from Inertial.calib_gyro import calibrate_gyro_bias
from Inertial.preintegrate import preintegrate
from Inertial.optimize_scale_gravity_velocity import (
    solve_scale_gravity_velocity,
)
from Inertial.refine_gravity import refine_gravity


def pose_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → quaternion [w, x, y, z]."""
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w, x, y, z = 0.25/s, (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w, x, y, z = (R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w, x, y, z = (R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w, x, y, z = (R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


class VisualInertialAlignment:

    def __init__(self, T_bc: np.ndarray, imu_noise_params: dict,
                 chunk_db: IMUChunkDatabase):
        """
        T_bc            : 4×4 camera-to-IMU extrinsic
        imu_noise_params: dict with sigma_a, sigma_w, sigma_ba, sigma_bw
        chunk_db        : shared IMUChunkDatabase
        """
        self.R_bc = T_bc[:3, :3]
        self.p_bc = T_bc[:3,  3]
        self.noise = imu_noise_params

        self.b_w = np.zeros(3)
        self.b_a = np.zeros(3)

        self._chunk_db = chunk_db

    # ── Primary entry point ───────────────────────────────────────────────

    def run_between(
        self,
        keyframe_poses: List[Tuple[np.ndarray, np.ndarray]],
        kf_timestamps:  List[float],
        tolerance:      float = 0.05,
    ):
        """
        For each consecutive keyframe pair (t_i, t_{i+1}):
          1. Fetch ALL per-frame chunks from chunk_db in that interval
          2. Concatenate their raw_samples into ONE flat list
          3. Pass to preintegrate() for a single fresh integration

        This gives VIA the complete, properly-integrated IMU measurements
        between keyframes — not stitched per-frame preintegrations.

        Parameters
        ──────────
        keyframe_poses : list of (p_bar, R)  length N
        kf_timestamps  : list of float       length N

        Returns
        ───────
        (scale s, gravity g_c0, velocities list)
        """
        assert len(kf_timestamps) == len(keyframe_poses), \
            "kf_timestamps must have one entry per keyframe pose"
        assert len(keyframe_poses) >= 5, \
            "Need at least 5 keyframes for reliable alignment"

        n_pairs      = len(keyframe_poses) - 1
        imu_segments = []

        for i in range(n_pairs):
            t_start = kf_timestamps[i]
            t_end   = kf_timestamps[i + 1]

            # Collect ALL per-frame chunks between these two keyframes
            chunks = self._chunk_db.get_chunks_between(
                t_start, t_end, tolerance=tolerance
            )

            if not chunks:
                raise RuntimeError(
                    f"[VIA] No IMU chunks found between "
                    f"t={t_start:.3f} and t={t_end:.3f}. "
                    f"DB has {self._chunk_db.get_chunk_count()} chunks total. "
                    f"Ensure vo_pipeline calls imu_pipeline.notify_frame() "
                    f"every camera frame."
                )

            # Sort chronologically then concatenate raw samples into one
            # flat list — this is what preintegrate() will integrate fresh
            chunks_sorted = sorted(chunks, key=lambda c: c.t_start)
            all_raw: list = []
            for chunk in chunks_sorted:
                if not chunk.raw_samples:
                    print(f"[VIA] Warning: chunk [{chunk.t_start:.3f}"
                          f"→{chunk.t_end:.3f}] has no raw samples — skipping")
                    continue
                all_raw.extend(chunk.raw_samples)

            if not all_raw:
                raise RuntimeError(
                    f"[VIA] All chunks between t={t_start:.3f} and "
                    f"t={t_end:.3f} have empty raw_samples."
                )

            imu_segments.append(all_raw)

            # print(f"[VIA] Pair {i}: {len(chunks_sorted)} chunks  "
            #       f"total_raw={len(all_raw)}  "
            #       f"span=[{chunks_sorted[0].t_start:.3f}"
            #       f"→{chunks_sorted[-1].t_end:.3f}]")

        # Hand off to core alignment — preintegrate() is called fresh
        # on the full concatenated raw sequence per keyframe pair
        return self.run(keyframe_poses, imu_segments)

    # ── Core alignment (run() unchanged from original) ────────────────────

    def run(
        self,
        keyframe_poses: List[Tuple[np.ndarray, np.ndarray]],
        imu_segments:   List[list],
    ):
        """
        keyframe_poses : list of (p_bar, R)
        imu_segments   : list of raw IMU segments, one per KF pair
                         each = [(dt, accel(3,), gyro(3,)), ...]

        Internally calls preintegrate() fresh on each segment.
        Returns (s, g_c0, velocities).
        """
        assert len(imu_segments) == len(keyframe_poses) - 1
        assert len(keyframe_poses) >= 5

        # ── Step 1: Fresh preintegration over each full KF interval ──────
        preint_results = []
        for segment in imu_segments:
            result = preintegrate(segment, self.b_a, self.b_w, 
                        sigma_a=self.noise.get('sigma_a', 0.02),
                      sigma_w=self.noise.get('sigma_w', 0.005),
                      )
            preint_results.append(result)

        alphas      = [r[0] for r in preint_results]
        betas       = [r[1] for r in preint_results]
        gammas      = [r[2] for r in preint_results]
        J_gamma_bws = [r[7] for r in preint_results]
        dts         = [sum(d for d, a, g in seg) for seg in imu_segments]

        # ── Step 2: Gyro bias calibration ────────────────────────────────
        visual_quats = [pose_to_quat(R) for _, R in keyframe_poses]
        delta_bw     = calibrate_gyro_bias(visual_quats, gammas, J_gamma_bws)
        self.b_w    += delta_bw

        # Re-preintegrate with corrected bias
        preint_results = [
            preintegrate(seg, self.b_a, self.b_w) for seg in imu_segments
        ]
        alphas = [r[0] for r in preint_results]
        betas  = [r[1] for r in preint_results]

        # ── Step 3: Solve for scale, gravity, velocities ──────────────────
        s, g_c0, velocities = solve_scale_gravity_velocity(
            keyframe_poses, alphas, betas, dts, self.p_bc, self.R_bc,
        )

        assert s > 0, "Scale must be positive"
        assert 9.0 < np.linalg.norm(g_c0) < 10.5, \
            f"Gravity magnitude {np.linalg.norm(g_c0):.2f} m/s² unreasonable"

        # ── Step 4: Gravity refinement ────────────────────────────────────
        g_c0 = refine_gravity(
            g_c0, keyframe_poses, alphas, betas, dts, self.p_bc, velocities
        )

        return s, g_c0, velocities

    def scale_visual_map(self, s, keyframe_poses, map_points):
        return [(s * p, R) for p, R in keyframe_poses], [s * X for X in map_points]