import numpy as np
from typing import List, Tuple, Optional

from Visual_odometry.Inertial.imu_chunk_db import IMUChunkDatabase
from Visual_odometry.Inertial.calib_gyro import calibrate_gyro_bias
from Visual_odometry.Inertial.preintegrate import preintegrate
from Visual_odometry.Inertial.optimize_scale_gravity_velocity import (
    solve_scale_gravity_velocity,
)
from Visual_odometry.Inertial.refine_gravity import refine_gravity


def pose_to_quat(R: np.ndarray) -> np.ndarray:
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s  = 0.5 / np.sqrt(trace + 1.0)
        w  = 0.25 / s
        x  = (R[2,1] - R[1,2]) * s
        y  = (R[0,2] - R[2,0]) * s
        z  = (R[1,0] - R[0,1]) * s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s  = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w  = (R[2,1] - R[1,2]) / s
        x  = 0.25 * s
        y  = (R[0,1] + R[1,0]) / s
        z  = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s  = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w  = (R[0,2] - R[2,0]) / s
        x  = (R[0,1] + R[1,0]) / s
        y  = 0.25 * s
        z  = (R[1,2] + R[2,1]) / s
    else:
        s  = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w  = (R[1,0] - R[0,1]) / s
        x  = (R[0,2] + R[2,0]) / s
        y  = (R[1,2] + R[2,1]) / s
        z  = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


class VisualInertialAlignment:

    def __init__(self, T_bc: np.ndarray, imu_noise_params: dict,
                 chunk_db: IMUChunkDatabase):
        self.R_bc = T_bc[:3, :3]
        self.p_bc = T_bc[:3,  3]
        self.noise = imu_noise_params

        self.b_w = np.zeros(3)
        self.b_a = np.zeros(3)

        self._chunk_db = chunk_db

    def run_between(
        self,
        keyframe_poses: List[Tuple[np.ndarray, np.ndarray]],
        kf_timestamps:  List[float],
        tolerance:      float = 0.05,
    ):
        assert len(kf_timestamps) == len(keyframe_poses), \
            "kf_timestamps must have one entry per keyframe pose"
        assert len(keyframe_poses) >= 5, \
            "Need at least 5 keyframes for reliable alignment"

        n_pairs = len(keyframe_poses) - 1
        imu_segments = []

        for i in range(n_pairs):
            t_start = kf_timestamps[i]
            t_end   = kf_timestamps[i + 1]

            chunks = self._chunk_db.get_chunks_between(
                t_start, t_end, tolerance=tolerance
            )

            if not chunks:
                raise RuntimeError(
                    f"[VIA] No IMU chunk found between "
                    f"t={t_start:.3f} and t={t_end:.3f}. "
                    f"DB has {self._chunk_db.get_chunk_count()} chunks total. "
                    f"Ensure imu_pipeline.notify_keyframe() is being called."
                )

            best_chunk = min(
                chunks,
                key=lambda c: abs(c.t_start - t_start) + abs(c.t_end - t_end)
            )

            if not best_chunk.raw_samples:
                raise RuntimeError(
                    f"[VIA] Chunk [{t_start:.3f}→{t_end:.3f}] has no raw "
                    f"samples. Check imu_pipeline._integrate_and_store()."
                )

            imu_segments.append(best_chunk.raw_samples)

            print(f"[VIA] Segment {i}: chunk [{best_chunk.t_start:.3f}"
                  f"→{best_chunk.t_end:.3f}]  "
                  f"n_raw={best_chunk.n_samples()}  "
                  f"dt={best_chunk.dt_total:.3f}s")

        return self.run(keyframe_poses, imu_segments)


    def run(
        self,
        keyframe_poses: List[Tuple[np.ndarray, np.ndarray]],
        imu_segments:   List[list],
    ):
        assert len(imu_segments) == len(keyframe_poses) - 1, \
            "Need one IMU segment per consecutive keyframe pair"
        assert len(keyframe_poses) >= 5, \
            "Need at least 5 keyframes for reliable initialization"

        preint_results = []
        for segment in imu_segments:
            result = preintegrate(segment, self.b_a, self.b_w)
            preint_results.append(result)

        alphas      = [r[0] for r in preint_results]
        betas       = [r[1] for r in preint_results]
        gammas      = [r[2] for r in preint_results]
        J_gamma_bws = [r[7] for r in preint_results]
        dts         = [sum(d for d, a, g in seg) for seg in imu_segments]

        visual_quats = [pose_to_quat(R) for _, R in keyframe_poses]
        delta_bw     = calibrate_gyro_bias(visual_quats, gammas, J_gamma_bws)
        self.b_w    += delta_bw

        preint_results = [
            preintegrate(seg, self.b_a, self.b_w) for seg in imu_segments
        ]
        alphas = [r[0] for r in preint_results]
        betas  = [r[1] for r in preint_results]

        s, g_c0, velocities = solve_scale_gravity_velocity(
            keyframe_poses, alphas, betas, dts,
            self.p_bc, self.R_bc,
        )

        assert s > 0, "Scale must be positive"
        assert 9.0 < np.linalg.norm(g_c0) < 10.5, \
            f"Gravity magnitude {np.linalg.norm(g_c0):.2f} m/s² is unreasonable"

        g_c0 = refine_gravity(
            g_c0, keyframe_poses, alphas, betas, dts, self.p_bc, velocities
        )

        return s, g_c0, velocities

    def scale_visual_map(
        self,
        s: float,
        keyframe_poses: List[Tuple[np.ndarray, np.ndarray]],
        map_points:     List[np.ndarray],
    ):
        scaled_poses  = [(s * p, R) for p, R in keyframe_poses]
        scaled_points = [s * X for X in map_points]
        return scaled_poses, scaled_points