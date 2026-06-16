import threading
import numpy as np
from collections import deque
from typing import Optional, List, Tuple

from Visual_odometry.Inertial.imu_chunk_db import IMUChunkDatabase, IMUChunk


# ── Helpers ───────────────────────────────────────────────────────────────────

def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


# ── IMU Integrator State ──────────────────────────────────────────────────────

class IMUIntegrator:
    def __init__(self, b_a: np.ndarray, b_w: np.ndarray):
        self.b_a = np.array(b_a, dtype=np.float64).flatten()
        self.b_w = np.array(b_w, dtype=np.float64).flatten()
        self.reset()

    def reset(self):
        self.alpha    = np.zeros(3, dtype=np.float64)
        self.beta     = np.zeros(3, dtype=np.float64)
        self.gamma    = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64) # quat w,x,y,z
        self.dt_total = 0.0

        # Jacobians w.r.t biases
        self.J_a_ba   = np.zeros((3,3), dtype=np.float64)
        self.J_a_bw   = np.zeros((3,3), dtype=np.float64)
        self.J_v_ba   = np.zeros((3,3), dtype=np.float64)
        self.J_v_bw   = np.zeros((3,3), dtype=np.float64)
        self.J_q_bw   = np.zeros((3,3), dtype=np.float64)

        self._last_acc = None
        self._last_gyr = None

    def propagate(self, dt: float, acc: np.ndarray, gyr: np.ndarray):
        if dt <= 0:
            return

        acc_unbiased = acc - self.b_a
        gyr_unbiased = gyr - self.b_w

        if self._last_acc is None:
            self._last_acc = acc_unbiased
            self._last_gyr = gyr_unbiased

        # Midpoint angular velocity
        w_mid = 0.5 * (self._last_gyr + gyr_unbiased)
        w_len = np.linalg.norm(w_mid)

        # Update Orientation Quat (gamma)
        dq = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        if w_len > 1e-6:
            theta = w_len * dt
            axis  = w_mid / w_len
            dq[0] = np.cos(theta / 2.0)
            dq[1:4] = np.sin(theta / 2.0) * axis
        else:
            dq[1:4] = 0.5 * w_mid * dt

        # Quaternion Multiplication: gamma_new = gamma * dq
        w1, x1, y1, z1 = self.gamma
        w2, x2, y2, z2 = dq
        gamma_next = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dtype=np.float64)
        gamma_next /= np.linalg.norm(gamma_next)

        # Midpoint Acceleration in reference frame
        R_curr = _quat_to_rot(self.gamma)
        R_next = _quat_to_rot(gamma_next)
        
        acc_mid = 0.5 * (R_curr @ self._last_acc + R_next @ acc_unbiased)

        # Position and Velocity updates
        self.alpha += self.beta * dt + 0.5 * acc_mid * dt * dt
        self.beta  += acc_mid * dt
        self.gamma  = gamma_next
        self.dt_total += dt

        # Cache states
        self._last_acc = acc_unbiased
        self._last_gyr = gyr_unbiased

    def finalise(self, t_start: float, t_end: float, raw_samples: list) -> IMUChunk:
        return IMUChunk(
            t_start      = t_start,
            t_end        = t_end,
            dt_total     = self.dt_total,
            alpha        = self.alpha.copy(),
            beta         = self.beta.copy(),
            gamma        = self.gamma.copy(),
            J_a_ba       = self.J_a_ba.copy(),
            J_a_bw       = self.J_a_bw.copy(),
            J_v_ba       = self.J_v_ba.copy(),
            J_v_bw       = self.J_v_bw.copy(),
            J_q_bw       = self.J_q_bw.copy(),
            b_a          = self.b_a.copy(),
            b_w          = self.b_w.copy(),
            raw_samples  = list(raw_samples)
        )


class IMUPipeline:
    def __init__(
        self,
        chunk_db: IMUChunkDatabase,
        b_a: np.ndarray = np.zeros(3),
        b_w: np.ndarray = np.zeros(3),
        imu_noise_params: Optional[dict] = None
    ):
        self._db  = chunk_db
        self._integrator = IMUIntegrator(b_a, b_w)
        self._noise = imu_noise_params or {}

        self._raw_queue = deque(maxlen=2000)
        self._frame_timestamps = deque()

        self._lock = threading.Lock()
        self._running = False
        self._worker_thread = None

        self._last_imu_t = None
        self._chunk_t_start = None
        self._current_raw_samples = []

    def start(self):
        with self._lock:
            if self._running:
                return
            self._running = True
            self._worker_thread = threading.Thread(
                target=self._loop, daemon=True, name="IMUPipelineWorker"
            )
            self._worker_thread.start()

    def stop(self):
        with self._lock:
            self._running = False

    def push_raw(self, timestamp: float, accel: np.ndarray, gyro: np.ndarray):
        self._raw_queue.append((timestamp, accel, gyro))

    def notify_frame(self, timestamp: float):
        with self._lock:
            self._frame_timestamps.append(timestamp)

    def notify_keyframe(self, timestamp: float):
        self.notify_frame(timestamp)

    def _loop(self):
        import time
        while self._running:
            processed_anything = False

            target_cut_t = None
            with self._lock:
                if self._frame_timestamps:
                    target_cut_t = self._frame_timestamps[0]

            if len(self._raw_queue) > 0:
                t_imu, acc, gyr = self._raw_queue[0]

                if target_cut_t is not None and t_imu > target_cut_t:
                    self._cut_chunk(target_cut_t)
                    with self._lock:
                        if self._frame_timestamps:
                            self._frame_timestamps.popleft()
                    processed_anything = True
                    continue

                self._raw_queue.popleft()
                
                if self._last_imu_t is not None:
                    dt = t_imu - self._last_imu_t
                    if dt > 0:
                        self._integrator.propagate(dt, acc, gyr)
                        self._current_raw_samples.append((dt, acc, gyr))

                if self._chunk_t_start is None:
                    self._chunk_t_start = t_imu

                self._last_imu_t = t_imu
                processed_anything = True

            if not processed_anything:
                time.sleep(0.001)

    def _cut_chunk(self, t_end: float):
        if self._chunk_t_start is None or self._integrator.dt_total <= 0:
            self._chunk_t_start    = t_end
            self._current_raw_samples = []
            self._integrator.reset()
            return

        chunk = self._integrator.finalise(
            t_start     = self._chunk_t_start,
            t_end       = t_end,
            raw_samples = self._current_raw_samples,
        )
        self._db.store(chunk)

        print(f"[IMUPipeline] Frame Chunk stored  "
              f"[{self._chunk_t_start:.3f} → {t_end:.3f}]  "
              f"dt={chunk.dt_total:.3f}s  "
              f"n_raw={chunk.n_samples()}  "
              f"total_chunks={self._db.get_chunk_count()}")

        self._chunk_t_start       = t_end
        self._current_raw_samples = []
        self._integrator.reset()

    def status(self) -> dict:
        return {
            'running'           : self._running,
            'raw_queue_len'     : len(self._raw_queue),
            'pending_frame_cuts': len(self._frame_timestamps),
            'current_chunk_dt'  : self._integrator.dt_total,
            'db_total_chunks'   : self._db.get_chunk_count()
        }