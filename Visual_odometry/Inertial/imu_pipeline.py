"""
imu_pipeline.py - Updated for reliable per-frame chunking
"""

import threading
import numpy as np
from collections import deque
from typing import Optional, List, Tuple

from Inertial.imu_chunk_db import IMUChunkDatabase, IMUChunk


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y)],
        [2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x)],
        [2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([
        [ 0,    -v[2],  v[1]],
        [ v[2],  0,    -v[0]],
        [-v[1],  v[0],  0   ],
    ], dtype=np.float64)


class IMUIntegrator:
    def __init__(self, b_a: np.ndarray, b_w: np.ndarray):
        self.b_a = np.array(b_a, dtype=np.float64).flatten()
        self.b_w = np.array(b_w, dtype=np.float64).flatten()
        self.reset()

    def reset(self):
        self.alpha    = np.zeros(3, dtype=np.float64)
        self.beta     = np.zeros(3, dtype=np.float64)
        self.gamma    = np.array([1., 0., 0., 0.], dtype=np.float64)
        self.dt_total = 0.0

        self.J_a_ba = np.zeros((3, 3), dtype=np.float64)
        self.J_a_bw = np.zeros((3, 3), dtype=np.float64)
        self.J_v_ba = np.zeros((3, 3), dtype=np.float64)
        self.J_v_bw = np.zeros((3, 3), dtype=np.float64)
        self.J_q_bw = np.zeros((3, 3), dtype=np.float64)

        self._last_acc_ub = None
        self._last_gyr_ub = None

    def propagate(self, dt: float, acc: np.ndarray, gyr: np.ndarray):
        if dt <= 0:
            return

        acc_ub = acc - self.b_a
        gyr_ub = gyr - self.b_w

        if self._last_acc_ub is None:
            self._last_acc_ub = acc_ub
            self._last_gyr_ub = gyr_ub

        # Midpoint integration (same as before)
        w_mid = 0.5 * (self._last_gyr_ub + gyr_ub)
        w_len = np.linalg.norm(w_mid)

        dq = np.array([1., 0., 0., 0.], dtype=np.float64)
        if w_len > 1e-6:
            theta = w_len * dt
            axis = w_mid / w_len
            dq[0] = np.cos(theta / 2)
            dq[1:4] = np.sin(theta / 2) * axis

        # Quaternion multiply
        w1, x1, y1, z1 = self.gamma
        w2, x2, y2, z2 = dq
        gamma_next = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])
        gamma_next /= np.linalg.norm(gamma_next)

        R_curr = _quat_to_rot(self.gamma)
        R_next = _quat_to_rot(gamma_next)
        acc_mid = 0.5 * (R_curr @ self._last_acc_ub + R_next @ acc_ub)

        self.alpha += self.beta * dt + 0.5 * acc_mid * dt**2
        self.beta  += acc_mid * dt
        self.gamma = gamma_next
        self.dt_total += dt

        # Jacobian updates (unchanged)
        skew_a = _skew(acc_ub)
        skew_w = _skew(gyr_ub)
        self.J_a_ba += self.J_v_ba * dt - 0.5 * R_curr * dt**2
        self.J_a_bw += (self.J_v_bw * dt - 0.5 * R_curr @ skew_a @ self.J_q_bw * dt**2)
        self.J_v_ba += -R_curr * dt
        self.J_v_bw += -R_curr @ skew_a @ self.J_q_bw * dt
        self.J_q_bw += -skew_w @ self.J_q_bw * dt - np.eye(3) * dt

        self._last_acc_ub = acc_ub
        self._last_gyr_ub = gyr_ub

    def finalise(self, t_start: float, t_end: float, raw_samples: list) -> IMUChunk:
        return IMUChunk(
            t_start=t_start, t_end=t_end, dt_total=self.dt_total,
            raw_samples=list(raw_samples),
            alpha=self.alpha.copy(), beta=self.beta.copy(), gamma=self.gamma.copy(),
            J_a_ba=self.J_a_ba.copy(), J_a_bw=self.J_a_bw.copy(),
            J_v_ba=self.J_v_ba.copy(), J_v_bw=self.J_v_bw.copy(),
            J_q_bw=self.J_q_bw.copy(),
            b_a=self.b_a.copy(), b_w=self.b_w.copy(),
        )


class _RawSample:
    __slots__ = ('stamp', 'accel', 'gyro')
    def __init__(self, stamp: float, accel: np.ndarray, gyro: np.ndarray):
        self.stamp = stamp
        self.accel = np.array(accel, dtype=np.float64)
        self.gyro  = np.array(gyro,  dtype=np.float64)


class IMUPipeline:
    _SENTINEL = object()

    def __init__(self, chunk_db: IMUChunkDatabase, b_a=None, b_w=None, imu_noise_params=None, raw_buffer_maxlen=5000):
        self._db = chunk_db
        _b_a = np.zeros(3) if b_a is None else np.array(b_a, dtype=np.float64)
        _b_w = np.zeros(3) if b_w is None else np.array(b_w, dtype=np.float64)
        self._integrator = IMUIntegrator(_b_a, _b_w)

        self._raw_queue = deque(maxlen=raw_buffer_maxlen)
        self._raw_event = threading.Event()

        self._frame_timestamps = deque()
        self._frame_lock = threading.Lock()

        self._current_raw_samples = []
        self._chunk_t_start = None
        self._last_stamp = None

        self._running = False
        self._thread = None

    def start(self):
        if self._running: return
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True, name='IMUPipeline')
        self._thread.start()

    def stop(self):
        self._running = False
        self._raw_queue.append(self._SENTINEL)
        self._raw_event.set()
        if self._thread: self._thread.join(timeout=2.0)

    def push_raw(self, stamp: float, accel: np.ndarray, gyro: np.ndarray):
        self._raw_queue.append(_RawSample(stamp, accel, gyro))
        self._raw_event.set()

    def notify_frame(self, timestamp: float):
        with self._frame_lock:
            self._frame_timestamps.append(timestamp)
        self._raw_event.set()

    def notify_keyframe(self, timestamp: float):
        self.notify_frame(timestamp)

    def _worker(self):
        while self._running:
            self._raw_event.wait(timeout=0.005)
            self._raw_event.clear()

            while self._raw_queue:
                item = self._raw_queue.popleft()
                if item is self._SENTINEL: return
                sample = item

                if self._chunk_t_start is None:
                    self._chunk_t_start = sample.stamp
                    self._last_stamp = sample.stamp
                    continue

                dt = sample.stamp - self._last_stamp
                if dt <= 0: continue

                cut_t = self._pop_frame_cut_before(sample.stamp)
                if cut_t is not None:
                    self._finalise_chunk(cut_t)
                    dt_rem = sample.stamp - cut_t
                    if dt_rem > 1e-6:
                        self._integrate_sample(dt_rem, sample.accel, sample.gyro)
                    self._last_stamp = sample.stamp
                    continue

                self._integrate_sample(dt, sample.accel, sample.gyro)
                self._last_stamp = sample.stamp

            # Safety cut
            if self._last_stamp is not None:
                cut_t = self._pop_frame_cut_before(self._last_stamp + 0.1)
                if cut_t is not None:
                    self._finalise_chunk(cut_t)

    def _integrate_sample(self, dt, accel, gyro):
        self._integrator.propagate(dt, accel, gyro)
        self._current_raw_samples.append((dt, accel.copy(), gyro.copy()))

    def _pop_frame_cut_before(self, stamp):
        with self._frame_lock:
            if self._frame_timestamps and self._frame_timestamps[0] <= stamp:
                return self._frame_timestamps.popleft()
        return None

    def _finalise_chunk(self, t_end):
        if self._chunk_t_start is None or self._integrator.dt_total <= 0:
            self._chunk_t_start = t_end
            self._current_raw_samples = []
            self._integrator.reset()
            return

        chunk = self._integrator.finalise(self._chunk_t_start, t_end, self._current_raw_samples)
        self._db.store(chunk)

        self._chunk_t_start = t_end
        self._current_raw_samples = []
        self._integrator.reset()