"""
imu_pipeline.py
───────────────
Autonomous IMU preintegration thread.

For every raw IMU sample that arrives:
  1. Integrate into running alpha / beta / gamma / Jacobians  (midpoint method)
  2. Append (dt, accel, gyro) to _current_raw_samples

When vo_pipeline fires notify_frame(t)  [called EVERY camera frame]:
  - Finalise current chunk → stores raw_samples + preint results in IMUChunk
  - Reset integrator and raw list for the next chunk

VIA then collects ALL chunks between two keyframe timestamps,
concatenates their raw_samples into one flat list, and calls
preintegrate() fresh on the full sequence.

Fixes applied
─────────────
  [Issue 1]  IMUChunk field names now match imu_chunk_db exactly
             (J_a_ba / J_a_bw / J_v_ba / J_v_bw / J_q_bw / b_a / b_w)
  [Issue 2]  Threading: replaced fragmented lock-peek-pop with a
             threading.Event + atomic pop pattern
  [Issue 3]  peek-then-pop race eliminated — pop happens in one lock scope
  [Issue 9]  Jacobians fully propagated inside propagate() using the same
             equations as preintegrate.py  (F-matrix / skew-symmetric method)
"""

import threading
import numpy as np
from collections import deque
from typing import Optional, List, Tuple
import time

from Inertial.imu_chunk_db import IMUChunkDatabase, IMUChunk


# ── Helpers ───────────────────────────────────────────────────────────────────

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


# ── IMU Integrator ────────────────────────────────────────────────────────────

class IMUIntegrator:
    """
    Incremental midpoint preintegrator.
    Matches field names used by IMUChunk in imu_chunk_db.py.
    Jacobians propagated using the same F-matrix equations as preintegrate.py.
    """

    def __init__(self, b_a: np.ndarray, b_w: np.ndarray):
        self.b_a = np.array(b_a, dtype=np.float64).flatten()
        self.b_w = np.array(b_w, dtype=np.float64).flatten()
        self.reset()

    def reset(self):
        self.alpha    = np.zeros(3, dtype=np.float64)
        self.beta     = np.zeros(3, dtype=np.float64)
        self.gamma    = np.array([1., 0., 0., 0.], dtype=np.float64)
        self.dt_total = 0.0

        # Jacobians wrt biases — same names as IMUChunk fields
        self.J_a_ba = np.zeros((3, 3), dtype=np.float64)   # d_alpha / d_b_a
        self.J_a_bw = np.zeros((3, 3), dtype=np.float64)   # d_alpha / d_b_w
        self.J_v_ba = np.zeros((3, 3), dtype=np.float64)   # d_beta  / d_b_a
        self.J_v_bw = np.zeros((3, 3), dtype=np.float64)   # d_beta  / d_b_w
        self.J_q_bw = np.zeros((3, 3), dtype=np.float64)   # d_gamma / d_b_w

        self._last_acc_ub: Optional[np.ndarray] = None
        self._last_gyr_ub: Optional[np.ndarray] = None

    def propagate(self, dt: float, acc: np.ndarray, gyr: np.ndarray):
        """Integrate one IMU sample using the midpoint method."""
        if dt <= 0:
            return

        acc_ub = acc - self.b_a
        gyr_ub = gyr - self.b_w

        if self._last_acc_ub is None:
            self._last_acc_ub = acc_ub
            self._last_gyr_ub = gyr_ub

        # ── Orientation update (midpoint angular velocity) ────────────────
        w_mid = 0.5 * (self._last_gyr_ub + gyr_ub)
        w_len = np.linalg.norm(w_mid)

        dq = np.array([1., 0., 0., 0.], dtype=np.float64)
        if w_len > 1e-6:
            theta   = w_len * dt
            axis    = w_mid / w_len
            dq[0]   = np.cos(theta / 2.0)
            dq[1:4] = np.sin(theta / 2.0) * axis
        else:
            dq[1:4] = 0.5 * w_mid * dt

        # gamma_new = gamma ⊗ dq
        w1, x1, y1, z1 = self.gamma
        w2, x2, y2, z2 = dq
        gamma_next = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ], dtype=np.float64)
        gamma_next /= np.linalg.norm(gamma_next)

        # ── Position / velocity update (midpoint acceleration) ────────────
        R_curr = _quat_to_rot(self.gamma)
        R_next = _quat_to_rot(gamma_next)
        acc_mid = 0.5 * (R_curr @ self._last_acc_ub + R_next @ acc_ub)

        self.alpha    += self.beta * dt + 0.5 * acc_mid * dt * dt
        self.beta     += acc_mid * dt
        self.gamma     = gamma_next
        self.dt_total += dt

        # ── Jacobian propagation (F-matrix, same as preintegrate.py) ─────
        # Use R_curr and current bias-corrected values for the linearisation
        skew_a = _skew(acc_ub)
        skew_w = _skew(gyr_ub)

        # d_alpha/d_b_a, d_alpha/d_b_w
        self.J_a_ba += self.J_v_ba * dt - 0.5 * R_curr * dt**2
        self.J_a_bw += (self.J_v_bw * dt
                        - 0.5 * R_curr @ skew_a @ self.J_q_bw * dt**2)

        # d_beta/d_b_a, d_beta/d_b_w
        self.J_v_ba += -R_curr * dt
        self.J_v_bw += -R_curr @ skew_a @ self.J_q_bw * dt

        # d_gamma/d_b_w
        self.J_q_bw += -skew_w @ self.J_q_bw * dt + (-np.eye(3)) * dt

        # Cache for next midpoint step
        self._last_acc_ub = acc_ub
        self._last_gyr_ub = gyr_ub

    def finalise(self, t_start: float, t_end: float,
                 raw_samples: list) -> IMUChunk:
        """Package accumulated state + raw samples into an IMUChunk."""
        return IMUChunk(
            t_start     = t_start,
            t_end       = t_end,
            dt_total    = self.dt_total,
            raw_samples = list(raw_samples),
            alpha       = self.alpha.copy(),
            beta        = self.beta.copy(),
            gamma       = self.gamma.copy(),
            J_a_ba      = self.J_a_ba.copy(),
            J_a_bw      = self.J_a_bw.copy(),
            J_v_ba      = self.J_v_ba.copy(),
            J_v_bw      = self.J_v_bw.copy(),
            J_q_bw      = self.J_q_bw.copy(),
            b_a         = self.b_a.copy(),
            b_w         = self.b_w.copy(),
        )


# ── Raw sample ────────────────────────────────────────────────────────────────

class _RawSample:
    __slots__ = ('stamp', 'accel', 'gyro')

    def __init__(self, stamp: float, accel: np.ndarray, gyro: np.ndarray):
        self.stamp = stamp
        self.accel = np.array(accel, dtype=np.float64)
        self.gyro  = np.array(gyro,  dtype=np.float64)


# ── IMUPipeline ───────────────────────────────────────────────────────────────

class IMUPipeline:
    """
    Autonomous background thread that:
      - Continuously drains raw IMU samples and integrates them
      - Simultaneously accumulates raw_samples for each chunk
      - On notify_frame(t): finalises and stores the current chunk,
        resets for the next one

    notify_frame() is called every camera frame (not just keyframes),
    so chunk_db gets one chunk per frame interval.
    notify_keyframe() is an alias for notify_frame() for compatibility.
    """

    _SENTINEL = object()

    def __init__(
        self,
        chunk_db:          IMUChunkDatabase,
        b_a:               np.ndarray        = None,
        b_w:               np.ndarray        = None,
        imu_noise_params:  Optional[dict]    = None,
        raw_buffer_maxlen: int               = 5000,
    ):
        self._db = chunk_db

        _b_a = np.zeros(3) if b_a is None else np.array(b_a, dtype=np.float64)
        _b_w = np.zeros(3) if b_w is None else np.array(b_w, dtype=np.float64)

        self._integrator = IMUIntegrator(_b_a, _b_w)

        # ── Raw IMU queue (producer: ROS callback, consumer: worker) ──────
        self._raw_queue: deque = deque(maxlen=raw_buffer_maxlen)
        self._raw_event = threading.Event()

        # ── Frame-cut timestamp queue ─────────────────────────────────────
        # Protected by _frame_lock; worker pops atomically
        self._frame_timestamps: deque = deque()
        self._frame_lock = threading.Lock()

        # ── Current open chunk state ──────────────────────────────────────
        self._current_raw_samples: List[Tuple[float, np.ndarray, np.ndarray]] = []
        self._chunk_t_start: Optional[float] = None
        self._last_stamp:    Optional[float] = None

        # ── Bias update lock ──────────────────────────────────────────────
        self._bias_lock = threading.Lock()

        self._running = False
        self._thread:  Optional[threading.Thread] = None

    # ── Public API ────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(
            target=self._worker,
            name='IMUPipeline',
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._raw_queue.append(self._SENTINEL)
        self._raw_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def push_raw(self, stamp: float, accel: np.ndarray, gyro: np.ndarray) -> None:
        """Called from ROS IMU callback. Thread-safe."""
        self._raw_queue.append(_RawSample(stamp, accel, gyro))
        self._raw_event.set()

    def notify_frame(self, timestamp: float) -> None:
        """
        Called by vo_pipeline for EVERY camera frame.
        Signals the worker to cut a chunk at this timestamp.
        """
        with self._frame_lock:
            self._frame_timestamps.append(timestamp)
        self._raw_event.set()

    def notify_keyframe(self, timestamp: float) -> None:
        """Alias for notify_frame() — kept for compatibility."""
        self.notify_frame(timestamp)

    def update_biases(self, b_a: np.ndarray, b_w: np.ndarray) -> None:
        """Update IMU biases. Takes effect from the next chunk."""
        with self._bias_lock:
            self._integrator.b_a = np.array(b_a, dtype=np.float64)
            self._integrator.b_w = np.array(b_w, dtype=np.float64)

    # ── Worker ────────────────────────────────────────────────────────────

    def _worker(self) -> None:
        while self._running:
            self._raw_event.wait(timeout=0.01)
            self._raw_event.clear()

            # Drain all available raw samples
            while self._raw_queue:
                item = self._raw_queue.popleft()

                if item is self._SENTINEL:
                    return

                sample: _RawSample = item

                # First sample ever — initialise chunk timing only
                if self._chunk_t_start is None:
                    self._chunk_t_start = sample.stamp
                    self._last_stamp    = sample.stamp
                    continue

                dt = sample.stamp - self._last_stamp
                if dt <= 0:
                    continue    # out-of-order / duplicate

                # ── Check for pending frame cut BEFORE integrating ────────
                # Atomic pop: peek and remove in one lock scope (fix Issue 3)
                cut_t = self._pop_frame_cut_before(sample.stamp)
                if cut_t is not None:
                    self._finalise_chunk(cut_t)
                    # Integrate the remainder of this sample into new chunk
                    dt_rem = sample.stamp - cut_t
                    if dt_rem > 0:
                        self._integrate_sample(dt_rem, sample.accel, sample.gyro)
                    self._last_stamp = sample.stamp
                    continue

                # Normal path: integrate and store raw sample
                self._integrate_sample(dt, sample.accel, sample.gyro)
                self._last_stamp = sample.stamp

            # After draining IMU queue, also check for any pending cuts
            # that haven't been triggered by a new IMU sample yet
            cut_t = self._pop_frame_cut_before(
                self._last_stamp + 1.0 if self._last_stamp else 0.0
            )
            if cut_t is not None and self._last_stamp is not None:
                if cut_t <= self._last_stamp:
                    self._finalise_chunk(cut_t)

    def _integrate_sample(self, dt: float,
                           accel: np.ndarray, gyro: np.ndarray) -> None:
        """Integrate one sample AND save raw tuple. Always together."""
        self._integrator.propagate(dt, accel, gyro)
        self._current_raw_samples.append((dt, accel.copy(), gyro.copy()))

    def _pop_frame_cut_before(self, stamp: float) -> Optional[float]:
        """
        Atomically pop and return the earliest frame-cut timestamp ≤ stamp.
        Returns None if none pending.  (Fix for Issues 2 & 3)
        """
        with self._frame_lock:
            if self._frame_timestamps and self._frame_timestamps[0] <= stamp:
                return self._frame_timestamps.popleft()
        return None

    def _finalise_chunk(self, t_end: float) -> None:
        """Seal current chunk, store in DB, reset for next chunk."""
        if self._chunk_t_start is None or self._integrator.dt_total <= 0:
            self._chunk_t_start       = t_end
            self._current_raw_samples = []
            self._integrator.reset()
            return

        chunk = self._integrator.finalise(
            t_start     = self._chunk_t_start,
            t_end       = t_end,
            raw_samples = self._current_raw_samples,
        )
        self._db.store(chunk)

        # print(f"[IMUPipeline] Chunk [{self._chunk_t_start:.3f}→{t_end:.3f}]  "
        #       f"dt={chunk.dt_total:.3f}s  n_raw={chunk.n_samples()}  "
        #       f"total={self._db.get_chunk_count()}")

        self._chunk_t_start       = t_end
        self._current_raw_samples = []
        self._integrator.reset()

    # ── Diagnostics ───────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            'running'           : self._running,
            'raw_queue_len'     : len(self._raw_queue),
            'pending_frame_cuts': len(self._frame_timestamps),
            'current_chunk_dt'  : self._integrator.dt_total,
            'current_raw_count' : len(self._current_raw_samples),
            'db_total_chunks'   : self._db.get_chunk_count(),
        }