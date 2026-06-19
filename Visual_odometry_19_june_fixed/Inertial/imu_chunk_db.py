"""
imu_chunk_db.py
───────────────
Thread-safe store for completed IMU preintegration chunks.

Each chunk covers the IMU measurements between two consecutive camera
frame timestamps. Each chunk stores BOTH:
  - raw_samples : real (dt, accel, gyro) tuples — fed to VIA for fresh
                  preintegration over the full keyframe-to-keyframe interval
  - preintegrated results — for diagnostics / covariance queries

Field names match imu_pipeline.IMUIntegrator exactly:
    J_a_ba, J_a_bw  (alpha Jacobians)
    J_v_ba, J_v_bw  (beta  Jacobians)
    J_q_bw          (gamma Jacobian)
    b_a, b_w        (biases active during this chunk)
"""

import threading
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

RawSample = Tuple[float, np.ndarray, np.ndarray]   # (dt, accel(3,), gyro(3,))


@dataclass
class IMUChunk:
    t_start:  float
    t_end:    float
    dt_total: float

    # ── Raw samples — primary data for VIA fresh preintegration ──────────
    raw_samples: List[RawSample]     # [(dt, accel(3,), gyro(3,)), ...]

    # ── Preintegrated results ─────────────────────────────────────────────
    alpha:    np.ndarray             # (3,)
    beta:     np.ndarray             # (3,)
    gamma:    np.ndarray             # (4,)  quaternion [w,x,y,z]

    # Jacobians — field names match IMUIntegrator
    J_a_ba:   np.ndarray             # (3,3)  d_alpha / d_b_a
    J_a_bw:   np.ndarray             # (3,3)  d_alpha / d_b_w
    J_v_ba:   np.ndarray             # (3,3)  d_beta  / d_b_a
    J_v_bw:   np.ndarray             # (3,3)  d_beta  / d_b_w
    J_q_bw:   np.ndarray             # (3,3)  d_gamma / d_b_w

    # Biases used during this chunk
    b_a:      np.ndarray             # (3,)
    b_w:      np.ndarray             # (3,)

    def n_samples(self) -> int:
        return len(self.raw_samples)

    def get_preint_tuple(self):
        """
        Return preintegrated values in preintegrate() return order:
        (alpha, beta, gamma,
         J_alpha_ba, J_alpha_bw,
         J_beta_ba,  J_beta_bw,
         J_gamma_bw, P=None)
        Field names are mapped to the preintegrate.py convention here.
        """
        return (
            self.alpha, self.beta, self.gamma,
            self.J_a_ba, self.J_a_bw,
            self.J_v_ba, self.J_v_bw,
            self.J_q_bw, None,          # P not computed in IMUIntegrator
        )


class IMUChunkDatabase:
    """
    Ordered, thread-safe store for IMUChunk objects.
    Chunks are kept in ascending t_start order.
    Old chunks are evicted when the store exceeds max_chunks.
    """

    def __init__(self, max_chunks: int = 500):
        # 500 chunks at ~30 ms each ≈ 15 s of history — plenty for VIA
        self._chunks: List[IMUChunk] = []
        self._lock       = threading.Lock()
        self._max_chunks = max_chunks

    # ── Write ─────────────────────────────────────────────────────────────

    def store(self, chunk: IMUChunk) -> None:
        with self._lock:
            # Almost always appended at the end; scan from back for safety
            inserted = False
            for i in range(len(self._chunks) - 1, -1, -1):
                if self._chunks[i].t_start <= chunk.t_start:
                    self._chunks.insert(i + 1, chunk)
                    inserted = True
                    break
            if not inserted:
                self._chunks.insert(0, chunk)
            if len(self._chunks) > self._max_chunks:
                self._chunks = self._chunks[-self._max_chunks:]

    # ── Read ──────────────────────────────────────────────────────────────

    def get_chunks_between(
        self,
        t_start:   float,
        t_end:     float,
        tolerance: float = 0.05,
    ) -> List[IMUChunk]:
        """
        Return ALL chunks whose interval falls within [t_start, t_end].
        Used by VIA to collect every per-frame chunk between two keyframes.
        """
        with self._lock:
            return [
                c for c in self._chunks
                if c.t_start >= t_start - tolerance
                and c.t_end  <= t_end   + tolerance
            ]

    def get_latest_n_chunks(self, n: int) -> List[IMUChunk]:
        with self._lock:
            return list(self._chunks[-n:])

    def get_chunk_count(self) -> int:
        with self._lock:
            return len(self._chunks)

    def trim_before(self, t: float) -> int:
        with self._lock:
            before = len(self._chunks)
            self._chunks = [c for c in self._chunks if c.t_end >= t]
            return before - len(self._chunks)

    def latest_timestamp(self) -> Optional[float]:
        with self._lock:
            return self._chunks[-1].t_end if self._chunks else None

    def earliest_timestamp(self) -> Optional[float]:
        with self._lock:
            return self._chunks[0].t_start if self._chunks else None

    def summary(self) -> dict:
        with self._lock:
            return {
                'n_chunks'     : len(self._chunks),
                't_start'      : self._chunks[0].t_start  if self._chunks else None,
                't_end'        : self._chunks[-1].t_end   if self._chunks else None,
                'total_dt'     : sum(c.dt_total for c in self._chunks),
                'n_raw_samples': sum(c.n_samples() for c in self._chunks),
            }