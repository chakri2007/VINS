"""
imu_chunk_db.py
───────────────
Thread-safe store for completed IMU preintegration chunks.

Each chunk covers the IMU measurements between two consecutive keyframe
timestamps.  Crucially, each chunk stores BOTH:
  - raw_samples  : the real (dt, accel, gyro) tuples collected during that
                   interval — fed directly to VIA.run() for preintegration
  - preintegrated results (alpha, beta, gamma, Jacobians, covariance)
                 — available for diagnostics and covariance queries

Chunks are stored in chronological order and evicted when the store
exceeds max_chunks.
"""

import threading
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# Each raw IMU sample: (dt, accel(3,), gyro(3,))
RawSample = Tuple[float, np.ndarray, np.ndarray]


@dataclass
class IMUChunk:
    """
    A completed preintegration segment between two keyframe timestamps.

    raw_samples   — real IMU data collected during [t_start, t_end].
                    Passed directly to VisualInertialAlignment.run() so
                    VIA can call preintegrate() on real data (not reconstructed).

    All Jacobians follow the VINS-Mono convention (b_k body frame).
    """
    t_start:    float
    t_end:      float
    dt_total:   float

    # ── Real raw samples — primary data source for VIA ────────────────────
    raw_samples: List[RawSample]        # [(dt, accel(3,), gyro(3,)), ...]

    # ── Preintegrated results — available for diagnostics / covariance ────
    alpha:      np.ndarray              # (3,)
    beta:       np.ndarray              # (3,)
    gamma:      np.ndarray              # (4,)  quaternion [w,x,y,z]

    J_alpha_ba: np.ndarray              # (3,3)
    J_alpha_bw: np.ndarray              # (3,3)
    J_beta_ba:  np.ndarray              # (3,3)
    J_beta_bw:  np.ndarray              # (3,3)
    J_gamma_bw: np.ndarray              # (3,3)

    P:          np.ndarray              # (9,9) covariance

    def get_preint_tuple(self):
        """
        Return preintegrated values in the same order as preintegrate()
        returns them — useful if a caller wants to skip re-integration.

        Returns:
            (alpha, beta, gamma,
             J_alpha_ba, J_alpha_bw,
             J_beta_ba,  J_beta_bw,
             J_gamma_bw, P)
        """
        return (
            self.alpha,    self.beta,      self.gamma,
            self.J_alpha_ba, self.J_alpha_bw,
            self.J_beta_ba,  self.J_beta_bw,
            self.J_gamma_bw, self.P,
        )

    def n_samples(self) -> int:
        return len(self.raw_samples)


class IMUChunkDatabase:
    """
    Ordered, thread-safe store for IMUChunk objects.

    Chunks are kept in ascending t_start order.
    Old chunks are evicted automatically when the window grows beyond
    max_chunks.
    """

    def __init__(self, max_chunks: int = 200):
        self._chunks: List[IMUChunk] = []
        self._lock        = threading.Lock()
        self._max_chunks  = max_chunks

    # ── Write ─────────────────────────────────────────────────────────────

    def store(self, chunk: IMUChunk) -> None:
        """Insert a completed chunk. Maintains chronological order."""
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

            # Evict oldest if over capacity
            if len(self._chunks) > self._max_chunks:
                self._chunks = self._chunks[-self._max_chunks:]

    # ── Read ──────────────────────────────────────────────────────────────

    def get_latest_n_chunks(self, n: int) -> List[IMUChunk]:
        """
        Return the n most recent completed chunks in chronological order.
        Used by VIA to get the IMU segments matching the latest N keyframe
        intervals.
        """
        with self._lock:
            return list(self._chunks[-n:])

    def get_chunks_between(
        self,
        t_start: float,
        t_end:   float,
        tolerance: float = 0.05,
    ) -> List[IMUChunk]:
        """
        Return all chunks whose time range falls within [t_start, t_end].
        tolerance (seconds) handles small camera/IMU clock mismatches.
        """
        with self._lock:
            return [
                c for c in self._chunks
                if c.t_start >= t_start - tolerance
                and c.t_end  <= t_end   + tolerance
            ]

    def get_chunk_count(self) -> int:
        with self._lock:
            return len(self._chunks)

    def trim_before(self, t: float) -> int:
        """Evict all chunks that ended before timestamp t. Returns count removed."""
        with self._lock:
            before = len(self._chunks)
            self._chunks = [c for c in self._chunks if c.t_end >= t]
            return before - len(self._chunks)

    def get_all_chunks(self) -> List[IMUChunk]:
        with self._lock:
            return list(self._chunks)

    def latest_timestamp(self) -> Optional[float]:
        with self._lock:
            return self._chunks[-1].t_end if self._chunks else None

    def earliest_timestamp(self) -> Optional[float]:
        with self._lock:
            return self._chunks[0].t_start if self._chunks else None

    def summary(self) -> dict:
        with self._lock:
            return {
                'n_chunks'  : len(self._chunks),
                't_start'   : self._chunks[0].t_start  if self._chunks else None,
                't_end'     : self._chunks[-1].t_end   if self._chunks else None,
                'total_dt'  : sum(c.dt_total for c in self._chunks),
                'n_raw_samples': sum(c.n_samples() for c in self._chunks),
            }