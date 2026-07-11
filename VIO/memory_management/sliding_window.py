"""
update_sliding_window — Python port of MATLAB helperFeaturePointManager.updateSlidingWindow.

Key invariant
-------------
`state.current_view_id` always holds the view_id of the last frame whose
observations were successfully written into all_observations / all_ids.
When a non-keyframe is replaced in the window the replaced frame's data is
already in all_observations (it was written earlier), and the NEW frame
takes its slot — but the new frame's data is being written right now, so
`state.current_view_id` must be updated to `view_id` only AFTER the write.

`state.current_sliding_window_index` == len(state.sliding_window_view_ids)
at all times.  Both are updated together.
"""

import bisect

import threading

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from vio_core.ransac import estimate_fundamental_matrix_ransac
from typing import Dict, Tuple

from vio_core.landmarks import Landmark
from imu.imu_measurement import IMUMeasurement
from imu.preintegration import IMUPreintegrator


@dataclass
class SlidingWindowState:
    window_size: int

    current_sliding_window_index: int = 0
    sliding_window_view_ids: list = field(default_factory=list)

    is_key_frame: dict = field(default_factory=dict)

    # view_id -> (N,2) float32 — 2-D point observations
    all_observations: dict = field(default_factory=dict)
    # view_id -> (N,2) int — columns: [view_id, point_id]
    all_ids: dict = field(default_factory=dict)
    # view_id -> (N,) bool — triangulated flag per point
    all_triangulated: dict = field(default_factory=dict)

    # point_id -> int — how many frames this point has survived in
    key_point_track_count: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Landmark database
    #
    # key:
    #     feature point id
    #
    # value:
    #     Landmark
    # ------------------------------------------------------------------
    landmarks: Dict[int, Landmark] = field(default_factory=dict)

    # last view_id that was fully written into all_observations / all_ids
    current_view_id: int = -1

    no_movement_at_start: bool = True

    # ------------------------------------------------------------------
    # IMU measurements
    #
    # A single continuous, timestamp-ordered buffer of every IMU sample
    # received so far (that hasn't been pruned yet). This mirrors the
    # MATLAB reference, which never chunks IMU data by frame-to-frame
    # pairs — it always re-slices the full gyroReadings/accelReadings
    # arrays by timestamp (helperExtractIMUDataBetweenViews). Slicing
    # this buffer by the timestamps of two arbitrary keyframes (see
    # extract_imu_between) works correctly regardless of how many
    # non-keyframes were dropped in between them, which per-pair
    # dict storage kept keyed by raw consecutive frame ids did not.
    #
    # Samples are pruned from the front once the sliding window's
    # oldest keyframe advances past them (see prune_imu_before) —
    # not when individual raw frames are dropped as non-keyframes,
    # since their IMU interval may still be needed to bridge two
    # surviving keyframes.
    # ------------------------------------------------------------------

    imu_buffer: List[IMUMeasurement] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Synchronization primitive used by the backend worker.
    #
    # The backend waits on this condition until the IMU buffer contains
    # measurements covering the timestamp of the frame it wants to
    # process. Every new IMU sample notifies this condition.
    # ------------------------------------------------------------------
    imu_condition: threading.Condition = field(
        default_factory=threading.Condition,
        repr=False,
        compare=False,
    )

    metric_scale: float = 1.0

    gravity: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, -9.81])
    )

    accelerometer_bias: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )

    gyroscope_bias: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )

    velocities: Dict[int, np.ndarray] = field(
        default_factory=dict
    )

    # ------------------------------------------------------------------
    # Per-keyframe bias estimates
    #
    # view_id -> (bias_g (3,), bias_a (3,))
    #
    # Phase 3 (BA_motion / windowed BA) refines a bias estimate for
    # every keyframe individually rather than sharing one global
    # estimate across the whole sequence. `gyroscope_bias` /
    # `accelerometer_bias` above remain as the pre-alignment bootstrap
    # values (zero) and as the fallback for any view_id not yet present
    # here (see VisualInertialOdometry._get_bias) -- they are no longer
    # written to once Phase 3 starts producing per-view estimates.
    # ------------------------------------------------------------------

    biases: Dict[int, Tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict
    )


def _within_image(points: np.ndarray, image_shape) -> np.ndarray:
    rows, cols = image_shape[0], image_shape[1]
    x, y = points[:, 0], points[:, 1]
    return (x >= 1) & (x <= cols) & (y >= 1) & (y <= rows)


def quick_check_parallax(m1: np.ndarray, m2: np.ndarray, parallax_threshold: float):
    diff = m1 - m2
    avg  = np.sqrt((diff ** 2).sum(axis=1)).mean()
    return avg, bool(avg > parallax_threshold)


def update_tracks(
    state: SlidingWindowState,
    image_shape,
    curr_points_tracked: np.ndarray,
    valid_idx: np.ndarray,
    view_id: int,
    F_loop: int,
    F_iterations: int,
    F_confidence: float,
    F_threshold: float,
    fundamental_matrix_ransac_fn: Optional[Callable] = None,
):
    """
    Frame-local half of what used to be update_sliding_window: RANSAC
    outlier rejection against the previously-stored view, plus writing
    this frame's observations/ids/track-count/current_view_id.

    Safe to call on the ROS callback thread, ahead of however far
    backend has gotten: it only ever reads/writes per-view entries
    (all_observations[view_id], all_ids[view_id], ...) and the
    single current_view_id continuity pointer. It never touches
    sliding_window_view_ids, so it cannot race with
    update_window_membership() below, which is the only thing allowed
    to mutate window membership -- and which must run on the backend
    thread, strictly in frame order, for exactly that reason.
    """

    if fundamental_matrix_ransac_fn is None:
        fundamental_matrix_ransac_fn = estimate_fundamental_matrix_ransac

    prev_stored_id = state.current_view_id
    prev_obs       = state.all_observations[prev_stored_id]

    ps_idx = _within_image(curr_points_tracked, image_shape)
    v1     = valid_idx & ps_idx

    # Need at least 8 point pairs for fundamental matrix estimation.
    # If fewer survive the validity+bounds filter, skip RANSAC entirely and
    # keep whatever valid points we have — the frame will likely be discarded
    # as a non-keyframe anyway.
    if v1.sum() >= 8:
        inl_f = None
        for _ in range(F_loop):
            _, inl_ff = fundamental_matrix_ransac_fn(
                prev_obs[v1],
                curr_points_tracked[v1],
                num_trials     = F_iterations,
                confidence     = F_confidence,
                dist_threshold = F_threshold,
            )
            if inl_f is None or np.count_nonzero(inl_ff) > np.count_nonzero(inl_f):
                inl_f = inl_ff

        inl_ff_full          = np.zeros_like(v1, dtype=bool)
        inl_ff_full[v1]      = inl_f
        v1                   = v1 & inl_ff_full

    # ── write this frame's observations ──────────────────────────────────
    state.all_observations[view_id]  = curr_points_tracked[v1]
    state.all_triangulated[view_id]  = state.all_triangulated.get(prev_stored_id, np.zeros(len(prev_obs), dtype=bool))[v1]

    p_ids = state.all_ids[prev_stored_id][v1, 1]
    for pid in p_ids:
        state.key_point_track_count[pid] = state.key_point_track_count.get(pid, 0) + 1

    state.all_ids[view_id]  = np.column_stack(
        [np.full(p_ids.shape[0], view_id), p_ids]
    ) if len(p_ids) > 0 else np.empty((0, 2), dtype=np.int64)
    state.current_view_id   = view_id


def update_window_membership(
    state: SlidingWindowState,
    view_id: int,
    key_frame_parallax: float,
):
    """
    Backend-only half of what used to be update_sliding_window: decides
    whether `view_id` (already written by update_tracks, above) joins
    sliding_window_view_ids, replaces a non-keyframe slot, or gets
    evicted, and updates keyframe flags accordingly.

    MUST run on the backend thread, and only ever in the same order
    backend consumes frames from its queue (FIFO) — this function is
    the sole mutator of sliding_window_view_ids / is_key_frame /
    current_sliding_window_index, and every decision it makes depends
    on the state left by the immediately preceding call. Running it
    out of order, or concurrently with itself from two threads, will
    corrupt the window the same way running it ahead of view_set
    commits did before this split.
    """

    window_state = {
        "isEnoughParallax": False,
        "isWindowFull":     False,
        "isFirstFewViews":  False,
    }

    # ── very first frame ─────────────────────────────────────────────────
    if state.current_sliding_window_index == 0:
        state.current_sliding_window_index = 1
        state.sliding_window_view_ids.append(view_id)
        state.current_view_id = view_id
        return -1, window_state

    no_move_window = 0.5
    warmup_cutoff  = int(np.floor(state.window_size * no_move_window))
    idx            = state.current_sliding_window_index   # alias, easier to read

    # default: oldest window slot (overwritten in every branch below)
    removed_frame_id = state.sliding_window_view_ids[0]

    # ── branch 1: warm-up ────────────────────────────────────────────────
    if (idx < warmup_cutoff and state.no_movement_at_start) or (idx < 2):

        at_warmup_boundary = (
            (idx == warmup_cutoff - 1 and state.no_movement_at_start)
            or (not state.no_movement_at_start and idx == 1)
        )

        if at_warmup_boundary:
            last_kf_view = state.sliding_window_view_ids[idx - 1]
            _, ia, ib = np.intersect1d(
                state.all_ids[last_kf_view][:, 1],
                state.all_ids[view_id][:, 1],
                return_indices=True,
            )
            if len(ia) > 1:
                m1 = state.all_observations[last_kf_view][ia]
                m2 = state.all_observations[view_id][ib]
                _, is_kf = quick_check_parallax(m1, m2, key_frame_parallax)
            else:
                is_kf = False

            if is_kf:
                state.is_key_frame[view_id] = True
                state.sliding_window_view_ids.append(view_id)
                state.current_sliding_window_index += 1
                removed_frame_id = -2
                window_state["isEnoughParallax"] = True
            else:
                # not enough parallax — discard current frame
                removed_frame_id = view_id
        else:
            state.sliding_window_view_ids.append(view_id)
            state.current_sliding_window_index += 1
            removed_frame_id = -3
            window_state["isFirstFewViews"] = True

    # ── branch 2: growing window ──────────────────────────────────────────
    elif (warmup_cutoff <= idx < state.window_size and state.no_movement_at_start) \
      or (idx < state.window_size and not state.no_movement_at_start):

        prev_window_view = state.sliding_window_view_ids[idx - 2]
        last_window_view = state.sliding_window_view_ids[idx - 1]

        ids_pw = state.all_ids.get(prev_window_view)
        ids_lw = state.all_ids.get(last_window_view)

        if ids_pw is not None and ids_lw is not None and len(ids_pw) > 0 and len(ids_lw) > 0:
            _, ia, ib = np.intersect1d(ids_pw[:, 1], ids_lw[:, 1], return_indices=True)
            if len(ia) > 1:
                m1 = state.all_observations[prev_window_view][ia]
                m2 = state.all_observations[last_window_view][ib]
                _, is_kf = quick_check_parallax(m1, m2, key_frame_parallax)
            else:
                is_kf = False
        else:
            is_kf = False

        if is_kf or state.is_key_frame.get(last_window_view, False):
            state.is_key_frame[last_window_view] = True
            state.sliding_window_view_ids.append(view_id)
            state.current_sliding_window_index += 1
            removed_frame_id = -2
            window_state["isEnoughParallax"] = True
        else:
            # replace last (non-KF) slot with current frame
            removed_frame_id = state.sliding_window_view_ids[idx - 1]
            state.sliding_window_view_ids[idx - 1] = view_id
            # current_sliding_window_index unchanged — same number of slots

    # ── branch 3: window full ─────────────────────────────────────────────
    else:
        window_state["isWindowFull"] = True
        prev_window_view = state.sliding_window_view_ids[-2]
        last_window_view = state.sliding_window_view_ids[-1]

        ids_pw = state.all_ids.get(prev_window_view)
        ids_lw = state.all_ids.get(last_window_view)

        if ids_pw is not None and ids_lw is not None and len(ids_pw) > 0 and len(ids_lw) > 0:
            _, ia, ib = np.intersect1d(ids_pw[:, 1], ids_lw[:, 1], return_indices=True)
            if len(ia) > 1:
                m1 = state.all_observations[prev_window_view][ia]
                m2 = state.all_observations[last_window_view][ib]
                _, is_kf = quick_check_parallax(m1, m2, key_frame_parallax)
            else:
                is_kf = False
        else:
            is_kf = False

        last_is_kf = is_kf or state.is_key_frame.get(last_window_view, False)

        if not last_is_kf:
            # replace the last (non-KF) slot — window length unchanged
            removed_frame_id = state.sliding_window_view_ids[-1]
            state.sliding_window_view_ids[-1] = view_id
        else:
            # last was a KF — slide: drop oldest, append current
            state.is_key_frame[last_window_view] = True
            removed_frame_id = state.sliding_window_view_ids[0]
            state.sliding_window_view_ids.pop(0)
            state.sliding_window_view_ids.append(view_id)
            # current_sliding_window_index unchanged — still window_size
            window_state["isEnoughParallax"] = True

    # Note: IMU data is no longer evicted here based on which raw frame
    # got dropped. The IMU buffer is timestamp-indexed (see imu_buffer
    # above) and pruned separately, by timestamp, once the sliding
    # window's oldest surviving keyframe actually advances — see
    # prune_imu_before. A dropped non-keyframe's IMU interval must stay
    # in the buffer, since it may still be needed to bridge the gap
    # between two keyframes that end up on either side of it.

    return removed_frame_id, window_state


# ============================================================
# IMU buffer — continuous, timestamp-ordered storage
# ============================================================

def append_imu_measurement(
    state: SlidingWindowState,
    measurement: IMUMeasurement,
):
    """
    Append one IMU sample to the continuous buffer.

    Every appended measurement notifies any backend thread waiting
    for additional IMU coverage before processing a frame.
    """

    with state.imu_condition:
        state.imu_buffer.append(measurement)

        # Wake up any backend thread waiting for more IMU data.
        state.imu_condition.notify_all()


def _nearest_index(timestamps: List[float], t: float) -> int:
    """
    Index of the buffer sample whose timestamp is closest to `t`.

    Mirrors MATLAB's
        [~,ind] = min(abs(timeStamps.imuTimeStamps - t))
    but exploits the fact that `timestamps` is sorted, via bisect,
    instead of scanning the whole array.
    """

    i = bisect.bisect_left(timestamps, t)

    if i <= 0:
        return 0
    if i >= len(timestamps):
        return len(timestamps) - 1

    before = timestamps[i - 1]
    after = timestamps[i]

    return (i - 1) if (t - before) <= (after - t) else i

def wait_until_imu_ready(
    state: SlidingWindowState,
    target_timestamp: float,
    timeout: float = 0.5,
) -> bool:
    """
    Block until the IMU buffer contains data at or beyond
    `target_timestamp`.

    This provides the same synchronization guarantee used by
    VINS-Mono: backend processing never begins until the IMU
    stream has caught up to the image timestamp.

    Parameters
    ----------
    target_timestamp : float
        Timestamp of the image/frame to be processed.

    timeout : float
        Maximum time (seconds) to wait.

    Returns
    -------
    bool
        True if IMU coverage exists.
        False if timeout occurred.
    """

    with state.imu_condition:

        #
        # Wait until:
        #
        #   imu_buffer is not empty
        #
        # and
        #
        #   latest imu timestamp >= target timestamp
        #
        ready = state.imu_condition.wait_for(
            lambda: (
                len(state.imu_buffer) > 0
                and state.imu_buffer[-1].timestamp >= target_timestamp
            ),
            timeout=timeout,
        )

    return ready


def get_synced_measurement(
    state: SlidingWindowState,
    target_timestamp: float,
    timeout: float = 0.5,
) -> Optional[List[IMUMeasurement]]:
    """
    Direct port of VINS-Mono's estimator_node.cpp::getMeasurements(),
    collapsed to the single-image case this backend needs.

    Supersedes the wait_until_imu_ready() + extract_imu_between() pair
    for the *boundary* call against a frame's own just-captured
    timestamp: those were two separate lock acquisitions, so nothing
    prevented _imu_callback() from mutating imu_buffer in the gap
    between "IMU looks ready" and "now go read it". This function does
    the readiness check and the extraction under the same held lock,
    exactly like getMeasurements() does under m_buf -- there is no
    window where the answer can go stale before it's used.

    It also implements the branch VINS-Mono's getMeasurements() has
    that wait_until_imu_ready() didn't: if the IMU buffer's *oldest*
    remaining sample is already newer than target_timestamp (imu_buffer
    got trimmed past this frame, e.g. the backend fell behind and a
    prior trim_imu_buffer() call already dropped everything up to a
    later frame), this frame's IMU coverage can never be satisfied no
    matter how long we wait. VINS-Mono's answer is to drop that image
    outright rather than hand it a wrong/partial window; the caller
    here should treat a None return exactly like a timeout -- skip the
    frame.

    Parameters
    ----------
    target_timestamp : float
        Timestamp of the image/frame to be processed (this frame's own
        timestamp, i.e. what used to be `t_to` in extract_imu_between).

    timeout : float
        Maximum time (seconds) to wait for the IMU stream to catch up.

    Returns
    -------
    list[IMUMeasurement], or None if:
      - timed out waiting for IMU coverage, or
      - the IMU buffer has already been trimmed past target_timestamp
        (frame is stale / unsatisfiable -- caller should skip it).
    """

    with state.imu_condition:

        def ready_or_stale() -> bool:
            if not state.imu_buffer:
                return False
            if state.imu_buffer[0].timestamp > target_timestamp:
                # stale case -- wake up immediately so the caller can
                # detect and skip, rather than waiting out the full
                # timeout for something that will never become ready.
                return True
            return state.imu_buffer[-1].timestamp >= target_timestamp

        if not state.imu_condition.wait_for(ready_or_stale, timeout=timeout):
            return None  # genuine timeout -- IMU stream never caught up

        if not state.imu_buffer or state.imu_buffer[0].timestamp > target_timestamp:
            return None  # stale -- buffer already moved past this frame

        # Still holding imu_condition's lock here: the extraction below
        # is atomic with the readiness check above, so no interleaved
        # _imu_callback() append/trim can invalidate it in between.
        timestamps = [m.timestamp for m in state.imu_buffer]
        ind = _nearest_index(timestamps, target_timestamp)
        return list(state.imu_buffer[: ind + 1])


def extract_imu_between(
    state: SlidingWindowState,
    t0: float,
    t1: float,
) -> List[IMUMeasurement]:
    """
    Return all IMU samples between two timestamps.

    Direct Python equivalent of
    helperExtractIMUDataBetweenViews for a single (i,j) pair: finds
    the buffer index nearest each endpoint timestamp and returns the
    half-open slice [ind1:ind2), matching MATLAB's ind1:(ind2-1) in
    1-based inclusive indexing.

    Parameters
    ----------
    t0, t1 : float
        Timestamps of the two views bounding the interval (t0 < t1).

    Returns
    -------
    list[IMUMeasurement]
    """

    # Locked for the same reason get_synced_measurement() is: this
    # reads state.imu_buffer, and _imu_callback()/prune_imu_before()
    # append to / mutate it concurrently from the ROS executor thread.
    # Previously this function had no lock at all -- it happened to
    # mostly work under CPython's GIL, but that's not a guarantee,
    # and it's not what VINS-Mono's getMeasurements() does (everything
    # touching imu_buf/feature_buf there is under m_buf).
    with state.imu_condition:
        if len(state.imu_buffer) < 2:
            return []

        timestamps = [m.timestamp for m in state.imu_buffer]

        ind1 = _nearest_index(timestamps, t0)
        ind2 = _nearest_index(timestamps, t1)

        if ind2 <= ind1:
            return []

        return list(state.imu_buffer[ind1:ind2])


def build_preintegration(
    state: SlidingWindowState,
    imu_calib: dict,
    t_from: float,
    t_to: float,
    bias_g: np.ndarray,
    bias_a: np.ndarray,
):
    """
    Preintegrate IMU samples between two timestamps, seeded at the
    given (bias_g, bias_a) linearization point.

    This is the single place that constructs an IMUPreintegrator from
    calibration + a bias estimate -- shared by
    VisualInertialOdometry._build_imu_preintegrations (vision-only /
    alignment window, multiple consecutive pairs),
    VisualInertialOdometry._build_single_imu_preintegration (BA_motion,
    one pair against a not-yet-added frame), and
    GraphBuilder.build_windowed_vio (Phase 3 windowed BA, one IMU
    factor per consecutive keyframe pair in the window) -- so bias
    linearization is handled identically in all three call sites
    instead of three independent copies.

    Returns
    -------
    PreintegratedIMU, or None if IMU coverage between t_from and t_to
    is insufficient (< 2 samples).
    """

    samples = extract_imu_between(state, t_from, t_to)

    if len(samples) < 2:
        return None

    print(f"[VIO DEBUG] extract_imu_between: t_from={t_from:.6f} t_to={t_to:.6f} "
          f"requested_span={t_to - t_from:.6f} n_samples={len(samples)} "
          f"samples_span=[{samples[0].timestamp:.6f},{samples[-1].timestamp:.6f}]="
          f"{samples[-1].timestamp - samples[0].timestamp:.6f}")

    preintegrator = IMUPreintegrator(
        gyro_noise=imu_calib.get(
            'gyroscope_noise_density', 1.0e-3
        ),
        accel_noise=imu_calib.get(
            'accelerometer_noise_density', 1.0e-2
        ),
        gyro_random_walk=imu_calib.get(
            'gyroscope_random_walk', 1.0e-5
        ),
        accel_random_walk=imu_calib.get(
            'accelerometer_random_walk', 1.0e-4
        ),
        bias_g=bias_g,
        bias_a=bias_a,
    )

    return preintegrator.integrate_measurements(samples)


def prune_imu_before(
    state: SlidingWindowState,
    keep_from_timestamp: float,
):
    """
    Drop buffer samples that can no longer be needed.

    Call this once the sliding window's oldest surviving keyframe
    advances (i.e. a keyframe permanently leaves the window), passing
    that new oldest keyframe's timestamp. Keeps one extra sample
    before the cutoff as a boundary margin for `extract_imu_between`.
    """

    with state.imu_condition:
        if not state.imu_buffer:
            return

        timestamps = [m.timestamp for m in state.imu_buffer]

        idx = bisect.bisect_left(timestamps, keep_from_timestamp)

        state.imu_buffer = state.imu_buffer[max(0, idx - 1):]