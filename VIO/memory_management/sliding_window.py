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

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from vio_core.ransac import estimate_fundamental_matrix_ransac
from typing import Dict, Tuple

from vio_core.landmarks import Landmark
from imu.imu_measurement import IMUMeasurement


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
    # key:
    #     (from_view, to_view)
    #
    # value:
    #     list[IMUMeasurement]
    # ------------------------------------------------------------------

    imu_measurements: Dict[tuple[int, int], list[IMUMeasurement]] = field(
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


def update_sliding_window(
    state: SlidingWindowState,
    image_shape,
    curr_points_tracked: np.ndarray,
    valid_idx: np.ndarray,
    view_id: int,
    F_loop: int,
    F_iterations: int,
    F_confidence: float,
    F_threshold: float,
    key_frame_parallax: float,
    fundamental_matrix_ransac_fn: Optional[Callable] = None,
):
    if fundamental_matrix_ransac_fn is None:
        fundamental_matrix_ransac_fn = estimate_fundamental_matrix_ransac

    window_state = {
        "isEnoughParallax": False,
        "isWindowFull":     False,
        "isFirstFewViews":  False,
    }

    # ── very first frame ─────────────────────────────────────────────────
    if state.current_sliding_window_index == 0:
        state.current_sliding_window_index = 1
        state.sliding_window_view_ids.append(view_id)
        # observations/ids were already written by the caller (_init_first_frame)
        state.current_view_id = view_id
        return -1, window_state

    # ── RANSAC outlier rejection ──────────────────────────────────────────
    # prev_obs must come from the LAST STORED view, not blindly view_id-1,
    # because a replaced (non-KF) frame may not be at view_id-1.
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

    # default: oldest window slot (overwritten in every branch below)
    removed_frame_id = state.sliding_window_view_ids[0]

    no_move_window = 0.5
    warmup_cutoff  = int(np.floor(state.window_size * no_move_window))
    idx            = state.current_sliding_window_index   # alias, easier to read

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

    #
    # Remove IMU data associated with frames that left the window
    #
    if removed_frame_id >= 0:

        keys_to_remove = []

        for key in list(state.imu_measurements):

            if removed_frame_id in key:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del state.imu_measurements[key]

    #
    # Remove feature observations belonging to frames
    # that left the sliding window
    #
    if removed_frame_id >= 0:

        state.all_observations.pop(removed_frame_id, None)
        state.all_ids.pop(removed_frame_id, None)
        state.all_triangulated.pop(removed_frame_id, None)
        state.is_key_frame.pop(removed_frame_id, None)

        return removed_frame_id, window_state

def add_imu_measurements(
    state: SlidingWindowState,
    from_view: int,
    to_view: int,
    measurements,
):
    """
    Store all IMU samples between two consecutive views.
    """

    state.imu_measurements[(from_view, to_view)] = list(measurements)

def get_imu_measurements(
    state: SlidingWindowState,
    from_view: int,
    to_view: int,
):
    """
    Return IMU samples between two views.
    """

    return state.imu_measurements.get(
        (from_view, to_view),
        [],
    )