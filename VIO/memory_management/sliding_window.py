"""
update_sliding_window — Python port of the MATLAB method
helperFeaturePointManager.updateSlidingWindow (see helperFeaturePointManager.m).

This file contains:
    - SlidingWindowState   (the persistent state the MATLAB code kept as
                             `obj.xxx` properties)
    - quick_check_parallax (port of the local function helperQuickCheckParallax)
    - update_sliding_window (port of the method body)

RANSAC fundamental-matrix estimation (MATLAB's estimateFundamentalMatrix
with Method='RANSAC') lives in ransac.py — see
ransac.estimate_fundamental_matrix_ransac, imported below and used as the
default for the fundamental_matrix_ransac_fn parameter.

Indexing note
-------------
MATLAB arrays are 1-indexed and `slidingWindowViewIDs` was a pre-allocated
fixed-size array (`zeros(params.windowSize,1)`) with `currentSlidingWindowIndex`
acting as a manual "number of filled slots" counter. In Python,
`sliding_window_view_ids` is a plain growing list (oldest -> newest), so:

    MATLAB obj.slidingWindowViewIDs(obj.currentSlidingWindowIndex)
        -> state.sliding_window_view_ids[-1]               (last filled slot)
    MATLAB obj.slidingWindowViewIDs(obj.currentSlidingWindowIndex - 1)
        -> state.sliding_window_view_ids[-2]
    MATLAB obj.slidingWindowViewIDs(end), (end-1)
        -> state.sliding_window_view_ids[-1], [-2]
    MATLAB obj.slidingWindowViewIDs = [obj.slidingWindowViewIDs(2:end); viewId]
        -> state.sliding_window_view_ids.pop(0); state.sliding_window_view_ids.append(view_id)

`current_sliding_window_index` is kept (as a count of filled slots, matching
the MATLAB semantics) purely so the warm-up / growing / full branch
conditions translate 1:1 and are easy to diff against the original. It is
always equal to len(state.sliding_window_view_ids) outside of the branch
where a slot is about to be appended.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from vio_core.ransac import estimate_fundamental_matrix_ransac


@dataclass
class SlidingWindowState:
    """Persistent state used across calls to update_sliding_window.

    Mirrors the relevant `obj.<property>` fields of helperFeaturePointManager.
    Dicts are keyed by view_id instead of MATLAB's preallocated
    cell(1, maxFrames) arrays, so there's no need to guess maxFrames upfront.
    """

    window_size: int  # MATLAB: params.windowSize

    # how many slots in sliding_window_view_ids are currently filled.
    # MATLAB: obj.currentSlidingWindowIndex
    current_sliding_window_index: int = 0

    # active sliding window view ids, oldest -> newest.
    # MATLAB: obj.slidingWindowViewIDs (fixed-size array, only first
    # current_sliding_window_index entries are valid)
    sliding_window_view_ids: list = field(default_factory=list)

    # view_id -> bool.  MATLAB: obj.isKeyFrame(viewId)
    is_key_frame: dict = field(default_factory=dict)

    # view_id -> (N,2) float array of tracked 2D points.
    # MATLAB: obj.AllObservations{viewId}
    all_observations: dict = field(default_factory=dict)

    # view_id -> (N,2) int array [view_id, point_id] per row.
    # MATLAB: obj.AllIds{viewId}
    all_ids: dict = field(default_factory=dict)

    # view_id -> (N,) bool array, per-point triangulated flag for this view.
    # MATLAB: obj.AllTriangulated{viewId}
    all_triangulated: dict = field(default_factory=dict)

    # point_id -> number of frames this point has been observed in.
    # MATLAB: obj.keyPointTrackCount(pIds)
    # Stored as a dict here (point ids are not assumed contiguous from 0);
    # adapt to a preallocated array if your point ids are dense.
    key_point_track_count: dict = field(default_factory=dict)

    # MATLAB: obj.currentViewID
    current_view_id: int = -1

    # MATLAB: obj.noMovementAtStart
    no_movement_at_start: bool = True


def _within_image(points: np.ndarray, image_shape) -> np.ndarray:
    """Port of local function helperWithinImage.

    points : (N,2) array of (x, y) pixel coordinates (1-based in MATLAB).
    image_shape : (rows, cols[, ...]) as returned by e.g. arr.shape.

    NOTE: MATLAB used 1-based pixel coordinates and bounds
    `points(:,1) >= 1 & points(:,1) <= cols` etc. If your points are
    0-based (typical in Python/OpenCV), adjust the lower bound to 0 to
    match your convention; left as >=1 here for direct fidelity with the
    original since the boundary semantics (not just the index base) can
    matter for downstream RANSAC behavior.
    """
    rows, cols = image_shape[0], image_shape[1]
    x = points[:, 0]
    y = points[:, 1]
    return (x >= 1) & (x <= cols) & (y >= 1) & (y <= rows)


def quick_check_parallax(m1: np.ndarray, m2: np.ndarray, parallax_threshold: float):
    """Port of local function helperQuickCheckParallax.

    m1, m2 : (N,2) matched point arrays between two views.

    Returns
    -------
    avg : float
        Mean Euclidean displacement between matched points.
    is_keyframe : bool
        True if avg > parallax_threshold.
    """
    diff = m1 - m2
    avg = np.sqrt((diff ** 2).sum(axis=1)).sum() / diff.shape[0]
    is_keyframe = avg > parallax_threshold
    return avg, is_keyframe


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
    """Port of helperFeaturePointManager.updateSlidingWindow.

    Parameters
    ----------
    state : SlidingWindowState
        Mutated in place AND returned via the function's side effects
        (caller keeps the same object across calls).
    image_shape : tuple
        Shape of current image I, e.g. I.shape (rows, cols, ...).
        MATLAB: size(I)
    curr_points_tracked : (N,2) float array
        Tracked 2D point locations in the current frame.
    valid_idx : (N,) bool array
        Tracker-reported validity per point (MATLAB: validIdx).
    view_id : int
        Current view id (MATLAB: viewId).
    F_loop, F_iterations, F_confidence, F_threshold : RANSAC params
        MATLAB: params.F_loop, params.F_Iterations, params.F_Confidence,
        params.F_Threshold
    key_frame_parallax : float
        MATLAB: params.keyFrameParallax
    fundamental_matrix_ransac_fn : callable, optional
        Defaults to ransac.estimate_fundamental_matrix_ransac (cv2-based
        RANSAC fundamental-matrix estimator). Pass your own implementation
        matching that contract if you want to swap it out.

    Returns
    -------
    removed_frame_id : int
        MATLAB: rmF.
        -1 : very first frame, accepted immediately, nothing removed.
        -2 : window grew (frame accepted, nothing removed yet).
        -3 : still in the "first few views" warm-up, nothing removed.
        Any other (non-negative) value: the view_id that was evicted from
        the sliding window.
    window_state : dict
        MATLAB: windowState, with keys 'isEnoughParallax', 'isWindowFull',
        'isFirstFewViews' (booleans). Starts as all-False each call, matching
        `windowState = obj.windowState;` followed by in-place edits in the
        original (the MATLAB code re-reads the stored struct first, but only
        ever turns flags on, never off, within a single call — replicated
        here by initializing fresh False values, which is observationally
        equivalent for a single call's return value).
    """
    if fundamental_matrix_ransac_fn is None:
        fundamental_matrix_ransac_fn = estimate_fundamental_matrix_ransac

    window_state = {
        "isEnoughParallax": False,
        "isWindowFull": False,
        "isFirstFewViews": False,
    }

    # --- very first frame: accept right away --------------------------
    # MATLAB: if obj.currentSlidingWindowIndex == 0
    if state.current_sliding_window_index == 0:
        state.current_sliding_window_index += 1
        state.sliding_window_view_ids.append(view_id)
        return -1, window_state

    # --- outlier rejection via repeated RANSAC fundamental matrix ------
    # MATLAB: psIdx = helperWithinImage(...); v1 = validIdx & psIdx;
    ps_idx = _within_image(curr_points_tracked, image_shape)
    v1 = valid_idx & ps_idx

    prev_view_id = view_id - 1  # MATLAB: viewId - 1 (previous view's observations)
    prev_obs = state.all_observations[prev_view_id]

    inl_f = None  # MATLAB: inlF = []
    for _ in range(F_loop):
        _, inl_ff = fundamental_matrix_ransac_fn(
            prev_obs[v1],
            curr_points_tracked[v1],
            num_trials=F_iterations,
            confidence=F_confidence,
            dist_threshold=F_threshold,
        )
        if inl_f is None or np.count_nonzero(inl_ff) > np.count_nonzero(inl_f):
            inl_f = inl_ff

    # scatter the v1-subset inlier mask back to full-length mask
    # MATLAB: inlFf = false(size(v1)); inlFf(v1) = inlF; v1 = v1 & inlFf;
    inl_ff_full = np.zeros_like(v1, dtype=bool)
    inl_ff_full[v1] = inl_f
    v1 = v1 & inl_ff_full

    # --- update feature tracks for this view ---------------------------
    # MATLAB: obj.AllObservations{viewId} = currPointsTracked(v1,:);
    state.all_observations[view_id] = curr_points_tracked[v1]

    # MATLAB: obj.AllTriangulated{viewId} = obj.AllTriangulated{max(1,viewId-1)}(v1,:);
    prev_triangulated_key = max(1, prev_view_id)
    state.all_triangulated[view_id] = state.all_triangulated[prev_triangulated_key][v1]

    # MATLAB: pIds = obj.AllIds{max(1,viewId-1)}(v1,2);
    prev_ids_key = max(1, prev_view_id)
    p_ids = state.all_ids[prev_ids_key][v1, 1]  # column 2 in MATLAB -> index 1

    # MATLAB: obj.keyPointTrackCount(pIds) = obj.keyPointTrackCount(pIds) + 1;
    for pid in p_ids:
        state.key_point_track_count[pid] = state.key_point_track_count.get(pid, 0) + 1

    # MATLAB: obj.AllIds{viewId} = [viewId*ones(size(pIds)), pIds];
    state.all_ids[view_id] = np.column_stack(
        [np.full(p_ids.shape[0], view_id), p_ids]
    )
    state.current_view_id = view_id

    # rmF defaults to the oldest frame in the window (overwritten below
    # in every branch, but matches the MATLAB initial assignment).
    # MATLAB: rmF = obj.slidingWindowViewIDs(1);
    removed_frame_id = state.sliding_window_view_ids[0]

    no_move_window = 0.5  # MATLAB: noMoveWindow = 0.5;
    warmup_cutoff = int(np.floor(state.window_size * no_move_window))

    # --- branch 1: warm-up (no movement at start) or very first couple frames
    # MATLAB:
    # if (obj.currentSlidingWindowIndex < floor(params.windowSize*noMoveWindow) && noMovementAtStart)
    #     || (obj.currentSlidingWindowIndex < 2)
    if (
        state.current_sliding_window_index < warmup_cutoff
        and state.no_movement_at_start
    ) or (state.current_sliding_window_index < 2):

        # MATLAB:
        # if (currentSlidingWindowIndex == (warmup_cutoff - 1) && noMovementAtStart)
        #     || (~noMovementAtStart && currentSlidingWindowIndex == 1)
        if (
            state.current_sliding_window_index == warmup_cutoff - 1
            and state.no_movement_at_start
        ) or (
            not state.no_movement_at_start
            and state.current_sliding_window_index == 1
        ):
            last_window_view = state.sliding_window_view_ids[
                state.current_sliding_window_index - 1
            ]
            _, ia, ib = np.intersect1d(
                state.all_ids[last_window_view][:, 1],
                state.all_ids[view_id][:, 1],
                return_indices=True,
            )
            m1 = state.all_observations[last_window_view][ia]
            m2 = state.all_observations[view_id][ib]
            _, is_kf = quick_check_parallax(m1, m2, key_frame_parallax)

            if is_kf:
                # last frame is a key frame -> grow window, accept current frame
                state.is_key_frame[view_id] = True
                state.current_sliding_window_index += 1
                state.sliding_window_view_ids.append(view_id)
                removed_frame_id = -2
                window_state["isEnoughParallax"] = True
            else:
                removed_frame_id = view_id
        else:
            state.current_sliding_window_index += 1
            state.sliding_window_view_ids.append(view_id)
            removed_frame_id = -3
            window_state["isFirstFewViews"] = True

    # --- branch 2: growing window (past warm-up, not yet full) ---------
    # MATLAB:
    # elseif (currentSlidingWindowIndex >= warmup_cutoff && currentSlidingWindowIndex < windowSize && noMovementAtStart)
    #     || (currentSlidingWindowIndex < windowSize && ~noMovementAtStart)
    elif (
        warmup_cutoff
        <= state.current_sliding_window_index
        < state.window_size
        and state.no_movement_at_start
    ) or (
        state.current_sliding_window_index < state.window_size
        and not state.no_movement_at_start
    ):
        prev_window_view = state.sliding_window_view_ids[
            state.current_sliding_window_index - 2
        ]
        last_window_view = state.sliding_window_view_ids[
            state.current_sliding_window_index - 1
        ]
        _, ia, ib = np.intersect1d(
            state.all_ids[prev_window_view][:, 1],
            state.all_ids[last_window_view][:, 1],
            return_indices=True,
        )
        m1 = state.all_observations[prev_window_view][ia]
        m2 = state.all_observations[last_window_view][ib]
        _, is_kf = quick_check_parallax(m1, m2, key_frame_parallax)

        if is_kf or state.is_key_frame.get(last_window_view, False):
            state.is_key_frame[last_window_view] = True
            state.current_sliding_window_index += 1
            state.sliding_window_view_ids.append(view_id)
            removed_frame_id = -2
            window_state["isEnoughParallax"] = True
        else:
            # remove last frame (it's not a keyframe), replace with current
            removed_frame_id = state.sliding_window_view_ids[
                state.current_sliding_window_index - 1
            ]
            state.sliding_window_view_ids[
                state.current_sliding_window_index - 1
            ] = view_id

    # --- branch 3: window is full ---------------------------------------
    # MATLAB: else  (window full)
    else:
        window_state["isWindowFull"] = True
        prev_window_view = state.sliding_window_view_ids[-2]
        last_window_view = state.sliding_window_view_ids[-1]
        _, ia, ib = np.intersect1d(
            state.all_ids[prev_window_view][:, 1],
            state.all_ids[last_window_view][:, 1],
            return_indices=True,
        )
        m1 = state.all_observations[prev_window_view][ia]
        m2 = state.all_observations[last_window_view][ib]
        _, is_kf = quick_check_parallax(m1, m2, key_frame_parallax)

        if (not is_kf) or (not state.is_key_frame.get(last_window_view, False)):
            # accept current frame, remove last frame (not a keyframe)
            removed_frame_id = state.sliding_window_view_ids[-1]
            state.sliding_window_view_ids[-1] = view_id
        else:
            # remove the oldest frame, slide window forward
            removed_frame_id = state.sliding_window_view_ids[0]
            state.sliding_window_view_ids.pop(0)
            state.sliding_window_view_ids.append(view_id)
            window_state["isEnoughParallax"] = True
            state.is_key_frame[state.sliding_window_view_ids[-2]] = True

    return removed_frame_id, window_state