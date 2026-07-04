"""
vi_alignment.py

Linear Visual-Inertial Alignment.

This module estimates the initial

    - metric scale
    - gravity
    - camera velocities
    - accelerometer bias

from

    - visual poses (ViewSet)
    - IMU preintegration

The implementation follows the initialization procedure used in

    Qin et al. (VINS-Mono)
    Forster et al.
    MATLAB Navigation Toolbox VIO

Author:
    VIO Project
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================
# Dataclasses
# ============================================================

@dataclass
class PairAlignmentData:
    """
    Data corresponding to one consecutive keyframe pair.

    Everything required to build the alignment equations is stored
    here so that the mathematics is completely independent of the
    ViewSet implementation.
    """

    view_i: int
    view_j: int

    dt: float

    # -----------------------------
    # Visual quantities (unscaled)
    # -----------------------------

    R_i: np.ndarray
    R_j: np.ndarray

    p_i: np.ndarray
    p_j: np.ndarray

    # -----------------------------
    # IMU preintegration
    # -----------------------------

    delta_R: np.ndarray

    delta_v: np.ndarray

    delta_p: np.ndarray

    # -----------------------------
    # Bias Jacobians
    # -----------------------------

    J_v_ba: np.ndarray

    J_p_ba: np.ndarray


@dataclass
class VIAlignmentResult:
    """
    Result returned by the linear VI alignment.

    velocities is keyed by view_id (NOT positional index), so it is
    safe to use directly against a ViewSet / SlidingWindowState whose
    keyframe ids are not contiguous starting at 0.
    """

    success: bool

    scale: float

    gravity: np.ndarray

    accel_bias: np.ndarray

    velocities: Dict[int, np.ndarray]


# ============================================================
# Camera <-> body/IMU extrinsic folding
# ============================================================
#
# vio_core stores ViewSet poses as camera-to-world (R_wc, t_wc). The
# alignment equations below assume the poses they're given are
# body/IMU-to-world, i.e. R_bc == I. When the rig's camera-to-body
# extrinsic (T_BS, "sensor expressed in body frame") is not identity,
# that assumption is wrong and must be corrected before the poses ever
# reach build_pair_equation — not by changing the (tested) linear
# algebra itself.
#
# T_BS convention (matches the calibration yaml files): a point in the
# camera/sensor frame is mapped into the body frame by
#     p_body = R_bs @ p_camera + t_bs
#
# Given camera-to-world (R_wc, t_wc), the equivalent body-to-world pose
# is derived by composing with the inverse of T_BS:
#
#     p_world = R_wc @ p_camera + t_wc
#             = R_wc @ (R_bs.T @ (p_body - t_bs)) + t_wc
#             = (R_wc @ R_bs.T) @ p_body + (t_wc - R_wc @ R_bs.T @ t_bs)
#
#     => R_wb = R_wc @ R_bs.T
#        t_wb = t_wc - R_wb @ t_bs
#

def camera_pose_to_body_pose(
    R_wc: np.ndarray,
    t_wc: np.ndarray,
    R_bs: np.ndarray,
    t_bs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a camera-to-world pose into the equivalent body/IMU-to-world
    pose, given the camera-to-body extrinsic (R_bs, t_bs) == T_BS.

    Parameters
    ----------
    R_wc, t_wc : camera-to-world rotation/translation.
    R_bs, t_bs : camera-to-body extrinsic (T_BS), i.e.
        p_body = R_bs @ p_camera + t_bs.

    Returns
    -------
    R_wb, t_wb : body-to-world rotation/translation.
    """

    R_wb = R_wc @ R_bs.T
    t_wb = t_wc - R_wb @ t_bs

    return R_wb, t_wb


def _split_sensor_transform(
    sensor_transform,
):
    """
    Accept either a (R_bs, t_bs) tuple or a 4x4 T_BS matrix and return
    (R_bs, t_bs). Returns None if sensor_transform is None.
    """

    if sensor_transform is None:
        return None

    if isinstance(sensor_transform, tuple) and len(sensor_transform) == 2:
        R_bs, t_bs = sensor_transform
        return np.asarray(R_bs, dtype=float), np.asarray(t_bs, dtype=float).reshape(3)

    T_BS = np.asarray(sensor_transform, dtype=float)

    if T_BS.shape != (4, 4):
        raise ValueError(
            "sensor_transform must be a (R,t) tuple or a 4x4 matrix, "
            f"got shape {T_BS.shape}"
        )

    return T_BS[:3, :3], T_BS[:3, 3]


# ============================================================
# Pair extraction
# ============================================================

def collect_alignment_pairs(
    view_set,
    imu_preintegrations: Dict,
    view_ids: Optional[List[int]] = None,
    sensor_transform=None,
) -> List[PairAlignmentData]:
    """
    Collect all consecutive keyframe pairs required for
    linear VI alignment.

    Parameters
    ----------
    view_set
        Existing ViewSet. Must expose:
            view_set.view_ids                -> iterable of int
            view_set.get_pose(view_id)        -> (R, t)
            view_set.get_timestamp(view_id)   -> float   (unused here,
                                                  dt comes from IMU)

    imu_preintegrations
        Dictionary

            (view_i, view_j)
                ->
            PreintegratedIMU

    view_ids : list[int], optional
        The exact set of views to align, in the order they should be
        treated as consecutive. If omitted, defaults to
        sorted(view_set.view_ids) (previous behaviour) — but callers
        driving a sliding-window pipeline should pass the sliding
        window's *keyframe* ids explicitly (e.g.
        sw_state.sliding_window_view_ids[:-1], matching MATLAB's
        swIDs(1:end-1)), since view_set generally accumulates a pose
        for every processed frame, not just keyframes.

    sensor_transform : (R_bs, t_bs) tuple or 4x4 ndarray, optional
        Camera-to-body extrinsic (T_BS). If given, every pose is
        converted from camera-to-world to body-to-world via
        camera_pose_to_body_pose before building the pair. If omitted,
        poses are used as-is (R_bc == I assumed, previous behaviour).

    Returns
    -------
    List[PairAlignmentData]
    """

    pairs = []

    if view_ids is None:
        #
        # Sort keyframes. sorted() is defensive: it doesn't matter whether
        # view_set.view_ids is already ordered, and it costs nothing.
        #
        view_ids = sorted(view_set.view_ids)
    else:
        view_ids = list(view_ids)

    extrinsic = _split_sensor_transform(sensor_transform)

    if len(view_ids) < 2:
        return pairs

    #
    # Build one PairAlignmentData per consecutive pair
    #
    for k in range(len(view_ids) - 1):

        i = view_ids[k]
        j = view_ids[k + 1]

        if (i, j) not in imu_preintegrations:
            raise KeyError(
                f"Missing IMU preintegration for ({i},{j})"
            )

        imu = imu_preintegrations[(i, j)]

        #
        # Pose access goes through the public ViewSet API. These are
        # camera-to-world poses; fold in the camera->body extrinsic (if
        # given) so downstream code can keep assuming body-to-world.
        #
        R_i, p_i = view_set.get_pose(i)
        R_j, p_j = view_set.get_pose(j)

        if extrinsic is not None:
            R_bs, t_bs = extrinsic
            R_i, p_i = camera_pose_to_body_pose(R_i, p_i, R_bs, t_bs)
            R_j, p_j = camera_pose_to_body_pose(R_j, p_j, R_bs, t_bs)

        #
        # Time interval comes from the IMU preintegration, since that's
        # the clock the delta_R / delta_v / delta_p were integrated on.
        #
        dt = imu.delta_t

        #
        # Store pair
        #
        pair = PairAlignmentData(

            view_i=i,
            view_j=j,

            dt=dt,

            R_i=R_i.copy(),
            R_j=R_j.copy(),

            p_i=p_i.copy(),
            p_j=p_j.copy(),

            delta_R=imu.delta_R.copy(),

            delta_v=imu.delta_v.copy(),

            delta_p=imu.delta_p.copy(),

            J_v_ba=imu.J_v_ba.copy(),

            J_p_ba=imu.J_p_ba.copy(),
        )

        pairs.append(pair)

    return pairs


# ============================================================
# Helpers - unknown vector layout
# ============================================================
#
# x = [ v_0 ... v_{N-1} | gravity(3) | scale(1) | accel_bias(3) ]
#
# Total size = 3N + 7
#

def number_of_unknowns(
    num_views: int,
):
    """
    Unknown vector size: 3N (velocities) + 3 (gravity) + 1 (scale)
    + 3 (accel bias) = 3N + 7.
    """

    return 3 * num_views + 7


def velocity_column(
    index: int,
):
    """
    Starting column of velocity at positional index `index`
    (index into the sorted view_ids list, NOT the view_id itself).
    """

    return 3 * index


def gravity_column(
    num_views: int,
):
    """
    Starting column of gravity.
    """

    return 3 * num_views


def scale_column(
    num_views: int,
):
    """
    Column of scale.
    """

    return 3 * num_views + 3


def accel_bias_column(
    num_views: int,
):
    """
    Starting column of accelerometer bias.
    """

    return 3 * num_views + 4


# ============================================================
# Per-pair linear equation
# ============================================================
#
# Preintegrated IMU measurement model (body/camera frame == IMU frame
# assumed here; if R_bc != I it must be folded into R_i/R_j upstream):
#
#   delta_p_ij = R_i^T ( s*p_j - s*p_i - v_i*dt - 0.5*g*dt^2 ) + J_p_ba*ba
#   delta_v_ij = R_i^T ( v_j - v_i - g*dt )                    + J_v_ba*ba
#
# p_i, p_j are the *unscaled* visual translations, and s is the metric
# scale factor such that p_metric = s * p_visual. Rearranged into
# A x = b form (6 rows per pair: 3 position, 3 velocity):
#
#   position rows:
#       -R_i^T*dt * v_i + 0*v_j - 0.5*R_i^T*dt^2 * g
#           + R_i^T*(p_j - p_i) * s - J_p_ba * ba = delta_p
#
#   velocity rows:
#       -R_i^T * v_i + R_i^T * v_j - R_i^T*dt * g
#           + 0 * s + J_v_ba * ba = delta_v
#

def build_pair_equation(
    pair: PairAlignmentData,
    index_i: int,
    num_views: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the (6, 3N+7) block and (6,) right-hand side for a single
    consecutive keyframe pair.

    Parameters
    ----------
    pair : PairAlignmentData

    index_i : int
        Positional index of view_i within the sorted view list
        (view_j is always index_i + 1, since pairs are consecutive).

    num_views : int
        Total number of keyframes N in the window.

    Returns
    -------
    A_pair : ndarray, shape (6, 3N+7)
    b_pair : ndarray, shape (6,)
    """

    n_cols = number_of_unknowns(num_views)

    A_pair = np.zeros((6, n_cols))
    b_pair = np.zeros(6)

    Ri_T = pair.R_i.T

    dt = pair.dt
    dt2 = dt * dt

    vi_col = velocity_column(index_i)
    vj_col = velocity_column(index_i + 1)
    g_col = gravity_column(num_views)
    s_col = scale_column(num_views)
    ba_col = accel_bias_column(num_views)

    # ---- position rows (0:3) ----

    A_pair[0:3, vi_col:vi_col + 3] = -Ri_T * dt
    A_pair[0:3, g_col:g_col + 3] = -0.5 * Ri_T * dt2
    A_pair[0:3, s_col] = Ri_T @ (pair.p_j - pair.p_i)
    A_pair[0:3, ba_col:ba_col + 3] = -pair.J_p_ba

    b_pair[0:3] = pair.delta_p

    # ---- velocity rows (3:6) ----

    A_pair[3:6, vi_col:vi_col + 3] = -Ri_T
    A_pair[3:6, vj_col:vj_col + 3] = Ri_T
    A_pair[3:6, g_col:g_col + 3] = -Ri_T * dt
    A_pair[3:6, ba_col:ba_col + 3] = pair.J_v_ba

    b_pair[3:6] = pair.delta_v

    return A_pair, b_pair


def build_alignment_system(
    pairs,
    num_views,
):
    """
    Assemble the global linear system

        A x = b

    from all consecutive keyframe pairs.

    Parameters
    ----------
    pairs : list[PairAlignmentData]

    num_views : int

    Returns
    -------
    A : ndarray
        (6*(N-1), 3N+7)

    b : ndarray
        (6*(N-1),)
    """

    if len(pairs) != num_views - 1:
        raise ValueError(
            "Expected one pair for every consecutive keyframe."
        )

    A_blocks = []
    b_blocks = []

    for index_i, pair in enumerate(pairs):

        A_pair, b_pair = build_pair_equation(
            pair,
            index_i=index_i,
            num_views=num_views,
        )

        A_blocks.append(A_pair)
        b_blocks.append(b_pair)

    A = np.vstack(A_blocks)
    b = np.concatenate(b_blocks)

    return A, b


# ============================================================
# Solve
# ============================================================

def solve_alignment(
    A,
    b,
    view_ids,
):
    """
    Solve the linear VI alignment system.

    Parameters
    ----------
    A, b : the assembled system
    view_ids : sequence[int]
        The same sorted view id list used to build `pairs` /
        `A`, in the same order. Positional index k in the unknown
        vector corresponds to view_ids[k], NOT to view id k.

    Returns
    -------
    VIAlignmentResult
        velocities is keyed by the actual view_id.
    """

    num_views = len(view_ids)

    if A.shape[0] < A.shape[1]:
        raise RuntimeError(
            "Alignment system is underdetermined "
            f"(rows={A.shape[0]}, cols={A.shape[1]}); "
            "need more keyframes/IMU pairs."
        )

    x, residuals, rank, singular_values = np.linalg.lstsq(
        A,
        b,
        rcond=None,
    )

    expected_rank = number_of_unknowns(num_views)

    if rank < expected_rank:
        return VIAlignmentResult(
            success=False,
            scale=1.0,
            gravity=np.zeros(3),
            accel_bias=np.zeros(3),
            velocities={},
        )

    velocities = {}

    for idx, view_id in enumerate(view_ids):

        c = velocity_column(idx)

        velocities[view_id] = x[c:c + 3].copy()

    gravity = x[
        gravity_column(num_views):
        gravity_column(num_views) + 3
    ].copy()

    scale = float(
        x[
            scale_column(num_views)
        ]
    )

    accel_bias = x[
        accel_bias_column(num_views):
        accel_bias_column(num_views) + 3
    ].copy()

    return VIAlignmentResult(
        success=True,
        scale=scale,
        gravity=gravity,
        accel_bias=accel_bias,
        velocities=velocities,
    )


# ============================================================
# Gravity refinement
# ============================================================
#
# The raw linear estimate of gravity has the right direction but its
# magnitude is not constrained to |g| = 9.81, because the linear
# system doesn't know that constraint. Standard fix (VINS-Mono style):
# reparameterize g as
#
#     g = G * normalize(g_hat) + B @ dg
#
# where B is a (3,2) basis spanning the plane tangent to g_hat, and
# dg is a small 2-DOF perturbation. Substitute into the system, solve
# for dg (and everything else) with lstsq, update g, and iterate a
# few times. This keeps |g| fixed at G by construction.
#

def gravity_tangent_basis(g0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build an orthonormal basis (b1, b2) spanning the plane
    perpendicular to g0.
    """

    a = g0 / np.linalg.norm(g0)

    tmp = np.array([1.0, 0.0, 0.0])
    if abs(a[0]) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])

    b1 = np.cross(tmp, a)
    b1 /= np.linalg.norm(b1)

    b2 = np.cross(a, b1)

    return b1, b2


def refine_gravity(
    A: np.ndarray,
    b: np.ndarray,
    num_views: int,
    g_init: np.ndarray,
    gravity_magnitude: float = 9.81,
    n_iters: int = 4,
) -> np.ndarray:
    """
    Refine a raw linear gravity estimate so that ||g|| ==
    gravity_magnitude, by re-solving the alignment system with gravity
    constrained to a 2-DOF perturbation around the current estimate.

    Parameters
    ----------
    A, b : the same system passed to solve_alignment
    num_views : int
    g_init : ndarray (3,)
        Initial (unconstrained) gravity estimate, e.g. from
        solve_alignment(...).gravity
    gravity_magnitude : float
        Target |g|, e.g. 9.81 for Earth.
    n_iters : int
        Number of refinement iterations. 3-5 is typically enough.

    Returns
    -------
    g_refined : ndarray (3,)
        Refined gravity vector with ||g_refined|| == gravity_magnitude.
    """

    norm_g = np.linalg.norm(g_init)

    if norm_g < 1e-6:
        g = np.array([0.0, 0.0, gravity_magnitude])
    else:
        g = gravity_magnitude * g_init / norm_g

    g_col = gravity_column(num_views)

    A_g = A[:, g_col:g_col + 3]

    # Columns before and after the 3 gravity columns stay fixed;
    # only the gravity block gets replaced by a 2-column tangent block
    # each iteration.
    A_before = A[:, :g_col]
    A_after = A[:, g_col + 3:]

    for _ in range(n_iters):

        b1, b2 = gravity_tangent_basis(g)
        B = np.column_stack([b1, b2])  # (3, 2)

        A_dg = A_g @ B  # (rows, 2)

        A_iter = np.hstack([A_before, A_dg, A_after])
        b_iter = b - A_g @ g

        x, *_ = np.linalg.lstsq(A_iter, b_iter, rcond=None)

        dg = x[g_col:g_col + 2]

        g = g + B @ dg
        g = gravity_magnitude * g / np.linalg.norm(g)

    return g


# ============================================================
# Apply results
# ============================================================

def apply_scale_to_visual_map(
    view_set,
    landmarks: Dict[int, np.ndarray],
    scale: float,
):
    """
    Scale the visual map (camera translations + landmark positions)
    from the arbitrary monocular scale into metric units.

    NOTE: this assumes `view_set` exposes a setter of the form
        view_set.set_pose(view_id, R, t)
    and that `landmarks` maps landmark_id -> 3D position (ndarray).
    Confirm these names against your actual ViewSet / map storage
    before wiring this in -- rename as needed, the scaling math itself
    (multiply all translations by `scale`) is what matters.

    Parameters
    ----------
    view_set : ViewSet
    landmarks : Dict[int, np.ndarray]
    scale : float

    Returns
    -------
    None (mutates in place)
    """

    if scale <= 0.0:
        raise ValueError(f"Invalid scale for map scaling: {scale}")

    for view_id in view_set.view_ids:

        R, t = view_set.get_pose(view_id)

        view_set.set_pose(view_id, R, t * scale)

    for landmark_id, position in landmarks.items():

        landmarks[landmark_id] = position * scale


def apply_alignment_result(
    result: VIAlignmentResult,
    sliding_window,
):
    """
    Store the initialized inertial state into the SlidingWindowState.

    Field names (metric_scale, gravity, accelerometer_bias, velocities)
    match memory_management.sliding_window.SlidingWindowState.
    """

    if not result.success:
        raise RuntimeError(
            "VI alignment failed."
        )

    sliding_window.metric_scale = float(result.scale)

    sliding_window.gravity = result.gravity.copy()

    sliding_window.accelerometer_bias = result.accel_bias.copy()

    sliding_window.velocities.clear()

    for view_id, velocity in result.velocities.items():

        sliding_window.velocities[view_id] = velocity.copy()


# ============================================================
# Top-level entry point
# ============================================================

def initialize_visual_inertial_state(
    view_set,
    sliding_window,
    imu_preintegrations,
    landmarks: Dict[int, np.ndarray] = None,
    view_ids: Optional[List[int]] = None,
    sensor_transform=None,
    gravity_magnitude: float = 9.81,
    refine_gravity_estimate: bool = True,
    apply_scale_to_map: bool = True,
):
    """
    Run complete linear visual-inertial alignment:

        1. collect consecutive keyframe pairs
        2. build and solve the linear system for
           [velocities, gravity, scale, accel_bias]
        3. (optional) refine gravity to ||g|| == gravity_magnitude
        4. (optional) apply the recovered scale to the visual map
        5. store the result into sliding_window

    Parameters
    ----------
    view_set : ViewSet
    sliding_window : SlidingWindowState
    imu_preintegrations : Dict[(int,int), PreintegratedIMU]
    landmarks : Dict[int, np.ndarray], optional
        Required if apply_scale_to_map=True.
    view_ids : list[int], optional
        Exact, ordered set of views to align (see collect_alignment_pairs).
        Defaults to sorted(view_set.view_ids) if omitted — callers driving
        a sliding window should pass the window's keyframe ids explicitly
        (e.g. sw_state.sliding_window_view_ids[:-1]).
    sensor_transform : (R_bs, t_bs) tuple or 4x4 ndarray, optional
        Camera-to-body extrinsic (T_BS). See collect_alignment_pairs.
    gravity_magnitude : float
    refine_gravity_estimate : bool
    apply_scale_to_map : bool

    Returns
    -------
    VIAlignmentResult
    """

    if view_ids is None:
        view_ids = sorted(view_set.view_ids)
    else:
        view_ids = list(view_ids)

    num_views = len(view_ids)

    pairs = collect_alignment_pairs(
        view_set,
        imu_preintegrations,
        view_ids=view_ids,
        sensor_transform=sensor_transform,
    )

    A, b = build_alignment_system(
        pairs,
        num_views,
    )

    result = solve_alignment(
        A,
        b,
        view_ids,
    )

    if not result.success:
        return result

    if refine_gravity_estimate:

        refined_g = refine_gravity(
            A,
            b,
            num_views,
            result.gravity,
            gravity_magnitude=gravity_magnitude,
        )

        result.gravity = refined_g

    if apply_scale_to_map:

        if landmarks is None:
            raise ValueError(
                "apply_scale_to_map=True requires `landmarks`."
            )

        apply_scale_to_visual_map(
            view_set,
            landmarks,
            result.scale,
        )

    apply_alignment_result(
        result,
        sliding_window,
    )

    return result