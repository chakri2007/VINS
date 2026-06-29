"""
triangulation.py

Incremental landmark triangulation.

This module follows the logic of MATLAB's triangulateNew3DPoints(),
but is implemented in smaller, testable steps.

Step 1 implemented here:
    - Find candidate feature tracks for triangulation.

Later steps will:
    - Build projection matrices
    - Triangulate with OpenCV
    - Cheirality check
    - Store landmarks
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import cv2

@dataclass
class TriangulationCandidate:
    """
    One feature that can potentially be triangulated.
    """

    point_id: int

    view1: int
    view2: int

    uv1: np.ndarray      # (2,)
    uv2: np.ndarray      # (2,)


def find_triangulation_candidates(
    sliding_window_state,
    view_set,
):
    """
    Find feature tracks that are ready to be triangulated.

    Conditions
    ----------
    1. Visible in the last two keyframes.
    2. Not already triangulated.

    Returns
    -------
    candidates : list[TriangulationCandidate]
    """

    sw_ids = sliding_window_state.sliding_window_view_ids

    if len(sw_ids) < 2:
        return []

    view1 = sw_ids[-2]
    view2 = sw_ids[-1]

    ids1 = sliding_window_state.all_ids[view1]
    ids2 = sliding_window_state.all_ids[view2]

    obs1 = sliding_window_state.all_observations[view1]
    obs2 = sliding_window_state.all_observations[view2]

    tri1 = sliding_window_state.all_triangulated[view1]
    tri2 = sliding_window_state.all_triangulated[view2]

    common_ids, idx1, idx2 = np.intersect1d(
        ids1[:, 1],
        ids2[:, 1],
        return_indices=True,
    )

    candidates = []

    for pid, i1, i2 in zip(common_ids, idx1, idx2):

        # already reconstructed?
        if tri1[i1] or tri2[i2]:
            continue

        candidates.append(
            TriangulationCandidate(
                point_id=int(pid),
                view1=view1,
                view2=view2,
                uv1=obs1[i1].copy(),
                uv2=obs2[i2].copy(),
            )
        )

    print(
        f"[Triangulation] "
        f"{view1} -> {view2} | "
        f"Shared={len(common_ids)} | "
        f"Candidates={len(candidates)}"
    )

    return candidates
def triangulate_candidates(
    candidates,
    view_set,
    K,
):
    """
    Triangulate feature correspondences between two keyframes.

    Parameters
    ----------
    candidates : list[TriangulationCandidate]

    Returns
    -------
    list[TriangulatedPoint]
    """

    if len(candidates) == 0:
        return []

    view1 = candidates[0].view1
    view2 = candidates[0].view2

    #
    # Projection matrices
    #
    P1, P2 = view_set.get_projection_matrices(
        view1,
        view2,
        K,
    )

    R1, t1 = view_set.get_pose(view1)
    R2, t2 = view_set.get_pose(view2)

    print("\n========== TRIANGULATION DEBUG ==========")
    print(f"Views : {view1} -> {view2}")
    print("\nCamera 1")
    print("R=\n", R1)
    print("t=", t1)

    print("\nCamera 2")
    print("R=\n", R2)
    print("t=", t2)

    print("\nP1=\n", P1)
    print("\nP2=\n", P2)
    print("=========================================\n")

    #
    # Build point arrays
    #
    pts1 = np.array([c.uv1 for c in candidates], dtype=np.float64).T
    pts2 = np.array([c.uv2 for c in candidates], dtype=np.float64).T

    #
    # Linear triangulation
    #
    X_h = cv2.triangulatePoints(
        P1,
        P2,
        pts1,
        pts2,
    )

    #
    # Homogeneous -> Euclidean
    #
    X = (X_h[:3] / X_h[3]).T

    #
    # Camera poses
    #
    R1, t1 = view_set.get_pose(view1)
    R2, t2 = view_set.get_pose(view2)

    triangulated = []

    rejected = 0

    depths = []

    for c, xyz in zip(candidates, X):

        #
        # Cheirality test
        #

        z1 = (R1.T @ (xyz - t1))[2]
        z2 = (R2.T @ (xyz - t2))[2]

        if z1 <= 0 or z2 <= 0:
            rejected += 1
            continue

        depths.append((z1 + z2) * 0.5)

        triangulated.append(
            TriangulatedPoint(
                point_id=c.point_id,
                xyz=xyz,

                view1=c.view1,
                view2=c.view2,

                uv1=c.uv1,
                uv2=c.uv2,
            )
        )

    print(
        f"[Triangulation] "
        f"Input={len(candidates)} | "
        f"Valid={len(triangulated)} | "
        f"Rejected={rejected}"
    )

    if len(depths):

        print(
            "[Triangulation] "
            f"Depth(min={np.min(depths):.2f}, "
            f"median={np.median(depths):.2f}, "
            f"max={np.max(depths):.2f})"
        )

    return triangulated

from vio_core.landmarks import Landmark

def add_landmarks(
    triangulated_points,
    sliding_window_state,
):
    """
    Insert newly triangulated points into the landmark database.

    Returns
    -------
    num_added
    """

    added = 0

    for p in triangulated_points:

        #
        # Skip if landmark already exists
        #
        if p.point_id in sliding_window_state.landmarks:
            continue

        landmark = Landmark(
            point_id=p.point_id,
            xyz=p.xyz.copy(),
            first_view=p.view1,
        )

        landmark.add_observation(
            p.view1,
            p.uv1,
        )

        landmark.add_observation(
            p.view2,
            p.uv2,
        )

        sliding_window_state.landmarks[p.point_id] = landmark

        #
        # Update triangulation flags
        #
        for view_id in (p.view1, p.view2):

            ids = sliding_window_state.all_ids[view_id]

            idx = np.where(ids[:, 1] == p.point_id)[0]

            if len(idx):

                sliding_window_state.all_triangulated[view_id][idx[0]] = True

        added += 1

    print(
        f"[Landmarks] Added {added} landmarks."
    )

    return added


@dataclass
class TriangulatedPoint:
    """
    One successfully reconstructed 3D point.
    """

    point_id: int

    xyz: np.ndarray          # (3,)

    view1: int
    view2: int

    uv1: np.ndarray
    uv2: np.ndarray