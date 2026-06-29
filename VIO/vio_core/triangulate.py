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