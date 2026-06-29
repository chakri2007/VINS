from dataclasses import dataclass
import numpy as np


@dataclass
class PnPCorrespondence:
    """
    One 3D ↔ 2D correspondence for solvePnPRansac().
    """

    point_id: int

    xyz: np.ndarray      # (3,) world coordinates

    uv: np.ndarray       # (2,) image coordinates

import numpy as np


def find_pnp_correspondences(
    sliding_window_state,
    current_view_id,
):
    """
    Collect all triangulated landmarks observed in the current frame.

    Returns
    -------
    correspondences : list[PnPCorrespondence]
    """

    correspondences = []

    #
    # Feature IDs currently tracked in this frame
    #
    ids = sliding_window_state.all_ids[current_view_id]

    #
    # Image observations
    #
    observations = sliding_window_state.all_observations[current_view_id]

    for row, uv in zip(ids, observations):

        #
        # Global feature ID
        #
        point_id = int(row[1])

        #
        # Skip features that have not been triangulated
        #
        if point_id not in sliding_window_state.landmarks:
            continue

        landmark = sliding_window_state.landmarks[point_id]

        correspondences.append(

            PnPCorrespondence(

                point_id=point_id,

                xyz=landmark.xyz.copy(),

                uv=uv.copy(),

            )
        )

    print(
        f"[PnP] "
        f"Current features={len(ids)} | "
        f"Landmark matches={len(correspondences)}"
    )
    print("\n========== PnP CORRESPONDENCES ==========")
    print(f"Current View : {current_view_id}")
    print(f"Tracked Features : {len(ids)}")
    print(f"3D Matches : {len(correspondences)}")

    for c in correspondences[:10]:
        print(
            f"ID={c.point_id:4d} "
            f"XYZ={np.round(c.xyz,2)} "
            f"UV={np.round(c.uv,1)}"
        )

    print("=========================================\n")

    return correspondences