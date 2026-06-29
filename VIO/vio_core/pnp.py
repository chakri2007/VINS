from dataclasses import dataclass
import numpy as np

import cv2


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

def solve_pnp(
    correspondences,
    K,
):
    """
    Estimate camera pose from 3D-2D correspondences.

    Returns
    -------
    PnPResult
    """

    if len(correspondences) < 6:
        print("[PnP] Not enough correspondences.")
        return None
    
    object_points = np.array(
        [c.xyz for c in correspondences],
        dtype=np.float64,
    )

    image_points = np.array(
        [c.uv for c in correspondences],
        dtype=np.float64,
    )
    print("\n========== PnP INPUT ==========")
    print("Object points :", object_points.shape)
    print("Image points  :", image_points.shape)
    print("===============================\n")

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            K,
            None,                       # images are already undistorted
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=4.0,
            confidence=0.99,
            iterationsCount=100,
        )
    

    #Frame conversion: OpenCV uses camera-to-world rotation, but we want world-to-camera rotation.

    Rcw, _ = cv2.Rodrigues(rvec)

    Rwc = Rcw.T

    C = (-Rwc @ tvec).reshape(3)





    print("\n========== PnP POSE ==========")

    print("Camera Center:")
    print(C)

    print("\nRotation:")
    print(Rwc)

    print("==============================\n")




    print("\n========== PnP RESULT ==========")
    print("Success :", success)

    if inliers is None:
        print("Inliers : 0")
    else:
        print("Inliers :", len(inliers))

    print("================================\n")


from dataclasses import dataclass
import numpy as np


@dataclass
class PnPResult:
    """
    Result returned by solvePnPRansac().
    """

    success: bool

    R: np.ndarray          # (3,3) Camera-to-world rotation

    t: np.ndarray          # (3,) Camera center in world

    inlier_ids: np.ndarray

    num_inliers: int

    reprojection_error: float