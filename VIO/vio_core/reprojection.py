import numpy as np

def reprojection_error(
    landmark,
    view_set,
    K,
):
    """
    Compute reprojection error of one landmark over all its observations.

    Returns
    -------
    mean_error
    errors
    """

    errors = []


    for obs in landmark.observations:

        uv_pred = view_set.project_point(
            obs.view_id,
            K,
            landmark.xyz,
        )

        if uv_pred is None:
            continue

        err = np.linalg.norm(
            uv_pred - obs.uv
        )

        errors.append(err)


    if len(errors) == 0:
        return np.inf, []

    return float(np.mean(errors)), errors

def validate_landmarks(
    sliding_window_state,
    view_set,
    K,
    max_error=3.0,
):
    """
    Validate every landmark using reprojection error.
    """

    good = 0
    bad = 0

    all_errors = []

    for landmark in sliding_window_state.landmarks.values():

        mean_error, errors = reprojection_error(
            landmark,
            view_set,
            K,
        )

        landmark.reprojection_error = mean_error

        if mean_error > max_error:

            landmark.is_bad = True
            bad += 1

        else:

            landmark.is_bad = False
            good += 1

        all_errors.extend(errors)

    print("\n========== Landmark Validation ==========")
    print(f"Good landmarks : {good}")
    print(f"Bad landmarks  : {bad}")

    if len(all_errors):

        print(
            f"Mean error : {np.mean(all_errors):.3f} px"
        )

        print(
            f"Median     : {np.median(all_errors):.3f} px"
        )

        print(
            f"Maximum    : {np.max(all_errors):.3f} px"
        )

    print("=========================================\n")

    print("\n========== TRIANGULATION ANGLES ==========")
    print(f"Min    : {np.min(angles):.2f} deg")
    print(f"Median : {np.median(angles):.2f} deg")
    print(f"Mean   : {np.mean(angles):.2f} deg")
    print(f"Max    : {np.max(angles):.2f} deg")
    print("==========================================")