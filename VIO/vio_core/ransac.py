"""
ransac.py — Python port of the RANSAC fundamental-matrix estimation used in
the MATLAB Phase 1 (Structure from Motion) script.

This single function is called from TWO places in the MATLAB reference,
both with identical parameter names/semantics:

1. Inside helperFeaturePointManager.updateSlidingWindow (helperFeaturePointManager.m, line 141):
       [~,inlFf] = estimateFundamentalMatrix(
           obj.AllObservations{viewId-1}(v1,:), currPointsTracked(v1,:), ...
           'Method','RANSAC','NumTrials',obj.params.F_Iterations, ...
           'Confidence',obj.params.F_Confidence,'DistanceThreshold',obj.params.F_Threshold);

2. In the map-initialization block of vio_matlab.mlx (lines 67-77), right
   before estrelpose, to recover the relative pose between the first two
   keyframes:
       [f,inliers] = estimateFundamentalMatrix( ...
           matches1,matches2,Method="RANSAC", ...
           NumTrials=params.F_Iterations,DistanceThreshold=params.F_Threshold, ...
           Confidence=params.F_Confidence);

Both call sites wrap the call in a `for k = 1:N` loop, keeping the result
with the most inliers across N independent RANSAC runs (N = params.F_loop
inside updateSlidingWindow; N = 10, hardcoded, in the mlx map-init block).
That retry loop lives in the CALLER (update_sliding_window, or the Phase 1
script translation) — this file only ports the single-call estimator
itself, matching the MATLAB function signature 1:1.

Parameter correspondence (kept EXACTLY as in the MATLAB reference):
    MATLAB params.F_Iterations   -> num_trials
    MATLAB params.F_Confidence   -> confidence       (0-100, NOT 0-1 — see note below)
    MATLAB params.F_Threshold    -> dist_threshold
"""

import cv2
import numpy as np


def estimate_fundamental_matrix_ransac(
    pts1: np.ndarray,
    pts2: np.ndarray,
    num_trials: int,
    confidence: float,
    dist_threshold: float,
):
    """Port of MATLAB's estimateFundamentalMatrix(pts1, pts2, 'Method','RANSAC', ...).

    Parameters
    ----------
    pts1, pts2 : (N, 2) float arrays
        Matched point correspondences, same N, row i in pts1 <-> row i in pts2.
    num_trials : int
        MATLAB: NumTrials = params.F_Iterations (e.g. 2000).
    confidence : float
        MATLAB: Confidence = params.F_Confidence, given as a PERCENTAGE
        (e.g. 99 means 99%), matching helperVIOParameters.m
        (`params.F_Confidence = 99;`). Converted internally to the
        0-1 range cv2.findFundamentalMat expects — the caller should
        keep passing 99 (or whatever params.F_Confidence is), exactly
        as in the MATLAB reference, not 0.99.
    dist_threshold : float
        MATLAB: DistanceThreshold = params.F_Threshold (e.g. 4), max
        distance in pixels from a point to its epipolar line to be
        considered an inlier.

    Returns
    -------
    F : (3, 3) float array, or None
        Estimated fundamental matrix. None if estimation failed (e.g.
        fewer than 8 valid correspondences, or cv2 could not find a
        solution) — matching MATLAB's behavior of returning an empty/
        invalid F and zero inliers in degenerate cases.
    inliers : (N,) bool array
        True where the correspondence is an inlier w.r.t. F. All-False
        (shape (N,)) if estimation failed, so callers can safely do
        len(find(inliers)) / np.count_nonzero(inliers) comparisons
        without special-casing None.
    """
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)
    n = pts1.shape[0]

    # cv2.findFundamentalMat needs at least 8 points for FM_RANSAC.
    # MATLAB's estimateFundamentalMatrix would similarly be unable to fit
    # with too few points; mirror that as a clean "no inliers" result
    # rather than letting cv2 raise.
    if n < 8:
        return None, np.zeros(n, dtype=bool)

    # MATLAB Confidence is a percentage (0-100); cv2 wants a 0-1 probability.
    cv2_confidence = confidence / 100.0

    F, mask = cv2.findFundamentalMat(
        pts1,
        pts2,
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=dist_threshold,
        confidence=cv2_confidence,
        maxIters=num_trials,
    )

    if F is None or mask is None:
        return None, np.zeros(n, dtype=bool)

    inliers = mask.ravel().astype(bool)
    return F, inliers