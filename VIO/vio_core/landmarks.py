"""
landmark.py

3D landmark representation used throughout the VIO pipeline.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Observation:
    """
    One observation of a landmark in an image.
    """

    view_id: int
    uv: np.ndarray        # (2,)


@dataclass
class Landmark:
    """
    One reconstructed 3D landmark.
    """

    point_id: int

    xyz: np.ndarray       # (3,)

    first_view: int

    observations: list[Observation] = field(default_factory=list)

    is_triangulated: bool = True

    # Consecutive windowed-BA cycles in which every observation of this
    # landmark failed the near/behind-camera depth check (see
    # ceres_bundle_adjustment_window.py's MIN_PROJECTION_DEPTH filter
    # and memory_management.sliding_window.cull_stale_landmarks). Reset
    # to 0 the moment the landmark is actually used in a solve again.
    # This existing before now would have caught the "same ~500
    # observations skipped every single window" pattern instead of
    # letting it accumulate forever.
    depth_fail_streak: int = 0

    def add_observation(
        self,
        view_id: int,
        uv: np.ndarray,
    ):

        self.observations.append(
            Observation(
                view_id=view_id,
                uv=np.asarray(uv, dtype=float),
            )
        )