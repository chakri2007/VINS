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