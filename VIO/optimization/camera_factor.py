from dataclasses import dataclass, field
import numpy as np


@dataclass
class CameraFactor:
    """
    One reprojection observation.

    Connects

        Camera Pose
              ↓
        3D Landmark
              ↓
      2D Image Measurement

    Equivalent to MATLAB's factorCameraSE3AndPointXYZ.
    """

    view_id: int
    point_id: int

    # observed image measurement (u,v)
    measurement: np.ndarray

    # information matrix (inverse covariance)
    information: np.ndarray = field(
        default_factory=lambda: np.eye(2)
    )

    def project(
        self,
        R: np.ndarray,
        C: np.ndarray,
        xyz: np.ndarray,
        K: np.ndarray,
    ):
        """
        Project a world point into the image.

        Parameters
        ----------
        R : (3,3)
            Camera-to-world rotation.

        C : (3,)
            Camera center in world coordinates.

        xyz : (3,)
            Landmark in world coordinates.

        K : (3,3)
            Camera intrinsics.

        Returns
        -------
        uv : (2,)
            Pixel coordinates.

        None
            If the landmark lies behind the camera.
        """

        #
        # World -> Camera
        #
        pc = R.T @ (xyz - C)

        #
        # Behind camera
        #
        if pc[2] <= 1e-8:
            return None

        # Perspective projection
        uv = K @ pc

        uv = uv[:2] / uv[2]

        return uv

    def residual(
        self,
        R: np.ndarray,
        C: np.ndarray,
        xyz: np.ndarray,
        K: np.ndarray,
    ):
        """
        Compute reprojection error.

        residual = observed - predicted
        """

        uv_pred = self.project(
            R,
            C,
            xyz,
            K,
        )

        if uv_pred is None:
            return np.zeros(2)

        return self.measurement - uv_pred

    def weighted_residual(
        self,
        R: np.ndarray,
        C: np.ndarray,
        xyz: np.ndarray,
        K: np.ndarray,
    ):
        """
        Information-weighted reprojection residual.
        """

        r = self.residual(
            R,
            C,
            xyz,
            K,
        )

        #
        # sqrt(information)
        #
        L = np.linalg.cholesky(self.information)

        return L @ r