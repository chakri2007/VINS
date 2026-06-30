import numpy as np
import cv2
from scipy.optimize import least_squares


class BundleAdjuster:

    def __init__(self, factor_graph):

        self.graph = factor_graph

        #
        # Optimization vector indexing
        #
        self.pose_index = {}
        self.landmark_index = {}

        # Fixed pose nodes
        self.fixed_pose_ids = set()

    # ------------------------------------------------------------ #
    # Pack graph → optimization vector
    # ------------------------------------------------------------ #

    def pack_variables(self):
        """
        Convert the current factor graph into one optimization vector.

        Pose parameterization
        ---------------------
        [rx ry rz tx ty tz]

        where
            r = Rodrigues rotation vector
            t = camera center in world coordinates

        Landmark parameterization
        -------------------------
        [x y z]

        Returns
        -------
        x : (N,) ndarray
            Optimization vector.
        """

        x = []

        self.pose_index.clear()
        self.landmark_index.clear()

        #
        # Pack camera poses
        #
        for view_id in sorted(self.graph.pose_nodes.keys()):

            #
            # Skip fixed poses
            #
            if view_id in self.fixed_pose_ids:
                continue

            pose = self.graph.pose_nodes[view_id]

            R = pose["R"]
            t = pose["t"]

            rvec, _ = cv2.Rodrigues(R)

            start = len(x)

            self.pose_index[view_id] = start

            x.extend(rvec.flatten())
            x.extend(t.flatten())

        #
        # Pack landmarks
        #
        for point_id in sorted(self.graph.landmark_nodes.keys()):

            xyz = self.graph.landmark_nodes[point_id]

            start = len(x)

            self.landmark_index[point_id] = start

            x.extend(xyz.flatten())

        return np.asarray(
            x,
            dtype=np.float64,
        )

    # ------------------------------------------------------------ #
    # Unpack optimization vector → graph
    # ------------------------------------------------------------ #

    def unpack_variables(
        self,
        x,
    ):
        """
        Write optimized variables back into the factor graph.

        Parameters
        ----------
        x : (N,) ndarray
            Optimization vector returned by least_squares().
        """

        #
        # Update camera poses
        #
        for view_id in sorted(self.pose_index.keys()):

            start = self.pose_index[view_id]

            rvec = x[start:start + 3]

            t = x[start + 3:start + 6]

            R, _ = cv2.Rodrigues(rvec)

            self.graph.update_pose(
                view_id,
                R,
                t,
            )

        #
        # Update landmarks
        #
        for point_id in sorted(self.landmark_index.keys()):
            start = self.landmark_index[point_id]

            xyz = x[start:start + 3]

            self.graph.update_landmark(
                point_id,
                xyz,
            )

    def compute_residuals(
        self,
        x,
    ):
        """
        Compute reprojection residuals for every camera factor.

        Parameters
        ----------
        x : ndarray
            Optimization vector.

        Returns
        -------
        residuals : ndarray
            Stacked reprojection residual vector.
        """


        #
        # Start with all graph poses
        #
        pose_cache = {}

        for view_id, pose in self.graph.pose_nodes.items():

            pose_cache[view_id] = (
                pose["R"],
                pose["t"],
            )

        #
        # Overwrite only optimized poses
        #
        for view_id, start in self.pose_index.items():

            rvec = x[start:start + 3]

            t = x[start + 3:start + 6]

            R, _ = cv2.Rodrigues(rvec)

            pose_cache[view_id] = (R, t)

        landmark_cache = {}

        # start from graph values
        for point_id, xyz in self.graph.landmark_nodes.items():
            landmark_cache[point_id] = xyz

        # overwrite optimized ones
        for point_id, start in self.landmark_index.items():

            landmark_cache[point_id] = x[start:start + 3].copy()

        #
        # Compute reprojection residuals
        #

        residuals = []

        for factor in self.graph.camera_factors:

            #
            # Pose
            #
            R, t = pose_cache[factor.view_id]

            #
            # Landmark
            #
            xyz = landmark_cache[factor.point_id]

            #
            # World -> Camera
            #
            pc = R.T @ (xyz - t)


            #
            # Camera -> Image
            #
            uv = self.graph.K @ pc

            uv = uv[:2] / uv[2]

            #
            # Reprojection error
            #
            error = uv - factor.measurement

            #
            # Weight using information matrix
            #
            weighted_error = factor.sqrt_information @ error

            residuals.extend(weighted_error.tolist())

        return np.asarray(
            residuals,
            dtype=np.float64,
        )

    def optimize(
        self,
        max_iterations=100,
        ):
        """
        Run bundle adjustment.

        Parameters
        ----------
        max_iterations : int

        Returns
        -------
        result : OptimizeResult
        """

        #
        # Initial optimization vector
        #
        x0 = self.pack_variables()

        if len(x0) == 0:
            print("Nothing to optimize.")
            return None

        print("\n========== BUNDLE ADJUSTMENT ==========")
        print(f"Variables : {len(x0)}")
        print(f"Residuals : {len(self.compute_residuals(x0))}")

        #
        # Nonlinear least squares
        #
        result = least_squares(
            fun=self.compute_residuals,
            x0=x0,
            method="trf",
            loss="huber",
            verbose=2,
            max_nfev=max_iterations,
        )

        #
        # Write optimized values back
        #
        self.unpack_variables(result.x)

        print("=======================================\n")

        return result
    
    def fix_pose(self, view_id):
        """
        Exclude a pose from optimization.
        """
        self.fixed_pose_ids.add(view_id)


    def unfix_pose(self, view_id):
        """
        Re-enable a pose for optimization.
        """
        self.fixed_pose_ids.discard(view_id)


    def clear_fixed_poses(self):
        """
        Remove all fixed-pose constraints.
        """
        self.fixed_pose_ids.clear()