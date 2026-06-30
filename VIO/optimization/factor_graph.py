from typing import Dict, List

from .camera_factor import CameraFactor


class FactorGraph:
    """
    Lightweight factor graph.

    Initially contains only camera factors.

    Later it will also contain

        IMU factors
        Velocity nodes
        Bias nodes
    """

    def __init__(self, K):

        self.K = K.copy()

        #
        # Optimization variables
        #

        self.pose_nodes = {}

        self.landmark_nodes = {}

        #
        # Factors
        #

        self.camera_factors: List[CameraFactor] = []

        self.imu_factors = []

    def add_pose(self, view_id, R, t):

        self.pose_nodes[view_id] = {
            "R": R.copy(),
            "t": t.copy(),
        }


    def add_landmark(self, point_id, xyz):

        self.landmark_nodes[point_id] = xyz.copy()


    def add_camera_factor(self, factor):

        self.camera_factors.append(factor)

    def print_summary(self):

        print("\n========== FACTOR GRAPH ==========")

        print(f"Pose Nodes      : {len(self.pose_nodes)}")

        print(f"Landmark Nodes  : {len(self.landmark_nodes)}")

        print(f"Camera Factors  : {len(self.camera_factors)}")

        print(f"IMU Factors     : {len(self.imu_factors)}")

        print("==================================")

    def get_pose(self, view_id):

        pose = self.pose_nodes[view_id]

        return pose["R"].copy(), pose["t"].copy()


    def get_landmark(self, point_id):

        return self.landmark_nodes[point_id].copy()
    
    def update_pose(self, view_id, R, t):

        self.pose_nodes[view_id]["R"] = R.copy()
        self.pose_nodes[view_id]["t"] = t.copy()


    def update_landmark(self, point_id, xyz):

        self.landmark_nodes[point_id] = xyz.copy()