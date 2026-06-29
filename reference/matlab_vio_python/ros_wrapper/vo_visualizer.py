import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge
from rclpy.time import Time

class VOFeatureVisualizer(Node):
    def __init__(self):
        super().__init__('vo_feature_visualizer')
        self.bridge = CvBridge()

        self.image_pub = self.create_publisher(
            Image,
            '/visual_odometry/features',
            10
        )

    def publish_feature_tracks(
        self,
        raw_frame,
        timestamp,
        active_tracks,
        K,
        distortion_coeffs
    ):

        undistorted_image = cv2.undistort(
            raw_frame,
            K,
            distortion_coeffs
        )

        for feature_id, point_history in active_tracks.items():

            if len(point_history) == 0:
                continue

            # point_history entries are:
            # (frame_idx, u, v)

            _, u, v = point_history[0]

            current_pt = (
                int(u),
                int(v)
            )

            if len(point_history) == 1:
                color = (0, 255, 0)
                cv2.circle(
                    undistorted_image,
                    current_pt,
                    4,
                    color,
                    -1
                )
            else:
                color = (255, 0, 0)
                cv2.circle(
                    undistorted_image,
                    current_pt,
                    4,
                    color,
                    -1
                )

                for i in range(len(point_history) - 1):

                    _, u1, v1 = point_history[i]
                    _, u2, v2 = point_history[i + 1]

                    pt_start = (
                        int(u1),
                        int(v1)
                    )

                    pt_end = (
                        int(u2),
                        int(v2)
                    )

                    cv2.line(
                        undistorted_image,
                        pt_start,
                        pt_end,
                        (0, 255, 255),
                        1
                    )

        img_msg = self.bridge.cv2_to_imgmsg(
            undistorted_image,
            encoding='bgr8'
        )

        ros_time_obj = Time(seconds=timestamp).to_msg()

        img_msg.header.stamp = ros_time_obj
        img_msg.header.frame_id = "camera_link"

        self.image_pub.publish(img_msg)