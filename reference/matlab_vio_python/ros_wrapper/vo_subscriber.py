import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
import yaml
import cv2
import numpy as np
from cv_bridge import CvBridge
import os, sys

from vo_visualizer import VOFeatureVisualizer
from vo_publisher import VOPosePublisher

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)


class VisualOdometryNode(Node):

    def __init__(self):
        super().__init__('vo_subscriber_node')

        self.bridge = CvBridge()

        config_path = "/home/icgel/vio/VINS/Visual_odometry/config/ros_config.yaml"
        with open(config_path, 'r') as f:
            self.ros_config = yaml.safe_load(f)

        self.mode = self.ros_config['vo_mode']
        self.get_logger().info(
            f"Initializing VIO Node in [{self.mode.upper()}] mode."
        )

        calibration_data = self.load_calibration_files()


        # VO pipeline receives the motion_estimator from estimation_core
        self.vo_pipeline = VisualOdometryPipeline(
            calibration_data = calibration_data,
            motion_estimator = self.estimation_core.motion_estimator,
            mode             = self.mode,
        )

        # [Fix Issues 4 & 5] — set_frame_callback fires notify_frame()
        # on EVERY camera frame so IMU pipeline cuts one chunk per frame
        self.vo_pipeline.set_frame_callback(
            self.imu_pipeline.notify_frame
        )

        self.visualizer     = VOFeatureVisualizer()
        self.pose_publisher = VOPosePublisher()

        # ── Camera subscriber ─────────────────────────────────────────────
        if self.mode == "mono":
            self.image_sub = self.create_subscription(
                Image,
                self.ros_config['left_camera_topic'],
                self.mono_image_callback,
                10,
            )
            self.get_logger().info(
                f"Subscribed to camera: {self.ros_config['left_camera_topic']}"
            )

        # ── IMU subscriber ────────────────────────────────────────────────
        imu_topic = self.ros_config.get('imu_topic', '/imu/imu/data')
        self.imu_sub = self.create_subscription(
            Imu,
            imu_topic,
            self.imu_callback,
            200,
        )
        self.get_logger().info(f"Subscribed to IMU: {imu_topic}")

    # ── Callbacks ─────────────────────────────────────────────────────────

    
    def mono_image_callback(self, msg: Image):
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        cv_image  = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        result = self.vo_pipeline.process_frame_mono(cv_image, timestamp)
        if result is None:
            return

        # Forward pose to vo_publisher — same direct-call pattern as visualizer
        if result.get('pose') is not None:
            self.pose_publisher.publish(result['pose'], msg.header.stamp)

        self.visualizer.publish_feature_tracks(
            cv_image, timestamp, result['tracks'], result['K'], result['D']
        )

    def load_calibration_files(self) -> dict:
        calib = {}
        with open(self.ros_config['left_camera_config_path'], 'r') as f:
            calib['left'] = yaml.safe_load(f)
        if self.mode == "stereo":
            with open(self.ros_config['right_camera_config_path'], 'r') as f:
                calib['right'] = yaml.safe_load(f)
        return calib



    def destroy_node(self):
        self.imu_pipeline.stop()
        self.vo_pipeline.shutdown()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VisualOdometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down VIO node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()