import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
import yaml
import cv2
import numpy as np
from cv_bridge import CvBridge
import os
import sys

# Make the project root importable regardless of working directory.
current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from vio_core.vio_core import VisualInertialOdometry
from ros_wrapper.vio_visualizer import VOFeatureVisualizer


class VisualOdometryNode(Node):
    """ROS 2 node: subscribes to a mono camera topic, runs the VIO
    initialisation pipeline, and publishes annotated feature-track images.
    """

    def __init__(self):
        super().__init__('vo_subscriber_node')

        self.bridge = CvBridge()

        # ── Load config ──────────────────────────────────────────────
        config_path = self.declare_parameter(
            'config_path',
            '/home/icgel/vio/VINS/VIO/config/ros_config.yaml',
        ).get_parameter_value().string_value

        with open(config_path, 'r') as f:
            self.ros_config = yaml.safe_load(f)

        self.mode = self.ros_config.get('vo_mode', 'mono')
        self.get_logger().info(
            f"Initializing VIO Node in [{self.mode.upper()}] mode."
        )

        # ── Load calibration ─────────────────────────────────────────
        calibration_data = self._load_calibration_files()

        # ── Build VIO pipeline ────────────────────────────────────────
        self.vio = VisualInertialOdometry(calibration_data)

        # ── Visualizer ────────────────────────────────────────────────
        self.visualizer = VOFeatureVisualizer()

        # ── Camera subscriber ─────────────────────────────────────────
        camera_topic = self.ros_config.get('left_camera_topic', '/camera/image_raw')
        if self.mode == 'mono':
            self.image_sub = self.create_subscription(
                Image,
                camera_topic,
                self._mono_image_callback,
                10,
            )
            self.get_logger().info(f"Subscribed to camera: {camera_topic}")

        # ── IMU subscriber (stored for Phase 2 use) ───────────────────
        imu_topic = self.ros_config.get('imu_topic', '/imu/imu/data')
        self.imu_sub = self.create_subscription(
            Imu,
            imu_topic,
            self._imu_callback,
            200,
        )
        self.get_logger().info(f"Subscribed to IMU: {imu_topic}")

    # ── Callbacks ─────────────────────────────────────────────────────

    def _mono_image_callback(self, msg: Image):
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        cv_image  = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')

        result = self.vio.process_frame_mono(cv_image, timestamp)

        # Publish annotated feature-track image.
        self.visualizer.publish_feature_tracks(
            cv_image,
            timestamp,
            result['tracks'],
            result['K'],
            result['D'],
        )

        if result.get('pose') is not None:
            self.get_logger().info(
                f"[Frame {self.vio.frameID}] Pose available."
            )

    def _imu_callback(self, msg: Imu):
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        accel = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
        ])
        gyro = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
        ])

        # Appends to sw_state.imu_buffer (timestamp-ordered); consumed on
        # demand by _build_imu_preintegrations via extract_imu_between.
        self.vio.process_imu(accel, gyro, timestamp)

    # ── Helpers ───────────────────────────────────────────────────────

    def _load_calibration_files(self) -> dict:
        calib = {}
        with open(self.ros_config['left_camera_config_path'], 'r') as f:
            calib['left'] = yaml.safe_load(f)
        if self.mode == 'stereo':
            with open(self.ros_config['right_camera_config_path'], 'r') as f:
                calib['right'] = yaml.safe_load(f)

        # IMU noise densities + T_BS (imu.yaml), consumed by
        # VisualInertialOdometry.__init__ as calib_data['imu'] and passed
        # through to IMUPreintegrator in _build_imu_preintegrations.
        imu_config_path = self.ros_config.get('imu_config_path')
        if imu_config_path:
            with open(imu_config_path, 'r') as f:
                calib['imu'] = yaml.safe_load(f)
        else:
            self.get_logger().warn(
                "No 'imu_config_path' in ros_config.yaml — "
                "IMUPreintegrator will fall back to its default noise params."
            )

        return calib

    def destroy_node(self):
        self.vio.feature_extractor.shutdown()
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