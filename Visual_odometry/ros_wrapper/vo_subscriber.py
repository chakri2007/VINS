import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
import yaml
import cv2
import numpy as np
from cv_bridge import CvBridge
import os, sys

from vo_visualizer import VOFeatureVisualizer

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from vo_core.vo_pipeline                    import VisualOdometryPipeline
from Inertial.imu_pipeline                  import IMUPipeline
from motion_estimation.motion_estimation    import VIOEstimationPipeline


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

        # [Fix Issue 8] — validate imu_config_path with a clear error
        imu_noise_params = self._load_imu_noise_params()

        # [Fix Issue 7] — VIOEstimationPipeline constructs and owns:
        #   chunk_db, vio_state, via, motion_estimator
        self.estimation_core = VIOEstimationPipeline(
            calibration_data=calibration_data,
            imu_noise_params=imu_noise_params,
        )

        # IMUPipeline uses the shared chunk_db from estimation_core
        self.imu_pipeline = IMUPipeline(
            chunk_db         = self.estimation_core.chunk_db,
            b_a              = np.zeros(3),
            b_w              = np.zeros(3),
            imu_noise_params = imu_noise_params,
        )
        self.imu_pipeline.start()

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

        self.visualizer = VOFeatureVisualizer()

        # External pose callback registered by vo_publisher
        self._external_pose_callback = None

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
        elif self.mode == "stereo":
            from message_filters import Subscriber, ApproximateTimeSynchronizer
            self.left_sub  = Subscriber(self, Image, self.ros_config['left_camera_topic'])
            self.right_sub = Subscriber(self, Image, self.ros_config['right_camera_topic'])
            self.ats = ApproximateTimeSynchronizer(
                [self.left_sub, self.right_sub], queue_size=10, slop=0.02
            )
            self.ats.registerCallback(self.stereo_image_callback)
            self.get_logger().info("Subscribed to synchronized stereo topics.")

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

    def imu_callback(self, msg: Imu):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        accel = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
        ], dtype=np.float64)
        gyro = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
        ], dtype=np.float64)
        self.imu_pipeline.push_raw(stamp, accel, gyro)

    def mono_image_callback(self, msg: Image):
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        cv_image  = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # process_frame_mono:
        #   1. calls imu_pipeline.notify_frame(timestamp)  — cuts IMU chunk
        #   2. calls motion_estimator.compute_pose(R, t)   — fast path
        #   3. returns result dict with 'pose' (Pose dataclass or None)
        result = self.vo_pipeline.process_frame_mono(cv_image, timestamp)
        if result is None:
            return

        # Forward pose to vo_publisher via registered callback
        if (self._external_pose_callback is not None
                and result.get('pose') is not None):
            self._external_pose_callback(result['pose'], msg.header.stamp)

        self.visualizer.publish_feature_tracks(
            cv_image, timestamp, result['tracks'], result['K'], result['D']
        )

    def stereo_image_callback(self, left_msg: Image, right_msg: Image):
        pass

    def register_pose_output_callback(self, cb):
        """Called by vo_publisher to receive Pose objects."""
        self._external_pose_callback = cb

    # ── Config helpers ────────────────────────────────────────────────────

    def load_calibration_files(self) -> dict:
        calib = {}
        with open(self.ros_config['left_camera_config_path'], 'r') as f:
            calib['left'] = yaml.safe_load(f)
        if self.mode == "stereo":
            with open(self.ros_config['right_camera_config_path'], 'r') as f:
                calib['right'] = yaml.safe_load(f)
        return calib

    def _load_imu_noise_params(self) -> dict:
        # [Fix Issue 8] — clear KeyError if imu_config_path missing
        imu_yaml_path = self.ros_config.get('imu_config_path')
        if imu_yaml_path is None:
            raise KeyError(
                "'imu_config_path' missing from ros_config.yaml. "
                "Add it pointing to your IMU calibration YAML."
            )
        with open(imu_yaml_path, 'r') as f:
            imu_data = yaml.safe_load(f)
        self.get_logger().info(f"Loaded IMU noise params from: {imu_yaml_path}")
        return {
            'sigma_a'  : float(imu_data['accelerometer_noise_density']),
            'sigma_w'  : float(imu_data['gyroscope_noise_density']),
            'sigma_ba' : float(imu_data['accelerometer_random_walk']),
            'sigma_bw' : float(imu_data['gyroscope_random_walk']),
        }

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