import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import Image, Imu
import yaml
import cv2
import numpy as np
from cv_bridge import CvBridge
import os
import sys
import threading
import queue
import traceback

VIO_DEBUG = os.environ.get("VIO_DEBUG", "0") == "1"

# Make the project root importable regardless of working directory.
current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from vio_core.vio_core import VisualInertialOdometry
from ros_wrapper.vio_visualizer import VOFeatureVisualizer
from ros_wrapper.vio_publisher import VIOOdometryPublisher
from imu.vi_alignment import camera_pose_to_body_pose


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

        # ── Odometry publisher (nav_msgs/Odometry + TF) ────────────────
        # Separate rclpy.Node instance -- only used for its publisher/
        # TransformBroadcaster (create_publisher etc. work without this
        # node ever being spun itself; this process's single executor
        # spins `self`, the outer VisualOdometryNode).
        self.odometry_publisher = VIOOdometryPublisher()

        # Backend work queue. Unbounded and NEVER drops an item: once
        # process_frontend() (see vio_loop_frontend) has called
        # update_sliding_window() and committed a frame into
        # sw_state.sliding_window_view_ids, that frame MUST eventually
        # get a view_set.add_view() call from the backend, or later
        # code (graph_builder.build(), triangulation, ...) will look up
        # a view_id that's in the window but not in view_set and raise
        # KeyError. A drop-oldest bounded queue silently violates that
        # invariant -- it was tried here before and broke exactly this
        # way. Bounding backend WORK (max_solver_time_in_seconds on the
        # Ceres calls) is the correct way to keep this from growing
        # latency unboundedly; bounding the QUEUE by dropping committed
        # frames is not.
        self._backend_running = True
        self._backend_queue = queue.Queue()
        self._backend_thread = threading.Thread(
            target=self._backend_worker,
            name='vio_backend_worker',
            daemon=True,
        )
        self._backend_thread.start()

        # ── Callback groups ────────────────────────────────────────────
        # Two separate MutuallyExclusiveCallbackGroups, not the default
        # single group. With a MultiThreadedExecutor and no explicit
        # groups, ALL callbacks share one implicit mutually-exclusive
        # group -- which is no better than single-threaded spin() for our
        # purposes and was the original bug. The naive fix of just adding
        # more executor threads with no groups is actually worse: ROS
        # would then be free to dispatch image callback N+1 before image
        # callback N finishes, so multiple vio_loop_frontend() calls can
        # run concurrently, fighting over state_lock and thrashing the
        # GIL -- that's the "laggy / not processing properly" regression.
        # What we actually want is: image callbacks stay serialized
        # against each other (frontend still processes one frame at a
        # time, in order), while IMU callbacks are free to run on a
        # different thread even while an image callback is in flight, so
        # imu_buffer never stalls behind a slow frontend/backend step.
        # One group per subscription achieves exactly that.
        self._image_cb_group = MutuallyExclusiveCallbackGroup()
        self._imu_cb_group   = MutuallyExclusiveCallbackGroup()

        # ── Camera subscriber ─────────────────────────────────────────
        camera_topic = self.ros_config.get('left_camera_topic', '/camera/image_raw')
        if self.mode == 'mono':
            self.image_sub = self.create_subscription(
                Image,
                camera_topic,
                self._mono_image_callback,
                10,
                callback_group=self._image_cb_group,
            )
            self.get_logger().info(f"Subscribed to camera: {camera_topic}")

        # ── IMU subscriber (stored for Phase 2 use) ───────────────────
        imu_topic = self.ros_config.get('imu_topic', '/imu/imu/data')
        self.imu_sub = self.create_subscription(
            Imu,
            imu_topic,
            self._imu_callback,
            200,
            callback_group=self._imu_cb_group,
        )
        self.get_logger().info(f"Subscribed to IMU: {imu_topic}")

    # ── Callbacks ─────────────────────────────────────────────────────

    def _mono_image_callback(self, msg: Image):
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        cv_image  = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')

        # Frontend only: KLT tracking + RANSAC + new-feature detection.
        # Stays on this (callback) thread -- bounded/cheap, same as
        # VINS-Fusion's trackImage() call inside inputImage().
        frontend_result = self.vio.vio_loop_frontend(cv_image, timestamp)

        # Publish annotated feature-track image immediately -- this only
        # needs the frontend's tracks, not the backend's pose/BA result.
        self.visualizer.publish_feature_tracks(
            cv_image,
            timestamp,
            self.vio.get_active_tracks(),
            self.vio.K,
            self.vio.distortion_coeffs,
        )

        if frontend_result is None:
            # First frame -- _init_first_frame() already handled it,
            # nothing to hand off to the backend yet.
            return

        frameID, ts = frontend_result

        # Hand off to the backend worker thread. Never dropped -- see
        # the comment on self._backend_queue's construction for why.
        self._backend_queue.put((frameID, ts))

        backlog = self._backend_queue.qsize()

        if VIO_DEBUG:
            # Unthrottled trend line -- lets you plot backlog vs. frameID
            # afterwards to see whether the backend is steadily falling
            # behind (queue growing) vs. just having occasional spikes.
            print(f"[QUEUE-DEBUG] frame={frameID} backlog={backlog}")

        if backlog >= 5:
            # Not fatal, but worth knowing about: the backend is falling
            # behind the camera rate and latency is growing. Throttled so
            # this doesn't spam the log every frame once it's backed up.
            self.get_logger().warn(
                f"[VIO] Backend queue backlog: {backlog} frames pending "
                f"(frontend at {frameID}) -- backend is falling behind.",
                throttle_duration_sec=2.0,
            )

    def _backend_worker(self):
        """
        Dedicated thread: pulls (frameID, timestamp) off the queue,
        strictly FIFO, and runs the backend (window-membership decision
        + phase selection / PnP / bundle adjustment). Mirrors
        VINS-Fusion's Estimator::processMeasurements() loop. Blocks
        freely on Ceres solves here -- that's the whole point of this
        thread existing. FIFO order matters: update_window_membership()
        (called inside vio_loop_backend) is only safe because this is
        the one place, in this one thread, that frames are ever
        finalized into the window, always in the order they were
        queued.
        """
        while self._backend_running:
            try:
                frameID, ts = self._backend_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                self.vio.vio_loop_backend(frameID, ts)
            except Exception:
                # rclpy's logger has no `exc_info` kwarg (it only accepts
                # throttle_duration_sec / throttle_time_source_type /
                # skip_first / once) -- format the traceback ourselves
                # and pass it as part of the message string instead.
                self.get_logger().error(
                    f"[VIO backend] exception processing frame {frameID}:\n"
                    f"{traceback.format_exc()}"
                )
                continue  # don't report "complete" below for a frame that errored

            self._publish_odometry_if_aligned(frameID, ts)

            self.get_logger().info(
                f"[Frame {frameID}] Backend step complete."
            )

    def _publish_odometry_if_aligned(self, frameID, timestamp):
        """
        Publish the latest pose/velocity estimate, gated on
        self.vio.isVI_aligned -- Phase 1/2 (pre-alignment) poses are
        vision-only/unscaled and not yet in the metric, gravity-aligned
        frame odometry consumers expect.

        ViewSet/VIO internals store camera-to-world poses; Odometry is
        conventionally published in the body/IMU frame, so convert via
        T_BS (camera->body extrinsic) before publishing, same
        convention as camera_pose_to_body_pose used elsewhere
        (imu/vi_alignment.py, vio_core._predict_pose_from_imu).
        """

        if not self.vio.isVI_aligned:
            return

        if frameID not in self.vio.view_set.view_ids:
            # Frame never got a pose committed (e.g. insufficient IMU
            # coverage entirely -- see visual_inertial_optimization).
            return

        R_wc, t_wc = self.vio.view_set.get_pose(frameID)

        R_bs = self.vio.T_BS[:3, :3]
        t_bs = self.vio.T_BS[:3, 3]

        R_wb, t_wb = camera_pose_to_body_pose(R_wc, t_wc, R_bs, t_bs)

        velocity = self.vio.sw_state.velocities.get(frameID)

        self.odometry_publisher.publish_odometry(
            timestamp,
            R_wb,
            t_wb,
            velocity=velocity,
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
        self._backend_running = False
        self._backend_thread.join(timeout=2.0)
        self.vio.feature_extractor.shutdown()
        self.odometry_publisher.destroy_node()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VisualOdometryNode()
    # MultiThreadedExecutor (not the default single-threaded spin()):
    # image and IMU messages arrive on separate subscriptions but were
    # both being dispatched by one executor thread. Under the old
    # single-threaded spin(), a slow image callback (vio_loop_frontend,
    # itself briefly serialized behind vio_loop_backend's Ceres solves
    # via the old shared state_lock) could hold the only executor thread
    # long enough that queued IMU callbacks simply couldn't run --
    # imu_buffer would silently stop growing for the duration, producing
    # the "future side" EMPTY SLICE / insufficient-IMU-coverage failures
    # seen in VIO_DEBUG logs even though IMU was arriving at a clean,
    # constant rate the whole time. A MultiThreadedExecutor lets the
    # image and IMU callbacks actually run concurrently, so
    # process_imu() (now guarded by its own imu_lock, decoupled from
    # backend's state_lock -- see VisualInertialOdometry.__init__) is
    # never starved by frontend/backend work.
    # 2 callback groups (image, IMU) need at most 2 concurrent threads to
    # both make progress; a couple of spares cover node/timer housekeeping
    # without over-subscribing threads that would just add scheduling
    # overhead for no benefit (this is bounding thread count, not doing
    # the actual serialization -- the callback groups above do that).
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down VIO node.")
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()