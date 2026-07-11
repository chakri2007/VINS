import rclpy
from rclpy.node import Node
from rclpy.time import Time
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
import numpy as np
from tf2_ros import TransformBroadcaster


class VIOOdometryPublisher(Node):
    """Publishes the current VIO pose/velocity estimate as a
    nav_msgs/Odometry message (and the corresponding TF transform).

    Topic: /vio/odometry

    Pose convention
    ---------------
    This publishes whatever (R, t) it's given as the world_frame_id ->
    body_frame_id transform, as-is -- it does NOT know about or apply
    the camera<->body (T_BS) extrinsic itself. The caller is
    responsible for passing the BODY pose (R_wb, t_wb), already
    converted from ViewSet's camera-to-world (R_wc, t_wc) via
    imu.vi_alignment.camera_pose_to_body_pose(R_wc, t_wc, R_bs, t_bs) --
    see vio_subscriber.py's call site. Publishing the raw camera pose
    here under the 'vio_body' child frame would be silently wrong for
    any consumer expecting IMU/body-frame odometry.

    velocity, if given, is also expected in the WORLD frame (same
    convention as sw_state.velocities) -- publish_odometry rotates it
    into the body frame internally using the same R passed for
    orientation, so it must be the R_wb used for that conversion, not
    R_wc.
    """

    # Fixed diagonal covariance placeholders (position: m^2, orientation:
    # rad^2, linear velocity: (m/s)^2). VIO doesn't currently track a
    # real per-pose uncertainty estimate anywhere upstream (Ceres'
    # Jacobian isn't retained after solve), so these are NOT derived
    # from anything -- they exist only so downstream consumers (e.g.
    # robot_localization) don't misread an all-zero covariance as
    # "perfectly known", which is arguably worse than a rough guess.
    # Tune to the actual sensor/pipeline if precise fusion matters.
    _POSITION_VARIANCE    = 0.05    # m^2
    _ORIENTATION_VARIANCE = 0.02    # rad^2
    _VELOCITY_VARIANCE    = 0.10    # (m/s)^2

    def __init__(self):
        super().__init__('vio_odometry_publisher')

        self.odom_pub = self.create_publisher(
            Odometry,
            '/vio/odometry',
            10,
        )

        self.tf_broadcaster = TransformBroadcaster(self)

        self.world_frame_id = 'world'
        self.body_frame_id  = 'vio_body'

    def publish_odometry(
        self,
        timestamp: float,
        R: np.ndarray,
        t: np.ndarray,
        velocity: np.ndarray = None,
    ) -> None:
        """Build and publish an Odometry message for the latest pose.

        Parameters
        ----------
        timestamp : seconds (float), same convention as vio_visualizer.
        R          : (3,3) BODY-to-world rotation (R_wb) -- already
                     converted from ViewSet's camera-to-world pose via
                     camera_pose_to_body_pose; see class docstring.
        t          : (3,) body-to-world translation (t_wb, body origin
                     in world coordinates), same conversion as R above.
        velocity   : (3,) world-frame velocity, optional. If None (e.g.
                     the frame never got a velocity estimate committed),
                     the twist fields are left zeroed and its covariance
                     is marked unknown rather than publishing a
                     confident-looking stale/zero value.
        """

        stamp = Time(seconds=timestamp).to_msg()

        qw, qx, qy, qz = self._rotation_to_quaternion(R)

        odom_msg = Odometry()
        odom_msg.header.stamp    = stamp
        odom_msg.header.frame_id = self.world_frame_id
        odom_msg.child_frame_id  = self.body_frame_id

        odom_msg.pose.pose.position.x = float(t[0])
        odom_msg.pose.pose.position.y = float(t[1])
        odom_msg.pose.pose.position.z = float(t[2])

        odom_msg.pose.pose.orientation.w = qw
        odom_msg.pose.pose.orientation.x = qx
        odom_msg.pose.pose.orientation.y = qy
        odom_msg.pose.pose.orientation.z = qz

        # Row-major 6x6 (x,y,z,rot_x,rot_y,rot_z), diagonal only --
        # see class docstring for why this isn't literally zero.
        pose_cov = [0.0] * 36
        for i in range(3):
            pose_cov[i * 6 + i] = self._POSITION_VARIANCE
        for i in range(3, 6):
            pose_cov[i * 6 + i] = self._ORIENTATION_VARIANCE
        odom_msg.pose.covariance = pose_cov

        if velocity is not None:
            # World-frame velocity; Odometry.twist is body-frame by
            # convention, so rotate into the body frame.
            v_body = R.T @ np.asarray(velocity, dtype=float)
            odom_msg.twist.twist.linear.x = float(v_body[0])
            odom_msg.twist.twist.linear.y = float(v_body[1])
            odom_msg.twist.twist.linear.z = float(v_body[2])

            twist_cov = [0.0] * 36
            for i in range(3):
                twist_cov[i * 6 + i] = self._VELOCITY_VARIANCE
            # Angular velocity isn't tracked as a per-frame state
            # anywhere upstream (no gyro-bias-corrected rate is stored
            # per keyframe), so rows/cols 3:6 are left at the
            # uninformative default rather than fabricating a number.
            odom_msg.twist.covariance = twist_cov
        else:
            # No velocity for this frame -- mark the twist covariance
            # as "unknown" (large) rather than implying a confident
            # zero velocity, since the linear fields above are also
            # left at their zeroed default in this branch.
            unknown_cov = [0.0] * 36
            for i in range(6):
                unknown_cov[i * 6 + i] = 1e6
            odom_msg.twist.covariance = unknown_cov

        self.odom_pub.publish(odom_msg)

        # ── TF ──────────────────────────────────────────────────────
        tf_msg = TransformStamped()
        tf_msg.header.stamp    = stamp
        tf_msg.header.frame_id = self.world_frame_id
        tf_msg.child_frame_id  = self.body_frame_id

        tf_msg.transform.translation.x = float(t[0])
        tf_msg.transform.translation.y = float(t[1])
        tf_msg.transform.translation.z = float(t[2])

        tf_msg.transform.rotation.w = qw
        tf_msg.transform.rotation.x = qx
        tf_msg.transform.rotation.y = qy
        tf_msg.transform.rotation.z = qz

        self.tf_broadcaster.sendTransform(tf_msg)

    @staticmethod
    def _rotation_to_quaternion(R: np.ndarray):
        """Standard R (3x3) -> (w, x, y, z) quaternion conversion."""

        R = np.asarray(R, dtype=float)
        trace = np.trace(R)

        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2
            qw = 0.25 * s
            qx = (R[2, 1] - R[1, 2]) / s
            qy = (R[0, 2] - R[2, 0]) / s
            qz = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s

        return float(qw), float(qx), float(qy), float(qz)