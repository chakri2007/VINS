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
    ViewSet stores (R, t) as the camera-to-world transform, i.e. `t` is
    already the camera center in world coordinates and `R` rotates
    camera axes into world axes. That maps directly onto Odometry's
    pose.pose (position = t, orientation = quaternion(R)) with no
    inversion needed.
    """

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
        R          : (3,3) camera-to-world rotation.
        t          : (3,) camera-to-world translation (position).
        velocity   : (3,) world-frame velocity, optional. If None (e.g.
                     BA_motion was skipped for this frame), the twist
                     fields are left zeroed rather than publishing a
                     stale value.
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

        if velocity is not None:
            # World-frame velocity; Odometry.twist is body-frame by
            # convention, so rotate into the body frame.
            v_body = R.T @ np.asarray(velocity, dtype=float)
            odom_msg.twist.twist.linear.x = float(v_body[0])
            odom_msg.twist.twist.linear.y = float(v_body[1])
            odom_msg.twist.twist.linear.z = float(v_body[2])

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