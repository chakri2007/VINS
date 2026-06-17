from rclpy.node import Node
from geometry_msgs.msg import PoseStamped


class VOPosePublisher(Node):

    def __init__(self):
        super().__init__('vo_pose_publisher')
        self.pose_pub = self.create_publisher(
            PoseStamped, '/vins/odometry_pose', 10
        )
        self.get_logger().info(
            "VOPosePublisher ready on /vins/odometry_pose"
        )

    def publish(self, pose_data, ros_stamp):
        msg = PoseStamped()
        msg.header.stamp    = ros_stamp
        msg.header.frame_id = 'map'

        msg.pose.position.x = pose_data.x
        msg.pose.position.y = pose_data.y
        msg.pose.position.z = pose_data.z

        msg.pose.orientation.x = pose_data.qx
        msg.pose.orientation.y = pose_data.qy
        msg.pose.orientation.z = pose_data.qz
        msg.pose.orientation.w = pose_data.qw

        self.pose_pub.publish(msg)