import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from vo_subscriber import VisualOdometryNode

class VisualOdometryPublisherNode(Node):

    def __init__(self, subscriber_node: VisualOdometryNode):
        super().__init__('vo_publisher_node')
        self._sub_node = subscriber_node
        
        self.pose_pub = self.create_publisher(PoseStamped, '/vins/odometry_pose', 10)
        self.get_logger().info("VO Pose Publisher initialized on topic: /vins/odometry_pose")
        
        self._sub_node.register_pose_output_callback(self.publish_metric_pose)

    def publish_metric_pose(self, pose_data, ros_stamp):
        msg = PoseStamped()
        msg.header.stamp = ros_stamp
        msg.header.frame_id = 'map'

        msg.pose.position.x = pose_data.x
        msg.pose.position.y = pose_data.y
        msg.pose.position.z = pose_data.z

        msg.pose.orientation.x = pose_data.qx
        msg.pose.orientation.y = pose_data.qy
        msg.pose.orientation.z = pose_data.qz
        msg.pose.orientation.w = pose_data.qw

        self.pose_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    
    sub_node = VisualOdometryNode()
    
    pub_node = VisualOdometryPublisherNode(subscriber_node=sub_node)
    
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(sub_node)
    executor.add_node(pub_node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        sub_node.destroy_node()
        pub_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()