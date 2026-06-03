import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import yaml
import cv2
from cv_bridge import CvBridge

# Import your core pipeline (which remains independent of ROS2)
from .core.pipeline import VisualOdometryPipeline

class VisualOdometryNode(Node):
    def __init__(self):
        super().__init__('vo_subscriber_node')
        self.bridge = CvBridge()
        config_path = "config/ros_config.yaml" 
        with open(config_path, 'r') as file:
            self.ros_config = yaml.safe_load(file)
            
        self.mode = self.ros_config['vo_mode']
        self.get_logger().info(f"Initializing Visual Odometry Node in [{self.mode.upper()}] mode.")
        calibration_data = self.load_calibration_files()
        self.vo_pipeline = VisualOdometryPipeline(calibration_data, mode=self.mode)

        if self.mode == "mono":
            self.image_sub = self.create_subscription(
                Image, 
                self.ros_config['left_camera_topic'], 
                self.mono_image_callback, 
                10
            )
            self.get_logger().info(f"Subscribed to mono topic: {self.ros_config['left_camera_topic']}")
            
        elif self.mode == "stereo":
            from message_filters import Subscriber, ApproximateTimeSynchronizer
            
            self.left_sub = Subscriber(self, Image, self.ros_config['left_camera_topic'])
            self.right_sub = Subscriber(self, Image, self.ros_config['right_camera_topic'])
            
            self.ats = ApproximateTimeSynchronizer([self.left_sub, self.right_sub], queue_size=10, slop=0.02)
            self.ats.registerCallback(self.stereo_image_callback)
            self.get_logger().info("Subscribed to synchronized stereo topics.")

    def load_calibration_files(self):
        calib = {}
        
        with open(self.ros_config['left_camera_config_path'], 'r') as f:
            calib['left'] = yaml.safe_load(f)
            
        if self.mode == "stereo":
            with open(self.ros_config['right_camera_config_path'], 'r') as f:
                calib['right'] = yaml.safe_load(f)
                
        return calib

    def mono_image_callback(self, msg):
        timestamp = msg.header.stamp.sec + (msg.header.stamp.nanosec * 1e-9)
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        tracks, K, D = self.vo_pipeline.process_frame_mono(cv_image,timestamp)
        self.visualizer.publish_feature_tracks(cv_image, timestamp, tracks, K, D)

    def stereo_image_callback(self, left_msg, right_msg):
        pass
def main(args=None):
    rclpy.init(args=args)
    node = VisualOdometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down vo_subscriber_node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()