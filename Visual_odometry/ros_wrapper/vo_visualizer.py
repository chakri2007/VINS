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
        
        self.image_pub = self.create_publisher(Image, '/visual_odometry/features', 10)

    def publish_feature_tracks(self, raw_frame, timestamp, active_tracks, K, distortion_coeffs):

        undistorted_image = cv2.undistort(raw_frame, K, distortion_coeffs)

        for feature_id, point_history in active_tracks.items():
            if len(point_history) == 0:
                continue

            current_pt = (int(point_history[0][0]), int(point_history[0][1]))
            if len(point_history) == 1:
                # BRAND NEW FEATURE: Draw a Bright Green circle
                color = (0, 255, 0) 
                cv2.circle(undistorted_image, current_pt, 4, color, -1)
            else:
                # OLD/MATURED FEATURE: Draw a Bright Blue circle
                color = (255, 0, 0)
                cv2.circle(undistorted_image, current_pt, 4, color, -1)
                
                # DRAW THE TRACE PATH (History Tail)
                for i in range(len(point_history) - 1):
                    pt_start = (int(point_history[i][0]), int(point_history[i][1]))
                    pt_end = (int(point_history[i+1][0]), int(point_history[i+1][1]))
                    
                    # Draw a fading track path line (Yellow-ish trailing color)
                    cv2.line(undistorted_image, pt_start, pt_end, (0, 255, 255), 1)

        img_msg = self.bridge.cv2_to_imgmsg(undistorted_image, encoding='bgr8')
        ros_time_obj = Time(seconds=timestamp).to_msg()
        img_msg.header.stamp = ros_time_obj
        img_msg.header.frame_id = "camera_link"
        
        self.image_pub.publish(img_msg)