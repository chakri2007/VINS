import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge
from rclpy.time import Time


class VOFeatureVisualizer(Node):
    """Publishes an annotated image showing current feature detections and
    their motion tracks across the active sliding window.

    Topic: /visual_odometry/features

    Rendering convention
    --------------------
    • Green filled circle  — point seen in the current frame only (new detection).
    • Blue filled circle   — current position of a tracked point (seen in ≥2 frames).
    • Yellow lines         — trail of the track from oldest to newest observation.
    """

    def __init__(self):
        super().__init__('vo_feature_visualizer')
        self.bridge = CvBridge()

        self.image_pub = self.create_publisher(
            Image,
            '/visual_odometry/features',
            10,
        )

    def publish_feature_tracks(
        self,
        raw_frame: np.ndarray,
        timestamp: float,
        active_tracks: dict,
        K: np.ndarray,
        distortion_coeffs: np.ndarray,
    ) -> None:
        """Draw feature detections and tracks, then publish.

        Parameters
        ----------
        raw_frame      : HxW or HxWx3 uint8 image (mono or BGR).
        timestamp      : seconds (float) for the header stamp.
        active_tracks  : dict[point_id] -> list[(frame_idx, u, v)],
                         oldest entry first (produced by get_active_tracks()).
        K              : (3,3) camera intrinsic matrix.
        distortion_coeffs : distortion coefficient array.
        """
        # Convert mono to BGR for coloured annotations.
        if raw_frame.ndim == 2 or (raw_frame.ndim == 3 and raw_frame.shape[2] == 1):
            vis = cv2.cvtColor(raw_frame, cv2.COLOR_GRAY2BGR)
        else:
            vis = raw_frame.copy()

        # Undistort for clean visualisation (same K used throughout).
        vis = cv2.undistort(vis, K, distortion_coeffs)

        for feature_id, point_history in active_tracks.items():
            if len(point_history) == 0:
                continue

            # Newest observation is the last entry (get_active_tracks appends
            # in increasing view-id order).
            _, u_cur, v_cur = point_history[-1]
            current_pt = (int(round(u_cur)), int(round(v_cur)))

            if len(point_history) == 1:
                # New detection — green dot.
                cv2.circle(vis, current_pt, 4, (0, 255, 0), -1)
            else:
                # Tracked point — draw trail then blue dot on top.
                for i in range(len(point_history) - 1):
                    _, u1, v1 = point_history[i]
                    _, u2, v2 = point_history[i + 1]
                    pt_a = (int(round(u1)), int(round(v1)))
                    pt_b = (int(round(u2)), int(round(v2)))
                    cv2.line(vis, pt_a, pt_b, (0, 255, 255), 1)   # yellow trail

                cv2.circle(vis, current_pt, 4, (255, 0, 0), -1)   # blue current pos

        # Overlay a small status string so it's easy to see in RViz.
        n_tracked = sum(1 for ph in active_tracks.values() if len(ph) > 1)
        n_new     = len(active_tracks) - n_tracked
        cv2.putText(
            vis,
            f"tracked: {n_tracked}  new: {n_new}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        print("Visualizer tracks:", len(active_tracks))

        img_msg = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
        img_msg.header.stamp    = Time(seconds=timestamp).to_msg()
        img_msg.header.frame_id = 'camera_link'
        self.image_pub.publish(img_msg)