#!/usr/bin/env python3
"""
ROS 2 CameraInfo Publisher

Publishes static CameraInfo messages for all cameras defined in:
- config/camera_calibration_parameters.yaml

Supports runtime reload on calibration completion signal:
- /camera_calibration_complete (Bool)

This is a ROS2 conversion of the original ROS1 implementation.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy

import yaml
import os
import traceback
import cv2 as cv
import numpy as np

from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Bool

from ament_index_python.packages import get_package_share_directory


# -------------------------------
# Utility Functions
# -------------------------------


def parse_rostopic(rostopic_str):
    """
    Extract base camera name from ROS topic string.
    Example: /fisheye_front/image_raw -> fisheye_front
    """
    try:
        parts = rostopic_str.strip("/").split("/")
        if not parts or not parts[0]:
            raise ValueError("Rostopic is empty or invalid")
        return parts[0]
    except Exception as e:
        print(f"[WARN] Could not parse rostopic '{rostopic_str}': {e}")
        return None


def create_camera_info_msg(cam_data, base_name, stamp):
    """
    Creates a sensor_msgs/CameraInfo message from camera calibration data.
    """
    required_keys = [
        "resolution",
        "intrinsics",
        "distortion_model",
        "distortion_coeffs",
    ]
    if not all(k in cam_data for k in required_keys):
        print(f"[WARN] Skipping camera '{base_name}': Missing keys {required_keys}")
        return None, None

    try:
        info_msg = CameraInfo()
        info_msg.header.stamp = stamp
        info_msg.header.frame_id = f"{base_name}_optical_frame"

        # Resolution
        res = cam_data["resolution"]
        if len(res) != 2:
            raise ValueError(f"Invalid resolution: {res}")
        info_msg.width = int(res[0])
        info_msg.height = int(res[1])

        # Distortion model
        d_model = cam_data["distortion_model"]
        if d_model == "radtan":
            info_msg.distortion_model = "plumb_bob"
        elif d_model == "equidistant":
            info_msg.distortion_model = "equidistant"
        else:
            info_msg.distortion_model = d_model

        # Distortion coefficients
        coeffs = cam_data["distortion_coeffs"]
        info_msg.d = [float(c) for c in coeffs]

        # Intrinsics (K)
        intr = cam_data["intrinsics"]
        if len(intr) != 4:
            raise ValueError(f"Invalid intrinsics (expected [fx, fy, cx, cy]): {intr}")
        fx, fy, cx, cy = map(float, intr)
        info_msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]

        # Rectification matrix (R) - identity for unrectified
        info_msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

        # Projection matrix (P) - using OpenCV optimal camera matrix
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        D = np.array(info_msg.d)

        newcameramtx, roi = cv.getOptimalNewCameraMatrix(
            K,
            D,
            (info_msg.width, info_msg.height),
            0,
            (info_msg.width, info_msg.height),
        )
        info_msg.p = [
            float(newcameramtx[0, 0]),
            0.0,
            float(newcameramtx[0, 2]),
            0.0,
            0.0,
            float(newcameramtx[1, 1]),
            float(newcameramtx[1, 2]),
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ]

        # Binning/ROI defaults
        info_msg.binning_x = 0
        info_msg.binning_y = 0
        info_msg.roi.x_offset = 0
        info_msg.roi.y_offset = 0
        info_msg.roi.width = 0
        info_msg.roi.height = 0
        info_msg.roi.do_rectify = False

        # Topic name
        topic_name = f"/{base_name}/camera_info_vf1228"

        return info_msg, topic_name

    except Exception as e:
        print(f"[ERROR] Failed to create CameraInfo for {base_name}: {e}")
        traceback.print_exc()
        return None, None


# -------------------------------
# Main Node
# -------------------------------


class CameraInfoPublisher(Node):
    """
    ROS2 node that publishes CameraInfo messages for all cameras in YAML.
    Supports reload when calibration completes.
    """

    def __init__(self):
        super().__init__("camera_info_publisher")

        # -----------------------
        # Parameters
        # -----------------------
        self.declare_parameter("config_file", "")
        pkg_share = get_package_share_directory("vf_camera_tf_publisher")
        self.yaml_file_path = self.get_parameter(
            "config_file"
        ).get_parameter_value().string_value or os.path.join(
            pkg_share, "config", "camera_calibration_parameters.yaml"
        )

        self.get_logger().info(f"Using Camera YAML: {self.yaml_file_path}")

        # -----------------------
        # Publishers dict
        # -----------------------
        self.cam_publishers = {}

        # -----------------------
        # Load and publish CameraInfo messages
        # -----------------------
        self.load_and_publish()

        # -----------------------
        # Subscriber for calibration reload
        # -----------------------
        self.create_subscription(
            Bool, "/camera_calibration_complete", self._reload_callback, 10
        )

    def load_and_publish(self):
        """Load YAML and publish CameraInfo for all cameras."""
        try:
            with open(self.yaml_file_path, "r") as f:
                cam_data_all = yaml.safe_load(f)
        except Exception as e:
            self.get_logger().error(f"Failed to load YAML: {e}")
            return

        if not cam_data_all:
            self.get_logger().warn(
                "YAML file empty or invalid. No CameraInfo published."
            )
            return

        current_time = self.get_clock().now().to_msg()

        for cam_key, cam_info in cam_data_all.items():
            if not isinstance(cam_info, dict) or "rostopic" not in cam_info:
                self.get_logger().debug(f"Skipping invalid camera key '{cam_key}'")
                continue

            base_name = parse_rostopic(cam_info["rostopic"])
            if not base_name:
                self.get_logger().warn(
                    f"Could not determine base name for '{cam_key}'. Skipping."
                )
                continue

            info_msg, topic_name = create_camera_info_msg(
                cam_info, base_name, current_time
            )

            if info_msg and topic_name:
                # ROS2 latch replacement: transient local QoS
                if topic_name not in self.cam_publishers:
                    qos = QoSProfile(depth=1)
                    qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
                    self.cam_publishers[topic_name] = self.create_publisher(
                        CameraInfo, topic_name, qos
                    )
                self.cam_publishers[topic_name].publish(info_msg)
                self.get_logger().info(
                    f"Published CameraInfo for '{base_name}' on '{topic_name}'"
                )
            else:
                self.get_logger().warn(
                    f"Failed to generate CameraInfo for '{base_name}'"
                )

    def _reload_callback(self, msg):
        """Reload YAML and republish CameraInfo on calibration update."""
        if msg.data:
            self.get_logger().info(
                "Calibration update received. Reloading CameraInfo..."
            )
            self.load_and_publish()


def main():
    rclpy.init()
    try:
        node = CameraInfoPublisher()
        rclpy.spin(node)
    except Exception:
        traceback.print_exc()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
