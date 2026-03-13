#!/usr/bin/env python3
"""
ROS 2 Static Camera TF Publisher

Publishes static TF transforms for:
- base_link -> d435_link -> d435_color_optical_frame
- chained fisheye cameras using calibration YAML

This is a ROS 2 conversion of the original ROS 1 implementation.
"""

import rclpy
from rclpy.node import Node

import yaml
import os
import traceback

import numpy as np

from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Bool

from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
import tf_transformations as tf_trans

from ament_index_python.packages import get_package_share_directory


class CameraTFPublisher(Node):
    """
    ROS 2 node that publishes STATIC transforms for a camera chain.
    """

    # ===============================
    # Fixed Optical ↔ Link transforms
    # ===============================

    # Optical → Link rotation matrix (camera optical frame → ROS link frame)
    R_LINK_OPTICAL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])

    T_LINK_OPTICAL = np.identity(4)
    T_LINK_OPTICAL[0:3, 0:3] = R_LINK_OPTICAL

    R_OPTICAL_LINK = R_LINK_OPTICAL.T
    T_OPTICAL_LINK = np.identity(4)
    T_OPTICAL_LINK[0:3, 0:3] = R_OPTICAL_LINK

    # ===============================

    def __init__(self):
        super().__init__("camera_tf_publisher")

        # Broadcaster for static TFs
        self.static_broadcaster = StaticTransformBroadcaster(self)
        self.transforms_to_publish = []

        # Load parameters and YAML paths
        self._declare_and_load_parameters()

        # Subscribe to calibration trigger topic
        self.create_subscription(
            Bool, "/camera_calibration_complete", self._restart_callback, 10
        )

        self.get_logger().info("Camera TF Publisher initialized")

    # ----------------------------------------------------
    # Parameter handling
    # ----------------------------------------------------
    def _declare_and_load_parameters(self):

        # Declare launch/parameter arguments
        self.declare_parameter("config_file", "")
        self.declare_parameter("d435_config_file", "")
        self.declare_parameter("base_link_frame", "d435_link")
        self.declare_parameter("publish_base_to_d435_tf", True)

        pkg_share = get_package_share_directory("vf_camera_tf_publisher")

        # YAML paths (allow overrides via parameters)
        self.yaml_file_path = self.get_parameter(
            "config_file"
        ).get_parameter_value().string_value or os.path.join(
            pkg_share, "config", "camera_calibration_parameters.yaml"
        )

        self.d435_yaml_file_path = self.get_parameter(
            "d435_config_file"
        ).get_parameter_value().string_value or os.path.join(
            pkg_share, "config", "d435_extrinsics.yaml"
        )

        # Camera chain parameters
        self.cam_chain_base_frame_id = self.get_parameter("base_link_frame").value
        self.publish_base_tf = self.get_parameter("publish_base_to_d435_tf").value

        self.get_logger().info(f"Camera YAML: {self.yaml_file_path}")
        self.get_logger().info(f"D435 YAML: {self.d435_yaml_file_path}")

    # ----------------------------------------------------
    # Convert 4x4 matrix → TransformStamped
    # ----------------------------------------------------
    @staticmethod
    def matrix_to_tf(matrix, parent, child, node_clock):
        """
        Convert 4x4 numpy matrix → geometry_msgs/TransformStamped
        """
        t = TransformStamped()
        # Optional: Use fixed timestamp (0) for bag recordings
        t.header.stamp = node_clock.now().to_msg()
        t.header.frame_id = parent
        t.child_frame_id = child

        trans = tf_trans.translation_from_matrix(matrix)
        quat = tf_trans.quaternion_from_matrix(matrix)

        # Normalize quaternion
        norm = np.linalg.norm(quat)
        if norm > 1e-9:
            quat = quat / norm
        else:
            quat = [0.0, 0.0, 0.0, 1.0]

        t.transform.translation.x = float(trans[0])
        t.transform.translation.y = float(trans[1])
        t.transform.translation.z = float(trans[2])

        t.transform.rotation.x = float(quat[0])
        t.transform.rotation.y = float(quat[1])
        t.transform.rotation.z = float(quat[2])
        t.transform.rotation.w = float(quat[3])

        return t

    # ----------------------------------------------------
    # Calibration reload trigger
    # ----------------------------------------------------
    def _restart_callback(self, msg):
        if msg.data:
            self.get_logger().info("Reloading TFs after calibration update")
            self._reload_and_publish()

    # ----------------------------------------------------
    # Core logic
    # ----------------------------------------------------
    def _reload_and_publish(self):
        """Reload YAML files and publish all static transforms"""
        self.transforms_to_publish.clear()
        self._load_base_transforms()
        self._load_camera_chain()
        self._publish_transforms()

    # ----------------------------------------------------
    # Load D435 base transforms
    # ----------------------------------------------------
    def _load_base_transforms(self):

        if not self.publish_base_tf:
            return

        try:
            with open(self.d435_yaml_file_path, "r") as f:
                data = yaml.safe_load(f)

            # Use improved default pitch (~1 rad) if missing
            roll = data.get("d435_roll_angle", 0.0)
            pitch = data.get("d435_pitch_angle", 1.047)  # ~60 deg
            yaw = data.get("d435_yaw_angle", 0.0)

            t_mat = tf_trans.compose_matrix(
                translate=[
                    data.get("d435_x_distance", 0.03),
                    data.get("d435_y_distance", 0.0),
                    data.get("d435_z_distance", 1.8),
                ],
                angles=[roll, pitch, yaw],
            )

            # base_link -> d435_link
            self.transforms_to_publish.append(
                self.matrix_to_tf(t_mat, "base_link", "d435_link", self.get_clock())
            )
            # d435_link -> d435_color_optical_frame
            self.transforms_to_publish.append(
                self.matrix_to_tf(
                    self.T_LINK_OPTICAL,
                    "d435_link",
                    "d435_color_optical_frame",
                    self.get_clock(),
                )
            )

        except Exception as e:
            self.get_logger().error(f"D435 TF load failed: {e}")

    # ----------------------------------------------------
    # Load fisheye camera chain
    # ----------------------------------------------------
    def _load_camera_chain(self):

        try:
            with open(self.yaml_file_path, "r") as f:
                cam_data = yaml.safe_load(f)
        except Exception as e:
            self.get_logger().error(f"Camera YAML error: {e}")
            return

        parent_frame = self.cam_chain_base_frame_id

        camera_keys = sorted(
            k for k in cam_data.keys() if k.startswith("cam") and k != "cam0"
        )

        for cam_key in camera_keys:
            cam = cam_data[cam_key]

            # ----- Safe YAML access -----
            if "T_cn_cnm1" not in cam:
                self.get_logger().warn(f"{cam_key} missing T_cn_cnm1, skipping")
                continue
            if "rostopic" not in cam:
                self.get_logger().warn(f"{cam_key} missing 'rostopic', skipping")
                continue

            rostopic = cam.get("rostopic", "").strip("/")
            if not rostopic:
                self.get_logger().warn(f"{cam_key} has empty rostopic, skipping")
                continue

            # Extract base name safely
            base = rostopic.split("/")[0]
            if not base:
                self.get_logger().warn(f"{cam_key} rostopic invalid, skipping")
                continue

            link_frame = f"{base}_link"
            optical_frame = f"{base}_optical_frame"

            T_yaml = np.array(cam["T_cn_cnm1"])
            # Compute link-to-link transform
            T_link = self.T_LINK_OPTICAL @ np.linalg.inv(T_yaml) @ self.T_OPTICAL_LINK

            # Add parent -> current_link
            self.transforms_to_publish.append(
                self.matrix_to_tf(T_link, parent_frame, link_frame, self.get_clock())
            )
            # Add current_link -> optical_frame
            self.transforms_to_publish.append(
                self.matrix_to_tf(
                    self.T_LINK_OPTICAL, link_frame, optical_frame, self.get_clock()
                )
            )

            parent_frame = link_frame  # Next camera in chain

    # ----------------------------------------------------
    # Publish all static transforms
    # ----------------------------------------------------
    def _publish_transforms(self):
        if self.transforms_to_publish:
            self.static_broadcaster.sendTransform(self.transforms_to_publish)
            self.get_logger().info(
                f"Published {len(self.transforms_to_publish)} static TFs"
            )
        else:
            self.get_logger().warn("No transforms to publish")

    # ----------------------------------------------------
    # Run node
    # ----------------------------------------------------
    def run(self):
        self._reload_and_publish()  # Publish once at startup
        rclpy.spin(self)


# ======================================================
def main():
    rclpy.init()
    try:
        node = CameraTFPublisher()
        node.run()
    except Exception:
        traceback.print_exc()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
