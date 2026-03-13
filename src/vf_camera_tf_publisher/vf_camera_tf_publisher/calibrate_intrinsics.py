#!/usr/bin/env python3

"""
ROS2 version of ROS1 intrinsic camera calibration service.
- Uses fisheye calibration via OpenCV
- Publishes annotated checkerboard frames
- Updates YAML camera config with new intrinsics and distortion
- Compatible with multiple cameras defined in YAML
"""

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

import os
import yaml
import numpy as np
import cv2

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Bool

from vfmessages.srv import StartCalibrationIntrinsics


# Global state to prevent concurrent calibrations
is_calibrating = False

# ------------------------------
# Default YAML configuration path (soft-coded)
# ------------------------------
from ament_index_python.packages import get_package_share_directory

CONFIG_FILE_DEFAULT = os.path.join(
    get_package_share_directory("vf_camera_tf_publisher"),
    "config",
    "camera_calibration_parameters.yaml",
)


# ------------------------------
# Utility: Load YAML config
# ------------------------------
def load_yaml_config(path: str):
    """Safely load YAML config"""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        return None


# ------------------------------
# Main Calibration Node
# ------------------------------
class IntrinsicCalibrator(Node):
    """ROS2 Node implementing intrinsic calibration as a service"""

    def __init__(self):
        super().__init__("intrinsic_calibrator_service")

        self.bridge = CvBridge()

        # Create service for calibration
        self.srv = self.create_service(
            StartCalibrationIntrinsics,
            "start_calibration",
            self.handle_start_calibration,
        )

        self.get_logger().info("Intrinsic calibration service ready.")

    # ------------------------------
    # Service callback
    # ------------------------------
    def handle_start_calibration(self, request, response):
        global is_calibrating

        if is_calibrating:
            self.get_logger().warn("Calibration already in progress.")
            response.success = False
            response.message = "Calibration already in progress."
            return response

        is_calibrating = True
        camera_id = request.camera_id
        self.get_logger().info(
            f"Starting intrinsic calibration for camera: {camera_id}"
        )

        # ------------------------------
        # Load YAML config
        # ------------------------------
        config = load_yaml_config(CONFIG_FILE_DEFAULT)
        if config is None or camera_id not in config:
            is_calibrating = False
            response.success = False
            response.message = f"Camera ID '{camera_id}' not found in YAML."
            return response

        cam_data = config[camera_id]
        rostopic = cam_data.get("rostopic", None)
        if not rostopic:
            is_calibrating = False
            response.success = False
            response.message = "Camera topic not found in YAML."
            return response

        # ------------------------------
        # Calibration setup
        # ------------------------------
        calib_state = {
            "last_frame": None,
            "objpoints": [],
            "imgpoints": [],
            "last_corners": None,
            "last_ret": False,
            "do_calibration": False,
        }

        CHECKERBOARD = (6, 5)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(
            -1, 2
        )

        # Publisher for annotated images
        viz_pub = self.create_publisher(
            Image, "/camera_calibration/intrinsics/image", 1
        )

        # ------------------------------
        # Image callback
        # ------------------------------
        def image_callback(msg):
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                calib_state["last_frame"] = cv_image.copy()
                gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

                ret, corners = cv2.findChessboardCorners(
                    gray,
                    CHECKERBOARD,
                    cv2.CALIB_CB_ADAPTIVE_THRESH
                    + cv2.CALIB_CB_FAST_CHECK
                    + cv2.CALIB_CB_NORMALIZE_IMAGE,
                )

                if ret:
                    corners_refined = cv2.cornerSubPix(
                        gray, corners, (11, 11), (-1, -1), criteria
                    )
                    calib_state["last_corners"] = corners_refined
                    calib_state["last_ret"] = True
                    cv2.drawChessboardCorners(
                        cv_image, CHECKERBOARD, corners_refined, ret
                    )
                else:
                    calib_state["last_ret"] = False

                n_views = len(calib_state["objpoints"])
                feedback_text = (
                    f"Press 'c' to capture ({n_views} views). Press 'q' to calibrate."
                )
                cv2.putText(
                    cv_image,
                    feedback_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

                # Publish annotated image
                viz_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
                viz_pub.publish(viz_msg)

            except CvBridgeError as e:
                self.get_logger().error(f"cv_bridge error: {e}")

        # ------------------------------
        # Subscribe to camera topic
        # ------------------------------
        sub_img = self.create_subscription(Image, rostopic, image_callback, 1)
        self.get_logger().info(f"Subscribed to {rostopic} for calibration images")

        # ------------------------------
        # Subscribers for user commands
        # ------------------------------
        def capture_callback(msg):
            if calib_state["last_ret"]:
                calib_state["objpoints"].append(objp)
                calib_state["imgpoints"].append(calib_state["last_corners"])
                self.get_logger().info(
                    f"Captured points. Total views: {len(calib_state['objpoints'])}"
                )
            else:
                self.get_logger().warn(
                    "No valid checkerboard detected. Cannot capture."
                )

        def calibrate_callback(msg):
            calib_state["do_calibration"] = True

        capture_sub = self.create_subscription(
            Bool, "/camera_calibration/capture", capture_callback, 1
        )
        calibrate_sub = self.create_subscription(
            Bool, "/camera_calibration/calibrate", calibrate_callback, 1
        )

        self.get_logger().info(
            "Listening for capture and calibrate commands on ROS2 topics"
        )

        # ------------------------------
        # Wait for calibration command
        # ------------------------------
        while rclpy.ok() and not calib_state["do_calibration"]:
            rclpy.spin_once(self, timeout_sec=0.1)

        # Unsubscribe
        sub_img.destroy()
        capture_sub.destroy()
        calibrate_sub.destroy()

        if not rclpy.ok():
            is_calibrating = False
            response.success = False
            response.message = "ROS shutdown during calibration."
            return response

        # ------------------------------
        # Perform calibration
        # ------------------------------
        if len(calib_state["objpoints"]) < 5:
            is_calibrating = False
            response.success = False
            response.message = "Not enough views captured."
            return response

        gray = cv2.cvtColor(calib_state["last_frame"], cv2.COLOR_BGR2GRAY)
        N_OK = len(calib_state["objpoints"])
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]
        calib_flags = (
            cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
            | cv2.fisheye.CALIB_CHECK_COND
            | cv2.fisheye.CALIB_FIX_SKEW
        )

        try:
            rms, _, _, _, _ = cv2.fisheye.calibrate(
                calib_state["objpoints"],
                calib_state["imgpoints"],
                gray.shape[::-1],
                K,
                D,
                rvecs,
                tvecs,
                calib_flags,
                criteria,
            )

            # Update YAML
            intrinsics_yaml = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
            distortion_yaml = D.flatten().tolist()

            config[camera_id]["intrinsics"] = [float(i) for i in intrinsics_yaml]
            config[camera_id]["distortion_coeffs"] = [float(d) for d in distortion_yaml]

            with open(CONFIG_FILE_DEFAULT, "w") as f:
                yaml.dump(config, f, default_flow_style=None)

            # Notify completion
            completion_pub = self.create_publisher(
                Bool, "/camera_calibration_complete", 1
            )
            completion_pub.publish(Bool(data=True))

            # Save verification image
            undistorted_img = cv2.fisheye.undistortImage(
                calib_state["last_frame"], K, D, Knew=K
            )
            h, w = calib_state["last_frame"].shape[:2]
            comparison_img = np.concatenate(
                (calib_state["last_frame"], undistorted_img), axis=1
            )
            cv2.putText(
                comparison_img,
                "Original",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                comparison_img,
                "Undistorted",
                (w + 10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.imwrite(request.output_file, comparison_img)

            message = f"Calibration successful with RMS error {rms:.4f}. YAML updated."
            is_calibrating = False
            response.success = True
            response.message = message
            self.get_logger().info(message)
            return response

        except Exception as e:
            is_calibrating = False
            response.success = False
            response.message = f"Calibration calculation failed: {e}"
            self.get_logger().error(f"Calibration failed: {e}")
            return response


# ------------------------------
# Main entrypoint
# ------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = IntrinsicCalibrator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
