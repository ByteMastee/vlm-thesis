# File: launch/publish_tfs.launch.py
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dir = get_package_share_directory("vf_camera_tf_publisher")
    config_file_default = os.path.join(
        pkg_dir, "config", "camera_calibration_parameters.yaml"
    )
    tf_launch_file = os.path.join(pkg_dir, "launch", "tf.launch.py")  # ROS2 launch

    return LaunchDescription(
        [
            # --- Launch Arguments ---
            DeclareLaunchArgument(
                "config_file",
                default_value=config_file_default,
                description="Path to the YAML file containing camera extrinsics and intrinsics.",
            ),
            DeclareLaunchArgument(
                "publish_base_tf",
                default_value="true",
                description="Set to true to publish the base_link -> d435_link -> d435_color_optical_frame transforms.",
            ),
            DeclareLaunchArgument(
                "base_link_frame",
                default_value="d435_link",
                description="Parent frame for the first camera link frame defined in the YAML.",
            ),
            # --- Include the tf.launch.py ---
            IncludeLaunchDescription(PythonLaunchDescriptionSource(tf_launch_file)),
            # --- Nodes ---
            Node(
                package="vf_camera_tf_publisher",
                executable="publish_camera_tfs",  # <-- remove .py
                name="camera_tf_publisher",
                output="screen",
                parameters=[
                    {
                        "config_file": config_file_default,
                        "publish_base_to_d435_tf": True,
                        "base_link_frame": "d435_link",
                    }
                ],
            ),
            Node(
                package="vf_camera_tf_publisher",
                executable="publish_cam_info",  # <-- remove .py
                name="camera_info_publisher",
                output="screen",
                parameters=[{"config_file": config_file_default}],
            ),
            Node(
                package="vf_camera_tf_publisher",
                executable="calibrate_cams",  # <-- remove .py
                name="camera_calibrator",
                output="screen",
            ),
        ]
    )
