# File: launch/tf.launch.py
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dir = get_package_share_directory("vf_camera_tf_publisher")

    # --- Launch Arguments ---
    return LaunchDescription(
        [
            DeclareLaunchArgument("t265_x_distance", default_value="0.55"),
            DeclareLaunchArgument("t265_y_distance", default_value="0.0"),
            DeclareLaunchArgument("t265_z_distance", default_value="0.5"),
            DeclareLaunchArgument("d435_x_distance", default_value="0.03"),
            DeclareLaunchArgument("d435_y_distance", default_value="0.0"),
            DeclareLaunchArgument(
                "d435_z_distance", default_value="0.0"
            ),  # default, can override with YAML
            DeclareLaunchArgument(
                "d435_pitch_angle", default_value="1.0470829130124137"
            ),
            DeclareLaunchArgument("d435_roll_angle", default_value="0.0"),
            # Static transform: base_link -> front_right_usound_link
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="base_link_to_front_right_usound",
                arguments=[
                    "0.0",
                    "-0.195",
                    "0.155",
                    "0",
                    "0",
                    "0",
                    "base_link",
                    "front_right_usound_link",
                ],
            ),
            # Static transform: base_link -> T265
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="base_link_to_t265",
                arguments=[
                    "0.55",
                    "0.0",
                    "0.5",
                    "3.141592654",
                    "0",
                    "0",
                    "base_link",
                    "t265_link",
                ],
            ),
            # Static transform: base_link -> d435
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="base_link_to_d435",
                arguments=[
                    "0.03",
                    "0.0",
                    "0.0",
                    "0",
                    "1.0470829130124137",
                    "0",
                    "base_link",
                    "d435_link",
                ],
            ),
        ]
    )
