#!/usr/bin/env python3
# =============================================================================
#  bringup_launch_amcl_d455.py  —  AMCL + D455 depthimage_to_laserscan
#
#  D455 is rear-facing with ZERO pitch → horizontal scan plane → clean AMCL
#  Simplest option. ~87° FOV backward.
#
#  INSTALL: sudo apt install ros-humble-depthimage-to-laserscan
#
#  USAGE:
#    Terminal 1: ros2 launch uvc1_gazebo uvc1_my_world_nav2_rtabmap.launch.py
#    Terminal 2: ros2 launch uvc1_gazebo bringup_launch_amcl_d455.py map_name:=house1
# =============================================================================

import os
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, IncludeLaunchDescription,
    OpaqueFunction, TimerAction, LogInfo,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def setup_map_path(context, *args, **kwargs):
    pkg = get_package_share_directory("uvc1_gazebo")
    map_name = LaunchConfiguration("map_name").perform(context)
    use_sim_time = LaunchConfiguration("use_sim_time")
    map_dir = Path(pkg) / "maps" / map_name
    if not map_dir.exists():
        raise RuntimeError(f"Map folder not found: {map_dir}")
    yaml_path = str(map_dir / f"{map_name}.yaml")
    params_file = os.path.join(pkg, "config", "nav2_params_amcl.yaml")
    pkg_nav2 = get_package_share_directory("nav2_bringup")

    return [TimerAction(period=10.0, actions=[
        LogInfo(msg=["Launching Nav2 with map: ", yaml_path]),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_nav2, "launch", "bringup_launch.py")),
            launch_arguments={
                "use_sim_time": use_sim_time,
                "map": yaml_path,
                "params_file": params_file,
                "autostart": "true",
            }.items(),
        ),
    ])]


def generate_launch_description():
    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time", default_value="true")
    declare_map_name = DeclareLaunchArgument(
        "map_name", default_value="house1")

    # Static TF: bridge Gazebo's "camera_depth_frame" → URDF's D455 optical frame
    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_tf_camera_depth_frame",
        arguments=[
            "--x", "0", "--y", "0", "--z", "0",
            "--roll", "0", "--pitch", "0", "--yaw", "0",
            "--frame-id", "camera_d455_depth_optical_frame",
            "--child-frame-id", "camera_depth_frame",
        ],
        parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
    )

    # D455 depth image → /scan
    depth_to_scan = Node(
        package="depthimage_to_laserscan",
        executable="depthimage_to_laserscan_node",
        name="depthimage_to_laserscan",
        output="screen",
        parameters=[{
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "scan_height": 10,
            "scan_time": 0.033,
            "range_min": 0.6,
            "range_max": 6.0,
            "output_frame_id": "camera_d455_depth_optical_frame",
        }],
        remappings=[
            ("depth", "/d455/depth/d455_depth/depth/image_raw"),
            ("depth_camera_info", "/d455/depth/d455_depth/depth/camera_info"),
            ("scan", "/scan"),
        ],
    )

    ld = LaunchDescription()
    ld.add_action(declare_use_sim_time)
    ld.add_action(declare_map_name)
    ld.add_action(static_tf)
    ld.add_action(depth_to_scan)
    ld.add_action(OpaqueFunction(function=setup_map_path))
    return ld
