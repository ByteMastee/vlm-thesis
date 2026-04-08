#!/usr/bin/env python3
# =============================================================================
#  bringup_launch_amcl_d435i.py  —  AMCL + D435i pointcloud_to_laserscan
#
#  D435i is tilted 60° down, so depthimage_to_laserscan won't work.
#  Instead we use pointcloud_to_laserscan which extracts a HORIZONTAL
#  slice from the 3D point cloud at a chosen height range — independent
#  of camera tilt. The point cloud is transformed to base_link first.
#
#  INSTALL: sudo apt install ros-humble-pointcloud-to-laserscan
#
#  USAGE:
#    Terminal 1: ros2 launch uvc1_gazebo uvc1_my_world_nav2_rtabmap.launch.py
#    Terminal 2: ros2 launch uvc1_gazebo bringup_launch_amcl_d435i.py map_name:=house1
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

    # D435i PointCloud2 → horizontal /scan
    # target_frame: base_link transforms the cloud to robot frame first,
    # then min_height/max_height slices horizontally regardless of camera tilt.
    # Height 0.2–0.8m captures walls and obstacles, ignores floor and ceiling.
    pc_to_scan = Node(
        package="pointcloud_to_laserscan",
        executable="pointcloud_to_laserscan_node",
        name="pointcloud_to_laserscan",
        output="screen",
        parameters=[{
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "target_frame": "base_link",
            "transform_tolerance": 0.1,
            "min_height": 0.2,
            "max_height": 0.8,
            "angle_min": -1.5708,   # -90°
            "angle_max": 1.5708,    # +90°
            "angle_increment": 0.0087,  # ~0.5°
            "scan_time": 0.033,
            "range_min": 0.3,
            "range_max": 4.0,
            "use_inf": True,
            "inf_epsilon": 1.0,
        }],
        remappings=[
            ("cloud_in", "/d435i/depth/d435i_depth/points"),
            ("scan", "/scan"),
        ],
    )

    ld = LaunchDescription()
    ld.add_action(declare_use_sim_time)
    ld.add_action(declare_map_name)
    ld.add_action(pc_to_scan)
    ld.add_action(OpaqueFunction(function=setup_map_path))
    return ld
