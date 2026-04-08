#!/usr/bin/env python3
# =============================================================================
#  bringup_launch_amcl_dual.py  —  AMCL + BOTH cameras merged into /scan
#
#  Uses pointcloud_to_laserscan on BOTH D455 and D435i point clouds,
#  producing /scan_d455 and /scan_d435i, then merges them with
#  ira_laser_tools into a single /scan topic.
#
#  This gives the widest FOV coverage:
#    D455:  rear-facing ~87° horizontal
#    D435i: forward/down-facing ~87° (point cloud slice is horizontal)
#  Combined: ~170°+ coverage depending on overlap
#
#  INSTALL:
#    sudo apt install ros-humble-pointcloud-to-laserscan
#    sudo apt install ros-humble-ira-laser-tools
#    # If ira-laser-tools not available via apt:
#    cd ~/vf_robot_model_ros2/src
#    git clone https://github.com/iralabdisco/ira_laser_tools.git -b humble
#    cd .. && colcon build --packages-select ira_laser_tools
#
#  USAGE:
#    Terminal 1: ros2 launch uvc1_gazebo uvc1_my_world_nav2_rtabmap.launch.py
#    Terminal 2: ros2 launch uvc1_gazebo bringup_launch_amcl_dual.py map_name:=house1
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

    return [TimerAction(period=12.0, actions=[
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

    # =========================================================================
    # D455 PointCloud2 → /scan_d455
    # D455 is horizontal (zero pitch), so height slice works cleanly.
    # =========================================================================
    pc_to_scan_d455 = Node(
        package="pointcloud_to_laserscan",
        executable="pointcloud_to_laserscan_node",
        name="pointcloud_to_laserscan_d455",
        output="screen",
        parameters=[{
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "target_frame": "base_link",
            "transform_tolerance": 0.1,
            "min_height": 0.2,
            "max_height": 0.8,
            "angle_min": -1.5708,
            "angle_max": 1.5708,
            "angle_increment": 0.0087,
            "scan_time": 0.033,
            "range_min": 0.3,
            "range_max": 4.0,
            "use_inf": True,
            "inf_epsilon": 1.0,
        }],
        remappings=[
            ("cloud_in", "/d455/depth/d455_depth/points"),
            ("scan", "/scan_d455"),
        ],
    )

    # =========================================================================
    # D435i PointCloud2 → /scan_d435i
    # D435i is tilted 60° but pointcloud_to_laserscan handles it via
    # target_frame=base_link + height slice.
    # =========================================================================
    pc_to_scan_d435i = Node(
        package="pointcloud_to_laserscan",
        executable="pointcloud_to_laserscan_node",
        name="pointcloud_to_laserscan_d435i",
        output="screen",
        parameters=[{
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "target_frame": "base_link",
            "transform_tolerance": 0.1,
            "min_height": 0.2,
            "max_height": 0.8,
            "angle_min": -1.5708,
            "angle_max": 1.5708,
            "angle_increment": 0.0087,
            "scan_time": 0.033,
            "range_min": 0.3,
            "range_max": 4.0,
            "use_inf": True,
            "inf_epsilon": 1.0,
        }],
        remappings=[
            ("cloud_in", "/d435i/depth/d435i_depth/points"),
            ("scan", "/scan_d435i"),
        ],
    )

    # =========================================================================
    # MERGE: /scan_d455 + /scan_d435i → /scan
    # Uses ira_laser_tools laserscan_multi_merger
    # =========================================================================
    laser_merger = Node(
        package="ira_laser_tools",
        executable="laserscan_multi_merger",
        name="laserscan_multi_merger",
        output="screen",
        parameters=[{
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "destination_frame": "base_link",
            "cloud_destination_topic": "/merged_cloud",
            "scan_destination_topic": "/scan",
            "laserscan_topics": "/scan_d455 /scan_d435i",
            "angle_min": -3.14159,
            "angle_max": 3.14159,
            "angle_increment": 0.0087,
            "scan_time": 0.033,
            "range_min": 0.3,
            "range_max": 4.0,
        }],
    )

    ld = LaunchDescription()
    ld.add_action(declare_use_sim_time)
    ld.add_action(declare_map_name)
    ld.add_action(pc_to_scan_d455)
    ld.add_action(pc_to_scan_d435i)
    ld.add_action(laser_merger)
    ld.add_action(OpaqueFunction(function=setup_map_path))
    return ld
