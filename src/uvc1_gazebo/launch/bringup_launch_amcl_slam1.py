#!/usr/bin/env python3
# =============================================================================
#  bringup_launch_amcl_slam.py  —  uvc1 Virofighter Robot
#
#  Standard Nav2 SLAM using slam_toolbox + depthimage_to_laserscan.
#  NO RTAB-Map dependency. Fully Nav2-native stack.
#
#  USAGE:
#    ros2 launch uvc1_gazebo bringup_launch_amcl_slam.py
#    ros2 launch uvc1_gazebo bringup_launch_amcl_slam.py use_sim_time:=false
#
#  WHAT IT LAUNCHES:
#    1. depthimage_to_laserscan (D455 depth → /scan)
#    2. slam_toolbox (online_async — builds map from /scan)
#    3. Nav2 navigation_launch.py (controller, planner, costmaps, behaviors)
#
#  PREREQUISITES:
#    sudo apt install ros-humble-depthimage-to-laserscan ros-humble-slam-toolbox
#
#  SAVING THE MAP:
#    ros2 run nav2_map_server map_saver_cli -f ~/maps/my_map
# =============================================================================

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_uvc1 = get_package_share_directory("uvc1_gazebo")
    pkg_nav2 = get_package_share_directory("nav2_bringup")

    use_sim_time = LaunchConfiguration("use_sim_time")
    params_file = os.path.join(pkg_uvc1, "config", "nav2_params.yaml")

    # --- Arguments ---
    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation clock",
    )

    # --- depthimage_to_laserscan (D455 → /scan) ---
    depth_to_scan = Node(
        package="depthimage_to_laserscan",
        executable="depthimage_to_laserscan_node",
        name="depthimage_to_laserscan",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "scan_height": 10,
                "scan_time": 0.033,
                "range_min": 0.6,
                "range_max": 6.0,
                "output_frame_id": "camera_d455_depth_optical_frame",
            }
        ],
        remappings=[
            ("depth", "/d455/depth/d455_depth/depth/image_raw"),
            ("depth_camera_info", "/d455/depth/d455_depth/camera_info"),
            ("scan", "/scan"),
        ],
    )

    # --- slam_toolbox (online async) ---
    slam_toolbox = Node(
        package="slam_toolbox",
        executable="async_slam_toolbox_node",
        name="slam_toolbox",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "solver_plugin": "solver_plugins::CeresSolver",
                "ceres_linear_solver": "SPARSE_NORMAL_CHOLESKY",
                "ceres_preconditioner": "SCHUR_JACOBI",
                "ceres_trust_strategy": "LEVENBERG_MARQUARDT",
                "ceres_dogleg_type": "TRADITIONAL_DOGLEG",
                "ceres_loss_function": "None",
                # Frame IDs
                "odom_frame": "odom",
                "map_frame": "map",
                "base_frame": "base_link",
                "scan_topic": "/scan",
                # Map update
                "mode": "mapping",
                "map_update_interval": 5.0,
                "resolution": 0.05,
                "max_laser_range": 4.0,    # match depth camera range
                "minimum_travel_distance": 0.3,
                "minimum_travel_heading": 0.3,
                # TF
                "transform_publish_period": 0.02,
                "tf_buffer_duration": 30.0,
                "transform_timeout": 0.2,
                # Performance
                "stack_size_to_use": 40000000,
                "enable_interactive_mode": True,
            }
        ],
    )

    # --- Nav2 navigation stack (delayed to let slam_toolbox publish map→odom) ---
    nav2_navigation = TimerAction(
        period=8.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(pkg_nav2, "launch", "navigation_launch.py")
                ),
                launch_arguments={
                    "use_sim_time": use_sim_time,
                    "params_file": params_file,
                }.items(),
            )
        ],
    )

    # --- Build launch description ---
    ld = LaunchDescription()
    ld.add_action(declare_use_sim_time)
    ld.add_action(depth_to_scan)
    ld.add_action(slam_toolbox)
    ld.add_action(nav2_navigation)

    return ld


"""
QUICK START:

1. Install dependencies:
   sudo apt install ros-humble-depthimage-to-laserscan ros-humble-slam-toolbox

2. Terminal 1 — Start Gazebo + robot:
   ros2 launch uvc1_gazebo uvc1_my_world_nav2_rtabmap.launch.py

3. Terminal 2 — Start SLAM + Nav2:
   ros2 launch uvc1_gazebo bringup_launch_amcl_slam.py

4. Drive around to build map (teleop or Nav2 goals)

5. Save the map:
   ros2 run nav2_map_server map_saver_cli -f ~/maps/house1

6. Next time, use localization mode:
   ros2 launch uvc1_gazebo bringup_launch_amcl.py map_name:=house1
"""
