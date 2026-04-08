#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch.substitutions import PythonExpression


def generate_launch_description():
    # --- Package paths ---
    pkg_uvc1 = get_package_share_directory("uvc1_gazebo")
    pkg_nav2 = get_package_share_directory("nav2_bringup")

    # --- Launch arguments ---
    use_sim_time = LaunchConfiguration("use_sim_time")
    map_name = LaunchConfiguration("map_name")
    run_mode = LaunchConfiguration("run_mode")  # 'slam' or 'localization'

    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation clock (True/False)",
    )

    declare_map_name = DeclareLaunchArgument(
        "map_name",
        default_value="house1",
        description="Map folder name inside uvc1_gazebo/maps/",
    )

    declare_run_mode = DeclareLaunchArgument(
        "run_mode",
        default_value="slam",
        description="Run mode: 'slam' for building map, 'localization' for using saved map",
    )

    # --- SLAM launch include with condition ---
    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_uvc1, "launch", "slam_launch_multicamera.py")
        ),
        condition=IfCondition(PythonExpression(["'", run_mode, "' == 'slam'"])),
    )

    # --- Localization launch include with condition ---
    localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_uvc1, "launch", "localization_launch_multicamera.py")
        ),
        launch_arguments={
            "map_name": map_name,
            "use_sim_time": use_sim_time,
        }.items(),
        condition=IfCondition(PythonExpression(["'", run_mode, "' == 'localization'"])),
    )

    # --- Nav2 navigation launch include ---
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_nav2, "launch", "navigation_launch.py")
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "params_file": os.path.join(pkg_uvc1, "config", "nav2_params_rtab2.yaml"),
        }.items(),
    )

    # --- LaunchDescription ---
    ld = LaunchDescription()
    ld.add_action(declare_use_sim_time)
    ld.add_action(declare_map_name)
    ld.add_action(declare_run_mode)

    # Add SLAM or Localization depending on run_mode
    ld.add_action(slam_launch)
    ld.add_action(localization_launch)

    # Always launch Nav2
    ld.add_action(nav2_launch)

    return ld


"""
Usage Examples

1. Default SLAM in simulation:

    ros2 launch uvc1_gazebo bringup_launch_multicamera.py

2. Localization with a saved map:

    ros2 launch uvc1_gazebo bringup_launch_multicamera.py run_mode:=localization map_name:=house1

3. Real robot mode:

    ros2 launch uvc1_gazebo bringup_launch_multicamera.py use_sim_time:=false run_mode:=localization map_name:=house1
"""
