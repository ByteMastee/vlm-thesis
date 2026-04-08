#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():

    pkg_uvc1 = get_package_share_directory("uvc1_gazebo")
    pkg_nav2 = get_package_share_directory("nav2_bringup")

    use_sim_time = LaunchConfiguration("use_sim_time")
    run_mode = LaunchConfiguration("run_mode")

    declare_use_sim_time = DeclareLaunchArgument("use_sim_time", default_value="true")

    declare_run_mode = DeclareLaunchArgument("run_mode", default_value="slam")

    # RTAB-Map SLAM or Localization
    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_uvc1, "launch", "slam_launch_multicamera.py")
        ),
    )

    localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_uvc1, "launch", "localization_launch_multicamera.py")
        ),
    )

    # Nav2 WITHOUT AMCL + map_server
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_nav2, "launch", "bringup_launch.py")
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "slam": "False",
            "map": "",  # IMPORTANT: no map yaml
            "params_file": os.path.join(pkg_uvc1, "config", "nav2_params_rtab.yaml"),
            "use_composition": "False",
        }.items(),
    )

    ld = LaunchDescription()
    ld.add_action(declare_use_sim_time)
    ld.add_action(declare_run_mode)

    ld.add_action(slam_launch)
    ld.add_action(localization_launch)
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
