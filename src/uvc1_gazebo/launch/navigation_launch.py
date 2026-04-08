#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():

    # Nav2 package
    pkg_nav2 = get_package_share_directory("nav2_bringup")

    # My package
    pkg_uvc1 = get_package_share_directory("uvc1_gazebo")

    # Path to Nav2 package navigation_launch.py
    nav2_launch = os.path.join(
        pkg_nav2,
        "launch",
        "navigation_launch.py",
    )

    # Default params file inside my package
    default_params = os.path.join(
        pkg_uvc1,
        "config",
        "nav2_params.yaml",
    )

    use_sim_time = LaunchConfiguration("use_sim_time")
    params_file = LaunchConfiguration("params_file")

    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation clock",
    )

    declare_params = DeclareLaunchArgument(
        "params_file",
        default_value=default_params,
        description="Nav2 parameters file",
    )

    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(nav2_launch),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "params_file": params_file,
            # "autostart": "true",  # ensures lifecycle nodes activate
            # "use_composition": "False",  # disables container issues
        }.items(),
    )

    ld = LaunchDescription()

    ld.add_action(declare_use_sim_time)
    ld.add_action(declare_params)
    ld.add_action(navigation)

    return ld


# Launch Works Like This
# Default
# ros2 launch uvc1_gazebo navigation_launch.py

# Uses:
# uvc1_gazebo/config/nav2_params.yaml

# Override if needed
# ros2 launch uvc1_gazebo navigation_launch.py params_file:=/home/pravin/test.yaml

# ros2 launch uvc1_gazebo navigation_launch.py \
# params_file:=/home/pravin/test.yaml
