#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    pkg_uvc1 = get_package_share_directory("uvc1_gazebo")

    # Launch configurations
    use_sim_time = LaunchConfiguration("use_sim_time")
    map_yaml_file = LaunchConfiguration("map")
    params_file = LaunchConfiguration("params_file")
    autostart = LaunchConfiguration("autostart")
    log_level = LaunchConfiguration("log_level")

    # Environment variable for logging
    stdout_linebuf_envvar = SetEnvironmentVariable(
        "RCUTILS_LOGGING_BUFFERED_STREAM", "1"
    )

    # Launch arguments
    declare_use_sim_time = DeclareLaunchArgument("use_sim_time", default_value="true")
    declare_map = DeclareLaunchArgument(
        "map", default_value=os.path.join(pkg_uvc1, "maps", "house.yaml")
    )
    declare_params = DeclareLaunchArgument(
        "params_file",
        default_value=os.path.join(pkg_uvc1, "config", "nav2_params.yaml"),
    )
    declare_autostart = DeclareLaunchArgument("autostart", default_value="true")
    declare_log_level = DeclareLaunchArgument("log_level", default_value="info")

    # Map server node
    map_server = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time, "yaml_filename": map_yaml_file}],
        arguments=["--ros-args", "--log-level", log_level],
    )

    # Lifecycle manager for map server
    lifecycle_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_localization",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "autostart": autostart,
                "node_names": ["map_server"],
            }
        ],
    )

    ld = LaunchDescription()
    ld.add_action(stdout_linebuf_envvar)
    ld.add_action(declare_use_sim_time)
    ld.add_action(declare_map)
    ld.add_action(declare_params)
    ld.add_action(declare_autostart)
    ld.add_action(declare_log_level)
    ld.add_action(map_server)
    ld.add_action(lifecycle_manager)

    return ld
