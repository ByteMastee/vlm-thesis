#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_nav = get_package_share_directory("uvc1_gazebo")
    params_file = os.path.join(pkg_nav, "config", "nav2_params.yaml")
    navigation_launch = os.path.join(pkg_nav, "launch", "navigation_launch.py")

    use_sim_time = LaunchConfiguration("use_sim_time")
    params = LaunchConfiguration("params_file")

    declare_sim_time = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
    )

    declare_params = DeclareLaunchArgument(
        "params_file",
        default_value=params_file,
    )

    # RGBD Sync for D455
    rgbd_sync = Node(
        package="rtabmap_sync",
        executable="rgbd_sync",
        name="rgbd_sync_d455",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
        remappings=[
            ("rgb/image", "/d455/rgb/d455_rgb/image_raw"),
            ("rgb/camera_info", "/d455/rgb/d455_rgb/camera_info"),
            ("depth/image", "/d455/depth/d455_depth/depth/image_raw"),
            ("rgbd_image", "/rgbd_image_d455"),
        ],
    )

    # RTABMAP SLAM (single camera)
    rtabmap = Node(
        package="rtabmap_slam",
        executable="rtabmap",
        name="rtabmap",
        output="screen",
        # parameters=[params],
        parameters=[
            os.path.join(pkg_nav, "config", "rtabmap_params_singlecamera.yaml")
        ],
        remappings=[
            ("rgbd_image", "/rgbd_image_d455"),
        ],
    )

    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(navigation_launch),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "params_file": params,
        }.items(),
    )

    ld = LaunchDescription()
    ld.add_action(declare_sim_time)
    ld.add_action(declare_params)
    ld.add_action(rgbd_sync)
    ld.add_action(rtabmap)
    ld.add_action(navigation)

    return ld
