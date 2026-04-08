#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_nav = get_package_share_directory("uvc1_gazebo")

    # Launch configuration
    use_sim_time = LaunchConfiguration("use_sim_time")
    params_file = LaunchConfiguration("params_file")

    # Default parameter file inside your package
    default_params = os.path.join(pkg_nav, "config", "nav2_params.yaml")

    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation clock",
    )

    declare_params_file = DeclareLaunchArgument(
        "params_file",
        default_value=default_params,
        description="Nav2 parameters file",
    )

    # --- RGBD Sync nodes for multiple cameras ---
    rgbd_sync_d455 = Node(
        package="rtabmap_sync",
        executable="rgbd_sync",
        name="rgbd_sync_d455",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time, "approx_sync": True}],
        remappings=[
            ("rgb/image", "/d455/rgb/d455_rgb/image_raw"),
            ("rgb/camera_info", "/d455/rgb/d455_rgb/camera_info"),
            ("depth/image", "/d455/depth/d455_depth/depth/image_raw"),
            ("rgbd_image", "/rgbd_image0"),
        ],
    )

    rgbd_sync_d435i = Node(
        package="rtabmap_sync",
        executable="rgbd_sync",
        name="rgbd_sync_d435i",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time, "approx_sync": True}],
        remappings=[
            ("rgb/image", "/d435i/rgb/d435i_rgb/image_raw"),
            ("rgb/camera_info", "/d435i/rgb/d435i_rgb/camera_info"),
            ("depth/image", "/d435i/depth/d435i_depth/depth/image_raw"),
            ("rgbd_image", "/rgbd_image1"),
        ],
    )

    # --- RTAB-Map SLAM node (multicamera support) ---
    rtabmap_node = Node(
        package="rtabmap_slam",
        executable="rtabmap",
        name="rtabmap",
        output="screen",
        parameters=[os.path.join(pkg_nav, "config", "rtabmap_params_multicamera.yaml")],
        remappings=[
            ("rgbd_image0", "/rgbd_image0"),
            ("rgbd_image1", "/rgbd_image1"),
        ],
    )

    # --- LaunchDescription ---
    ld = LaunchDescription()
    ld.add_action(declare_use_sim_time)
    ld.add_action(declare_params_file)
    ld.add_action(rgbd_sync_d455)
    ld.add_action(rgbd_sync_d435i)
    ld.add_action(rtabmap_node)

    return ld
