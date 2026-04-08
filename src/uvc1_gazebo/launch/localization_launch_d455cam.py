#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node


def generate_launch_description():

    pkg_nav = get_package_share_directory("uvc1_gazebo")
    params_file = os.path.join(pkg_nav, "config", "nav2_params.yaml")
    navigation_launch = os.path.join(pkg_nav, "launch", "navigation_launch.py")

    # Launch arguments
    use_sim_time = LaunchConfiguration("use_sim_time")
    map_db = LaunchConfiguration("map_db")
    params = LaunchConfiguration("params_file")

    declare_sim_time = DeclareLaunchArgument(
        "use_sim_time", default_value="true", description="Use simulation time"
    )
    declare_params = DeclareLaunchArgument(
        "params_file", default_value=params_file, description="Nav2 parameter file"
    )
    declare_map_db = DeclareLaunchArgument(
        "map_db", description="Absolute path to the RTAB-Map database (.db)"
    )

    # RGBD Sync D455
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

    # RGBD Sync D435i
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

    # RTAB-Map Localization
    rtabmap_localization = Node(
        package="rtabmap_slam",
        executable="rtabmap",
        name="rtabmap",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "frame_id": "base_link",
                "odom_frame_id": "odom",
                "map_frame_id": "map",
                "database_path": map_db,  # from launch argument
                "subscribe_rgbd": True,
                "rgbd_cameras": 2,
                "approx_sync": True,
                "subscribe_scan": False,
                "subscribe_rgb": False,
                "subscribe_depth": False,
                "grid": True,  # publish 2D occupancy grid for Nav2
            }
        ],
        remappings=[
            ("rgbd_image0", "/rgbd_image0"),
            ("rgbd_image1", "/rgbd_image1"),
        ],
    )

    # Nav2 launch
    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(navigation_launch),
        launch_arguments={"use_sim_time": use_sim_time, "params_file": params}.items(),
    )

    ld = LaunchDescription()
    ld.add_action(declare_sim_time)
    ld.add_action(declare_params)
    ld.add_action(declare_map_db)
    ld.add_action(rgbd_sync_d455)
    ld.add_action(rgbd_sync_d435i)
    ld.add_action(rtabmap_localization)
    ld.add_action(navigation)

    return ld
