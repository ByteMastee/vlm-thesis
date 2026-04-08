#!/usr/bin/env python3

import os
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def setup_paths(context, *args, **kwargs):
    pkg = get_package_share_directory("uvc1_gazebo")

    # --- Arguments ---
    map_name = LaunchConfiguration("map_name").perform(context)
    use_sim_time = LaunchConfiguration("use_sim_time")

    # --- Map folder path ---
    map_dir = Path(pkg) / "maps" / map_name
    if not map_dir.exists():
        raise RuntimeError(f"Map folder not found: {map_dir}")

    db_path = str(map_dir / f"{map_name}.db")  # only DB is needed for localization

    # --- RTAB-MAP localization node ---
    return [
        Node(
            package="rtabmap_slam",
            executable="rtabmap",
            name="rtabmap",
            output="screen",
            parameters=[
                {
                    "use_sim_time": use_sim_time,
                    "database_path": db_path,
                    # Localization mode (strings, not bools!)
                    "Mem/IncrementalMemory": "false",
                    "Mem/InitWMWithAllNodes": "true",
                    # Multi-camera support
                    "subscribe_rgbd": True,
                    "rgbd_cameras": 2,
                    # TF configuration
                    "frame_id": "base_link",
                    "odom_frame_id": "odom",
                    # Nav2 integration
                    "publish_tf": True,
                    "publish_odom": True,
                    "grid_map": True,
                }
            ],
            remappings=[
                ("rgbd_image0", "/rgbd_image0"),
                ("rgbd_image1", "/rgbd_image1"),
                # allow RViz pose updates
                ("/initialpose", "/rtabmap/initial_pose"),
            ],
        )
    ]


def generate_launch_description():
    # -------------------
    # Environment variable for ROS 2 logging
    # -------------------
    stdout_linebuf_envvar = SetEnvironmentVariable(
        "RCUTILS_LOGGING_BUFFERED_STREAM", "1"
    )

    # -------------------
    # Launch arguments
    # -------------------
    declare_use_sim_time = DeclareLaunchArgument("use_sim_time", default_value="true")
    declare_map_name = DeclareLaunchArgument(
        "map_name",
        default_value="house1",
        description="Folder inside maps/ containing .db, .yaml, .pgm",
    )

    # -------------------
    # Multi-camera RGBD sync nodes
    # -------------------
    rgbd_sync_d455 = Node(
        package="rtabmap_sync",
        executable="rgbd_sync",
        name="rgbd_sync_d455",
        output="screen",
        parameters=[
            {"use_sim_time": LaunchConfiguration("use_sim_time"), "approx_sync": True}
        ],
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
        parameters=[
            {"use_sim_time": LaunchConfiguration("use_sim_time"), "approx_sync": True}
        ],
        remappings=[
            ("rgb/image", "/d435i/rgb/d435i_rgb/image_raw"),
            ("rgb/camera_info", "/d435i/rgb/d435i_rgb/camera_info"),
            ("depth/image", "/d435i/depth/d435i_depth/depth/image_raw"),
            ("rgbd_image", "/rgbd_image1"),
        ],
    )

    # -------------------
    # Build launch description
    # -------------------
    ld = LaunchDescription()
    ld.add_action(stdout_linebuf_envvar)
    ld.add_action(declare_use_sim_time)
    ld.add_action(declare_map_name)
    ld.add_action(rgbd_sync_d455)
    ld.add_action(rgbd_sync_d435i)
    # Dynamic RTAB-MAP node with map database
    ld.add_action(OpaqueFunction(function=setup_paths))

    return ld


"""
Usage:

ros2 launch uvc1_gazebo localization_launch_rtab_multicamera.py map_name:=house1
ros2 launch uvc1_gazebo localization_launch_rtab_multicamera.py map_name:=my_map
"""
