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

    map_name = LaunchConfiguration("map_name").perform(context)
    use_sim_time = LaunchConfiguration("use_sim_time")

    map_dir = Path(pkg) / "maps" / map_name

    if not map_dir.exists():
        raise RuntimeError(f"Map folder not found: {map_dir}")

    yaml_path = str(map_dir / f"{map_name}.yaml")
    db_path = str(map_dir / f"{map_name}.db")

    return [
        # -------------------
        # MAP SERVER (Nav2)
        # -------------------
        Node(
            package="nav2_map_server",
            executable="map_server",
            name="map_server",
            output="screen",
            parameters=[{"use_sim_time": use_sim_time, "yaml_filename": yaml_path}],
        ),
        Node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="lifecycle_manager_localization",
            output="screen",
            parameters=[
                {
                    "use_sim_time": use_sim_time,
                    "autostart": True,
                    "node_names": ["map_server"],
                }
            ],
        ),
        # -------------------
        # RTABMAP LOCALIZATION (KEY PART)
        # -------------------
        Node(
            package="rtabmap_slam",
            executable="rtabmap",
            name="rtabmap",
            output="screen",
            parameters=[
                {
                    "use_sim_time": use_sim_time,
                    # LOAD MY SAVED MAP DATABASE
                    "database_path": db_path,
                    # LOCALIZATION MODE
                    "Mem/IncrementalMemory": "false",
                    "Mem/InitWMWithAllNodes": "true",
                    # CRITICAL FIX
                    "RGBD/StartAtOrigin": "true",
                    "Reg/Strategy": "1",  # remove this for default PnP, 1=Icp+Feature2d, 2=Icp, 3=Feature2d
                    # "Icp/MaxCorrespondenceDistance": "0.1", # 10cm is a good start for indoors, increase if you have fast motion or noisy depth
                    # Sensors
                    "subscribe_rgbd": True,
                    "rgbd_cameras": 2,
                    # TF setup
                    "frame_id": "base_link",
                    "Odom/FrameId": "odom",
                    "Map/FrameId": "map",
                }
            ],
            remappings=[
                ("rgbd_image0", "/rgbd_image0"),
                ("rgbd_image1", "/rgbd_image1"),
                # allow RViz pose estimate
                ("/initialpose", "/rtabmap/initial_pose"),
                # prevent map conflict
                ("map", "/slam_map"),
            ],
        ),
    ]


def generate_launch_description():
    # -------------------
    # ENV
    # -------------------
    stdout_linebuf_envvar = SetEnvironmentVariable(
        "RCUTILS_LOGGING_BUFFERED_STREAM", "1"
    )

    # -------------------
    # ARGUMENTS
    # -------------------
    declare_use_sim_time = DeclareLaunchArgument("use_sim_time", default_value="true")

    declare_map_name = DeclareLaunchArgument(
        "map_name", default_value="my_map", description="Map folder name inside maps/"
    )

    # -------------------
    # RGBD SYNC (MULTICAMERA)
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
    # BUILD LAUNCH
    # -------------------
    ld = LaunchDescription()

    ld.add_action(stdout_linebuf_envvar)
    ld.add_action(declare_use_sim_time)
    ld.add_action(declare_map_name)

    ld.add_action(rgbd_sync_d455)
    ld.add_action(rgbd_sync_d435i)

    # dynamic nodes
    ld.add_action(OpaqueFunction(function=setup_paths))

    return ld


"""
ros2 launch uvc1_gazebo localization_launch_multicamera.py map_name:=house1
ros2 launch uvc1_gazebo localization_launch_multicamera.py map_name:=my_map
"""
